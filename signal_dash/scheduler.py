"""Background scheduler for signal_dash.

Wires APScheduler to periodically trigger ingest and classify jobs,
persisting results to SQLite on a configurable interval.

The scheduler runs an ``ingest_and_classify`` job for each configured
source (Reddit, Mastodon) on a fixed interval defined by
``settings.refresh_interval_seconds``.  Each run:

1. Fetches fresh posts from the configured source(s).
2. Filters out posts already present in the database (deduplication).
3. Classifies new posts using either the LLM or stub classifier.
4. Persists raw posts and resulting signals to SQLite.
5. Optionally prunes old signals to bound database growth.

Usage
-----
Call :func:`create_scheduler` to obtain a configured
:class:`~apscheduler.schedulers.asyncio.AsyncIOScheduler`, then call
``.start()`` / ``.shutdown()`` around the application lifetime.  The
FastAPI application lifecycle in ``main.py`` owns start/stop.

All scheduler state is module-level so that the FastAPI lifespan can
reference a single shared instance.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from signal_dash.classifier import classify_posts
from signal_dash.config import Settings, get_settings
from signal_dash.db import (
    delete_old_signals,
    get_connection,
    init_db,
    insert_posts,
    insert_signals,
)
from signal_dash.ingest import fetch_mastodon_posts, fetch_reddit_posts
from signal_dash.models import Source

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level scheduler instance
# ---------------------------------------------------------------------------

#: Shared scheduler instance created by :func:`create_scheduler`.
_scheduler: Optional[AsyncIOScheduler] = None

#: Shared SQLite connection used by scheduler jobs.
_db_conn: Optional[sqlite3.Connection] = None

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

#: Maximum number of signals to retain per keyword (prune threshold).
_MAX_SIGNALS_PER_KEYWORD: int = 2000

#: Job IDs for each source job.
_JOB_ID_REDDIT = "ingest_reddit"
_JOB_ID_MASTODON = "ingest_mastodon"


# ---------------------------------------------------------------------------
# Core job function
# ---------------------------------------------------------------------------


async def ingest_and_classify(
    source: Source,
    settings: Optional[Settings] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, int]:
    """Fetch, classify, and persist posts from a single *source*.

    This coroutine is the primary payload executed by the APScheduler jobs.
    It handles its own exceptions so that a transient network failure does
    not kill the scheduler.

    Parameters
    ----------
    source:
        The social media platform to ingest from
        (:attr:`~signal_dash.models.Source.REDDIT` or
        :attr:`~signal_dash.models.Source.MASTODON`).
    settings:
        Application settings.  When ``None`` the module-level singleton
        from :func:`~signal_dash.config.get_settings` is used.
    conn:
        An open SQLite connection to use for persistence.  When ``None``
        the module-level shared connection is used (falling back to
        opening a new one if the shared connection has not been
        initialised yet).

    Returns
    -------
    dict[str, int]
        A summary dict with keys:

        - ``"fetched"``   — posts retrieved from the source API.
        - ``"inserted"``  — new posts written to ``posts`` table.
        - ``"classified"`` — signals written to ``signals`` table.
    """
    cfg = settings or get_settings()
    db = conn or _get_shared_conn(cfg)

    summary = {"fetched": 0, "inserted": 0, "classified": 0}

    logger.info(
        "[scheduler] Starting ingest job for source=%s keyword='%s'.",
        source.value,
        cfg.keyword,
    )

    try:
        # ------------------------------------------------------------------ #
        # 1. Fetch posts from the source
        # ------------------------------------------------------------------ #
        async with httpx.AsyncClient(timeout=20.0) as client:
            if source == Source.REDDIT:
                posts = await fetch_reddit_posts(
                    keyword=cfg.keyword,
                    subreddit=cfg.reddit_subreddit,
                    limit=cfg.reddit_post_limit,
                    base_url=cfg.reddit_base_url,
                    user_agent=cfg.reddit_user_agent,
                    client=client,
                )
            elif source == Source.MASTODON:
                posts = await fetch_mastodon_posts(
                    keyword=cfg.keyword,
                    base_url=cfg.mastodon_base_url,
                    limit=cfg.mastodon_post_limit,
                    client=client,
                )
            else:
                logger.warning("[scheduler] Unknown source: %s — skipping.", source)
                return summary

        summary["fetched"] = len(posts)
        logger.info(
            "[scheduler] Fetched %d posts from %s.",
            len(posts),
            source.value,
        )

        if not posts:
            return summary

        # ------------------------------------------------------------------ #
        # 2. Persist raw posts (deduplication happens inside insert_posts)
        # ------------------------------------------------------------------ #
        inserted_count = insert_posts(posts, conn=db)
        summary["inserted"] = inserted_count
        logger.info(
            "[scheduler] Inserted %d/%d new posts from %s.",
            inserted_count,
            len(posts),
            source.value,
        )

        # ------------------------------------------------------------------ #
        # 3. Classify ALL fetched posts
        # (insert_signal uses REPLACE so re-classifying duplicates is safe)
        # ------------------------------------------------------------------ #
        signals = await classify_posts(
            posts,
            openai_api_key=cfg.openai_api_key,
            openai_model=cfg.openai_model,
            batch_size=cfg.classifier_batch_size,
        )
        classified_count = insert_signals(signals, conn=db)
        summary["classified"] = classified_count
        logger.info(
            "[scheduler] Classified and stored %d signals from %s.",
            classified_count,
            source.value,
        )

        # ------------------------------------------------------------------ #
        # 4. Prune old signals to bound database growth
        # ------------------------------------------------------------------ #
        pruned = delete_old_signals(
            keep_latest=_MAX_SIGNALS_PER_KEYWORD,
            keyword=cfg.keyword,
            conn=db,
        )
        if pruned > 0:
            logger.info(
                "[scheduler] Pruned %d old signals for keyword='%s'.",
                pruned,
                cfg.keyword,
            )

    except httpx.HTTPStatusError as exc:
        logger.error(
            "[scheduler] HTTP error fetching %s (status %d): %s",
            source.value,
            exc.response.status_code,
            exc,
        )
    except httpx.RequestError as exc:
        logger.error(
            "[scheduler] Network error fetching %s: %s",
            source.value,
            exc,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[scheduler] Unexpected error in ingest job for %s: %s",
            source.value,
            exc,
        )

    return summary


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------


def create_scheduler(
    settings: Optional[Settings] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> AsyncIOScheduler:
    """Build and return a fully-configured :class:`AsyncIOScheduler`.

    One interval job is registered per enabled source.  Each job calls
    :func:`ingest_and_classify` with the appropriate
    :class:`~signal_dash.models.Source` value.

    The scheduler is **not** started by this function — call
    ``scheduler.start()`` explicitly (e.g. inside the FastAPI lifespan
    context manager).

    Parameters
    ----------
    settings:
        Application settings to use when configuring jobs.  Defaults to
        the module-level singleton from :func:`~signal_dash.config.get_settings`.
    conn:
        SQLite connection to pass through to each job.  When ``None`` jobs
        will use the module-level shared connection.

    Returns
    -------
    AsyncIOScheduler
        Configured scheduler ready to be started.
    """
    global _scheduler, _db_conn

    cfg = settings or get_settings()

    # Initialise the shared database connection if not already done.
    if conn is not None:
        _db_conn = conn
    elif _db_conn is None:
        _db_conn = get_connection(cfg.database_url)

    init_db(_db_conn)

    scheduler = AsyncIOScheduler(
        job_defaults={
            "coalesce": True,       # skip missed runs (don't pile up)
            "max_instances": 1,     # only one concurrent instance per job
            "misfire_grace_time": 60,  # seconds of tolerance for late runs
        }
    )

    interval_seconds = cfg.refresh_interval_seconds
    sources: list[Source] = [Source(s) for s in cfg.sources]

    for source in sources:
        job_id = (
            _JOB_ID_REDDIT if source == Source.REDDIT else _JOB_ID_MASTODON
        )
        scheduler.add_job(
            ingest_and_classify,
            trigger=IntervalTrigger(seconds=interval_seconds),
            id=job_id,
            name=f"ingest_{source.value}",
            kwargs={"source": source, "settings": cfg, "conn": _db_conn},
            replace_existing=True,
        )
        logger.info(
            "[scheduler] Registered job '%s' every %d seconds.",
            job_id,
            interval_seconds,
        )

    _scheduler = scheduler
    return scheduler


# ---------------------------------------------------------------------------
# Convenience start / shutdown helpers
# ---------------------------------------------------------------------------


async def start_scheduler(
    settings: Optional[Settings] = None,
    conn: Optional[sqlite3.Connection] = None,
    run_immediately: bool = True,
) -> AsyncIOScheduler:
    """Create, (optionally) fire immediately, and start the scheduler.

    This is a convenience wrapper for use in the FastAPI lifespan or
    standalone scripts.  It calls :func:`create_scheduler` and then
    starts the scheduler.

    Parameters
    ----------
    settings:
        Application settings.  Defaults to the module-level singleton.
    conn:
        SQLite connection for persistence.  Defaults to the shared conn.
    run_immediately:
        When ``True``, each registered job is triggered once immediately
        after the scheduler starts (so the dashboard has data right away
        rather than waiting for the first interval to elapse).

    Returns
    -------
    AsyncIOScheduler
        The running scheduler instance.
    """
    scheduler = create_scheduler(settings=settings, conn=conn)
    scheduler.start()
    logger.info("[scheduler] APScheduler started.")

    if run_immediately:
        cfg = settings or get_settings()
        sources: list[Source] = [Source(s) for s in cfg.sources]
        for source in sources:
            job_id = (
                _JOB_ID_REDDIT if source == Source.REDDIT else _JOB_ID_MASTODON
            )
            job = scheduler.get_job(job_id)
            if job is not None:
                # Trigger the job to run now without waiting for the interval.
                scheduler.modify_job(job_id, next_run_time=_now_utc())
                logger.info(
                    "[scheduler] Triggered immediate run for job '%s'.", job_id
                )

    return scheduler


async def shutdown_scheduler() -> None:
    """Stop the module-level scheduler and close the shared DB connection.

    Safe to call even if the scheduler was never started.
    """
    global _scheduler, _db_conn

    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("[scheduler] APScheduler shut down.")
        _scheduler = None

    if _db_conn is not None:
        try:
            _db_conn.close()
        except Exception:  # noqa: BLE001
            pass
        _db_conn = None
        logger.info("[scheduler] Database connection closed.")


def get_scheduler() -> Optional[AsyncIOScheduler]:
    """Return the current module-level scheduler instance, or ``None``.

    Returns
    -------
    AsyncIOScheduler or None
        The scheduler if it has been created, otherwise ``None``.
    """
    return _scheduler


def get_shared_conn() -> Optional[sqlite3.Connection]:
    """Return the shared SQLite connection used by the scheduler.

    Returns
    -------
    sqlite3.Connection or None
        The shared connection if initialised, otherwise ``None``.
    """
    return _db_conn


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_shared_conn(cfg: Settings) -> sqlite3.Connection:
    """Return (or lazily create) the module-level shared DB connection.

    Parameters
    ----------
    cfg:
        Application settings used to determine the database path.

    Returns
    -------
    sqlite3.Connection
        An initialised, shared SQLite connection.
    """
    global _db_conn
    if _db_conn is None:
        _db_conn = get_connection(cfg.database_url)
        init_db(_db_conn)
    return _db_conn


def _now_utc():
    """Return the current UTC datetime (used to trigger immediate job runs).

    Returns
    -------
    datetime.datetime
        Current UTC-aware datetime.
    """
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc)
