"""Unit tests for signal_dash.scheduler.

Tests cover job registration, the ingest_and_classify coroutine,
and the start/shutdown lifecycle helpers using mocked dependencies
to avoid real network calls or APScheduler timing behaviour.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from signal_dash.config import Settings
from signal_dash.db import (
    count_posts,
    count_signals,
    get_connection,
    get_post,
    get_signal,
    init_db,
)
from signal_dash.models import Post, Sentiment, Signal, Source
from signal_dash.scheduler import (
    _JOB_ID_MASTODON,
    _JOB_ID_REDDIT,
    create_scheduler,
    get_scheduler,
    get_shared_conn,
    ingest_and_classify,
    shutdown_scheduler,
    start_scheduler,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def memory_conn() -> sqlite3.Connection:
    """Provide a fresh in-memory SQLite connection with schema."""
    conn = get_connection(":memory:")
    init_db(conn)
    return conn


@pytest.fixture()
def stub_settings() -> Settings:
    """Return a Settings instance configured for stub mode."""
    return Settings(
        keyword="pytest",
        sources=["reddit", "mastodon"],
        refresh_interval_seconds=60,
        openai_api_key="",
        database_url=":memory:",
        reddit_subreddit="all",
        reddit_post_limit=5,
        mastodon_post_limit=5,
    )


@pytest.fixture(autouse=True)
async def reset_scheduler_state():
    """Ensure module-level scheduler state is clean before and after each test."""
    # Shutdown any lingering scheduler from previous tests
    await shutdown_scheduler()
    yield
    await shutdown_scheduler()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_post(
    source_id: str = "t3_test",
    source: Source = Source.REDDIT,
    keyword: str = "pytest",
) -> Post:
    """Build a minimal Post for testing."""
    return Post(
        source_id=source_id,
        source=source,
        body="This is a test post about Python.",
        url=f"https://reddit.com/r/python/{source_id}",
        keyword=keyword,
        fetched_at=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


def _make_signal(
    source_id: str = "t3_test",
    source: Source = Source.REDDIT,
    keyword: str = "pytest",
) -> Signal:
    """Build a minimal Signal for testing."""
    return Signal(
        source_id=source_id,
        source=source,
        keyword=keyword,
        url=f"https://reddit.com/r/python/{source_id}",
        sentiment_score=0.5,
        sentiment_label=Sentiment.POSITIVE,
        topics=["python"],
        signal_strength=0.7,
    )


# ---------------------------------------------------------------------------
# create_scheduler
# ---------------------------------------------------------------------------


class TestCreateScheduler:
    """Tests for the create_scheduler() factory."""

    def test_returns_async_scheduler(self, stub_settings, memory_conn) -> None:
        """create_scheduler() should return an AsyncIOScheduler."""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        scheduler = create_scheduler(settings=stub_settings, conn=memory_conn)
        assert isinstance(scheduler, AsyncIOScheduler)
        scheduler.shutdown(wait=False) if scheduler.running else None

    def test_registers_reddit_job(self, stub_settings, memory_conn) -> None:
        """A Reddit job should be registered when 'reddit' is in sources."""
        scheduler = create_scheduler(settings=stub_settings, conn=memory_conn)
        job = scheduler.get_job(_JOB_ID_REDDIT)
        assert job is not None

    def test_registers_mastodon_job(self, stub_settings, memory_conn) -> None:
        """A Mastodon job should be registered when 'mastodon' is in sources."""
        scheduler = create_scheduler(settings=stub_settings, conn=memory_conn)
        job = scheduler.get_job(_JOB_ID_MASTODON)
        assert job is not None

    def test_only_reddit_job_when_single_source(self, memory_conn) -> None:
        """Only a Reddit job should exist when sources=['reddit']."""
        settings = Settings(
            keyword="test",
            sources=["reddit"],
            refresh_interval_seconds=60,
            openai_api_key="",
            database_url=":memory:",
        )
        scheduler = create_scheduler(settings=settings, conn=memory_conn)
        assert scheduler.get_job(_JOB_ID_REDDIT) is not None
        assert scheduler.get_job(_JOB_ID_MASTODON) is None

    def test_only_mastodon_job_when_single_source(self, memory_conn) -> None:
        """Only a Mastodon job should exist when sources=['mastodon']."""
        settings = Settings(
            keyword="test",
            sources=["mastodon"],
            refresh_interval_seconds=60,
            openai_api_key="",
            database_url=":memory:",
        )
        scheduler = create_scheduler(settings=settings, conn=memory_conn)
        assert scheduler.get_job(_JOB_ID_MASTODON) is not None
        assert scheduler.get_job(_JOB_ID_REDDIT) is None

    def test_scheduler_not_started(self, stub_settings, memory_conn) -> None:
        """create_scheduler() should NOT start the scheduler automatically."""
        scheduler = create_scheduler(settings=stub_settings, conn=memory_conn)
        assert not scheduler.running

    def test_get_scheduler_returns_instance(self, stub_settings, memory_conn) -> None:
        """get_scheduler() should return the created scheduler instance."""
        scheduler = create_scheduler(settings=stub_settings, conn=memory_conn)
        assert get_scheduler() is scheduler

    def test_initialises_db_schema(self, stub_settings, memory_conn) -> None:
        """create_scheduler() should ensure the DB schema exists."""
        create_scheduler(settings=stub_settings, conn=memory_conn)
        # Schema exists if posts table is present
        row = memory_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='posts'"
        ).fetchone()
        assert row is not None


# ---------------------------------------------------------------------------
# ingest_and_classify — Reddit
# ---------------------------------------------------------------------------


class TestIngestAndClassifyReddit:
    """Tests for ingest_and_classify() with Source.REDDIT."""

    async def test_inserts_posts(self, stub_settings, memory_conn) -> None:
        """Fetched posts should be inserted into the database."""
        posts = [_make_post(source_id=f"t3_{i}") for i in range(3)]

        with patch(
            "signal_dash.scheduler.fetch_reddit_posts",
            new=AsyncMock(return_value=posts),
        ):
            summary = await ingest_and_classify(
                Source.REDDIT, settings=stub_settings, conn=memory_conn
            )

        assert summary["fetched"] == 3
        assert summary["inserted"] == 3
        assert count_posts(conn=memory_conn) == 3

    async def test_inserts_signals(self, stub_settings, memory_conn) -> None:
        """Classified signals should be inserted into the database."""
        posts = [_make_post(source_id=f"t3_{i}") for i in range(2)]
        signals = [_make_signal(source_id=f"t3_{i}") for i in range(2)]

        with patch(
            "signal_dash.scheduler.fetch_reddit_posts",
            new=AsyncMock(return_value=posts),
        ), patch(
            "signal_dash.scheduler.classify_posts",
            new=AsyncMock(return_value=signals),
        ):
            summary = await ingest_and_classify(
                Source.REDDIT, settings=stub_settings, conn=memory_conn
            )

        assert summary["classified"] == 2
        assert count_signals(conn=memory_conn) == 2

    async def test_returns_summary_dict(self, stub_settings, memory_conn) -> None:
        """ingest_and_classify() should return a dict with correct keys."""
        posts = [_make_post()]

        with patch(
            "signal_dash.scheduler.fetch_reddit_posts",
            new=AsyncMock(return_value=posts),
        ):
            summary = await ingest_and_classify(
                Source.REDDIT, settings=stub_settings, conn=memory_conn
            )

        assert set(summary.keys()) == {"fetched", "inserted", "classified"}

    async def test_empty_fetch_returns_zero_counts(self, stub_settings, memory_conn) -> None:
        """When no posts are fetched, all summary counts should be 0."""
        with patch(
            "signal_dash.scheduler.fetch_reddit_posts",
            new=AsyncMock(return_value=[]),
        ):
            summary = await ingest_and_classify(
                Source.REDDIT, settings=stub_settings, conn=memory_conn
            )

        assert summary["fetched"] == 0
        assert summary["inserted"] == 0
        assert summary["classified"] == 0

    async def test_deduplication_skips_existing(self, stub_settings, memory_conn) -> None:
        """Posts already in the DB should not be counted as inserted twice."""
        posts = [_make_post(source_id="t3_dup")]

        with patch(
            "signal_dash.scheduler.fetch_reddit_posts",
            new=AsyncMock(return_value=posts),
        ):
            # First run — inserts the post
            await ingest_and_classify(
                Source.REDDIT, settings=stub_settings, conn=memory_conn
            )
            # Second run — same post, should be deduplicated
            summary = await ingest_and_classify(
                Source.REDDIT, settings=stub_settings, conn=memory_conn
            )

        # Only 1 row in posts table
        assert count_posts(conn=memory_conn) == 1
        # Second run inserted 0 new posts
        assert summary["inserted"] == 0

    async def test_http_error_handled_gracefully(self, stub_settings, memory_conn) -> None:
        """An HTTP error during fetch should be caught and not re-raised."""
        import httpx

        with patch(
            "signal_dash.scheduler.fetch_reddit_posts",
            new=AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Too Many Requests",
                    request=MagicMock(),
                    response=MagicMock(status_code=429),
                )
            ),
        ):
            # Should not raise
            summary = await ingest_and_classify(
                Source.REDDIT, settings=stub_settings, conn=memory_conn
            )

        assert summary["fetched"] == 0

    async def test_network_error_handled_gracefully(self, stub_settings, memory_conn) -> None:
        """A network error during fetch should be caught and not re-raised."""
        import httpx

        with patch(
            "signal_dash.scheduler.fetch_reddit_posts",
            new=AsyncMock(side_effect=httpx.ConnectError("Connection refused")),
        ):
            summary = await ingest_and_classify(
                Source.REDDIT, settings=stub_settings, conn=memory_conn
            )

        assert summary["fetched"] == 0

    async def test_unexpected_error_handled_gracefully(
        self, stub_settings, memory_conn
    ) -> None:
        """Any unexpected error should be caught and not re-raised."""
        with patch(
            "signal_dash.scheduler.fetch_reddit_posts",
            new=AsyncMock(side_effect=RuntimeError("something broke")),
        ):
            summary = await ingest_and_classify(
                Source.REDDIT, settings=stub_settings, conn=memory_conn
            )

        assert summary["fetched"] == 0


# ---------------------------------------------------------------------------
# ingest_and_classify — Mastodon
# ---------------------------------------------------------------------------


class TestIngestAndClassifyMastodon:
    """Tests for ingest_and_classify() with Source.MASTODON."""

    async def test_inserts_mastodon_posts(self, stub_settings, memory_conn) -> None:
        """Mastodon posts should be inserted into the database."""
        posts = [
            Post(
                source_id=f"masto_{i}",
                source=Source.MASTODON,
                body="Mastodon test post.",
                url=f"https://mastodon.social/@alice/masto_{i}",
                keyword="pytest",
                fetched_at=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            )
            for i in range(2)
        ]

        with patch(
            "signal_dash.scheduler.fetch_mastodon_posts",
            new=AsyncMock(return_value=posts),
        ):
            summary = await ingest_and_classify(
                Source.MASTODON, settings=stub_settings, conn=memory_conn
            )

        assert summary["fetched"] == 2
        assert count_posts(conn=memory_conn) == 2

    async def test_mastodon_signals_stored(self, stub_settings, memory_conn) -> None:
        """Signals from Mastodon posts should be stored in the signals table."""
        posts = [
            Post(
                source_id="masto_99",
                source=Source.MASTODON,
                body="Great discussion happening here.",
                url="https://mastodon.social/@bob/masto_99",
                keyword="pytest",
            )
        ]

        with patch(
            "signal_dash.scheduler.fetch_mastodon_posts",
            new=AsyncMock(return_value=posts),
        ):
            summary = await ingest_and_classify(
                Source.MASTODON, settings=stub_settings, conn=memory_conn
            )

        assert summary["classified"] >= 1
        assert count_signals(conn=memory_conn) >= 1

    async def test_http_error_handled(self, stub_settings, memory_conn) -> None:
        """HTTP errors from Mastodon should be caught gracefully."""
        import httpx

        with patch(
            "signal_dash.scheduler.fetch_mastodon_posts",
            new=AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Forbidden",
                    request=MagicMock(),
                    response=MagicMock(status_code=403),
                )
            ),
        ):
            summary = await ingest_and_classify(
                Source.MASTODON, settings=stub_settings, conn=memory_conn
            )

        assert summary["fetched"] == 0


# ---------------------------------------------------------------------------
# ingest_and_classify — unknown source
# ---------------------------------------------------------------------------


class TestIngestAndClassifyUnknownSource:
    """Tests for edge-cases around the source parameter."""

    async def test_unknown_source_returns_zero_summary(self, stub_settings, memory_conn) -> None:
        """Passing an unrecognised source should return a zero summary."""
        # Patch Source to have an extra member by creating a fake object
        fake_source = MagicMock(spec=Source)
        fake_source.value = "twitter"  # not a real Source member

        # Monkeypatch the equality checks used in the job
        with patch.object(Source, "REDDIT", Source.REDDIT), \
             patch.object(Source, "MASTODON", Source.MASTODON):
            summary = await ingest_and_classify(
                fake_source, settings=stub_settings, conn=memory_conn  # type: ignore
            )

        assert summary["fetched"] == 0


# ---------------------------------------------------------------------------
# start_scheduler and shutdown_scheduler
# ---------------------------------------------------------------------------


class TestStartAndShutdownScheduler:
    """Tests for start_scheduler() and shutdown_scheduler() lifecycle."""

    async def test_start_scheduler_starts_scheduler(
        self, stub_settings, memory_conn
    ) -> None:
        """start_scheduler() should return a running scheduler."""
        scheduler = await start_scheduler(
            settings=stub_settings,
            conn=memory_conn,
            run_immediately=False,
        )
        assert scheduler.running
        scheduler.shutdown(wait=False)

    async def test_shutdown_stops_running_scheduler(
        self, stub_settings, memory_conn
    ) -> None:
        """shutdown_scheduler() should stop a running scheduler."""
        scheduler = await start_scheduler(
            settings=stub_settings,
            conn=memory_conn,
            run_immediately=False,
        )
        assert scheduler.running
        await shutdown_scheduler()
        assert not scheduler.running

    async def test_shutdown_idempotent_when_not_started(self) -> None:
        """Calling shutdown_scheduler() before starting should not raise."""
        # Should not raise even if no scheduler was ever created
        await shutdown_scheduler()  # idempotent

    async def test_get_scheduler_returns_none_before_create(self) -> None:
        """get_scheduler() returns None before create_scheduler() is called."""
        # After reset_scheduler_state fixture runs, state is clean
        result = get_scheduler()
        assert result is None

    async def test_get_shared_conn_none_before_create(self) -> None:
        """get_shared_conn() should return None before scheduler is created."""
        result = get_shared_conn()
        assert result is None

    async def test_get_shared_conn_after_create(
        self, stub_settings, memory_conn
    ) -> None:
        """get_shared_conn() should return the connection after create_scheduler()."""
        create_scheduler(settings=stub_settings, conn=memory_conn)
        assert get_shared_conn() is memory_conn


# ---------------------------------------------------------------------------
# Integration: ingest job uses settings keyword
# ---------------------------------------------------------------------------


class TestIngestJobUsesSettingsKeyword:
    """Verify that the keyword from Settings is propagated to posts and signals."""

    async def test_posts_use_settings_keyword(self, memory_conn) -> None:
        """Posts inserted by the job should carry the keyword from settings."""
        settings = Settings(
            keyword="signal_dash_test",
            sources=["reddit"],
            refresh_interval_seconds=60,
            openai_api_key="",
            database_url=":memory:",
        )
        posts = [
            Post(
                source_id="t3_kw1",
                source=Source.REDDIT,
                body="test content",
                url="https://reddit.com/r/test/t3_kw1",
                keyword="signal_dash_test",
                fetched_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            )
        ]

        with patch(
            "signal_dash.scheduler.fetch_reddit_posts",
            new=AsyncMock(return_value=posts),
        ):
            await ingest_and_classify(
                Source.REDDIT, settings=settings, conn=memory_conn
            )

        from signal_dash.db import get_posts
        results = get_posts(keyword="signal_dash_test", conn=memory_conn)
        assert len(results) == 1
        assert results[0].keyword == "signal_dash_test"


# ---------------------------------------------------------------------------
# Stub classifier integration (end-to-end without mocking classify_posts)
# ---------------------------------------------------------------------------


class TestStubClassifierIntegration:
    """End-to-end tests using real stub classifier (no OpenAI key)."""

    async def test_signals_created_with_stub(self, memory_conn) -> None:
        """Full pipeline with stub classifier should produce valid signals."""
        settings = Settings(
            keyword="integration",
            sources=["reddit"],
            refresh_interval_seconds=60,
            openai_api_key="",  # stub mode
            database_url=":memory:",
        )
        posts = [
            Post(
                source_id="t3_integ_1",
                source=Source.REDDIT,
                title="Great Python library",
                body="I love using this library. It is amazing and fast.",
                url="https://reddit.com/r/python/t3_integ_1",
                keyword="integration",
                score=100,
                fetched_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            ),
            Post(
                source_id="t3_integ_2",
                source=Source.REDDIT,
                title="Terrible bug report",
                body="This crashed and burned. Awful experience.",
                url="https://reddit.com/r/python/t3_integ_2",
                keyword="integration",
                score=5,
                fetched_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            ),
        ]

        with patch(
            "signal_dash.scheduler.fetch_reddit_posts",
            new=AsyncMock(return_value=posts),
        ):
            summary = await ingest_and_classify(
                Source.REDDIT, settings=settings, conn=memory_conn
            )

        assert summary["fetched"] == 2
        assert summary["inserted"] == 2
        assert summary["classified"] == 2

        sig1 = get_signal("t3_integ_1", Source.REDDIT, conn=memory_conn)
        sig2 = get_signal("t3_integ_2", Source.REDDIT, conn=memory_conn)

        assert sig1 is not None
        assert sig2 is not None

        # Positive post should have positive score
        assert sig1.sentiment_score > 0
        assert sig1.sentiment_label == Sentiment.POSITIVE

        # Negative post should have negative score
        assert sig2.sentiment_score < 0
        assert sig2.sentiment_label == Sentiment.NEGATIVE

        # Signal strengths should be valid
        assert 0.0 <= sig1.signal_strength <= 1.0
        assert 0.0 <= sig2.signal_strength <= 1.0
