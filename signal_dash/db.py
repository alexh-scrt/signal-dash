"""SQLite persistence layer for signal_dash.

Provides schema initialisation and CRUD helpers for storing raw
:class:`~signal_dash.models.Post` objects and classified
:class:`~signal_dash.models.Signal` objects with deduplication by
source post ID.

All public functions accept an optional ``conn`` parameter so callers
(including tests) can supply an existing connection — enabling the use
of in-memory databases for isolation.  When ``conn`` is ``None`` the
module opens a new connection to the path returned by
:func:`~signal_dash.config.get_settings`.

Schema notes
------------
- ``posts``   — raw ingested posts; ``(source_id, source)`` is the
  unique deduplication key.
- ``signals`` — classified signals; ``(source_id, source)`` is also
  the unique deduplication key (one signal per post).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator, Optional

from signal_dash.models import Post, Sentiment, Signal, Source

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_POSTS_TABLE = """
CREATE TABLE IF NOT EXISTS posts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    author      TEXT,
    title       TEXT,
    body        TEXT    NOT NULL DEFAULT '',
    url         TEXT    NOT NULL,
    score       INTEGER NOT NULL DEFAULT 0,
    keyword     TEXT    NOT NULL,
    fetched_at  TEXT    NOT NULL,
    UNIQUE (source_id, source)
);
"""

_CREATE_SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS signals (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id        TEXT    NOT NULL,
    source           TEXT    NOT NULL,
    keyword          TEXT    NOT NULL,
    title            TEXT,
    body             TEXT    NOT NULL DEFAULT '',
    url              TEXT    NOT NULL,
    author           TEXT,
    post_score       INTEGER NOT NULL DEFAULT 0,
    sentiment_score  REAL    NOT NULL,
    sentiment_label  TEXT    NOT NULL,
    topics           TEXT    NOT NULL DEFAULT '[]',
    signal_strength  REAL    NOT NULL,
    classified_at    TEXT    NOT NULL,
    UNIQUE (source_id, source)
);
"""

_CREATE_POSTS_IDX_KEYWORD = """
CREATE INDEX IF NOT EXISTS idx_posts_keyword
    ON posts (keyword);
"""

_CREATE_POSTS_IDX_FETCHED = """
CREATE INDEX IF NOT EXISTS idx_posts_fetched_at
    ON posts (fetched_at);
"""

_CREATE_SIGNALS_IDX_KEYWORD = """
CREATE INDEX IF NOT EXISTS idx_signals_keyword
    ON signals (keyword);
"""

_CREATE_SIGNALS_IDX_CLASSIFIED = """
CREATE INDEX IF NOT EXISTS idx_signals_classified_at
    ON signals (classified_at);
"""

_CREATE_SIGNALS_IDX_SENTIMENT = """
CREATE INDEX IF NOT EXISTS idx_signals_sentiment_label
    ON signals (sentiment_label);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def get_connection(database_url: Optional[str] = None) -> sqlite3.Connection:
    """Open and return a new SQLite connection.

    Parameters
    ----------
    database_url:
        Path to the SQLite database file, or ``':memory:'`` for an in-memory
        database.  When ``None`` the value from
        :func:`~signal_dash.config.get_settings` is used.

    Returns
    -------
    sqlite3.Connection
        A connection with ``row_factory`` set to :class:`sqlite3.Row` so
        that rows can be accessed by column name.
    """
    if database_url is None:
        from signal_dash.config import get_settings
        database_url = get_settings().database_url

    conn = sqlite3.connect(database_url, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def _transaction(conn: sqlite3.Connection) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that commits on success or rolls back on error.

    Parameters
    ----------
    conn:
        An open SQLite connection.

    Yields
    ------
    sqlite3.Connection
        The same connection, ready for use within the transaction.
    """
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


def init_db(conn: Optional[sqlite3.Connection] = None) -> sqlite3.Connection:
    """Create database tables and indexes if they do not already exist.

    This function is idempotent — calling it multiple times is safe.

    Parameters
    ----------
    conn:
        An existing SQLite connection to use.  When ``None`` a new
        connection is opened using :func:`get_connection`.

    Returns
    -------
    sqlite3.Connection
        The connection that was used (either the supplied one or a newly
        created one).
    """
    if conn is None:
        conn = get_connection()

    with _transaction(conn):
        conn.execute(_CREATE_POSTS_TABLE)
        conn.execute(_CREATE_SIGNALS_TABLE)
        conn.execute(_CREATE_POSTS_IDX_KEYWORD)
        conn.execute(_CREATE_POSTS_IDX_FETCHED)
        conn.execute(_CREATE_SIGNALS_IDX_KEYWORD)
        conn.execute(_CREATE_SIGNALS_IDX_CLASSIFIED)
        conn.execute(_CREATE_SIGNALS_IDX_SENTIMENT)

    logger.debug("Database schema initialised.")
    return conn


# ---------------------------------------------------------------------------
# Post CRUD
# ---------------------------------------------------------------------------


def insert_post(
    post: Post,
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """Insert a post, ignoring the row if it already exists.

    Deduplication is performed on the ``(source_id, source)`` unique
    constraint using ``INSERT OR IGNORE``.

    Parameters
    ----------
    post:
        The :class:`~signal_dash.models.Post` to persist.
    conn:
        An existing SQLite connection.  When ``None`` a new connection is
        opened.

    Returns
    -------
    bool
        ``True`` if the row was inserted, ``False`` if it was a duplicate
        and was ignored.
    """
    _conn = conn or get_connection()
    sql = """
        INSERT OR IGNORE INTO posts
            (source_id, source, author, title, body, url, score, keyword, fetched_at)
        VALUES
            (:source_id, :source, :author, :title, :body, :url, :score, :keyword, :fetched_at)
    """
    params = {
        "source_id": post.source_id,
        "source": post.source.value,
        "author": post.author,
        "title": post.title,
        "body": post.body,
        "url": post.url,
        "score": post.score,
        "keyword": post.keyword,
        "fetched_at": post.fetched_at.isoformat(),
    }
    with _transaction(_conn):
        cursor = _conn.execute(sql, params)
        inserted = cursor.rowcount > 0

    if inserted:
        logger.debug("Inserted post %s from %s.", post.source_id, post.source.value)
    else:
        logger.debug(
            "Skipped duplicate post %s from %s.", post.source_id, post.source.value
        )
    return inserted


def insert_posts(
    posts: list[Post],
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """Insert multiple posts in a single transaction, ignoring duplicates.

    Parameters
    ----------
    posts:
        A list of :class:`~signal_dash.models.Post` objects to persist.
    conn:
        An existing SQLite connection.  When ``None`` a new connection is
        opened.

    Returns
    -------
    int
        The number of rows actually inserted (duplicates are not counted).
    """
    _conn = conn or get_connection()
    sql = """
        INSERT OR IGNORE INTO posts
            (source_id, source, author, title, body, url, score, keyword, fetched_at)
        VALUES
            (:source_id, :source, :author, :title, :body, :url, :score, :keyword, :fetched_at)
    """
    params_list = [
        {
            "source_id": p.source_id,
            "source": p.source.value,
            "author": p.author,
            "title": p.title,
            "body": p.body,
            "url": p.url,
            "score": p.score,
            "keyword": p.keyword,
            "fetched_at": p.fetched_at.isoformat(),
        }
        for p in posts
    ]
    inserted_count = 0
    with _transaction(_conn):
        for params in params_list:
            cursor = _conn.execute(sql, params)
            inserted_count += cursor.rowcount

    logger.debug("Bulk-inserted %d/%d posts.", inserted_count, len(posts))
    return inserted_count


def get_post(
    source_id: str,
    source: Source,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[Post]:
    """Retrieve a single post by its source identifier.

    Parameters
    ----------
    source_id:
        The platform-native post identifier.
    source:
        The social media platform the post originated from.
    conn:
        An existing SQLite connection.  When ``None`` a new connection is
        opened.

    Returns
    -------
    Post or None
        The matching :class:`~signal_dash.models.Post`, or ``None`` if no
        such post exists.
    """
    _conn = conn or get_connection()
    sql = """
        SELECT source_id, source, author, title, body, url, score, keyword, fetched_at
        FROM posts
        WHERE source_id = ? AND source = ?
    """
    row = _conn.execute(sql, (source_id, source.value)).fetchone()
    if row is None:
        return None
    return _row_to_post(row)


def get_posts(
    keyword: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    conn: Optional[sqlite3.Connection] = None,
) -> list[Post]:
    """Retrieve posts, optionally filtered by keyword.

    Parameters
    ----------
    keyword:
        When supplied, only posts matching this keyword are returned.
        ``None`` returns posts for all keywords.
    limit:
        Maximum number of posts to return.  Defaults to 100.
    offset:
        Number of posts to skip (for pagination).  Defaults to 0.
    conn:
        An existing SQLite connection.  When ``None`` a new connection is
        opened.

    Returns
    -------
    list[Post]
        Posts ordered by ``fetched_at`` descending (newest first).
    """
    _conn = conn or get_connection()
    if keyword is not None:
        sql = """
            SELECT source_id, source, author, title, body, url, score, keyword, fetched_at
            FROM posts
            WHERE keyword = ?
            ORDER BY fetched_at DESC
            LIMIT ? OFFSET ?
        """
        rows = _conn.execute(sql, (keyword, limit, offset)).fetchall()
    else:
        sql = """
            SELECT source_id, source, author, title, body, url, score, keyword, fetched_at
            FROM posts
            ORDER BY fetched_at DESC
            LIMIT ? OFFSET ?
        """
        rows = _conn.execute(sql, (limit, offset)).fetchall()
    return [_row_to_post(row) for row in rows]


def count_posts(
    keyword: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """Return the total number of stored posts, optionally filtered by keyword.

    Parameters
    ----------
    keyword:
        When supplied, only posts matching this keyword are counted.
    conn:
        An existing SQLite connection.  When ``None`` a new connection is
        opened.

    Returns
    -------
    int
        The number of matching posts.
    """
    _conn = conn or get_connection()
    if keyword is not None:
        row = _conn.execute(
            "SELECT COUNT(*) FROM posts WHERE keyword = ?", (keyword,)
        ).fetchone()
    else:
        row = _conn.execute("SELECT COUNT(*) FROM posts").fetchone()
    return int(row[0])


# ---------------------------------------------------------------------------
# Signal CRUD
# ---------------------------------------------------------------------------


def insert_signal(
    signal: Signal,
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """Insert a signal, replacing the existing row if one already exists.

    Unlike posts (which use ``INSERT OR IGNORE``), signals use
    ``INSERT OR REPLACE`` so that re-classification of the same post
    updates the stored signal with fresh analysis.

    Parameters
    ----------
    signal:
        The :class:`~signal_dash.models.Signal` to persist.
    conn:
        An existing SQLite connection.  When ``None`` a new connection is
        opened.

    Returns
    -------
    bool
        Always ``True`` (the row is always written, either inserted or
        replaced).
    """
    _conn = conn or get_connection()
    sql = """
        INSERT OR REPLACE INTO signals
            (source_id, source, keyword, title, body, url, author,
             post_score, sentiment_score, sentiment_label, topics,
             signal_strength, classified_at)
        VALUES
            (:source_id, :source, :keyword, :title, :body, :url, :author,
             :post_score, :sentiment_score, :sentiment_label, :topics,
             :signal_strength, :classified_at)
    """
    params = {
        "source_id": signal.source_id,
        "source": signal.source.value,
        "keyword": signal.keyword,
        "title": signal.title,
        "body": signal.body,
        "url": signal.url,
        "author": signal.author,
        "post_score": signal.post_score,
        "sentiment_score": signal.sentiment_score,
        "sentiment_label": signal.sentiment_label.value,
        "topics": json.dumps(signal.topics),
        "signal_strength": signal.signal_strength,
        "classified_at": signal.classified_at.isoformat(),
    }
    with _transaction(_conn):
        _conn.execute(sql, params)

    logger.debug(
        "Upserted signal %s from %s.", signal.source_id, signal.source.value
    )
    return True


def insert_signals(
    signals: list[Signal],
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """Insert or replace multiple signals in a single transaction.

    Parameters
    ----------
    signals:
        A list of :class:`~signal_dash.models.Signal` objects to persist.
    conn:
        An existing SQLite connection.  When ``None`` a new connection is
        opened.

    Returns
    -------
    int
        The number of rows written (each row is always written).
    """
    _conn = conn or get_connection()
    sql = """
        INSERT OR REPLACE INTO signals
            (source_id, source, keyword, title, body, url, author,
             post_score, sentiment_score, sentiment_label, topics,
             signal_strength, classified_at)
        VALUES
            (:source_id, :source, :keyword, :title, :body, :url, :author,
             :post_score, :sentiment_score, :sentiment_label, :topics,
             :signal_strength, :classified_at)
    """
    params_list = [
        {
            "source_id": s.source_id,
            "source": s.source.value,
            "keyword": s.keyword,
            "title": s.title,
            "body": s.body,
            "url": s.url,
            "author": s.author,
            "post_score": s.post_score,
            "sentiment_score": s.sentiment_score,
            "sentiment_label": s.sentiment_label.value,
            "topics": json.dumps(s.topics),
            "signal_strength": s.signal_strength,
            "classified_at": s.classified_at.isoformat(),
        }
        for s in signals
    ]
    with _transaction(_conn):
        for params in params_list:
            _conn.execute(sql, params)

    logger.debug("Bulk-upserted %d signals.", len(signals))
    return len(signals)


def get_signal(
    source_id: str,
    source: Source,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[Signal]:
    """Retrieve a single signal by its source identifier.

    Parameters
    ----------
    source_id:
        The platform-native post identifier.
    source:
        The social media platform the signal originated from.
    conn:
        An existing SQLite connection.  When ``None`` a new connection is
        opened.

    Returns
    -------
    Signal or None
        The matching :class:`~signal_dash.models.Signal`, or ``None``.
    """
    _conn = conn or get_connection()
    sql = """
        SELECT source_id, source, keyword, title, body, url, author,
               post_score, sentiment_score, sentiment_label, topics,
               signal_strength, classified_at
        FROM signals
        WHERE source_id = ? AND source = ?
    """
    row = _conn.execute(sql, (source_id, source.value)).fetchone()
    if row is None:
        return None
    return _row_to_signal(row)


def get_signals(
    keyword: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    conn: Optional[sqlite3.Connection] = None,
) -> list[Signal]:
    """Retrieve signals, optionally filtered by keyword.

    Parameters
    ----------
    keyword:
        When supplied, only signals matching this keyword are returned.
    limit:
        Maximum number of signals to return.  Defaults to 100.
    offset:
        Number of signals to skip (for pagination).  Defaults to 0.
    conn:
        An existing SQLite connection.  When ``None`` a new connection is
        opened.

    Returns
    -------
    list[Signal]
        Signals ordered by ``classified_at`` descending (newest first).
    """
    _conn = conn or get_connection()
    if keyword is not None:
        sql = """
            SELECT source_id, source, keyword, title, body, url, author,
                   post_score, sentiment_score, sentiment_label, topics,
                   signal_strength, classified_at
            FROM signals
            WHERE keyword = ?
            ORDER BY classified_at DESC
            LIMIT ? OFFSET ?
        """
        rows = _conn.execute(sql, (keyword, limit, offset)).fetchall()
    else:
        sql = """
            SELECT source_id, source, keyword, title, body, url, author,
                   post_score, sentiment_score, sentiment_label, topics,
                   signal_strength, classified_at
            FROM signals
            ORDER BY classified_at DESC
            LIMIT ? OFFSET ?
        """
        rows = _conn.execute(sql, (limit, offset)).fetchall()
    return [_row_to_signal(row) for row in rows]


def get_signals_by_sentiment(
    sentiment: Sentiment,
    keyword: Optional[str] = None,
    limit: int = 50,
    conn: Optional[sqlite3.Connection] = None,
) -> list[Signal]:
    """Retrieve signals filtered by sentiment label.

    Parameters
    ----------
    sentiment:
        The :class:`~signal_dash.models.Sentiment` label to filter by.
    keyword:
        When supplied, further filter by keyword.
    limit:
        Maximum number of signals to return.  Defaults to 50.
    conn:
        An existing SQLite connection.  When ``None`` a new connection is
        opened.

    Returns
    -------
    list[Signal]
        Matching signals ordered by ``signal_strength`` descending.
    """
    _conn = conn or get_connection()
    if keyword is not None:
        sql = """
            SELECT source_id, source, keyword, title, body, url, author,
                   post_score, sentiment_score, sentiment_label, topics,
                   signal_strength, classified_at
            FROM signals
            WHERE sentiment_label = ? AND keyword = ?
            ORDER BY signal_strength DESC
            LIMIT ?
        """
        rows = _conn.execute(sql, (sentiment.value, keyword, limit)).fetchall()
    else:
        sql = """
            SELECT source_id, source, keyword, title, body, url, author,
                   post_score, sentiment_score, sentiment_label, topics,
                   signal_strength, classified_at
            FROM signals
            WHERE sentiment_label = ?
            ORDER BY signal_strength DESC
            LIMIT ?
        """
        rows = _conn.execute(sql, (sentiment.value, limit)).fetchall()
    return [_row_to_signal(row) for row in rows]


def get_sentiment_timeseries(
    keyword: Optional[str] = None,
    limit: int = 200,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict]:
    """Return a time-ordered list of ``(classified_at, sentiment_score)`` pairs.

    This is used to populate the Chart.js time-series chart on the dashboard.

    Parameters
    ----------
    keyword:
        When supplied, only data for this keyword is returned.
    limit:
        Maximum number of data points to return.  Defaults to 200.
    conn:
        An existing SQLite connection.  When ``None`` a new connection is
        opened.

    Returns
    -------
    list[dict]
        Each dict has keys ``'classified_at'`` (ISO 8601 string) and
        ``'sentiment_score'`` (float), ordered oldest-first.
    """
    _conn = conn or get_connection()
    if keyword is not None:
        sql = """
            SELECT classified_at, sentiment_score
            FROM signals
            WHERE keyword = ?
            ORDER BY classified_at DESC
            LIMIT ?
        """
        rows = _conn.execute(sql, (keyword, limit)).fetchall()
    else:
        sql = """
            SELECT classified_at, sentiment_score
            FROM signals
            ORDER BY classified_at DESC
            LIMIT ?
        """
        rows = _conn.execute(sql, (limit,)).fetchall()
    # Reverse so the result is oldest-first for Chart.js
    result = [
        {"classified_at": row["classified_at"], "sentiment_score": row["sentiment_score"]}
        for row in reversed(rows)
    ]
    return result


def get_top_topics(
    keyword: Optional[str] = None,
    limit: int = 10,
    conn: Optional[sqlite3.Connection] = None,
) -> list[dict]:
    """Return the most frequently occurring topic tags.

    Because topics are stored as a JSON array in a single column, this
    function loads the recent signals in Python and counts tags there
    rather than attempting JSON manipulation in SQLite.

    Parameters
    ----------
    keyword:
        When supplied, only signals for this keyword are considered.
    limit:
        Maximum number of topics to return.  Defaults to 10.
    conn:
        An existing SQLite connection.  When ``None`` a new connection is
        opened.

    Returns
    -------
    list[dict]
        Each dict has keys ``'topic'`` (str) and ``'count'`` (int),
        ordered by count descending.
    """
    signals = get_signals(keyword=keyword, limit=500, conn=conn)
    counts: dict[str, int] = {}
    for sig in signals:
        for tag in sig.topics:
            counts[tag] = counts.get(tag, 0) + 1
    sorted_topics = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [{"topic": tag, "count": cnt} for tag, cnt in sorted_topics[:limit]]


def count_signals(
    keyword: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """Return the total number of stored signals, optionally filtered by keyword.

    Parameters
    ----------
    keyword:
        When supplied, only signals matching this keyword are counted.
    conn:
        An existing SQLite connection.  When ``None`` a new connection is
        opened.

    Returns
    -------
    int
        The number of matching signals.
    """
    _conn = conn or get_connection()
    if keyword is not None:
        row = _conn.execute(
            "SELECT COUNT(*) FROM signals WHERE keyword = ?", (keyword,)
        ).fetchone()
    else:
        row = _conn.execute("SELECT COUNT(*) FROM signals").fetchone()
    return int(row[0])


def delete_old_signals(
    keep_latest: int = 1000,
    keyword: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """Delete signals beyond the most recent *keep_latest* rows.

    Useful for bounding database growth on long-running deployments.

    Parameters
    ----------
    keep_latest:
        Number of most-recent signals to retain per keyword (or globally
        when no keyword is supplied).  Defaults to 1000.
    keyword:
        When supplied, pruning is limited to this keyword's signals.
    conn:
        An existing SQLite connection.  When ``None`` a new connection is
        opened.

    Returns
    -------
    int
        Number of rows deleted.
    """
    _conn = conn or get_connection()
    if keyword is not None:
        sql = """
            DELETE FROM signals
            WHERE (source_id, source) NOT IN (
                SELECT source_id, source
                FROM signals
                WHERE keyword = ?
                ORDER BY classified_at DESC
                LIMIT ?
            )
            AND keyword = ?
        """
        with _transaction(_conn):
            cursor = _conn.execute(sql, (keyword, keep_latest, keyword))
    else:
        sql = """
            DELETE FROM signals
            WHERE (source_id, source) NOT IN (
                SELECT source_id, source
                FROM signals
                ORDER BY classified_at DESC
                LIMIT ?
            )
        """
        with _transaction(_conn):
            cursor = _conn.execute(sql, (keep_latest,))
    deleted = cursor.rowcount
    logger.debug("Pruned %d old signal rows.", deleted)
    return deleted


# ---------------------------------------------------------------------------
# Private row-conversion helpers
# ---------------------------------------------------------------------------


def _row_to_post(row: sqlite3.Row) -> Post:
    """Convert a raw SQLite row to a :class:`~signal_dash.models.Post`.

    Parameters
    ----------
    row:
        A :class:`sqlite3.Row` from the ``posts`` table.

    Returns
    -------
    Post
        The reconstructed Pydantic model.
    """
    return Post(
        source_id=row["source_id"],
        source=Source(row["source"]),
        author=row["author"],
        title=row["title"],
        body=row["body"] or "",
        url=row["url"],
        score=row["score"],
        keyword=row["keyword"],
        fetched_at=row["fetched_at"],
    )


def _row_to_signal(row: sqlite3.Row) -> Signal:
    """Convert a raw SQLite row to a :class:`~signal_dash.models.Signal`.

    Parameters
    ----------
    row:
        A :class:`sqlite3.Row` from the ``signals`` table.

    Returns
    -------
    Signal
        The reconstructed Pydantic model.
    """
    raw_topics = row["topics"]
    try:
        topics = json.loads(raw_topics) if raw_topics else []
    except (json.JSONDecodeError, TypeError):
        # Fallback: treat as comma-separated string (legacy rows)
        topics = [t.strip() for t in str(raw_topics).split(",") if t.strip()]

    return Signal(
        source_id=row["source_id"],
        source=Source(row["source"]),
        keyword=row["keyword"],
        title=row["title"],
        body=row["body"] or "",
        url=row["url"],
        author=row["author"],
        post_score=row["post_score"],
        sentiment_score=row["sentiment_score"],
        sentiment_label=Sentiment(row["sentiment_label"]),
        topics=topics,
        signal_strength=row["signal_strength"],
        classified_at=row["classified_at"],
    )
