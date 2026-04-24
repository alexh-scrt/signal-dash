"""Unit tests for signal_dash.db — schema creation, insert, and query helpers.

All tests use an in-memory SQLite database to ensure full isolation and
zero filesystem side-effects.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Generator

import pytest

from signal_dash.db import (
    count_posts,
    count_signals,
    delete_old_signals,
    get_connection,
    get_post,
    get_posts,
    get_sentiment_timeseries,
    get_signal,
    get_signals,
    get_signals_by_sentiment,
    get_top_topics,
    init_db,
    insert_post,
    insert_posts,
    insert_signal,
    insert_signals,
)
from signal_dash.models import Post, Sentiment, Signal, Source


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def conn():
    """Provide a fresh in-memory SQLite connection for each test."""
    connection = get_connection(":memory:")
    init_db(connection)
    yield connection
    connection.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_post(
    source_id: str = "t3_abc123",
    source: Source = Source.REDDIT,
    keyword: str = "python",
    **kwargs,
) -> Post:
    """Return a minimal valid Post."""
    defaults = {
        "source_id": source_id,
        "source": source,
        "body": "Hello from Reddit!",
        "url": f"https://reddit.com/r/python/{source_id}",
        "keyword": keyword,
        "fetched_at": datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
    }
    defaults.update(kwargs)
    return Post(**defaults)


def _make_signal(
    source_id: str = "t3_abc123",
    source: Source = Source.REDDIT,
    keyword: str = "python",
    sentiment_score: float = 0.5,
    signal_strength: float = 0.7,
    topics: list[str] | None = None,
    **kwargs,
) -> Signal:
    """Return a minimal valid Signal."""
    defaults = {
        "source_id": source_id,
        "source": source,
        "keyword": keyword,
        "url": f"https://reddit.com/r/python/{source_id}",
        "sentiment_score": sentiment_score,
        "sentiment_label": Sentiment.POSITIVE,
        "topics": topics if topics is not None else ["python", "coding"],
        "signal_strength": signal_strength,
        "classified_at": datetime(2024, 6, 1, 12, 5, 0, tzinfo=timezone.utc),
    }
    defaults.update(kwargs)
    return Signal(**defaults)


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


class TestInitDb:
    """Tests for the init_db() function."""

    def test_creates_posts_table(self, conn) -> None:
        """The posts table should exist after init_db()."""
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='posts'"
        ).fetchone()
        assert row is not None

    def test_creates_signals_table(self, conn) -> None:
        """The signals table should exist after init_db()."""
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='signals'"
        ).fetchone()
        assert row is not None

    def test_creates_posts_indexes(self, conn) -> None:
        """Index on posts.keyword should be created."""
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_posts_keyword'"
        ).fetchone()
        assert row is not None

    def test_creates_signals_indexes(self, conn) -> None:
        """Index on signals.keyword should be created."""
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_signals_keyword'"
        ).fetchone()
        assert row is not None

    def test_idempotent(self, conn) -> None:
        """Calling init_db() twice should not raise an error."""
        init_db(conn)  # second call on already-initialised db
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='posts'"
        ).fetchone()
        assert row is not None

    def test_returns_connection(self) -> None:
        """init_db() should return the connection it was given."""
        c = get_connection(":memory:")
        result = init_db(c)
        assert result is c
        c.close()


# ---------------------------------------------------------------------------
# Post insert
# ---------------------------------------------------------------------------


class TestInsertPost:
    """Tests for insert_post()."""

    def test_insert_single_post(self, conn) -> None:
        """Inserting a valid post should succeed and return True."""
        post = _make_post()
        result = insert_post(post, conn=conn)
        assert result is True

    def test_duplicate_post_ignored(self, conn) -> None:
        """Inserting the same post twice should return False on the second call."""
        post = _make_post()
        insert_post(post, conn=conn)
        result = insert_post(post, conn=conn)
        assert result is False

    def test_duplicate_not_counted_twice(self, conn) -> None:
        """Duplicate posts should not increase the count."""
        post = _make_post()
        insert_post(post, conn=conn)
        insert_post(post, conn=conn)
        assert count_posts(conn=conn) == 1

    def test_different_source_same_id_allowed(self, conn) -> None:
        """The same source_id from a different source is NOT a duplicate."""
        reddit_post = _make_post(source_id="12345", source=Source.REDDIT)
        mastodon_post = _make_post(source_id="12345", source=Source.MASTODON)
        r1 = insert_post(reddit_post, conn=conn)
        r2 = insert_post(mastodon_post, conn=conn)
        assert r1 is True
        assert r2 is True
        assert count_posts(conn=conn) == 2

    def test_post_with_title_stored(self, conn) -> None:
        """A post with a title should persist the title correctly."""
        post = _make_post(title="My Awesome Post")
        insert_post(post, conn=conn)
        fetched = get_post("t3_abc123", Source.REDDIT, conn=conn)
        assert fetched is not None
        assert fetched.title == "My Awesome Post"

    def test_post_with_author_stored(self, conn) -> None:
        """Author field should be stored and retrieved."""
        post = _make_post(author="alice")
        insert_post(post, conn=conn)
        fetched = get_post("t3_abc123", Source.REDDIT, conn=conn)
        assert fetched is not None
        assert fetched.author == "alice"

    def test_post_score_stored(self, conn) -> None:
        """Platform engagement score should persist correctly."""
        post = _make_post(score=42)
        insert_post(post, conn=conn)
        fetched = get_post("t3_abc123", Source.REDDIT, conn=conn)
        assert fetched is not None
        assert fetched.score == 42


# ---------------------------------------------------------------------------
# Bulk post insert
# ---------------------------------------------------------------------------


class TestInsertPosts:
    """Tests for insert_posts()."""

    def test_bulk_insert_returns_count(self, conn) -> None:
        """insert_posts() should return the number of new rows inserted."""
        posts = [_make_post(source_id=f"t3_{i}") for i in range(5)]
        inserted = insert_posts(posts, conn=conn)
        assert inserted == 5

    def test_bulk_insert_deduplicates(self, conn) -> None:
        """Duplicate posts in the batch should be skipped."""
        posts = [_make_post(source_id="t3_dup")] * 3
        inserted = insert_posts(posts, conn=conn)
        assert inserted == 1
        assert count_posts(conn=conn) == 1

    def test_bulk_insert_empty_list(self, conn) -> None:
        """Inserting an empty list should return 0."""
        inserted = insert_posts([], conn=conn)
        assert inserted == 0

    def test_bulk_insert_persists_all(self, conn) -> None:
        """All distinct posts in the batch should be retrievable."""
        posts = [_make_post(source_id=f"t3_{i}") for i in range(3)]
        insert_posts(posts, conn=conn)
        assert count_posts(conn=conn) == 3


# ---------------------------------------------------------------------------
# Post retrieval
# ---------------------------------------------------------------------------


class TestGetPost:
    """Tests for get_post()."""

    def test_returns_none_when_absent(self, conn) -> None:
        """get_post() should return None for an unknown source_id."""
        result = get_post("nonexistent", Source.REDDIT, conn=conn)
        assert result is None

    def test_returns_correct_post(self, conn) -> None:
        """get_post() should return the post matching the given ID and source."""
        post = _make_post(source_id="t3_xyz", keyword="fastapi")
        insert_post(post, conn=conn)
        fetched = get_post("t3_xyz", Source.REDDIT, conn=conn)
        assert fetched is not None
        assert fetched.source_id == "t3_xyz"
        assert fetched.keyword == "fastapi"

    def test_source_discriminates(self, conn) -> None:
        """get_post() with a different source should return None."""
        post = _make_post(source_id="42", source=Source.REDDIT)
        insert_post(post, conn=conn)
        result = get_post("42", Source.MASTODON, conn=conn)
        assert result is None

    def test_roundtrip_preserves_fields(self, conn) -> None:
        """All Post fields should survive a write-read roundtrip."""
        post = _make_post(
            source_id="t3_rt",
            source=Source.MASTODON,
            author="bob",
            title="Test title",
            body="Test body.",
            url="https://mastodon.social/@bob/1",
            score=99,
            keyword="mastodon",
        )
        insert_post(post, conn=conn)
        fetched = get_post("t3_rt", Source.MASTODON, conn=conn)
        assert fetched is not None
        assert fetched.source == Source.MASTODON
        assert fetched.author == "bob"
        assert fetched.title == "Test title"
        assert fetched.body == "Test body."
        assert fetched.score == 99
        assert fetched.keyword == "mastodon"


class TestGetPosts:
    """Tests for get_posts()."""

    def test_returns_all_when_no_filter(self, conn) -> None:
        """Without a keyword filter, all posts are returned."""
        insert_posts(
            [_make_post(source_id=f"t3_{i}", keyword="py") for i in range(3)],
            conn=conn,
        )
        results = get_posts(conn=conn)
        assert len(results) == 3

    def test_filters_by_keyword(self, conn) -> None:
        """Only posts matching the given keyword should be returned."""
        insert_post(_make_post(source_id="t3_1", keyword="python"), conn=conn)
        insert_post(_make_post(source_id="t3_2", keyword="rust"), conn=conn)
        results = get_posts(keyword="python", conn=conn)
        assert len(results) == 1
        assert results[0].keyword == "python"

    def test_ordered_newest_first(self, conn) -> None:
        """Posts should be returned newest fetched_at first."""
        old = _make_post(
            source_id="t3_old",
            fetched_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        new = _make_post(
            source_id="t3_new",
            fetched_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        insert_post(old, conn=conn)
        insert_post(new, conn=conn)
        results = get_posts(conn=conn)
        assert results[0].source_id == "t3_new"
        assert results[1].source_id == "t3_old"

    def test_limit(self, conn) -> None:
        """The limit parameter should cap the number of returned posts."""
        insert_posts(
            [_make_post(source_id=f"t3_{i}") for i in range(10)], conn=conn
        )
        results = get_posts(limit=3, conn=conn)
        assert len(results) == 3

    def test_offset(self, conn) -> None:
        """The offset parameter should skip the specified number of posts."""
        insert_posts(
            [_make_post(source_id=f"t3_{i}") for i in range(5)], conn=conn
        )
        all_results = get_posts(limit=5, conn=conn)
        offset_results = get_posts(limit=5, offset=2, conn=conn)
        assert len(offset_results) == 3
        assert offset_results[0].source_id == all_results[2].source_id

    def test_empty_db_returns_empty_list(self, conn) -> None:
        """An empty database should return an empty list."""
        assert get_posts(conn=conn) == []


class TestCountPosts:
    """Tests for count_posts()."""

    def test_zero_on_empty(self, conn) -> None:
        """Count should be 0 on an empty database."""
        assert count_posts(conn=conn) == 0

    def test_counts_all(self, conn) -> None:
        """count_posts() without a filter should count all posts."""
        insert_posts([_make_post(source_id=f"t3_{i}") for i in range(4)], conn=conn)
        assert count_posts(conn=conn) == 4

    def test_counts_by_keyword(self, conn) -> None:
        """count_posts() with a keyword should count only matching posts."""
        insert_post(_make_post(source_id="a", keyword="python"), conn=conn)
        insert_post(_make_post(source_id="b", keyword="rust"), conn=conn)
        assert count_posts(keyword="python", conn=conn) == 1
        assert count_posts(keyword="rust", conn=conn) == 1
        assert count_posts(keyword="go", conn=conn) == 0


# ---------------------------------------------------------------------------
# Signal insert
# ---------------------------------------------------------------------------


class TestInsertSignal:
    """Tests for insert_signal()."""

    def test_insert_single_signal(self, conn) -> None:
        """Inserting a valid signal should succeed and return True."""
        sig = _make_signal()
        result = insert_signal(sig, conn=conn)
        assert result is True
        assert count_signals(conn=conn) == 1

    def test_insert_replaces_on_duplicate(self, conn) -> None:
        """Re-inserting the same signal should replace (upsert) the existing row."""
        sig_v1 = _make_signal(sentiment_score=0.5, signal_strength=0.5)
        sig_v2 = _make_signal(sentiment_score=0.9, signal_strength=0.9)
        insert_signal(sig_v1, conn=conn)
        insert_signal(sig_v2, conn=conn)
        # Should still be only 1 row
        assert count_signals(conn=conn) == 1
        fetched = get_signal("t3_abc123", Source.REDDIT, conn=conn)
        assert fetched is not None
        assert fetched.sentiment_score == 0.9

    def test_topics_stored_as_json(self, conn) -> None:
        """Topics should survive a roundtrip as a JSON array."""
        sig = _make_signal(topics=["ai", "llm", "python"])
        insert_signal(sig, conn=conn)
        fetched = get_signal("t3_abc123", Source.REDDIT, conn=conn)
        assert fetched is not None
        assert fetched.topics == ["ai", "llm", "python"]

    def test_empty_topics_stored(self, conn) -> None:
        """A signal with no topics should store and retrieve an empty list."""
        sig = _make_signal(topics=[])
        insert_signal(sig, conn=conn)
        fetched = get_signal("t3_abc123", Source.REDDIT, conn=conn)
        assert fetched is not None
        assert fetched.topics == []

    def test_negative_sentiment_stored(self, conn) -> None:
        """Negative sentiment_score and NEGATIVE label should persist correctly."""
        sig = _make_signal(sentiment_score=-0.8)
        insert_signal(sig, conn=conn)
        fetched = get_signal("t3_abc123", Source.REDDIT, conn=conn)
        assert fetched is not None
        assert fetched.sentiment_score == pytest.approx(-0.8)
        assert fetched.sentiment_label == Sentiment.NEGATIVE


# ---------------------------------------------------------------------------
# Bulk signal insert
# ---------------------------------------------------------------------------


class TestInsertSignals:
    """Tests for insert_signals()."""

    def test_bulk_insert_returns_count(self, conn) -> None:
        """insert_signals() should return the number of rows written."""
        signals = [_make_signal(source_id=f"t3_{i}") for i in range(4)]
        count = insert_signals(signals, conn=conn)
        assert count == 4

    def test_bulk_insert_upserts(self, conn) -> None:
        """Re-inserting signals should replace existing rows."""
        signals_v1 = [_make_signal(source_id=f"t3_{i}", sentiment_score=0.1) for i in range(3)]
        signals_v2 = [_make_signal(source_id=f"t3_{i}", sentiment_score=0.9) for i in range(3)]
        insert_signals(signals_v1, conn=conn)
        insert_signals(signals_v2, conn=conn)
        assert count_signals(conn=conn) == 3
        fetched = get_signal("t3_0", Source.REDDIT, conn=conn)
        assert fetched is not None
        assert fetched.sentiment_score == pytest.approx(0.9)

    def test_bulk_insert_empty(self, conn) -> None:
        """Inserting an empty list should return 0."""
        assert insert_signals([], conn=conn) == 0


# ---------------------------------------------------------------------------
# Signal retrieval
# ---------------------------------------------------------------------------


class TestGetSignal:
    """Tests for get_signal()."""

    def test_returns_none_when_absent(self, conn) -> None:
        """get_signal() should return None for an unknown source_id."""
        assert get_signal("not_there", Source.REDDIT, conn=conn) is None

    def test_returns_correct_signal(self, conn) -> None:
        """get_signal() should return the correct signal."""
        sig = _make_signal(source_id="t3_xyz", keyword="fastapi")
        insert_signal(sig, conn=conn)
        fetched = get_signal("t3_xyz", Source.REDDIT, conn=conn)
        assert fetched is not None
        assert fetched.keyword == "fastapi"

    def test_roundtrip_preserves_all_fields(self, conn) -> None:
        """All Signal fields should survive a write-read roundtrip."""
        sig = Signal(
            source_id="masto_99",
            source=Source.MASTODON,
            keyword="climate",
            title="Climate update",
            body="Things are changing.",
            url="https://mastodon.social/@news/99",
            author="journalist",
            post_score=10,
            sentiment_score=-0.6,
            sentiment_label=Sentiment.NEGATIVE,
            topics=["climate", "environment"],
            signal_strength=0.85,
            classified_at=datetime(2024, 5, 1, 8, 0, 0, tzinfo=timezone.utc),
        )
        insert_signal(sig, conn=conn)
        fetched = get_signal("masto_99", Source.MASTODON, conn=conn)
        assert fetched is not None
        assert fetched.source == Source.MASTODON
        assert fetched.keyword == "climate"
        assert fetched.title == "Climate update"
        assert fetched.body == "Things are changing."
        assert fetched.author == "journalist"
        assert fetched.post_score == 10
        assert fetched.sentiment_score == pytest.approx(-0.6)
        assert fetched.sentiment_label == Sentiment.NEGATIVE
        assert fetched.topics == ["climate", "environment"]
        assert fetched.signal_strength == pytest.approx(0.85)


class TestGetSignals:
    """Tests for get_signals()."""

    def test_returns_all_without_filter(self, conn) -> None:
        """Without a keyword filter, all signals are returned."""
        sigs = [_make_signal(source_id=f"t3_{i}") for i in range(5)]
        insert_signals(sigs, conn=conn)
        results = get_signals(conn=conn)
        assert len(results) == 5

    def test_filters_by_keyword(self, conn) -> None:
        """Only signals matching the keyword should be returned."""
        insert_signal(_make_signal(source_id="a", keyword="python"), conn=conn)
        insert_signal(_make_signal(source_id="b", keyword="rust"), conn=conn)
        results = get_signals(keyword="python", conn=conn)
        assert len(results) == 1
        assert results[0].keyword == "python"

    def test_ordered_newest_first(self, conn) -> None:
        """Signals should be returned newest classified_at first."""
        old = _make_signal(
            source_id="old",
            classified_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        new = _make_signal(
            source_id="new",
            classified_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        insert_signal(old, conn=conn)
        insert_signal(new, conn=conn)
        results = get_signals(conn=conn)
        assert results[0].source_id == "new"
        assert results[1].source_id == "old"

    def test_limit_and_offset(self, conn) -> None:
        """limit and offset should page results correctly."""
        sigs = [_make_signal(source_id=f"t3_{i}") for i in range(6)]
        insert_signals(sigs, conn=conn)
        page1 = get_signals(limit=3, offset=0, conn=conn)
        page2 = get_signals(limit=3, offset=3, conn=conn)
        assert len(page1) == 3
        assert len(page2) == 3
        # No overlap
        ids_p1 = {s.source_id for s in page1}
        ids_p2 = {s.source_id for s in page2}
        assert ids_p1.isdisjoint(ids_p2)

    def test_empty_db_returns_empty(self, conn) -> None:
        """An empty signals table should return an empty list."""
        assert get_signals(conn=conn) == []


class TestGetSignalsBySentiment:
    """Tests for get_signals_by_sentiment()."""

    def test_filters_positive(self, conn) -> None:
        """Only signals with POSITIVE label should be returned."""
        pos = _make_signal(source_id="pos", sentiment_score=0.8)
        neg = _make_signal(source_id="neg", sentiment_score=-0.8)
        insert_signal(pos, conn=conn)
        insert_signal(neg, conn=conn)
        results = get_signals_by_sentiment(Sentiment.POSITIVE, conn=conn)
        assert all(s.sentiment_label == Sentiment.POSITIVE for s in results)
        assert len(results) == 1

    def test_filters_negative(self, conn) -> None:
        """Only signals with NEGATIVE label should be returned."""
        pos = _make_signal(source_id="pos", sentiment_score=0.8)
        neg = _make_signal(source_id="neg", sentiment_score=-0.8)
        insert_signal(pos, conn=conn)
        insert_signal(neg, conn=conn)
        results = get_signals_by_sentiment(Sentiment.NEGATIVE, conn=conn)
        assert len(results) == 1
        assert results[0].source_id == "neg"

    def test_filters_neutral(self, conn) -> None:
        """Only signals with NEUTRAL label should be returned."""
        neu = _make_signal(source_id="neu", sentiment_score=0.0)
        pos = _make_signal(source_id="pos", sentiment_score=0.9)
        insert_signal(neu, conn=conn)
        insert_signal(pos, conn=conn)
        results = get_signals_by_sentiment(Sentiment.NEUTRAL, conn=conn)
        assert len(results) == 1
        assert results[0].source_id == "neu"

    def test_filters_by_keyword_and_sentiment(self, conn) -> None:
        """Both sentiment and keyword filters should be applied together."""
        s1 = _make_signal(source_id="a", keyword="python", sentiment_score=0.8)
        s2 = _make_signal(source_id="b", keyword="rust", sentiment_score=0.7)
        insert_signal(s1, conn=conn)
        insert_signal(s2, conn=conn)
        results = get_signals_by_sentiment(
            Sentiment.POSITIVE, keyword="python", conn=conn
        )
        assert len(results) == 1
        assert results[0].keyword == "python"

    def test_returns_empty_when_none_match(self, conn) -> None:
        """Should return empty list when no signals match."""
        results = get_signals_by_sentiment(Sentiment.NEGATIVE, conn=conn)
        assert results == []


# ---------------------------------------------------------------------------
# Sentiment time-series
# ---------------------------------------------------------------------------


class TestGetSentimentTimeseries:
    """Tests for get_sentiment_timeseries()."""

    def test_returns_correct_structure(self, conn) -> None:
        """Each item should have 'classified_at' and 'sentiment_score' keys."""
        sig = _make_signal()
        insert_signal(sig, conn=conn)
        results = get_sentiment_timeseries(conn=conn)
        assert len(results) == 1
        assert "classified_at" in results[0]
        assert "sentiment_score" in results[0]

    def test_ordered_oldest_first(self, conn) -> None:
        """Time-series should be ordered oldest classified_at first."""
        old = _make_signal(
            source_id="old",
            classified_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        new = _make_signal(
            source_id="new",
            classified_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        insert_signal(old, conn=conn)
        insert_signal(new, conn=conn)
        results = get_sentiment_timeseries(conn=conn)
        assert "2024-01" in results[0]["classified_at"]
        assert "2024-06" in results[1]["classified_at"]

    def test_filters_by_keyword(self, conn) -> None:
        """Only time-series data for the given keyword should be returned."""
        s1 = _make_signal(source_id="a", keyword="python")
        s2 = _make_signal(source_id="b", keyword="rust")
        insert_signal(s1, conn=conn)
        insert_signal(s2, conn=conn)
        results = get_sentiment_timeseries(keyword="python", conn=conn)
        assert len(results) == 1

    def test_empty_returns_empty(self, conn) -> None:
        """Empty signals table should return empty list."""
        assert get_sentiment_timeseries(conn=conn) == []

    def test_limit_respected(self, conn) -> None:
        """The limit parameter should cap data points returned."""
        sigs = [
            _make_signal(
                source_id=f"t3_{i}",
                classified_at=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
            )
            for i in range(10)
        ]
        insert_signals(sigs, conn=conn)
        results = get_sentiment_timeseries(limit=5, conn=conn)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# Top topics
# ---------------------------------------------------------------------------


class TestGetTopTopics:
    """Tests for get_top_topics()."""

    def test_counts_topics_correctly(self, conn) -> None:
        """Topic counts should reflect tag frequency across signals."""
        s1 = _make_signal(source_id="a", topics=["python", "coding"])
        s2 = _make_signal(source_id="b", topics=["python", "ai"])
        s3 = _make_signal(source_id="c", topics=["ai"])
        insert_signals([s1, s2, s3], conn=conn)
        results = get_top_topics(conn=conn)
        topic_map = {r["topic"]: r["count"] for r in results}
        assert topic_map["python"] == 2
        assert topic_map["ai"] == 2
        assert topic_map["coding"] == 1

    def test_ordered_by_count_desc(self, conn) -> None:
        """Topics should be ranked highest count first."""
        s1 = _make_signal(source_id="a", topics=["python", "ml"])
        s2 = _make_signal(source_id="b", topics=["python"])
        insert_signals([s1, s2], conn=conn)
        results = get_top_topics(conn=conn)
        assert results[0]["topic"] == "python"
        assert results[0]["count"] == 2

    def test_limit_respected(self, conn) -> None:
        """limit parameter should cap the number of topics returned."""
        sigs = [
            _make_signal(source_id=f"t3_{i}", topics=[f"topic_{i}"])
            for i in range(15)
        ]
        insert_signals(sigs, conn=conn)
        results = get_top_topics(limit=5, conn=conn)
        assert len(results) == 5

    def test_filters_by_keyword(self, conn) -> None:
        """Only signals for the given keyword should contribute to counts."""
        s1 = _make_signal(source_id="a", keyword="python", topics=["python"])
        s2 = _make_signal(source_id="b", keyword="rust", topics=["rust"])
        insert_signals([s1, s2], conn=conn)
        results = get_top_topics(keyword="python", conn=conn)
        assert len(results) == 1
        assert results[0]["topic"] == "python"

    def test_empty_returns_empty(self, conn) -> None:
        """Empty signals table should return empty list."""
        assert get_top_topics(conn=conn) == []


# ---------------------------------------------------------------------------
# Count signals
# ---------------------------------------------------------------------------


class TestCountSignals:
    """Tests for count_signals()."""

    def test_zero_on_empty(self, conn) -> None:
        """Should return 0 on empty signals table."""
        assert count_signals(conn=conn) == 0

    def test_counts_all(self, conn) -> None:
        """count_signals() without filter should count all signals."""
        insert_signals([_make_signal(source_id=f"t3_{i}") for i in range(3)], conn=conn)
        assert count_signals(conn=conn) == 3

    def test_counts_by_keyword(self, conn) -> None:
        """count_signals() with keyword should only count matching signals."""
        insert_signal(_make_signal(source_id="a", keyword="python"), conn=conn)
        insert_signal(_make_signal(source_id="b", keyword="rust"), conn=conn)
        assert count_signals(keyword="python", conn=conn) == 1
        assert count_signals(keyword="rust", conn=conn) == 1


# ---------------------------------------------------------------------------
# Delete old signals
# ---------------------------------------------------------------------------


class TestDeleteOldSignals:
    """Tests for delete_old_signals()."""

    def test_deletes_excess_rows(self, conn) -> None:
        """Signals beyond keep_latest should be removed."""
        sigs = [
            _make_signal(
                source_id=f"t3_{i}",
                classified_at=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
            )
            for i in range(10)
        ]
        insert_signals(sigs, conn=conn)
        deleted = delete_old_signals(keep_latest=5, conn=conn)
        assert deleted == 5
        assert count_signals(conn=conn) == 5

    def test_no_deletion_when_within_limit(self, conn) -> None:
        """No rows should be deleted if total count <= keep_latest."""
        insert_signals([_make_signal(source_id=f"t3_{i}") for i in range(3)], conn=conn)
        deleted = delete_old_signals(keep_latest=10, conn=conn)
        assert deleted == 0
        assert count_signals(conn=conn) == 3

    def test_delete_with_keyword_filter(self, conn) -> None:
        """Deletion should be scoped to the specified keyword."""
        py_sigs = [
            _make_signal(
                source_id=f"py_{i}",
                keyword="python",
                classified_at=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
            )
            for i in range(6)
        ]
        rs_sigs = [
            _make_signal(
                source_id=f"rs_{i}",
                keyword="rust",
                classified_at=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
            )
            for i in range(4)
        ]
        insert_signals(py_sigs + rs_sigs, conn=conn)
        delete_old_signals(keep_latest=3, keyword="python", conn=conn)
        assert count_signals(keyword="python", conn=conn) == 3
        # Rust signals should be untouched
        assert count_signals(keyword="rust", conn=conn) == 4

    def test_empty_db_no_error(self, conn) -> None:
        """delete_old_signals() on an empty table should not raise."""
        deleted = delete_old_signals(keep_latest=10, conn=conn)
        assert deleted == 0
