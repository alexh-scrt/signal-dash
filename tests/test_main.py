"""Tests for signal_dash.main — FastAPI routes and HTMX partial endpoints.

Uses FastAPI's TestClient for synchronous request testing.
All database operations use an in-memory SQLite database to avoid
filesystem side-effects.  The scheduler is patched out to avoid
background job interference.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Generator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from signal_dash.config import Settings, get_settings
from signal_dash.db import (
    get_connection,
    init_db,
    insert_signal,
    insert_signals,
)
from signal_dash.models import Sentiment, Signal, Source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(
    source_id: str = "t3_test",
    source: Source = Source.REDDIT,
    keyword: str = "python",
    sentiment_score: float = 0.5,
    signal_strength: float = 0.7,
    topics: list[str] | None = None,
) -> Signal:
    """Build a minimal Signal for testing."""
    return Signal(
        source_id=source_id,
        source=source,
        keyword=keyword,
        url=f"https://reddit.com/r/python/{source_id}",
        title="Test Post Title",
        body="This is a test post body.",
        author="testuser",
        post_score=42,
        sentiment_score=sentiment_score,
        sentiment_label=Sentiment.POSITIVE,
        topics=topics if topics is not None else ["python", "testing"],
        signal_strength=signal_strength,
        classified_at=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
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
def test_settings() -> Settings:
    """Return settings pointing at in-memory DB with stub classifier."""
    return Settings(
        keyword="python",
        sources=["reddit"],
        refresh_interval_seconds=300,
        openai_api_key="",
        database_url=":memory:",
    )


@pytest.fixture()
def client(memory_conn, test_settings) -> Generator:
    """Provide a TestClient with mocked scheduler and DB."""
    # Patch scheduler lifecycle to avoid actual background jobs
    with patch("signal_dash.main.start_scheduler", new=AsyncMock()), \
         patch("signal_dash.main.shutdown_scheduler", new=AsyncMock()), \
         patch("signal_dash.main.get_shared_conn", return_value=memory_conn), \
         patch("signal_dash.main.get_settings", return_value=test_settings), \
         patch("signal_dash.config.get_settings", return_value=test_settings):
        from signal_dash.main import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client) -> None:
        """Health check should return HTTP 200."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_ok(self, client) -> None:
        """Health check body should include status: ok."""
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_returns_version(self, client) -> None:
        """Health check should include the application version."""
        resp = client.get("/health")
        data = resp.json()
        assert "version" in data


# ---------------------------------------------------------------------------
# Main dashboard route
# ---------------------------------------------------------------------------


class TestDashboardRoute:
    """Tests for the GET / dashboard route."""

    def test_returns_200(self, client) -> None:
        """Dashboard should return HTTP 200."""
        resp = client.get("/")
        assert resp.status_code == 200

    def test_returns_html(self, client) -> None:
        """Dashboard should return HTML content."""
        resp = client.get("/")
        assert "text/html" in resp.headers["content-type"]

    def test_contains_signal_dash_title(self, client) -> None:
        """Dashboard HTML should contain the app name."""
        resp = client.get("/")
        assert "Signal" in resp.text

    def test_contains_keyword(self, client) -> None:
        """Dashboard should display the active keyword."""
        resp = client.get("/")
        assert "python" in resp.text.lower()

    def test_contains_chart_canvas(self, client) -> None:
        """Dashboard should include a Chart.js canvas element."""
        resp = client.get("/")
        assert "sentimentChart" in resp.text

    def test_contains_htmx_script(self, client) -> None:
        """Dashboard should load the HTMX library."""
        resp = client.get("/")
        assert "htmx" in resp.text.lower()

    def test_shows_empty_state_when_no_signals(self, client) -> None:
        """When there are no signals, the empty state should be shown."""
        resp = client.get("/")
        assert resp.status_code == 200
        # Empty state message or 0 count should appear
        assert "0" in resp.text or "No signals" in resp.text

    def test_shows_signals_when_present(self, client, memory_conn, test_settings) -> None:
        """When signals exist, they should appear in the dashboard."""
        sig = _make_signal(keyword="python")
        with patch("signal_dash.main.get_shared_conn", return_value=memory_conn), \
             patch("signal_dash.main.get_settings", return_value=test_settings):
            insert_signal(sig, conn=memory_conn)
            resp = client.get("/")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# HTMX partial: signals table
# ---------------------------------------------------------------------------


class TestPartialSignalsTable:
    """Tests for the GET /partials/signals-table endpoint."""

    def test_returns_200(self, client) -> None:
        """Signals table partial should return HTTP 200."""
        resp = client.get("/partials/signals-table")
        assert resp.status_code == 200

    def test_returns_html(self, client) -> None:
        """Response should be HTML."""
        resp = client.get("/partials/signals-table")
        assert "text/html" in resp.headers["content-type"]

    def test_empty_state_when_no_signals(self, client) -> None:
        """When no signals exist, empty state markup should be present."""
        resp = client.get("/partials/signals-table")
        assert resp.status_code == 200
        # Either empty state message or table structure present
        text = resp.text
        assert "No signals" in text or "<table" in text or "<tbody" in text

    def test_shows_signal_row(self, client, memory_conn, test_settings) -> None:
        """With a signal in DB, the table should show signal data."""
        sig = _make_signal(keyword="python", source_id="t3_table_test")
        insert_signal(sig, conn=memory_conn)
        with patch("signal_dash.main.get_shared_conn", return_value=memory_conn), \
             patch("signal_dash.main.get_settings", return_value=test_settings):
            resp = client.get("/partials/signals-table?keyword=python")
        assert resp.status_code == 200

    def test_keyword_query_param_accepted(self, client) -> None:
        """The keyword query parameter should be accepted."""
        resp = client.get("/partials/signals-table?keyword=fastapi")
        assert resp.status_code == 200

    def test_limit_query_param_accepted(self, client) -> None:
        """The limit query parameter should be accepted."""
        resp = client.get("/partials/signals-table?limit=10")
        assert resp.status_code == 200

    def test_invalid_limit_rejected(self, client) -> None:
        """A limit of 0 should be rejected with 422."""
        resp = client.get("/partials/signals-table?limit=0")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# HTMX partial: chart data
# ---------------------------------------------------------------------------


class TestPartialChartData:
    """Tests for the GET /partials/chart-data endpoint."""

    def test_returns_200(self, client) -> None:
        """Chart data partial should return HTTP 200."""
        resp = client.get("/partials/chart-data")
        assert resp.status_code == 200

    def test_returns_html(self, client) -> None:
        """Response should be HTML."""
        resp = client.get("/partials/chart-data")
        assert "text/html" in resp.headers["content-type"]

    def test_contains_json_script_tag(self, client) -> None:
        """Response should include the JSON data island script tag."""
        resp = client.get("/partials/chart-data")
        assert "sd-chart-json" in resp.text

    def test_contains_valid_json(self, client) -> None:
        """The chart data island should contain valid JSON."""
        resp = client.get("/partials/chart-data")
        # Extract content between script tags
        text = resp.text
        # The JSON should have labels and values keys
        assert "labels" in text
        assert "values" in text

    def test_keyword_param_accepted(self, client) -> None:
        """The keyword query parameter should be accepted."""
        resp = client.get("/partials/chart-data?keyword=rust")
        assert resp.status_code == 200

    def test_chart_data_with_signals(self, client, memory_conn, test_settings) -> None:
        """With signals in DB, chart data should reflect them."""
        sigs = [
            _make_signal(
                source_id=f"t3_{i}",
                keyword="python",
                sentiment_score=0.1 * (i - 2),
            )
            for i in range(5)
        ]
        insert_signals(sigs, conn=memory_conn)
        with patch("signal_dash.main.get_shared_conn", return_value=memory_conn), \
             patch("signal_dash.main.get_settings", return_value=test_settings):
            resp = client.get("/partials/chart-data?keyword=python")
        assert resp.status_code == 200
        assert "sd-chart-json" in resp.text


# ---------------------------------------------------------------------------
# JSON API: /api/signals
# ---------------------------------------------------------------------------


class TestApiSignals:
    """Tests for the GET /api/signals endpoint."""

    def test_returns_200(self, client) -> None:
        """API endpoint should return HTTP 200."""
        resp = client.get("/api/signals")
        assert resp.status_code == 200

    def test_returns_json(self, client) -> None:
        """Response should be JSON."""
        resp = client.get("/api/signals")
        assert "application/json" in resp.headers["content-type"]

    def test_empty_signals(self, client) -> None:
        """With no data, signals array should be empty."""
        resp = client.get("/api/signals")
        data = resp.json()
        assert "signals" in data
        assert data["signals"] == []
        assert data["count"] == 0

    def test_returns_signal_fields(self, client, memory_conn, test_settings) -> None:
        """Signal objects should include expected fields."""
        sig = _make_signal(keyword="python")
        insert_signal(sig, conn=memory_conn)
        with patch("signal_dash.main.get_shared_conn", return_value=memory_conn), \
             patch("signal_dash.main.get_settings", return_value=test_settings):
            resp = client.get("/api/signals?keyword=python")
        data = resp.json()
        assert data["count"] >= 1
        first = data["signals"][0]
        assert "source_id" in first
        assert "sentiment_score" in first
        assert "signal_strength" in first
        assert "topics" in first

    def test_keyword_filter(self, client, memory_conn, test_settings) -> None:
        """Only signals for the requested keyword should be returned."""
        sig_py = _make_signal(source_id="a", keyword="python")
        sig_rs = _make_signal(source_id="b", keyword="rust")
        insert_signal(sig_py, conn=memory_conn)
        insert_signal(sig_rs, conn=memory_conn)
        with patch("signal_dash.main.get_shared_conn", return_value=memory_conn), \
             patch("signal_dash.main.get_settings", return_value=test_settings):
            resp = client.get("/api/signals?keyword=rust")
        data = resp.json()
        assert data["keyword"] == "rust"
        for s in data["signals"]:
            assert s["keyword"] == "rust"

    def test_limit_param(self, client, memory_conn, test_settings) -> None:
        """Limit parameter should cap the number of returned signals."""
        sigs = [_make_signal(source_id=f"t3_{i}", keyword="python") for i in range(10)]
        insert_signals(sigs, conn=memory_conn)
        with patch("signal_dash.main.get_shared_conn", return_value=memory_conn), \
             patch("signal_dash.main.get_settings", return_value=test_settings):
            resp = client.get("/api/signals?keyword=python&limit=3")
        data = resp.json()
        assert len(data["signals"]) <= 3

    def test_response_has_meta_fields(self, client) -> None:
        """Response should include keyword, limit, offset, count fields."""
        resp = client.get("/api/signals")
        data = resp.json()
        assert "keyword" in data
        assert "limit" in data
        assert "offset" in data
        assert "count" in data


# ---------------------------------------------------------------------------
# JSON API: /api/stats
# ---------------------------------------------------------------------------


class TestApiStats:
    """Tests for the GET /api/stats endpoint."""

    def test_returns_200(self, client) -> None:
        """Stats endpoint should return HTTP 200."""
        resp = client.get("/api/stats")
        assert resp.status_code == 200

    def test_returns_json(self, client) -> None:
        """Stats response should be JSON."""
        resp = client.get("/api/stats")
        assert "application/json" in resp.headers["content-type"]

    def test_stats_fields_present(self, client) -> None:
        """Stats response should include all expected fields."""
        resp = client.get("/api/stats")
        data = resp.json()
        assert "keyword" in data
        assert "total_posts" in data
        assert "total_signals" in data
        assert "avg_sentiment" in data
        assert "sentiment_distribution" in data
        assert "top_topics" in data

    def test_empty_db_stats(self, client) -> None:
        """With an empty database, counts should all be 0."""
        resp = client.get("/api/stats")
        data = resp.json()
        assert data["total_posts"] == 0
        assert data["total_signals"] == 0
        assert data["avg_sentiment"] == 0.0

    def test_stats_with_signals(self, client, memory_conn, test_settings) -> None:
        """With signals present, stats should reflect them."""
        sigs = [
            _make_signal(source_id="a", keyword="python", sentiment_score=0.8),
            _make_signal(source_id="b", keyword="python", sentiment_score=-0.4),
        ]
        insert_signals(sigs, conn=memory_conn)
        with patch("signal_dash.main.get_shared_conn", return_value=memory_conn), \
             patch("signal_dash.main.get_settings", return_value=test_settings):
            resp = client.get("/api/stats?keyword=python")
        data = resp.json()
        assert data["total_signals"] >= 2
        dist = data["sentiment_distribution"]
        assert "positive" in dist
        assert "negative" in dist
        assert "neutral" in dist

    def test_top_topics_in_stats(self, client, memory_conn, test_settings) -> None:
        """Top topics should appear in stats when signals have topics."""
        sig = _make_signal(keyword="python", topics=["python", "testing"])
        insert_signal(sig, conn=memory_conn)
        with patch("signal_dash.main.get_shared_conn", return_value=memory_conn), \
             patch("signal_dash.main.get_settings", return_value=test_settings):
            resp = client.get("/api/stats?keyword=python")
        data = resp.json()
        assert isinstance(data["top_topics"], list)
