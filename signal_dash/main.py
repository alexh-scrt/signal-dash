"""FastAPI application factory, route definitions, and HTMX partial endpoints.

This module defines the Signal Dash web application including:

- The main dashboard route (``GET /``) that renders the full Jinja2 template.
- HTMX partial endpoints for live-updating the signals table and chart data.
- A JSON API endpoint for programmatic access to signals.
- Application lifespan management (scheduler start/stop, DB init).
- A ``run()`` entry point for the ``signal-dash`` CLI script.

HTMX partials
-------------
``GET /partials/signals-table``
    Returns an HTML fragment containing the updated signals table rows.
    Triggered by the auto-refresh polling on the frontend.

``GET /partials/chart-data``
    Returns an HTML fragment containing the updated JSON data island
    consumed by Chart.js for the sentiment time-series chart.

JSON API
--------
``GET /api/signals``  — paginated list of classified signals.
``GET /api/stats``    — summary statistics (counts, avg sentiment, top topics).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from signal_dash import __version__
from signal_dash.config import get_settings
from signal_dash.db import (
    count_posts,
    count_signals,
    get_connection,
    get_sentiment_timeseries,
    get_signals,
    get_top_topics,
    init_db,
)
from signal_dash.models import DashboardConfig, Signal, Source
from signal_dash.scheduler import (
    get_shared_conn,
    shutdown_scheduler,
    start_scheduler,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template directory
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).parent / "templates"

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_db() -> sqlite3.Connection:
    """Return the shared DB connection, or open a new one as fallback.

    Returns
    -------
    sqlite3.Connection
        An initialised SQLite connection.
    """
    conn = get_shared_conn()
    if conn is None:
        settings = get_settings()
        conn = get_connection(settings.database_url)
        init_db(conn)
    return conn


def _build_dashboard_context(
    request: Request,
    conn: sqlite3.Connection,
    keyword: Optional[str] = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Build the Jinja2 template context dictionary for dashboard views.

    Parameters
    ----------
    request:
        The incoming FastAPI request (required by Jinja2Templates).
    conn:
        SQLite connection for data queries.
    keyword:
        Keyword filter override.  Defaults to the settings keyword.
    limit:
        Maximum number of signals to include in the table.

    Returns
    -------
    dict[str, Any]
        Template context with signals, chart data, topics, and config.
    """
    settings = get_settings()
    kw = keyword or settings.keyword
    dashboard_cfg = DashboardConfig.from_settings(settings)

    signals = get_signals(keyword=kw, limit=limit, conn=conn)
    timeseries = get_sentiment_timeseries(keyword=kw, limit=200, conn=conn)
    top_topics = get_top_topics(keyword=kw, limit=10, conn=conn)
    total_posts = count_posts(keyword=kw, conn=conn)
    total_signals = count_signals(keyword=kw, conn=conn)

    # Compute average sentiment across recent signals
    avg_sentiment: float = 0.0
    if signals:
        avg_sentiment = sum(s.sentiment_score for s in signals) / len(signals)

    # Build Chart.js compatible data
    chart_labels = [row["classified_at"][:16].replace("T", " ") for row in timeseries]
    chart_values = [round(row["sentiment_score"], 4) for row in timeseries]
    chart_data_json = json.dumps(
        {"labels": chart_labels, "values": chart_values}
    )

    # Sentiment distribution
    pos_count = sum(1 for s in signals if s.sentiment_score > 0.2)
    neg_count = sum(1 for s in signals if s.sentiment_score < -0.2)
    neu_count = len(signals) - pos_count - neg_count

    return {
        "request": request,
        "config": dashboard_cfg,
        "keyword": kw,
        "signals": signals,
        "top_topics": top_topics,
        "total_posts": total_posts,
        "total_signals": total_signals,
        "avg_sentiment": round(avg_sentiment, 3),
        "chart_data_json": chart_data_json,
        "chart_labels": chart_labels,
        "chart_values": chart_values,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "neu_count": neu_count,
        "refresh_interval_ms": settings.refresh_interval_seconds * 1000,
        "app_version": __version__,
    }


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager.

    Starts the APScheduler background job on startup and shuts it down
    cleanly on application shutdown.

    Parameters
    ----------
    app:
        The FastAPI application instance (unused directly but required by
        the lifespan protocol).

    Yields
    ------
    None
        Yields control to the application while it is running.
    """
    settings = get_settings()
    logger.info(
        "Signal Dash %s starting up — keyword='%s', sources=%s.",
        __version__,
        settings.keyword,
        settings.sources,
    )

    try:
        await start_scheduler(settings=settings, run_immediately=True)
        logger.info("Background scheduler started successfully.")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to start scheduler: %s", exc)

    yield  # application is running

    logger.info("Signal Dash shutting down.")
    await shutdown_scheduler()
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns
    -------
    FastAPI
        Fully configured application instance with all routes registered.
    """
    settings = get_settings()

    app = FastAPI(
        title="Signal Dash",
        description=(
            "Lightweight social media intelligence dashboard "
            "powered by Reddit, Mastodon, and LLM classification."
        ),
        version=__version__,
        debug=settings.debug,
        lifespan=_lifespan,
    )

    # ------------------------------------------------------------------ #
    # Routes
    # ------------------------------------------------------------------ #

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard(request: Request) -> HTMLResponse:
        """Render the main dashboard page.

        Parameters
        ----------
        request:
            Incoming HTTP request.

        Returns
        -------
        HTMLResponse
            Fully rendered HTML dashboard.
        """
        conn = _get_db()
        context = _build_dashboard_context(request, conn)
        return templates.TemplateResponse("index.html", context)

    @app.get(
        "/partials/signals-table",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def partial_signals_table(
        request: Request,
        keyword: Optional[str] = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> HTMLResponse:
        """Return an HTMX partial HTML fragment for the signals table.

        This endpoint is polled by the HTMX auto-refresh on the dashboard.
        It renders only the table rows partial, not the full page.

        Parameters
        ----------
        request:
            Incoming HTTP request.
        keyword:
            Keyword filter override.  Defaults to the settings keyword.
        limit:
            Maximum number of signals to show.

        Returns
        -------
        HTMLResponse
            HTML fragment with signal table rows.
        """
        conn = _get_db()
        context = _build_dashboard_context(request, conn, keyword=keyword, limit=limit)
        return templates.TemplateResponse(
            "partials/signals_table.html", context
        )

    @app.get(
        "/partials/chart-data",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def partial_chart_data(
        request: Request,
        keyword: Optional[str] = Query(default=None),
    ) -> HTMLResponse:
        """Return an HTMX partial HTML fragment with Chart.js data.

        Renders a ``<script>`` tag containing an updated JSON data island
        that the Chart.js instance on the dashboard reads to refresh the
        sentiment time-series chart.

        Parameters
        ----------
        request:
            Incoming HTTP request.
        keyword:
            Keyword filter override.  Defaults to the settings keyword.

        Returns
        -------
        HTMLResponse
            HTML script fragment with updated chart data.
        """
        conn = _get_db()
        context = _build_dashboard_context(request, conn, keyword=keyword)
        return templates.TemplateResponse(
            "partials/chart_data.html", context
        )

    @app.get("/api/signals", response_class=JSONResponse)
    async def api_signals(
        keyword: Optional[str] = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> JSONResponse:
        """Return paginated classified signals as JSON.

        Parameters
        ----------
        keyword:
            Keyword filter.  Defaults to the settings keyword.
        limit:
            Maximum number of results.
        offset:
            Number of results to skip (for pagination).

        Returns
        -------
        JSONResponse
            JSON array of signal objects.
        """
        settings = get_settings()
        kw = keyword or settings.keyword
        conn = _get_db()
        signals = get_signals(keyword=kw, limit=limit, offset=offset, conn=conn)
        return JSONResponse(
            content={
                "keyword": kw,
                "limit": limit,
                "offset": offset,
                "count": len(signals),
                "signals": [
                    {
                        "source_id": s.source_id,
                        "source": s.source.value,
                        "keyword": s.keyword,
                        "title": s.title,
                        "body": s.body[:200] if s.body else "",
                        "url": s.url,
                        "author": s.author,
                        "post_score": s.post_score,
                        "sentiment_score": s.sentiment_score,
                        "sentiment_label": s.sentiment_label.value,
                        "topics": s.topics,
                        "signal_strength": s.signal_strength,
                        "classified_at": s.classified_at.isoformat(),
                    }
                    for s in signals
                ],
            }
        )

    @app.get("/api/stats", response_class=JSONResponse)
    async def api_stats(
        keyword: Optional[str] = Query(default=None),
    ) -> JSONResponse:
        """Return summary statistics for the dashboard.

        Parameters
        ----------
        keyword:
            Keyword filter.  Defaults to the settings keyword.

        Returns
        -------
        JSONResponse
            JSON object with post count, signal count, avg sentiment,
            sentiment distribution, and top topics.
        """
        settings = get_settings()
        kw = keyword or settings.keyword
        conn = _get_db()

        signals = get_signals(keyword=kw, limit=200, conn=conn)
        total_posts = count_posts(keyword=kw, conn=conn)
        total_signals = count_signals(keyword=kw, conn=conn)
        top_topics = get_top_topics(keyword=kw, limit=10, conn=conn)

        avg_sentiment = 0.0
        if signals:
            avg_sentiment = sum(s.sentiment_score for s in signals) / len(signals)

        pos_count = sum(1 for s in signals if s.sentiment_score > 0.2)
        neg_count = sum(1 for s in signals if s.sentiment_score < -0.2)
        neu_count = len(signals) - pos_count - neg_count

        return JSONResponse(
            content={
                "keyword": kw,
                "total_posts": total_posts,
                "total_signals": total_signals,
                "avg_sentiment": round(avg_sentiment, 4),
                "sentiment_distribution": {
                    "positive": pos_count,
                    "neutral": neu_count,
                    "negative": neg_count,
                },
                "top_topics": top_topics,
            }
        )

    @app.get("/health", response_class=JSONResponse, include_in_schema=False)
    async def health_check() -> JSONResponse:
        """Simple health-check endpoint.

        Returns
        -------
        JSONResponse
            ``{"status": "ok", "version": "..."}``
        """
        return JSONResponse(content={"status": "ok", "version": __version__})

    return app


# ---------------------------------------------------------------------------
# Module-level app instance
# ---------------------------------------------------------------------------

app = create_app()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Run the Signal Dash application with uvicorn.

    This function is the target of the ``signal-dash`` CLI entry point
    defined in ``pyproject.toml``.
    """
    settings = get_settings()
    uvicorn.run(
        "signal_dash.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    run()
