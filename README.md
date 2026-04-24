# Signal Dash

> Lightweight social media intelligence dashboard powered by Reddit, Mastodon, and LLM classification.

Signal Dash pulls public posts from Reddit and Mastodon, classifies them using GPT-4o-mini (or a fully-offline rule-based stub), and surfaces trending signals, sentiment shifts, and emerging topics for any brand or keyword — all on an auto-refreshing dashboard with Chart.js time-series charts and HTMX-powered live updates.

---

## Table of Contents

1. [Features](#features)
2. [Architecture Overview](#architecture-overview)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Running the App](#running-the-app)
6. [Dashboard Tour](#dashboard-tour)
7. [API Reference](#api-reference)
8. [Development & Testing](#development--testing)
9. [Offline / Stub Mode](#offline--stub-mode)
10. [Project Structure](#project-structure)
11. [Troubleshooting](#troubleshooting)
12. [License](#license)

---

## Features

| Feature | Details |
|---|---|
| **Multi-source ingestion** | Reddit public JSON API (no auth) and Mastodon public timeline search |
| **LLM classification** | GPT-4o-mini assigns sentiment (−1 → +1), up to 3 topic tags, and a signal-strength score per post |
| **Offline stub mode** | Deterministic rule-based classifier so the app runs fully without an OpenAI key |
| **Auto-refreshing dashboard** | HTMX polls partial endpoints; Chart.js renders sentiment time-series |
| **SQLite persistence** | Zero-config local database with deduplication by source post ID |
| **Background scheduler** | APScheduler fires ingest + classify jobs on a configurable interval |
| **JSON API** | Programmatic access to signals and summary statistics |

---

## Architecture Overview

```
┌──────────────┐    HTTP poll     ┌──────────────────┐
│   Reddit     │ ◄──────────────► │                  │
│  (JSON API)  │                  │   signal_dash    │
└──────────────┘                  │   (FastAPI app)  │
                                  │                  │
┌──────────────┐    HTTP poll     │  ┌────────────┐  │
│  Mastodon    │ ◄──────────────► │  │ APScheduler│  │
│ (public API) │                  │  └─────┬──────┘  │
└──────────────┘                  │        │          │
                                  │  ┌─────▼──────┐  │
┌──────────────┐    API call      │  │ Classifier │  │
│  OpenAI API  │ ◄──────────────► │  │(LLM/Stub)  │  │
│ (gpt-4o-mini)│                  │  └─────┬──────┘  │
└──────────────┘                  │        │          │
                                  │  ┌─────▼──────┐  │
                                  │  │   SQLite   │  │
                                  │  └────────────┘  │
                                  │        │          │
                                  │  ┌─────▼──────┐  │
                                  │  │  Jinja2 /  │  │
                                  │  │   HTMX     │  │
                                  └──┴────────────┘──┘
                                           │
                                     Browser (Chart.js)
```

**Data flow per scheduler tick:**

1. `ingest.py` fetches up to N posts from Reddit / Mastodon.
2. New posts are persisted to `posts` table (deduplicated by `source_id + source`).
3. `classifier.py` batches posts → OpenAI API or stub → `signals` table.
4. The dashboard HTMX partials poll `/partials/signals-table` and `/partials/chart-data` and update the UI without a page reload.

---

## Quick Start

### Prerequisites

- Python 3.11 or 3.12
- `pip` (or `pipx` / `uv`)
- An OpenAI API key *(optional — app works fully without one in stub mode)*

### 1. Clone and install

```bash
git clone https://github.com/your-org/signal_dash.git
cd signal_dash

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install in editable mode with all dependencies
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set KEYWORD to your brand/term of interest.
# Add OPENAI_API_KEY if you want LLM-backed classification.
```

### 3. Run

```bash
signal-dash
# or: python -m signal_dash.main
# or: uvicorn signal_dash.main:app --reload
```

Open **http://localhost:8000** in your browser.

The dashboard will show an empty state for the first polling interval (default 300 s) then populate automatically as posts are ingested and classified.

---

## Configuration

All settings are read from environment variables or a `.env` file at the project root.

| Variable | Default | Description |
|---|---|---|
| `KEYWORD` | `python` | Primary keyword / brand to track across all sources |
| `SOURCES` | `reddit,mastodon` | Comma-separated list of sources to enable (`reddit`, `mastodon`) |
| `REFRESH_INTERVAL_SECONDS` | `300` | How often (seconds) the background job polls each source (min 30) |
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key; leave blank for offline stub mode |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model identifier used for classification |
| `CLASSIFIER_BATCH_SIZE` | `10` | Posts per LLM API call |
| `DATABASE_URL` | `signal_dash.db` | SQLite database file path (use `:memory:` for tests) |
| `REDDIT_SUBREDDIT` | `all` | Subreddit to search within |
| `REDDIT_POST_LIMIT` | `25` | Maximum Reddit posts per poll (1–100) |
| `REDDIT_BASE_URL` | `https://www.reddit.com` | Reddit API base URL |
| `REDDIT_USER_AGENT` | `signal_dash/0.1.0 …` | User-Agent header sent to Reddit |
| `MASTODON_BASE_URL` | `https://mastodon.social` | Mastodon instance base URL |
| `MASTODON_POST_LIMIT` | `20` | Maximum Mastodon statuses per poll (1–40) |
| `HOST` | `0.0.0.0` | Uvicorn bind host |
| `PORT` | `8000` | Uvicorn bind port |
| `DEBUG` | `false` | Enable FastAPI debug mode and uvicorn `--reload` |

See `.env.example` for a fully-annotated example.

### Minimal `.env` for LLM mode

```dotenv
KEYWORD=fastapi
OPENAI_API_KEY=sk-...
REFRESH_INTERVAL_SECONDS=120
```

### Minimal `.env` for offline / stub mode

```dotenv
KEYWORD=python
# No OPENAI_API_KEY — stub classifier is used automatically
```

---

## Running the App

### Via the installed CLI script

```bash
signal-dash
```

### Via uvicorn directly

```bash
uvicorn signal_dash.main:app --host 0.0.0.0 --port 8000

# With auto-reload for development:
uvicorn signal_dash.main:app --reload
```

### Via Python module

```bash
python -m signal_dash.main
```

### With Docker (example — no Dockerfile included, adapt as needed)

```bash
docker run -it --rm \
  -p 8000:8000 \
  -e KEYWORD=python \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/data:/app/data \
  -e DATABASE_URL=/app/data/signal_dash.db \
  your-image-name
```

---

## Dashboard Tour

```
┌─────────────────────────────────────────────────────────────────┐
│  SignalDash  ● Live  v0.1.0  [STUB]  [REDDIT]  [MASTODON]       │  ← Header
├─────────────────────────────────────────────────────────────────┤
│  Monitoring  ⬡ python              Auto-refreshes every 300s    │
├────────┬──────────┬─────────────┬──────────┬─────────┬─────────┤
│ Posts  │ Signals  │ Avg Sent.   │ Positive │Negative │ Neutral │  ← Stat cards
│  142   │   138    │   +0.12     │    89    │   21    │   28    │
├─────────────────────────────────────┬───────────────────────────┤
│                                     │  Trending Topics           │
│   Sentiment over Time (Chart.js)    │  1 #python      ████ 42   │
│                                     │  2 #ai          ███  31   │
│   ▁▃▅▇▅▃▁▃▅▇▅▆▄▂▁▃▅▇▅▃▁           │  3 #web         ██   18   │
│                                     │  4 #cloud       █    9    │
├─────────────────────────────────────┴───────────────────────────┤
│  Recent Signals                                      138 shown   │
│  Source  Post                  Author  Sent.  Str.  Topics  ⏱   │
│  Reddit  Amazing Python lib…   @alice  ▲+0.82 ████  #python …   │
│  Mastod  New release of Fast…  @bob   ▲+0.61 ███   #web    …   │
│  Reddit  This crashed badly…   @carol ▼-0.74 ██    #python …   │
└─────────────────────────────────────────────────────────────────┘
```

**Key UI elements:**

- **Header badge** shows `STUB` (no API key) or `LLM` (OpenAI active).
- **Stat cards** update on every HTMX refresh cycle.
- **Sentiment chart** — line chart with colour-coded points (green = positive, red = negative, grey = neutral). Refreshed via HTMX without reloading the page.
- **Trending topics** — ranked by mention count across recent signals.
- **Signals table** — sortable by classification time; each row links to the original post.

---

## API Reference

The FastAPI auto-generated docs are available at:

- **Swagger UI** → http://localhost:8000/docs
- **ReDoc** → http://localhost:8000/redoc

### `GET /api/signals`

Paginated list of classified signals.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `keyword` | string | settings default | Keyword filter |
| `limit` | int | 50 | Max results (1–200) |
| `offset` | int | 0 | Pagination offset |

**Example response:**

```json
{
  "keyword": "python",
  "limit": 3,
  "offset": 0,
  "count": 3,
  "signals": [
    {
      "source_id": "t3_abc123",
      "source": "reddit",
      "keyword": "python",
      "title": "Amazing new Python library",
      "body": "Just released a new async library…",
      "url": "https://reddit.com/r/python/…",
      "author": "alice",
      "post_score": 1204,
      "sentiment_score": 0.82,
      "sentiment_label": "positive",
      "topics": ["python", "open-source"],
      "signal_strength": 0.91,
      "classified_at": "2024-06-01T12:34:56+00:00"
    }
  ]
}
```

### `GET /api/stats`

Summary statistics for the active keyword.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `keyword` | string | settings default | Keyword filter |

**Example response:**

```json
{
  "keyword": "python",
  "total_posts": 142,
  "total_signals": 138,
  "avg_sentiment": 0.1234,
  "sentiment_distribution": {
    "positive": 89,
    "neutral": 28,
    "negative": 21
  },
  "top_topics": [
    {"topic": "python", "count": 42},
    {"topic": "ai", "count": 31}
  ]
}
```

### `GET /health`

Simple liveness probe.

```json
{"status": "ok", "version": "0.1.0"}
```

### HTMX Partials (internal)

| Endpoint | Purpose |
|---|---|
| `GET /partials/signals-table` | HTML fragment — signals table body |
| `GET /partials/chart-data` | HTML fragment — JSON data island for Chart.js |

These are consumed by HTMX on the dashboard and are not intended for direct API use, but they return valid HTML and can be inspected for debugging.

---

## Development & Testing

### Install dev dependencies

```bash
pip install -e .
# Test dependencies are declared in pyproject.toml
pip install pytest pytest-asyncio respx
```

### Run all tests

```bash
pytest
```

### Run tests with verbose output

```bash
pytest -v
```

### Run a specific test module

```bash
pytest tests/test_classifier.py -v
pytest tests/test_ingest.py -v
pytest tests/test_db.py -v
pytest tests/test_scheduler.py -v
pytest tests/test_main.py -v
```

### Run tests with coverage

```bash
pip install pytest-cov
pytest --cov=signal_dash --cov-report=term-missing
```

### Test design

| Test module | What it covers |
|---|---|
| `tests/test_config.py` | Settings loading, validation, defaults |
| `tests/test_models.py` | Pydantic model validation, computed fields |
| `tests/test_db.py` | Schema creation, CRUD, deduplication (in-memory SQLite) |
| `tests/test_ingest.py` | Reddit + Mastodon parsers, HTTP mocking via `respx` |
| `tests/test_classifier.py` | Stub classifier sentiment/topics, LLM response parser, `classify_posts()` |
| `tests/test_scheduler.py` | Job registration, `ingest_and_classify()`, error handling |
| `tests/test_main.py` | FastAPI routes, HTMX partials, JSON API (TestClient) |

All tests use **in-memory SQLite** and **mocked HTTP / scheduler** so they run offline with no external dependencies.

### pytest configuration

The project uses `asyncio_mode = "auto"` (set in `pyproject.toml`) so `async def test_*` functions work without any additional decorators.

---

## Offline / Stub Mode

When `OPENAI_API_KEY` is not set (or empty), the app automatically switches to the **stub classifier** — a deterministic, zero-network rule-based algorithm:

1. Tokenises post text (title + body) into lowercase words.
2. Counts positive / negative sentiment words from curated word lists.
3. Applies small adjustments for exclamation marks and question marks.
4. Produces a normalised sentiment score in `[−1.0, 1.0]`.
5. Extracts up to 3 topic tags by matching against domain keyword lists (ai, python, security, performance, web, cloud, data, open-source, community, jobs).
6. Derives signal strength from log-scaled platform engagement score and sentiment magnitude.

The stub produces plausible scores and is suitable for:

- Local development without incurring API costs.
- CI/CD pipelines where an API key is unavailable.
- Demos and testing.

To verify which mode is active, check the header badge on the dashboard (`STUB` vs `LLM`) or the startup log:

```
INFO signal_dash.classifier — Classifying 25 posts via stub classifier.
```

---

## Project Structure

```
signal_dash/
├── __init__.py          # Package init — exposes __version__
├── main.py              # FastAPI app factory, routes, HTMX partials
├── config.py            # Settings loader (pydantic-settings)
├── models.py            # Pydantic models: Post, Signal, DashboardConfig
├── db.py                # SQLite schema + CRUD helpers
├── ingest.py            # Reddit + Mastodon async ingestion
├── classifier.py        # LLM + stub classifiers
├── scheduler.py         # APScheduler background jobs
└── templates/
    ├── index.html                   # Main dashboard template
    └── partials/
        ├── signals_table.html       # HTMX partial — signals table
        └── chart_data.html          # HTMX partial — Chart.js data island

tests/
├── __init__.py
├── test_config.py
├── test_models.py
├── test_db.py
├── test_ingest.py
├── test_classifier.py
├── test_scheduler.py
└── test_main.py

pyproject.toml           # Project metadata + dependencies
.env.example             # Annotated example environment file
README.md                # This file
```

---

## Troubleshooting

### Dashboard shows "No signals yet" after startup

**Cause:** The first ingest job has not completed yet.

**Fix:** Wait for one polling interval (default 5 minutes) or lower `REFRESH_INTERVAL_SECONDS` to `60` for faster first-run data.

The scheduler fires an immediate run on startup (`run_immediately=True`), so data should appear within the time it takes to fetch and classify posts (typically 5–15 seconds with a good connection).

---

### `429 Too Many Requests` from Reddit

**Cause:** Reddit rate-limits aggressive bots.

**Fix:**
- Set a descriptive `REDDIT_USER_AGENT` (e.g. `my_app/1.0 (contact: me@example.com)`).
- Increase `REFRESH_INTERVAL_SECONDS` to 300 or more.
- Reduce `REDDIT_POST_LIMIT`.

---

### OpenAI classification errors

**Cause:** Invalid API key, quota exhaustion, or model unavailability.

**Fix:**
- Check your API key at https://platform.openai.com/api-keys.
- The app automatically falls back to the stub classifier per-batch on any OpenAI error, so the dashboard remains functional.
- Monitor logs for `WARNING signal_dash.classifier — LLM classification failed … falling back to stub`.

---

### Mastodon returns `403 Forbidden`

**Cause:** Some Mastodon instances require authentication for search.

**Fix:** Set `MASTODON_BASE_URL` to a more permissive instance (e.g. `https://mastodon.social`) or disable Mastodon in `SOURCES=reddit`.

---

### Database grows too large

**Cause:** High polling frequency with many results over time.

**Fix:** The scheduler automatically prunes signals older than the 2,000 most-recent per keyword. You can also manually clear the database:

```bash
rm signal_dash.db
# Restart the app — schema is recreated automatically.
```

---

### Tests fail with `RuntimeError: no running event loop`

**Cause:** `asyncio_mode` is not set to `"auto"` in pytest config.

**Fix:** Ensure `pyproject.toml` contains:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

---

### `ModuleNotFoundError: No module named 'signal_dash'`

**Fix:** Install the package in editable mode:

```bash
pip install -e .
```

---

## License

MIT — see `pyproject.toml` for details.

---

*Built with FastAPI, HTMX, Chart.js, APScheduler, and ❤️*
