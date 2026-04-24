# Signal Dash 📡

> Real-time social listening for the rest of us — no enterprise contract required.

Signal Dash pulls public posts from Reddit and Mastodon, classifies them using GPT-4o-mini (or a fully-offline rule-based fallback), and surfaces trending signals, sentiment shifts, and emerging topics for any keyword or brand. Results appear on an auto-refreshing dashboard with Chart.js time-series sentiment charts and HTMX-powered live updates — no page reloads, no bloat.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-org/signal_dash.git
cd signal_dash
pip install -e .

# 2. Configure (copy the example and edit as needed)
cp .env.example .env
# At minimum, set KEYWORD — everything else has sensible defaults.
# Add OPENAI_API_KEY to enable LLM classification (optional).

# 3. Run
signal-dash
```

Open **http://localhost:8000** — the dashboard starts polling immediately.

> **No OpenAI key?** Signal Dash runs fully offline using the built-in stub classifier. Just leave `OPENAI_API_KEY` unset.

---

## Features

- **Keyword-driven ingestion** from Reddit (public JSON API, no auth) and Mastodon public timelines with configurable polling intervals.
- **Batched LLM classification** via GPT-4o-mini — each post gets a sentiment score (−1 to 1), up to 3 topic tags, and a signal strength score.
- **Auto-refreshing HTMX dashboard** with a Chart.js time-series sentiment chart and a ranked emerging-topics table, updated without page reloads.
- **Lightweight SQLite persistence** with deduplication by source post ID — no external database needed.
- **Zero-config offline mode** — a deterministic rule-based stub classifier lets you develop and test without any API key or network access.

---

## Usage Examples

### Monitor a keyword from the CLI

```bash
# Track mentions of "fastapi" across Reddit and Mastodon
KEYWORD=fastapi signal-dash

# Limit to Reddit only, refresh every 2 minutes
KEYWORD=fastapi SOURCES=reddit REFRESH_INTERVAL_SECONDS=120 signal-dash
```

### Query signals via the JSON API

```bash
# Fetch the 20 most recent classified signals
curl http://localhost:8000/api/signals

# Filter by sentiment and limit results
curl "http://localhost:8000/api/signals?limit=10&min_sentiment=0.2"
```

```json
[
  {
    "source_id": "1abc23",
    "source": "reddit",
    "author": "u/dev_person",
    "text": "FastAPI's dependency injection is genuinely excellent.",
    "sentiment_score": 0.82,
    "topics": ["fastapi", "python", "web-framework"],
    "signal_strength": 0.74,
    "classified_at": "2024-11-01T10:22:00Z"
  }
]
```

### HTMX partials (auto-polled by the dashboard)

```bash
# Signals table fragment
curl http://localhost:8000/partials/signals-table

# Chart data island (JSON consumed by Chart.js)
curl http://localhost:8000/partials/chart-data
```

### Run tests

```bash
pip install -e ".[dev]"
pytest
```

---

## Project Structure

```
signal_dash/
├── __init__.py                  # Package init, version constant
├── main.py                      # FastAPI app factory, routes, HTMX partials
├── config.py                    # Settings loader (env vars + .env via pydantic-settings)
├── models.py                    # Pydantic models: Post, Signal, DashboardConfig
├── ingest.py                    # Async Reddit + Mastodon fetch functions (httpx)
├── classifier.py                # LLM classifier (GPT-4o-mini) + offline stub fallback
├── db.py                        # SQLite schema, CRUD helpers, deduplication
├── scheduler.py                 # APScheduler background ingest+classify job
└── templates/
    ├── index.html               # Main Jinja2 dashboard template
    └── partials/
        ├── signals_table.html   # Live-updating signals table fragment
        └── chart_data.html      # Chart.js JSON data island fragment
tests/
├── test_ingest.py               # Ingest unit tests (respx HTTP mocks)
├── test_classifier.py           # Classifier unit tests (stubbed LLM responses)
├── test_db.py                   # DB tests (in-memory SQLite)
├── test_scheduler.py            # Scheduler lifecycle + job tests
├── test_models.py               # Pydantic model validation tests
├── test_config.py               # Settings loading and validation tests
└── test_main.py                 # FastAPI route tests (TestClient)
.env.example                     # Documented config template
pyproject.toml                   # Project metadata and dependencies
```

---

## Configuration

Copy `.env.example` to `.env` and adjust values as needed. All settings can also be passed as plain environment variables.

| Variable | Default | Description |
|---|---|---|
| `KEYWORD` | `python` | Brand or keyword to monitor |
| `SOURCES` | `reddit,mastodon` | Comma-separated list of sources (`reddit`, `mastodon`) |
| `REFRESH_INTERVAL_SECONDS` | `300` | How often (in seconds) to poll for new posts |
| `OPENAI_API_KEY` | *(unset)* | OpenAI key for LLM classification; omit for offline stub mode |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use for classification |
| `MASTODON_INSTANCE` | `mastodon.social` | Mastodon instance base URL |
| `DB_PATH` | `signal_dash.db` | Path to the SQLite database file |
| `MAX_SIGNALS_STORED` | `1000` | Maximum number of signals retained (older records pruned) |
| `HOST` | `0.0.0.0` | Uvicorn bind host |
| `PORT` | `8000` | Uvicorn bind port |

**Minimal `.env` for offline development:**

```dotenv
KEYWORD=your-brand
SOURCES=reddit
```

**Minimal `.env` for LLM-backed classification:**

```dotenv
KEYWORD=your-brand
OPENAI_API_KEY=sk-...
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built with [Jitter](https://github.com/jitter-ai) — an AI agent that ships code daily.*
