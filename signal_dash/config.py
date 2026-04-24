"""Configuration loader for signal_dash.

Reads settings from environment variables (or a .env file) using
pydantic-settings, providing typed, validated configuration with
sensible defaults for local development.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings derived from environment variables.

    All variables can be set directly in the environment or via a ``.env``
    file placed at the project root.  Defaults are chosen so the app runs
    fully offline (stub classifier, SQLite in a local file) without any
    additional configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Core keyword / topic configuration
    # ------------------------------------------------------------------ #
    keyword: str = Field(
        default="python",
        description="Primary keyword or brand name to monitor across sources.",
    )

    # ------------------------------------------------------------------ #
    # Source selection
    # ------------------------------------------------------------------ #
    sources: list[Literal["reddit", "mastodon"]] = Field(
        default=["reddit", "mastodon"],
        description=(
            "Comma-separated list of sources to ingest from. "
            "Accepted values: reddit, mastodon."
        ),
    )

    @field_validator("sources", mode="before")
    @classmethod
    def _parse_sources(cls, value: object) -> list[str]:
        """Allow sources to be supplied as a comma-separated string."""
        if isinstance(value, str):
            return [s.strip().lower() for s in value.split(",") if s.strip()]
        return list(value)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # Polling / scheduler
    # ------------------------------------------------------------------ #
    refresh_interval_seconds: int = Field(
        default=300,
        ge=30,
        description="How often (in seconds) the background job polls each source.",
    )

    # ------------------------------------------------------------------ #
    # Reddit ingestion
    # ------------------------------------------------------------------ #
    reddit_base_url: str = Field(
        default="https://www.reddit.com",
        description="Base URL for the Reddit JSON API (override for testing).",
    )
    reddit_subreddit: str = Field(
        default="all",
        description="Subreddit to search within (default: r/all).",
    )
    reddit_post_limit: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Maximum number of posts to fetch per Reddit poll.",
    )
    reddit_user_agent: str = Field(
        default="signal_dash/0.1.0 (social listening bot)",
        description="User-Agent header sent to Reddit.",
    )

    # ------------------------------------------------------------------ #
    # Mastodon ingestion
    # ------------------------------------------------------------------ #
    mastodon_base_url: str = Field(
        default="https://mastodon.social",
        description="Base URL of the Mastodon instance to query.",
    )
    mastodon_post_limit: int = Field(
        default=20,
        ge=1,
        le=40,
        description="Maximum number of statuses to fetch per Mastodon poll.",
    )

    # ------------------------------------------------------------------ #
    # LLM / OpenAI classifier
    # ------------------------------------------------------------------ #
    openai_api_key: str = Field(
        default="",
        description=(
            "OpenAI API key.  Leave empty to use the built-in rule-based "
            "stub classifier (no network calls, safe for offline use)."
        ),
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name to use for classification.",
    )
    classifier_batch_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of posts sent in a single LLM classification request.",
    )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    database_url: str = Field(
        default="signal_dash.db",
        description=(
            "Path to the SQLite database file.  Use ':memory:' for an "
            "in-memory database (useful in tests)."
        ),
    )

    # ------------------------------------------------------------------ #
    # Server
    # ------------------------------------------------------------------ #
    host: str = Field(default="0.0.0.0", description="Host address for uvicorn.")
    port: int = Field(default=8000, ge=1, le=65535, description="Port for uvicorn.")
    debug: bool = Field(
        default=False,
        description="Enable FastAPI debug mode and auto-reload.",
    )

    # ------------------------------------------------------------------ #
    # Computed helpers
    # ------------------------------------------------------------------ #
    @property
    def use_stub_classifier(self) -> bool:
        """Return ``True`` when no OpenAI key is configured.

        When ``True`` the application falls back to the built-in rule-based
        classifier so it can operate fully offline without an API key.
        """
        return not self.openai_api_key.strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application settings singleton.

    The result is cached after the first call so that environment variables
    are read only once per process.  Call ``get_settings.cache_clear()`` in
    tests to force a fresh read.

    Returns
    -------
    Settings
        Validated, fully-populated settings object.
    """
    return Settings()
