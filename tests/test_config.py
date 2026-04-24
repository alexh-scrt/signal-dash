"""Unit tests for signal_dash.config — settings loading and validation."""

from __future__ import annotations

import pytest

from signal_dash.config import Settings, get_settings


class TestSettingsDefaults:
    """Verify that the default values satisfy expected constraints."""

    def setup_method(self) -> None:
        """Clear the LRU cache before each test so env vars are re-read."""
        get_settings.cache_clear()

    def test_default_keyword(self) -> None:
        """Default keyword should be 'python'."""
        s = Settings()
        assert s.keyword == "python"

    def test_default_sources(self) -> None:
        """Default sources should include both reddit and mastodon."""
        s = Settings()
        assert "reddit" in s.sources
        assert "mastodon" in s.sources

    def test_default_refresh_interval(self) -> None:
        """Default refresh interval should be 300 seconds."""
        s = Settings()
        assert s.refresh_interval_seconds == 300

    def test_default_database_url(self) -> None:
        """Default database URL should point to a local SQLite file."""
        s = Settings()
        assert s.database_url == "signal_dash.db"

    def test_default_openai_model(self) -> None:
        """Default OpenAI model should be gpt-4o-mini."""
        s = Settings()
        assert s.openai_model == "gpt-4o-mini"

    def test_default_port(self) -> None:
        """Default port should be 8000."""
        s = Settings()
        assert s.port == 8000


class TestStubClassifierProperty:
    """Verify the use_stub_classifier computed property."""

    def test_stub_when_no_key(self) -> None:
        """Stub mode active when openai_api_key is empty."""
        s = Settings(openai_api_key="")
        assert s.use_stub_classifier is True

    def test_stub_when_whitespace_only(self) -> None:
        """Stub mode active when openai_api_key contains only whitespace."""
        s = Settings(openai_api_key="   ")
        assert s.use_stub_classifier is True

    def test_llm_when_key_provided(self) -> None:
        """LLM mode active when a non-empty API key is supplied."""
        s = Settings(openai_api_key="sk-test-1234")
        assert s.use_stub_classifier is False


class TestSourcesValidator:
    """Test the comma-string parsing for the sources field."""

    def test_parse_comma_string(self) -> None:
        """A comma-separated string should be split into a list."""
        s = Settings(sources="reddit,mastodon")  # type: ignore[arg-type]
        assert s.sources == ["reddit", "mastodon"]

    def test_parse_list(self) -> None:
        """A list value should pass through unchanged."""
        s = Settings(sources=["reddit"])  # type: ignore[arg-type]
        assert s.sources == ["reddit"]

    def test_strips_whitespace(self) -> None:
        """Leading/trailing whitespace around source names should be stripped."""
        s = Settings(sources=" reddit , mastodon ")  # type: ignore[arg-type]
        assert s.sources == ["reddit", "mastodon"]

    def test_invalid_source_raises(self) -> None:
        """An unrecognised source name should raise a ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(sources=["twitter"])  # type: ignore[arg-type]


class TestGetSettingsSingleton:
    """Verify caching behaviour of get_settings()."""

    def setup_method(self) -> None:
        get_settings.cache_clear()

    def teardown_method(self) -> None:
        get_settings.cache_clear()

    def test_returns_settings_instance(self) -> None:
        """get_settings() should return a Settings instance."""
        s = get_settings()
        assert isinstance(s, Settings)

    def test_cached_identity(self) -> None:
        """Repeated calls should return the identical cached object."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clear_returns_new_instance(self) -> None:
        """After cache_clear(), a fresh Settings object is created."""
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        # They should be equal in value but not the same object in memory
        assert s1 is not s2
        assert s1.keyword == s2.keyword
