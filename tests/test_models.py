"""Unit tests for signal_dash.models — Post, Signal, and DashboardConfig."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from signal_dash.models import (
    DashboardConfig,
    Post,
    Sentiment,
    Signal,
    Source,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_post(**overrides: object) -> Post:
    """Return a minimal valid Post, optionally overriding fields."""
    defaults: dict[str, object] = {
        "source_id": "t3_abc123",
        "source": Source.REDDIT,
        "body": "Hello world",
        "url": "https://reddit.com/r/python/t3_abc123",
        "keyword": "python",
    }
    defaults.update(overrides)
    return Post(**defaults)  # type: ignore[arg-type]


def _make_signal(**overrides: object) -> Signal:
    """Return a minimal valid Signal, optionally overriding fields."""
    defaults: dict[str, object] = {
        "source_id": "t3_abc123",
        "source": Source.REDDIT,
        "keyword": "python",
        "url": "https://reddit.com/r/python/t3_abc123",
        "sentiment_score": 0.5,
        "sentiment_label": Sentiment.POSITIVE,
        "topics": ["python", "programming"],
        "signal_strength": 0.8,
    }
    defaults.update(overrides)
    return Signal(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Post tests
# ---------------------------------------------------------------------------


class TestPost:
    """Tests for the Post Pydantic model."""

    def test_minimal_valid_post(self) -> None:
        """A post with required fields should be created without errors."""
        post = _make_post()
        assert post.source_id == "t3_abc123"
        assert post.source == Source.REDDIT
        assert post.keyword == "python"

    def test_defaults(self) -> None:
        """Optional fields should have sensible defaults."""
        post = _make_post()
        assert post.author is None
        assert post.title is None
        assert post.body == "Hello world"
        assert post.score == 0

    def test_fetched_at_defaults_to_utc_now(self) -> None:
        """fetched_at should default to a UTC-aware datetime close to now."""
        before = datetime.now(tz=timezone.utc)
        post = _make_post()
        after = datetime.now(tz=timezone.utc)
        assert before <= post.fetched_at <= after
        assert post.fetched_at.tzinfo is not None

    def test_naive_fetched_at_coerced_to_utc(self) -> None:
        """A naive datetime supplied for fetched_at should be made UTC-aware."""
        naive = datetime(2024, 1, 1, 12, 0, 0)
        post = _make_post(fetched_at=naive)
        assert post.fetched_at.tzinfo == timezone.utc

    def test_iso_string_fetched_at(self) -> None:
        """An ISO 8601 string for fetched_at should be parsed correctly."""
        post = _make_post(fetched_at="2024-06-01T10:00:00")
        assert post.fetched_at.year == 2024
        assert post.fetched_at.month == 6

    def test_full_text_title_and_body(self) -> None:
        """full_text should concatenate title and body with a space."""
        post = _make_post(title="My Title", body="My body text.")
        assert post.full_text == "My Title My body text."

    def test_full_text_title_only(self) -> None:
        """full_text should return just the title when body is empty."""
        post = _make_post(title="Only Title", body="")
        assert post.full_text == "Only Title"

    def test_full_text_body_only(self) -> None:
        """full_text should return just the body when title is None."""
        post = _make_post(title=None, body="Only body.")
        assert post.full_text == "Only body."

    def test_full_text_empty(self) -> None:
        """full_text should be empty string when both title and body are empty."""
        post = _make_post(title=None, body="")
        assert post.full_text == ""

    def test_missing_source_id_raises(self) -> None:
        """Omitting source_id should raise ValidationError."""
        with pytest.raises(ValidationError):
            Post(
                source=Source.REDDIT,
                body="hello",
                url="https://reddit.com/abc",
                keyword="python",
            )

    def test_missing_url_raises(self) -> None:
        """Omitting url should raise ValidationError."""
        with pytest.raises(ValidationError):
            Post(
                source_id="t3_abc",
                source=Source.REDDIT,
                body="hello",
                keyword="python",
            )

    def test_mastodon_source(self) -> None:
        """Source.MASTODON should be accepted."""
        post = _make_post(source=Source.MASTODON, source_id="109876")
        assert post.source == Source.MASTODON

    def test_string_whitespace_stripped(self) -> None:
        """String fields should have surrounding whitespace stripped."""
        post = _make_post(keyword="  python  ", source_id="  t3_abc123  ")
        assert post.keyword == "python"
        assert post.source_id == "t3_abc123"


# ---------------------------------------------------------------------------
# Signal tests
# ---------------------------------------------------------------------------


class TestSignal:
    """Tests for the Signal Pydantic model."""

    def test_minimal_valid_signal(self) -> None:
        """A signal with required fields should be created without errors."""
        sig = _make_signal()
        assert sig.source_id == "t3_abc123"
        assert sig.sentiment_score == 0.5

    def test_sentiment_label_derived_positive(self) -> None:
        """Scores above 0.2 should yield a POSITIVE label."""
        sig = _make_signal(sentiment_score=0.3)
        assert sig.sentiment_label == Sentiment.POSITIVE

    def test_sentiment_label_derived_negative(self) -> None:
        """Scores below -0.2 should yield a NEGATIVE label."""
        sig = _make_signal(sentiment_score=-0.5)
        assert sig.sentiment_label == Sentiment.NEGATIVE

    def test_sentiment_label_derived_neutral_positive_boundary(self) -> None:
        """Score of exactly 0.2 should yield NEUTRAL (boundary is inclusive)."""
        sig = _make_signal(sentiment_score=0.2)
        assert sig.sentiment_label == Sentiment.NEUTRAL

    def test_sentiment_label_derived_neutral_negative_boundary(self) -> None:
        """Score of exactly -0.2 should yield NEUTRAL."""
        sig = _make_signal(sentiment_score=-0.2)
        assert sig.sentiment_label == Sentiment.NEUTRAL

    def test_sentiment_label_overwritten_by_validator(self) -> None:
        """Even if a wrong label is supplied, the validator corrects it."""
        # Supply POSITIVE label but a negative score — validator should fix it.
        sig = _make_signal(
            sentiment_score=-0.9,
            sentiment_label=Sentiment.POSITIVE,  # intentionally wrong
        )
        assert sig.sentiment_label == Sentiment.NEGATIVE

    def test_sentiment_score_out_of_range_raises(self) -> None:
        """Scores outside [-1.0, 1.0] should raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_signal(sentiment_score=1.1)
        with pytest.raises(ValidationError):
            _make_signal(sentiment_score=-1.1)

    def test_signal_strength_out_of_range_raises(self) -> None:
        """Signal strength outside [0.0, 1.0] should raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_signal(signal_strength=1.1)
        with pytest.raises(ValidationError):
            _make_signal(signal_strength=-0.1)

    def test_topics_normalised_to_lowercase(self) -> None:
        """Topic tags should be lowercased by the validator."""
        sig = _make_signal(topics=["Python", "ML", "AI"])
        assert sig.topics == ["python", "ml", "ai"]

    def test_topics_capped_at_three(self) -> None:
        """More than 3 topics should be silently truncated to 3."""
        sig = _make_signal(topics=["a", "b", "c", "d", "e"])
        assert len(sig.topics) == 3

    def test_topics_deduplicated(self) -> None:
        """Duplicate topic tags should be removed, keeping first occurrence."""
        sig = _make_signal(topics=["python", "python", "ml"])
        assert sig.topics == ["python", "ml"]

    def test_topics_from_comma_string(self) -> None:
        """A comma-separated string should be parsed into a list of topics."""
        sig = _make_signal(topics="python, ml, ai")  # type: ignore[arg-type]
        assert sig.topics == ["python", "ml", "ai"]

    def test_topics_none_becomes_empty_list(self) -> None:
        """None for topics should produce an empty list."""
        sig = _make_signal(topics=None)  # type: ignore[arg-type]
        assert sig.topics == []

    def test_classified_at_defaults_to_utc_now(self) -> None:
        """classified_at should default to a UTC-aware datetime."""
        before = datetime.now(tz=timezone.utc)
        sig = _make_signal()
        after = datetime.now(tz=timezone.utc)
        assert before <= sig.classified_at <= after
        assert sig.classified_at.tzinfo is not None

    def test_naive_classified_at_coerced(self) -> None:
        """A naive datetime for classified_at should be coerced to UTC-aware."""
        sig = _make_signal(classified_at=datetime(2024, 3, 15, 8, 0, 0))
        assert sig.classified_at.tzinfo == timezone.utc

    def test_from_post_factory(self) -> None:
        """Signal.from_post() should produce a consistent Signal from a Post."""
        post = _make_post(
            source_id="t3_xyz",
            source=Source.REDDIT,
            title="Great library",
            body="I love using this.",
            url="https://reddit.com/r/python/t3_xyz",
            author="alice",
            score=42,
            keyword="python",
        )
        sig = Signal.from_post(
            post=post,
            sentiment_score=0.8,
            topics=["library", "python"],
            signal_strength=0.9,
        )
        assert sig.source_id == "t3_xyz"
        assert sig.source == Source.REDDIT
        assert sig.keyword == "python"
        assert sig.title == "Great library"
        assert sig.body == "I love using this."
        assert sig.author == "alice"
        assert sig.post_score == 42
        assert sig.sentiment_score == 0.8
        assert sig.sentiment_label == Sentiment.POSITIVE
        assert sig.topics == ["library", "python"]
        assert sig.signal_strength == 0.9

    def test_from_post_factory_negative_sentiment(self) -> None:
        """from_post() with a negative score should produce a NEGATIVE label."""
        post = _make_post()
        sig = Signal.from_post(
            post=post,
            sentiment_score=-0.7,
            topics=[],
            signal_strength=0.5,
        )
        assert sig.sentiment_label == Sentiment.NEGATIVE


# ---------------------------------------------------------------------------
# DashboardConfig tests
# ---------------------------------------------------------------------------


class TestDashboardConfig:
    """Tests for the DashboardConfig model."""

    def test_valid_config(self) -> None:
        """A fully populated DashboardConfig should be accepted."""
        cfg = DashboardConfig(
            keyword="openai",
            sources=[Source.REDDIT],
            refresh_interval_seconds=60,
            classifier_mode="llm",
        )
        assert cfg.keyword == "openai"
        assert cfg.classifier_mode == "llm"

    def test_stub_classifier_mode(self) -> None:
        """classifier_mode='stub' should be accepted."""
        cfg = DashboardConfig(
            keyword="test",
            sources=[Source.MASTODON],
            refresh_interval_seconds=120,
            classifier_mode="stub",
        )
        assert cfg.classifier_mode == "stub"

    def test_invalid_classifier_mode_raises(self) -> None:
        """An unrecognised classifier_mode should raise ValidationError."""
        with pytest.raises(ValidationError):
            DashboardConfig(
                keyword="test",
                sources=[Source.REDDIT],
                refresh_interval_seconds=60,
                classifier_mode="openai",  # not 'llm' or 'stub'
            )

    def test_refresh_interval_below_minimum_raises(self) -> None:
        """refresh_interval_seconds < 30 should raise ValidationError."""
        with pytest.raises(ValidationError):
            DashboardConfig(
                keyword="test",
                sources=[Source.REDDIT],
                refresh_interval_seconds=10,
                classifier_mode="stub",
            )

    def test_default_app_version(self) -> None:
        """app_version should default to '0.1.0'."""
        cfg = DashboardConfig(
            keyword="x",
            sources=[Source.REDDIT],
            refresh_interval_seconds=30,
            classifier_mode="stub",
        )
        assert cfg.app_version == "0.1.0"

    def test_from_settings_stub_mode(self) -> None:
        """from_settings() with no OpenAI key should produce 'stub' mode."""
        from signal_dash.config import Settings

        settings = Settings(openai_api_key="", keyword="fastapi", sources=["reddit"])
        cfg = DashboardConfig.from_settings(settings)
        assert cfg.keyword == "fastapi"
        assert cfg.classifier_mode == "stub"
        assert Source.REDDIT in cfg.sources

    def test_from_settings_llm_mode(self) -> None:
        """from_settings() with an OpenAI key should produce 'llm' mode."""
        from signal_dash.config import Settings

        settings = Settings(
            openai_api_key="sk-test-xyz",
            keyword="mastodon",
            sources=["mastodon"],
        )
        cfg = DashboardConfig.from_settings(settings)
        assert cfg.classifier_mode == "llm"
        assert Source.MASTODON in cfg.sources

    def test_from_settings_both_sources(self) -> None:
        """from_settings() with both sources should include REDDIT and MASTODON."""
        from signal_dash.config import Settings

        settings = Settings(sources=["reddit", "mastodon"])
        cfg = DashboardConfig.from_settings(settings)
        assert Source.REDDIT in cfg.sources
        assert Source.MASTODON in cfg.sources
