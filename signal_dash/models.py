"""Pydantic models for signal_dash.

Defines the core data structures used across ingestion, classification,
persistence, and API response layers:

- ``Post``           — a raw social media post fetched from Reddit or Mastodon.
- ``Signal``         — a classified post enriched with sentiment, topic tags,
                       and a signal strength score.
- ``DashboardConfig`` — runtime configuration snapshot surfaced to the
                       Jinja2 templates and HTMX partial endpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Source(str, Enum):
    """Supported social media sources."""

    REDDIT = "reddit"
    MASTODON = "mastodon"


class Sentiment(str, Enum):
    """Human-readable sentiment bucket derived from the numeric score.

    The numeric ``sentiment_score`` on :class:`Signal` is the authoritative
    value; this enum provides a coarse label for display purposes.
    """

    POSITIVE = "positive"   # score >  0.2
    NEUTRAL = "neutral"     # score in [-0.2, 0.2]
    NEGATIVE = "negative"   # score < -0.2


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Sentiment score in the closed interval [-1.0, 1.0].
SentimentScore = Annotated[float, Field(ge=-1.0, le=1.0)]

# Signal strength in the closed interval [0.0, 1.0].
SignalStrength = Annotated[float, Field(ge=0.0, le=1.0)]

# A topic tag is a short, lowercase, non-empty string.
TopicTag = Annotated[str, Field(min_length=1, max_length=64)]


# ---------------------------------------------------------------------------
# Post model
# ---------------------------------------------------------------------------


class Post(BaseModel):
    """A raw social media post fetched from Reddit or Mastodon.

    Attributes
    ----------
    source_id:
        The platform-native identifier for the post (e.g. Reddit fullname
        ``t3_abc123`` or Mastodon status ID).  Used for deduplication.
    source:
        Which platform the post came from.
    author:
        Display name or username of the post author.  ``None`` when the
        author information is unavailable or redacted.
    title:
        Post title (Reddit) or ``None`` for platforms without a title field.
    body:
        Main textual content of the post.  May be empty for link-only posts.
    url:
        Canonical URL to the original post, stored as a plain string to
        avoid serialisation round-trip issues with Pydantic's ``HttpUrl``.
    score:
        Platform-native engagement score (upvotes, boosts, etc.).  Defaults
        to ``0`` when not provided by the source API.
    keyword:
        The search keyword that surfaced this post.
    fetched_at:
        UTC timestamp recording when the post was ingested.  Defaults to
        *now* if not supplied.
    """

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    source_id: str = Field(
        ...,
        description="Platform-native post identifier used for deduplication.",
        min_length=1,
    )
    source: Source = Field(
        ...,
        description="Social media platform this post was fetched from.",
    )
    author: str | None = Field(
        default=None,
        description="Author display name; None when unavailable.",
    )
    title: str | None = Field(
        default=None,
        description="Post title (Reddit-style); None for title-less platforms.",
    )
    body: str = Field(
        default="",
        description="Main textual content of the post.",
    )
    url: str = Field(
        ...,
        description="Canonical URL to the original post.",
        min_length=1,
    )
    score: int = Field(
        default=0,
        description="Platform engagement score (upvotes, boosts, etc.).",
    )
    keyword: str = Field(
        ...,
        description="Search keyword that surfaced this post.",
        min_length=1,
    )
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp of ingestion.",
    )

    @field_validator("fetched_at", mode="before")
    @classmethod
    def _ensure_utc(cls, value: object) -> datetime:
        """Coerce naive datetimes to UTC-aware.

        Parameters
        ----------
        value:
            Raw value passed to the field validator.

        Returns
        -------
        datetime
            A timezone-aware UTC datetime.
        """
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        return value  # let Pydantic raise if the type is truly wrong

    @property
    def full_text(self) -> str:
        """Return the combined title + body for classification.

        Returns
        -------
        str
            Whitespace-separated concatenation of non-empty title and body.
        """
        parts = [p for p in (self.title, self.body) if p]
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Signal model
# ---------------------------------------------------------------------------


class Signal(BaseModel):
    """A classified social media post enriched with LLM-derived metadata.

    A ``Signal`` is produced by passing a :class:`Post` through the
    classifier.  It extends the post's identifiers with sentiment analysis,
    topic tags, and a signal strength score.

    Attributes
    ----------
    source_id:
        Foreign key linking back to the originating :class:`Post`.
    source:
        Platform the post came from (denormalised for efficient querying).
    keyword:
        The search keyword that surfaced the originating post.
    title:
        Post title carried forward from the :class:`Post`.
    body:
        Post body carried forward from the :class:`Post`.
    url:
        Canonical URL carried forward from the :class:`Post`.
    author:
        Author display name carried forward from the :class:`Post`.
    post_score:
        Platform engagement score carried forward from the :class:`Post`.
    sentiment_score:
        Float in ``[-1.0, 1.0]``.  Negative values indicate negative
        sentiment; positive values indicate positive sentiment.
    sentiment_label:
        Coarse sentiment bucket derived from ``sentiment_score``.
    topics:
        Up to three short topic tags assigned by the classifier.
    signal_strength:
        Float in ``[0.0, 1.0]`` representing how strongly this post
        represents a trending signal (engagement × sentiment magnitude).
    classified_at:
        UTC timestamp of classification.  Defaults to *now*.
    """

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    source_id: str = Field(
        ...,
        description="Foreign key to the originating Post.",
        min_length=1,
    )
    source: Source = Field(
        ...,
        description="Platform the post originated from.",
    )
    keyword: str = Field(
        ...,
        description="Keyword that surfaced the post.",
        min_length=1,
    )
    title: str | None = Field(
        default=None,
        description="Post title carried forward from Post.",
    )
    body: str = Field(
        default="",
        description="Post body carried forward from Post.",
    )
    url: str = Field(
        ...,
        description="Canonical URL to the original post.",
        min_length=1,
    )
    author: str | None = Field(
        default=None,
        description="Author display name.",
    )
    post_score: int = Field(
        default=0,
        description="Platform engagement score.",
    )
    sentiment_score: SentimentScore = Field(
        ...,
        description="Numeric sentiment in [-1.0, 1.0].",
    )
    sentiment_label: Sentiment = Field(
        ...,
        description="Coarse sentiment label derived from sentiment_score.",
    )
    topics: list[TopicTag] = Field(
        default_factory=list,
        description="Up to 3 topic tags assigned by the classifier.",
        max_length=3,
    )
    signal_strength: SignalStrength = Field(
        ...,
        description="Trending signal strength in [0.0, 1.0].",
    )
    classified_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="UTC timestamp of classification.",
    )

    @field_validator("classified_at", mode="before")
    @classmethod
    def _ensure_utc(cls, value: object) -> datetime:
        """Coerce naive datetimes to UTC-aware.

        Parameters
        ----------
        value:
            Raw value passed to the field validator.

        Returns
        -------
        datetime
            A timezone-aware UTC datetime.
        """
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        return value

    @field_validator("topics", mode="before")
    @classmethod
    def _normalise_topics(
        cls, value: list[str] | str | None
    ) -> list[str]:
        """Normalise topics to a lowercase, deduplicated list of max 3.

        Accepts a raw list of strings or a comma-separated string (useful
        when deserialising from SQLite rows).

        Parameters
        ----------
        value:
            Raw topics value before validation.

        Returns
        -------
        list[str]
            Cleaned, deduplicated list of up to 3 topic tags.
        """
        if value is None:
            return []
        if isinstance(value, str):
            items: list[str] = [t.strip().lower() for t in value.split(",") if t.strip()]
        else:
            items = [t.strip().lower() for t in value if t.strip()]
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique[:3]

    @model_validator(mode="after")
    def _derive_sentiment_label(self) -> "Signal":
        """Ensure ``sentiment_label`` is consistent with ``sentiment_score``.

        This validator overwrites any user-supplied ``sentiment_label`` with
        the value derived from ``sentiment_score``, making the two fields
        always consistent.

        Returns
        -------
        Signal
            The validated model instance with a consistent sentiment label.
        """
        score = self.sentiment_score
        if score > 0.2:
            self.sentiment_label = Sentiment.POSITIVE
        elif score < -0.2:
            self.sentiment_label = Sentiment.NEGATIVE
        else:
            self.sentiment_label = Sentiment.NEUTRAL
        return self

    @classmethod
    def from_post(
        cls,
        post: Post,
        sentiment_score: float,
        topics: list[str],
        signal_strength: float,
    ) -> "Signal":
        """Construct a :class:`Signal` from a :class:`Post` and classifier output.

        Parameters
        ----------
        post:
            The raw post that was classified.
        sentiment_score:
            Numeric sentiment score in ``[-1.0, 1.0]``.
        topics:
            Up to three topic tags.
        signal_strength:
            Signal strength score in ``[0.0, 1.0]``.

        Returns
        -------
        Signal
            Fully populated signal ready for persistence.
        """
        return cls(
            source_id=post.source_id,
            source=post.source,
            keyword=post.keyword,
            title=post.title,
            body=post.body,
            url=post.url,
            author=post.author,
            post_score=post.score,
            sentiment_score=sentiment_score,
            # The model_validator will overwrite this; set a placeholder.
            sentiment_label=Sentiment.NEUTRAL,
            topics=topics,
            signal_strength=signal_strength,
        )


# ---------------------------------------------------------------------------
# DashboardConfig model
# ---------------------------------------------------------------------------


class DashboardConfig(BaseModel):
    """Runtime configuration snapshot surfaced to the dashboard UI.

    This lightweight model is instantiated from :class:`~signal_dash.config.Settings`
    and passed as context to Jinja2 templates and HTMX partial endpoints so
    that the UI can display active settings without exposing the full
    ``Settings`` object (which contains the OpenAI key).

    Attributes
    ----------
    keyword:
        Active monitoring keyword.
    sources:
        List of enabled source platforms.
    refresh_interval_seconds:
        Dashboard auto-refresh interval in seconds.
    classifier_mode:
        ``"llm"`` when backed by OpenAI, ``"stub"`` for the offline fallback.
    app_version:
        Package version string.
    """

    model_config = ConfigDict(populate_by_name=True)

    keyword: str = Field(
        ...,
        description="Active monitoring keyword.",
        min_length=1,
    )
    sources: list[Source] = Field(
        ...,
        description="Enabled source platforms.",
    )
    refresh_interval_seconds: int = Field(
        ...,
        ge=30,
        description="Auto-refresh interval in seconds.",
    )
    classifier_mode: str = Field(
        ...,
        description="'llm' or 'stub' indicating which classifier is active.",
        pattern=r"^(llm|stub)$",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application version string.",
    )

    @classmethod
    def from_settings(cls, settings: object) -> "DashboardConfig":
        """Build a :class:`DashboardConfig` from a ``Settings`` instance.

        This factory avoids a hard import of :class:`~signal_dash.config.Settings`
        at module level to prevent circular imports (``config`` → ``models``
        would create a cycle if ``models`` imported ``config``).

        Parameters
        ----------
        settings:
            A :class:`~signal_dash.config.Settings` instance (typed as
            ``object`` to avoid the circular import; duck-typed access is
            used instead).

        Returns
        -------
        DashboardConfig
            Populated dashboard configuration snapshot.

        Raises
        ------
        AttributeError
            If *settings* does not expose the expected attributes.
        """
        from signal_dash import __version__  # local import to avoid circularity

        return cls(
            keyword=settings.keyword,  # type: ignore[union-attr]
            sources=[Source(s) for s in settings.sources],  # type: ignore[union-attr]
            refresh_interval_seconds=settings.refresh_interval_seconds,  # type: ignore[union-attr]
            classifier_mode="stub" if settings.use_stub_classifier else "llm",  # type: ignore[union-attr]
            app_version=__version__,
        )
