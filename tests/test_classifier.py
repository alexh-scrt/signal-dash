"""Unit tests for signal_dash.classifier.

Tests cover:
- The stub (rule-based) classifier with various sentiment inputs.
- The LLM response parser with mocked/stubbed responses.
- The public classify_posts() entry point.
- Topic extraction logic.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from signal_dash.classifier import (
    _classify_stub,
    _extract_topics,
    _parse_llm_response,
    classify_posts,
)
from signal_dash.models import Post, Sentiment, Signal, Source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_post(
    source_id: str = "t3_test",
    title: str = "",
    body: str = "",
    score: int = 0,
    source: Source = Source.REDDIT,
    keyword: str = "python",
) -> Post:
    """Build a minimal Post for testing."""
    return Post(
        source_id=source_id,
        source=source,
        title=title or None,
        body=body,
        url=f"https://example.com/{source_id}",
        score=score,
        keyword=keyword,
        fetched_at=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# _extract_topics
# ---------------------------------------------------------------------------


class TestExtractTopics:
    """Tests for the topic extraction helper."""

    def test_detects_python_topic(self) -> None:
        """'python' keyword should trigger the python topic tag."""
        topics = _extract_topics("i love python and fastapi")
        assert "python" in topics

    def test_detects_ai_topic(self) -> None:
        """'gpt' or 'llm' should trigger the ai topic tag."""
        topics = _extract_topics("chatgpt is an interesting llm")
        assert "ai" in topics

    def test_detects_security_topic(self) -> None:
        """Security-related keywords should trigger the security tag."""
        topics = _extract_topics("there is a new cve vulnerability discovered")
        assert "security" in topics

    def test_caps_at_three_topics(self) -> None:
        """At most 3 topics should be returned even if more match."""
        # This text mentions ai, python, security, web — should get exactly 3
        text = "python gpt vulnerability api web cloud data open source"
        topics = _extract_topics(text)
        assert len(topics) <= 3

    def test_no_match_returns_empty(self) -> None:
        """Text with no recognisable keywords should return empty list."""
        topics = _extract_topics("xyzzy quux frobble")
        assert topics == []

    def test_case_insensitive_matching(self) -> None:
        """Topics should be matched on lowercased text."""
        # _extract_topics receives already-lowercased text
        topics = _extract_topics("python is great")
        assert "python" in topics

    def test_web_topic_detected(self) -> None:
        """'api' keyword should trigger the web topic."""
        topics = _extract_topics("built a rest api with fastapi")
        assert "web" in topics

    def test_cloud_topic_detected(self) -> None:
        """Cloud-related keywords should trigger the cloud topic."""
        topics = _extract_topics("deploying containers on kubernetes")
        assert "cloud" in topics


# ---------------------------------------------------------------------------
# _classify_stub — sentiment scoring
# ---------------------------------------------------------------------------


class TestClassifyStubSentiment:
    """Tests for stub classifier sentiment scoring."""

    def test_positive_text_yields_positive_score(self) -> None:
        """Text with positive words should produce a positive sentiment score."""
        post = _make_post(body="This is absolutely great and awesome and wonderful!")
        signal = _classify_stub(post)
        assert signal.sentiment_score > 0.0
        assert signal.sentiment_label == Sentiment.POSITIVE

    def test_negative_text_yields_negative_score(self) -> None:
        """Text with negative words should produce a negative sentiment score."""
        post = _make_post(
            body="This is terrible and awful. It crashed and failed badly."
        )
        signal = _classify_stub(post)
        assert signal.sentiment_score < 0.0
        assert signal.sentiment_label == Sentiment.NEGATIVE

    def test_neutral_text_yields_neutral_score(self) -> None:
        """Neutral text should produce a score close to zero."""
        post = _make_post(body="This is a post about something that happened today.")
        signal = _classify_stub(post)
        assert signal.sentiment_label == Sentiment.NEUTRAL

    def test_empty_text_yields_neutral(self) -> None:
        """A post with no text should produce a neutral score of 0.0."""
        post = _make_post(title=None, body="")
        signal = _classify_stub(post)
        assert signal.sentiment_score == 0.0
        assert signal.sentiment_label == Sentiment.NEUTRAL

    def test_score_in_valid_range(self) -> None:
        """Sentiment score must always be in [-1.0, 1.0]."""
        for body in [
            "great awesome excellent amazing wonderful best perfect",
            "terrible awful horrible bad worst hate useless broken",
            "",
        ]:
            post = _make_post(body=body)
            signal = _classify_stub(post)
            assert -1.0 <= signal.sentiment_score <= 1.0

    def test_title_contributes_to_sentiment(self) -> None:
        """Title should be included in sentiment analysis via full_text."""
        post = _make_post(title="Great news!", body="")
        signal = _classify_stub(post)
        assert signal.sentiment_score > 0.0

    def test_mixed_text_is_balanced(self) -> None:
        """Text with equal positive and negative words should be near neutral."""
        post = _make_post(
            body="love hate great terrible awesome awful good bad"
        )
        signal = _classify_stub(post)
        # Score should be somewhere in the neutral or near-neutral range
        assert -0.6 <= signal.sentiment_score <= 0.6


# ---------------------------------------------------------------------------
# _classify_stub — signal strength
# ---------------------------------------------------------------------------


class TestClassifyStubSignalStrength:
    """Tests for stub classifier signal strength scoring."""

    def test_strength_in_valid_range(self) -> None:
        """Signal strength must always be in [0.0, 1.0]."""
        for score in [0, 1, 10, 100, 1000, 10000]:
            post = _make_post(body="great python library", score=score)
            signal = _classify_stub(post)
            assert 0.0 <= signal.signal_strength <= 1.0

    def test_high_engagement_increases_strength(self) -> None:
        """A post with higher score should generally have higher signal strength."""
        low_post = _make_post(body="great library", score=0)
        high_post = _make_post(body="great library", score=5000)
        low_sig = _classify_stub(low_post)
        high_sig = _classify_stub(high_post)
        assert high_sig.signal_strength >= low_sig.signal_strength

    def test_strong_sentiment_contributes_to_strength(self) -> None:
        """A post with strong sentiment should have higher strength than neutral."""
        strong_post = _make_post(
            body="absolutely terrible and broken crash fail error bug"
        )
        neutral_post = _make_post(body="a thing happened today")
        strong_sig = _classify_stub(strong_post)
        neutral_sig = _classify_stub(neutral_post)
        assert strong_sig.signal_strength >= neutral_sig.signal_strength


# ---------------------------------------------------------------------------
# _classify_stub — topic extraction
# ---------------------------------------------------------------------------


class TestClassifyStubTopics:
    """Tests for topic extraction within the stub classifier."""

    def test_topics_extracted(self) -> None:
        """Topics should be extracted from the post text."""
        post = _make_post(body="I love python and building rest apis with fastapi")
        signal = _classify_stub(post)
        assert len(signal.topics) >= 1

    def test_topics_are_lowercase(self) -> None:
        """All extracted topics should be lowercase strings."""
        post = _make_post(body="Python LLM cloud kubernetes")
        signal = _classify_stub(post)
        for topic in signal.topics:
            assert topic == topic.lower()

    def test_no_topics_for_generic_text(self) -> None:
        """Generic text without topic keywords should yield empty or minimal topics."""
        post = _make_post(body="the weather today is nice and sunny")
        signal = _classify_stub(post)
        # Should have 0 topics since no domain keywords appear
        assert len(signal.topics) == 0

    def test_at_most_three_topics(self) -> None:
        """Stub classifier should never produce more than 3 topic tags."""
        post = _make_post(
            body="python ai cloud security web data open-source community jobs performance"
        )
        signal = _classify_stub(post)
        assert len(signal.topics) <= 3


# ---------------------------------------------------------------------------
# _classify_stub — Signal structure
# ---------------------------------------------------------------------------


class TestClassifyStubSignalStructure:
    """Tests that verify the Signal object produced by the stub is well-formed."""

    def test_returns_signal_instance(self) -> None:
        """_classify_stub() should return a Signal instance."""
        post = _make_post(body="test post")
        result = _classify_stub(post)
        assert isinstance(result, Signal)

    def test_source_id_propagated(self) -> None:
        """source_id should be carried from Post to Signal."""
        post = _make_post(source_id="t3_unique", body="test")
        signal = _classify_stub(post)
        assert signal.source_id == "t3_unique"

    def test_source_propagated(self) -> None:
        """source should be carried from Post to Signal."""
        post = _make_post(source=Source.MASTODON, body="test")
        signal = _classify_stub(post)
        assert signal.source == Source.MASTODON

    def test_keyword_propagated(self) -> None:
        """keyword should be carried from Post to Signal."""
        post = _make_post(keyword="fastapi", body="test")
        signal = _classify_stub(post)
        assert signal.keyword == "fastapi"

    def test_url_propagated(self) -> None:
        """URL should be carried from Post to Signal."""
        post = _make_post(source_id="t3_url_test", body="test")
        signal = _classify_stub(post)
        assert signal.url == post.url

    def test_post_score_propagated(self) -> None:
        """Platform score should be carried from Post to Signal as post_score."""
        post = _make_post(body="test", score=77)
        signal = _classify_stub(post)
        assert signal.post_score == 77

    def test_sentiment_label_consistent_with_score(self) -> None:
        """sentiment_label must be consistent with sentiment_score."""
        for body in [
            "great awesome wonderful amazing",
            "terrible broken failed crash error",
            "the cat sat on the mat",
        ]:
            post = _make_post(body=body)
            signal = _classify_stub(post)
            if signal.sentiment_score > 0.2:
                assert signal.sentiment_label == Sentiment.POSITIVE
            elif signal.sentiment_score < -0.2:
                assert signal.sentiment_label == Sentiment.NEGATIVE
            else:
                assert signal.sentiment_label == Sentiment.NEUTRAL


# ---------------------------------------------------------------------------
# _parse_llm_response
# ---------------------------------------------------------------------------


class TestParseLlmResponse:
    """Tests for the LLM JSON response parser."""

    def _make_llm_json(self, items: list[dict[str, Any]]) -> str:
        """Serialise a list of classification dicts to JSON string."""
        return json.dumps(items)

    def test_parses_valid_response(self) -> None:
        """A well-formed JSON array should produce matching Signal objects."""
        posts = [_make_post(source_id="t3_1", body="great python library")]
        raw = self._make_llm_json(
            [{"sentiment_score": 0.8, "topics": ["python"], "signal_strength": 0.7}]
        )
        signals = _parse_llm_response(raw, posts)
        assert len(signals) == 1
        assert signals[0].sentiment_score == pytest.approx(0.8)
        assert "python" in signals[0].topics
        assert signals[0].signal_strength == pytest.approx(0.7)

    def test_parses_multiple_posts(self) -> None:
        """Parser should handle multiple posts in a single response."""
        posts = [_make_post(source_id=f"t3_{i}") for i in range(3)]
        raw = self._make_llm_json(
            [
                {"sentiment_score": 0.5, "topics": ["python"], "signal_strength": 0.6},
                {"sentiment_score": -0.3, "topics": ["security"], "signal_strength": 0.4},
                {"sentiment_score": 0.0, "topics": [], "signal_strength": 0.1},
            ]
        )
        signals = _parse_llm_response(raw, posts)
        assert len(signals) == 3

    def test_strips_markdown_fences(self) -> None:
        """Markdown code fences around the JSON should be stripped."""
        posts = [_make_post(body="test")]
        raw = "```json\n[{\"sentiment_score\": 0.3, \"topics\": [], \"signal_strength\": 0.4}]\n```"
        signals = _parse_llm_response(raw, posts)
        assert len(signals) == 1
        assert signals[0].sentiment_score == pytest.approx(0.3)

    def test_clamped_sentiment_score(self) -> None:
        """Sentiment scores outside [-1, 1] from LLM should be clamped."""
        posts = [_make_post(body="test")]
        raw = self._make_llm_json(
            [{"sentiment_score": 2.5, "topics": [], "signal_strength": 0.5}]
        )
        signals = _parse_llm_response(raw, posts)
        assert signals[0].sentiment_score <= 1.0

    def test_clamped_signal_strength(self) -> None:
        """Signal strength outside [0, 1] from LLM should be clamped."""
        posts = [_make_post(body="test")]
        raw = self._make_llm_json(
            [{"sentiment_score": 0.0, "topics": [], "signal_strength": -5.0}]
        )
        signals = _parse_llm_response(raw, posts)
        assert signals[0].signal_strength >= 0.0

    def test_mismatched_length_falls_back_to_stub(self) -> None:
        """If LLM returns wrong number of results, stub fallback is used."""
        posts = [_make_post(source_id=f"t3_{i}") for i in range(3)]
        # Only 2 results for 3 posts
        raw = self._make_llm_json(
            [
                {"sentiment_score": 0.5, "topics": [], "signal_strength": 0.5},
                {"sentiment_score": 0.5, "topics": [], "signal_strength": 0.5},
            ]
        )
        signals = _parse_llm_response(raw, posts)
        # Should still return 3 signals (stub fallback for all)
        assert len(signals) == 3

    def test_invalid_json_falls_back_to_stub(self) -> None:
        """Invalid JSON should trigger stub fallback for all posts."""
        posts = [_make_post(body="test"), _make_post(source_id="t3_2", body="test2")]
        signals = _parse_llm_response("this is not json", posts)
        assert len(signals) == 2
        # Stub fallback should produce valid signals
        for sig in signals:
            assert isinstance(sig, Signal)

    def test_empty_array_falls_back_for_nonempty_posts(self) -> None:
        """An empty JSON array for non-empty posts should trigger stub fallback."""
        posts = [_make_post(body="test")]
        signals = _parse_llm_response("[]", posts)
        assert len(signals) == 1
        assert isinstance(signals[0], Signal)

    def test_topics_lowercased(self) -> None:
        """Topics from LLM should be lowercased."""
        posts = [_make_post(body="test")]
        raw = self._make_llm_json(
            [{"sentiment_score": 0.1, "topics": ["Python", "AI"], "signal_strength": 0.5}]
        )
        signals = _parse_llm_response(raw, posts)
        for topic in signals[0].topics:
            assert topic == topic.lower()

    def test_topics_capped_at_three(self) -> None:
        """More than 3 topics from LLM should be truncated to 3."""
        posts = [_make_post(body="test")]
        raw = self._make_llm_json(
            [
                {
                    "sentiment_score": 0.0,
                    "topics": ["a", "b", "c", "d", "e"],
                    "signal_strength": 0.5,
                }
            ]
        )
        signals = _parse_llm_response(raw, posts)
        assert len(signals[0].topics) <= 3

    def test_no_json_array_falls_back(self) -> None:
        """Response with no JSON array brackets should use stub fallback."""
        posts = [_make_post(body="test")]
        signals = _parse_llm_response("no brackets here at all", posts)
        assert len(signals) == 1
        assert isinstance(signals[0], Signal)


# ---------------------------------------------------------------------------
# classify_posts — integration
# ---------------------------------------------------------------------------


class TestClassifyPosts:
    """Integration tests for the classify_posts() entry point."""

    async def test_empty_posts_returns_empty(self) -> None:
        """classify_posts([]) should immediately return an empty list."""
        result = await classify_posts([])
        assert result == []

    async def test_stub_mode_without_key(self) -> None:
        """Without an API key, classify_posts() should use the stub classifier."""
        posts = [_make_post(body="great python library") for _ in range(3)]
        signals = await classify_posts(posts, openai_api_key="")
        assert len(signals) == 3
        for sig in signals:
            assert isinstance(sig, Signal)
            assert -1.0 <= sig.sentiment_score <= 1.0

    async def test_stub_returns_one_signal_per_post(self) -> None:
        """Stub mode should return exactly one signal per input post."""
        posts = [_make_post(source_id=f"t3_{i}", body=f"post number {i}") for i in range(7)]
        signals = await classify_posts(posts, openai_api_key="")
        assert len(signals) == len(posts)

    async def test_stub_preserves_source_ids(self) -> None:
        """Each signal's source_id should match the corresponding post."""
        posts = [
            _make_post(source_id="t3_a", body="great"),
            _make_post(source_id="t3_b", body="terrible"),
        ]
        signals = await classify_posts(posts, openai_api_key="")
        assert signals[0].source_id == "t3_a"
        assert signals[1].source_id == "t3_b"

    async def test_llm_mode_called_with_key(self) -> None:
        """With an API key, classify_posts() should call the LLM path."""
        posts = [_make_post(body="test post")]
        # Patch the internal LLM function so we don't make real API calls
        mock_signal = _classify_stub(posts[0])

        with patch(
            "signal_dash.classifier._classify_with_llm",
            new=AsyncMock(return_value=[mock_signal]),
        ) as mock_llm:
            signals = await classify_posts(
                posts, openai_api_key="sk-test-key"
            )
            mock_llm.assert_awaited_once()
        assert len(signals) == 1

    async def test_whitespace_only_key_uses_stub(self) -> None:
        """A whitespace-only API key should be treated as absent (stub mode)."""
        posts = [_make_post(body="test")]
        signals = await classify_posts(posts, openai_api_key="   ")
        assert len(signals) == 1
        assert isinstance(signals[0], Signal)

    async def test_signals_have_valid_sentiment_labels(self) -> None:
        """All returned signals should have a valid Sentiment label."""
        posts = [
            _make_post(body="this is great and awesome"),
            _make_post(source_id="t3_b", body="this is terrible and broken"),
            _make_post(source_id="t3_c", body="something happened today"),
        ]
        signals = await classify_posts(posts, openai_api_key="")
        valid_labels = {Sentiment.POSITIVE, Sentiment.NEGATIVE, Sentiment.NEUTRAL}
        for sig in signals:
            assert sig.sentiment_label in valid_labels

    async def test_signals_have_valid_strength_range(self) -> None:
        """All signal_strength values should be in [0.0, 1.0]."""
        posts = [_make_post(source_id=f"t3_{i}", body=f"post {i}", score=i * 10) for i in range(5)]
        signals = await classify_posts(posts, openai_api_key="")
        for sig in signals:
            assert 0.0 <= sig.signal_strength <= 1.0
