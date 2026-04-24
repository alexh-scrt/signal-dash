"""Unit tests for signal_dash.ingest.

All tests use respx to mock HTTP responses, avoiding any real network calls.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
import respx

from signal_dash.ingest import (
    _parse_mastodon_response,
    _parse_reddit_response,
    _strip_html,
    _truncate,
    fetch_mastodon_posts,
    fetch_reddit_posts,
)
from signal_dash.models import Source


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _reddit_response(children: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap children in the standard Reddit search response envelope."""
    return {"data": {"children": children}}


def _reddit_child(
    name: str = "t3_abc123",
    title: str = "Test Post",
    selftext: str = "Some body text.",
    author: str = "alice",
    score: int = 42,
    permalink: str = "/r/python/comments/abc123/test_post/",
    created_utc: float = 1717200000.0,
) -> dict[str, Any]:
    """Build a single Reddit search result child dict."""
    return {
        "kind": "t3",
        "data": {
            "name": name,
            "title": title,
            "selftext": selftext,
            "author": author,
            "score": score,
            "permalink": permalink,
            "url": f"https://www.reddit.com{permalink}",
            "created_utc": created_utc,
            "subreddit": "python",
        },
    }


def _mastodon_response(statuses: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap statuses in the standard Mastodon search response envelope."""
    return {"statuses": statuses, "accounts": [], "hashtags": []}


def _mastodon_status(
    status_id: str = "109876543",
    content: str = "<p>Hello <strong>world</strong>!</p>",
    url: str = "https://mastodon.social/@alice/109876543",
    acct: str = "alice",
    favourites_count: int = 5,
    reblogs_count: int = 2,
    created_at: str = "2024-06-01T10:00:00.000Z",
) -> dict[str, Any]:
    """Build a single Mastodon status dict."""
    return {
        "id": status_id,
        "content": content,
        "url": url,
        "uri": url,
        "account": {"id": "1", "username": "alice", "acct": acct},
        "favourites_count": favourites_count,
        "reblogs_count": reblogs_count,
        "created_at": created_at,
        "visibility": "public",
    }


# ---------------------------------------------------------------------------
# _strip_html
# ---------------------------------------------------------------------------


class TestStripHtml:
    """Unit tests for the _strip_html helper."""

    def test_removes_simple_tags(self) -> None:
        """HTML tags should be removed from the string."""
        assert _strip_html("<p>Hello</p>") == "Hello"

    def test_decodes_entities(self) -> None:
        """HTML entities like &amp; should be decoded."""
        assert "&" in _strip_html("Hello &amp; World")

    def test_handles_nested_tags(self) -> None:
        """Nested tags should all be stripped."""
        result = _strip_html("<p>Hello <strong>world</strong>!</p>")
        assert "<" not in result
        assert "Hello" in result
        assert "world" in result

    def test_collapses_whitespace(self) -> None:
        """Multiple whitespace characters should be collapsed."""
        result = _strip_html("Hello   world")
        assert result == "Hello world"

    def test_empty_string(self) -> None:
        """Empty string input should return empty string."""
        assert _strip_html("") == ""

    def test_plain_text_unchanged(self) -> None:
        """Plain text without HTML should pass through unchanged."""
        assert _strip_html("Just plain text.") == "Just plain text."


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    """Unit tests for the _truncate helper."""

    def test_short_text_unchanged(self) -> None:
        """Text shorter than max_chars should not be modified."""
        assert _truncate("Hello", max_chars=100) == "Hello"

    def test_exact_length_unchanged(self) -> None:
        """Text exactly max_chars long should not be truncated."""
        text = "a" * 50
        assert _truncate(text, max_chars=50) == text

    def test_long_text_truncated(self) -> None:
        """Text longer than max_chars should be truncated with ellipsis."""
        text = "a" * 200
        result = _truncate(text, max_chars=100)
        assert len(result) <= 102  # 100 chars + ellipsis char
        assert result.endswith("…")

    def test_empty_string(self) -> None:
        """Empty string should remain empty."""
        assert _truncate("", max_chars=100) == ""


# ---------------------------------------------------------------------------
# _parse_reddit_response
# ---------------------------------------------------------------------------


class TestParseRedditResponse:
    """Unit tests for the Reddit response parser."""

    def test_parses_single_post(self) -> None:
        """A well-formed Reddit response with one post should yield one Post."""
        data = _reddit_response([_reddit_child()])
        posts = _parse_reddit_response(data, keyword="python")
        assert len(posts) == 1
        assert posts[0].source == Source.REDDIT
        assert posts[0].keyword == "python"

    def test_source_id_set_correctly(self) -> None:
        """The source_id should match the Reddit 'name' field."""
        data = _reddit_response([_reddit_child(name="t3_xyz999")])
        posts = _parse_reddit_response(data, keyword="python")
        assert posts[0].source_id == "t3_xyz999"

    def test_title_parsed(self) -> None:
        """Post title should be extracted from the 'title' field."""
        data = _reddit_response([_reddit_child(title="Amazing Python library")])
        posts = _parse_reddit_response(data, keyword="python")
        assert posts[0].title == "Amazing Python library"

    def test_body_parsed(self) -> None:
        """Post body should be extracted from the 'selftext' field."""
        data = _reddit_response([_reddit_child(selftext="Great body text.")])
        posts = _parse_reddit_response(data, keyword="python")
        assert posts[0].body == "Great body text."

    def test_author_parsed(self) -> None:
        """Author should be extracted from the 'author' field."""
        data = _reddit_response([_reddit_child(author="bob")])
        posts = _parse_reddit_response(data, keyword="python")
        assert posts[0].author == "bob"

    def test_score_parsed(self) -> None:
        """Platform score should be extracted from the 'score' field."""
        data = _reddit_response([_reddit_child(score=999)])
        posts = _parse_reddit_response(data, keyword="python")
        assert posts[0].score == 999

    def test_url_uses_permalink(self) -> None:
        """The canonical URL should be constructed from the permalink."""
        data = _reddit_response(
            [_reddit_child(permalink="/r/python/comments/abc/test/")]
        )
        posts = _parse_reddit_response(
            data, keyword="python", base_url="https://www.reddit.com"
        )
        assert "/r/python/comments/abc/test/" in posts[0].url

    def test_created_utc_parsed(self) -> None:
        """The created_utc field should be converted to a UTC datetime."""
        data = _reddit_response([_reddit_child(created_utc=1717200000.0)])
        posts = _parse_reddit_response(data, keyword="python")
        assert posts[0].fetched_at.tzinfo is not None

    def test_multiple_posts(self) -> None:
        """Multiple children should produce multiple Post objects."""
        children = [
            _reddit_child(name=f"t3_{i}", title=f"Post {i}") for i in range(5)
        ]
        data = _reddit_response(children)
        posts = _parse_reddit_response(data, keyword="python")
        assert len(posts) == 5

    def test_deleted_author_becomes_none(self) -> None:
        """'[deleted]' author should be mapped to None."""
        data = _reddit_response([_reddit_child(author="[deleted]")])
        posts = _parse_reddit_response(data, keyword="python")
        assert posts[0].author is None

    def test_removed_author_becomes_none(self) -> None:
        """'[removed]' author should be mapped to None."""
        data = _reddit_response([_reddit_child(author="[removed]")])
        posts = _parse_reddit_response(data, keyword="python")
        assert posts[0].author is None

    def test_malformed_child_skipped(self) -> None:
        """A child without a name/id should be skipped without error."""
        bad_child: dict[str, Any] = {"kind": "t3", "data": {"title": "no id"}}
        good_child = _reddit_child(name="t3_good")
        data = _reddit_response([bad_child, good_child])
        posts = _parse_reddit_response(data, keyword="python")
        assert len(posts) == 1
        assert posts[0].source_id == "t3_good"

    def test_empty_children(self) -> None:
        """An empty children list should return an empty list of Posts."""
        data = _reddit_response([])
        posts = _parse_reddit_response(data, keyword="python")
        assert posts == []

    def test_malformed_response_returns_empty(self) -> None:
        """A response that doesn't match expected structure should return []."""
        posts = _parse_reddit_response({"unexpected": "data"}, keyword="python")
        assert posts == []

    def test_html_stripped_from_body(self) -> None:
        """HTML tags in selftext should be stripped."""
        data = _reddit_response([_reddit_child(selftext="<p>Hello &amp; world</p>")])
        posts = _parse_reddit_response(data, keyword="python")
        assert "<p>" not in posts[0].body
        assert "&amp;" not in posts[0].body


# ---------------------------------------------------------------------------
# _parse_mastodon_response
# ---------------------------------------------------------------------------


class TestParseMastodonResponse:
    """Unit tests for the Mastodon response parser."""

    def test_parses_single_status(self) -> None:
        """A well-formed Mastodon response should yield one Post."""
        data = _mastodon_response([_mastodon_status()])
        posts = _parse_mastodon_response(data, keyword="python")
        assert len(posts) == 1
        assert posts[0].source == Source.MASTODON

    def test_source_id_set_correctly(self) -> None:
        """source_id should match the Mastodon status 'id' field."""
        data = _mastodon_response([_mastodon_status(status_id="999111")])
        posts = _parse_mastodon_response(data, keyword="python")
        assert posts[0].source_id == "999111"

    def test_html_stripped_from_content(self) -> None:
        """HTML tags in status content should be stripped."""
        data = _mastodon_response(
            [_mastodon_status(content="<p>Hello <b>world</b></p>")]
        )
        posts = _parse_mastodon_response(data, keyword="python")
        assert "<p>" not in posts[0].body
        assert "Hello" in posts[0].body
        assert "world" in posts[0].body

    def test_author_from_acct(self) -> None:
        """Author should come from the account 'acct' field."""
        data = _mastodon_response([_mastodon_status(acct="alice@mastodon.social")])
        posts = _parse_mastodon_response(data, keyword="python")
        assert posts[0].author == "alice@mastodon.social"

    def test_score_is_sum_of_favourites_and_reblogs(self) -> None:
        """Score should equal favourites_count + reblogs_count."""
        data = _mastodon_response(
            [_mastodon_status(favourites_count=10, reblogs_count=3)]
        )
        posts = _parse_mastodon_response(data, keyword="python")
        assert posts[0].score == 13

    def test_url_set_correctly(self) -> None:
        """URL should come from the 'url' field in the status."""
        url = "https://mastodon.social/@bob/112233"
        data = _mastodon_response([_mastodon_status(url=url)])
        posts = _parse_mastodon_response(data, keyword="python")
        assert posts[0].url == url

    def test_created_at_parsed(self) -> None:
        """created_at should be parsed into a UTC-aware datetime."""
        data = _mastodon_response(
            [_mastodon_status(created_at="2024-06-01T10:00:00.000Z")]
        )
        posts = _parse_mastodon_response(data, keyword="python")
        assert posts[0].fetched_at.year == 2024
        assert posts[0].fetched_at.tzinfo is not None

    def test_title_is_none(self) -> None:
        """Mastodon statuses should have no title (None)."""
        data = _mastodon_response([_mastodon_status()])
        posts = _parse_mastodon_response(data, keyword="python")
        assert posts[0].title is None

    def test_multiple_statuses(self) -> None:
        """Multiple statuses should produce multiple Post objects."""
        statuses = [_mastodon_status(status_id=str(i)) for i in range(4)]
        data = _mastodon_response(statuses)
        posts = _parse_mastodon_response(data, keyword="python")
        assert len(posts) == 4

    def test_empty_statuses(self) -> None:
        """Empty status list should return empty list."""
        data = _mastodon_response([])
        posts = _parse_mastodon_response(data, keyword="python")
        assert posts == []

    def test_malformed_response_returns_empty(self) -> None:
        """A response with no 'statuses' key should return []."""
        posts = _parse_mastodon_response({"accounts": []}, keyword="python")
        assert posts == []

    def test_status_missing_id_skipped(self) -> None:
        """A status with no id should be skipped."""
        bad: dict[str, Any] = {
            "id": "",
            "content": "<p>no id</p>",
            "url": "https://mastodon.social/@x/0",
            "account": {"acct": "x"},
            "favourites_count": 0,
            "reblogs_count": 0,
            "created_at": "2024-01-01T00:00:00.000Z",
        }
        good = _mastodon_status(status_id="123")
        data = _mastodon_response([bad, good])
        posts = _parse_mastodon_response(data, keyword="python")
        assert len(posts) == 1
        assert posts[0].source_id == "123"


# ---------------------------------------------------------------------------
# fetch_reddit_posts (integration with mocked HTTP)
# ---------------------------------------------------------------------------


class TestFetchRedditPosts:
    """Tests for fetch_reddit_posts() using respx HTTP mocking."""

    @respx.mock
    async def test_successful_fetch(self) -> None:
        """A successful Reddit API response should return Post objects."""
        payload = _reddit_response([_reddit_child(name="t3_r1", title="Reddit Post 1")])
        respx.get(
            url__startswith="https://www.reddit.com/r/all/search.json"
        ).mock(return_value=httpx.Response(200, json=payload))

        async with httpx.AsyncClient() as client:
            posts = await fetch_reddit_posts(
                "python",
                base_url="https://www.reddit.com",
                client=client,
            )
        assert len(posts) == 1
        assert posts[0].source_id == "t3_r1"
        assert posts[0].source == Source.REDDIT
        assert posts[0].keyword == "python"

    @respx.mock
    async def test_empty_response(self) -> None:
        """An empty Reddit search result should return an empty list."""
        payload = _reddit_response([])
        respx.get(
            url__startswith="https://www.reddit.com/r/all/search.json"
        ).mock(return_value=httpx.Response(200, json=payload))

        async with httpx.AsyncClient() as client:
            posts = await fetch_reddit_posts(
                "python",
                base_url="https://www.reddit.com",
                client=client,
            )
        assert posts == []

    @respx.mock
    async def test_http_error_raises(self) -> None:
        """A non-2xx response should raise HTTPStatusError."""
        respx.get(
            url__startswith="https://www.reddit.com/r/all/search.json"
        ).mock(return_value=httpx.Response(429, text="Too Many Requests"))

        with pytest.raises(httpx.HTTPStatusError):
            async with httpx.AsyncClient() as client:
                await fetch_reddit_posts(
                    "python",
                    base_url="https://www.reddit.com",
                    client=client,
                )

    @respx.mock
    async def test_subreddit_in_url(self) -> None:
        """The subreddit parameter should appear in the request URL."""
        payload = _reddit_response([])
        route = respx.get(
            url__startswith="https://www.reddit.com/r/learnpython/search.json"
        ).mock(return_value=httpx.Response(200, json=payload))

        async with httpx.AsyncClient() as client:
            await fetch_reddit_posts(
                "python",
                subreddit="learnpython",
                base_url="https://www.reddit.com",
                client=client,
            )
        assert route.called

    @respx.mock
    async def test_multiple_posts_returned(self) -> None:
        """Multiple children in the response should produce multiple posts."""
        children = [_reddit_child(name=f"t3_{i}") for i in range(5)]
        payload = _reddit_response(children)
        respx.get(
            url__startswith="https://www.reddit.com/r/all/search.json"
        ).mock(return_value=httpx.Response(200, json=payload))

        async with httpx.AsyncClient() as client:
            posts = await fetch_reddit_posts(
                "python",
                base_url="https://www.reddit.com",
                client=client,
            )
        assert len(posts) == 5


# ---------------------------------------------------------------------------
# fetch_mastodon_posts (integration with mocked HTTP)
# ---------------------------------------------------------------------------


class TestFetchMastodonPosts:
    """Tests for fetch_mastodon_posts() using respx HTTP mocking."""

    @respx.mock
    async def test_successful_fetch(self) -> None:
        """A successful Mastodon API response should return Post objects."""
        payload = _mastodon_response([_mastodon_status(status_id="111")])
        respx.get(
            url__startswith="https://mastodon.social/api/v2/search"
        ).mock(return_value=httpx.Response(200, json=payload))

        async with httpx.AsyncClient() as client:
            posts = await fetch_mastodon_posts(
                "python",
                base_url="https://mastodon.social",
                client=client,
            )
        assert len(posts) == 1
        assert posts[0].source == Source.MASTODON
        assert posts[0].source_id == "111"

    @respx.mock
    async def test_empty_response(self) -> None:
        """An empty Mastodon search result should return an empty list."""
        payload = _mastodon_response([])
        respx.get(
            url__startswith="https://mastodon.social/api/v2/search"
        ).mock(return_value=httpx.Response(200, json=payload))

        async with httpx.AsyncClient() as client:
            posts = await fetch_mastodon_posts(
                "python",
                base_url="https://mastodon.social",
                client=client,
            )
        assert posts == []

    @respx.mock
    async def test_http_error_raises(self) -> None:
        """A non-2xx response should raise HTTPStatusError."""
        respx.get(
            url__startswith="https://mastodon.social/api/v2/search"
        ).mock(return_value=httpx.Response(403, text="Forbidden"))

        with pytest.raises(httpx.HTTPStatusError):
            async with httpx.AsyncClient() as client:
                await fetch_mastodon_posts(
                    "python",
                    base_url="https://mastodon.social",
                    client=client,
                )

    @respx.mock
    async def test_multiple_statuses_returned(self) -> None:
        """Multiple statuses in the response should produce multiple posts."""
        statuses = [_mastodon_status(status_id=str(i)) for i in range(3)]
        payload = _mastodon_response(statuses)
        respx.get(
            url__startswith="https://mastodon.social/api/v2/search"
        ).mock(return_value=httpx.Response(200, json=payload))

        async with httpx.AsyncClient() as client:
            posts = await fetch_mastodon_posts(
                "python",
                base_url="https://mastodon.social",
                client=client,
            )
        assert len(posts) == 3

    @respx.mock
    async def test_custom_instance(self) -> None:
        """A custom Mastodon base URL should be used in the request."""
        payload = _mastodon_response([])
        route = respx.get(
            url__startswith="https://fosstodon.org/api/v2/search"
        ).mock(return_value=httpx.Response(200, json=payload))

        async with httpx.AsyncClient() as client:
            await fetch_mastodon_posts(
                "python",
                base_url="https://fosstodon.org",
                client=client,
            )
        assert route.called

    @respx.mock
    async def test_keyword_propagated(self) -> None:
        """The keyword should be set on each returned Post."""
        payload = _mastodon_response([_mastodon_status()])
        respx.get(
            url__startswith="https://mastodon.social/api/v2/search"
        ).mock(return_value=httpx.Response(200, json=payload))

        async with httpx.AsyncClient() as client:
            posts = await fetch_mastodon_posts(
                "climate",
                base_url="https://mastodon.social",
                client=client,
            )
        assert posts[0].keyword == "climate"
