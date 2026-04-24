"""Async ingestion functions for Reddit and Mastodon public APIs.

Provides two primary coroutines:

- :func:`fetch_reddit_posts` — queries the Reddit JSON search API for a
  keyword without requiring authentication.
- :func:`fetch_mastodon_posts` — queries a Mastodon instance's public
  timeline search endpoint without requiring authentication.

Both functions return a list of :class:`~signal_dash.models.Post` objects
ready for classification and persistence.  All network calls are made via
:mod:`httpx` with configurable timeouts and a descriptive ``User-Agent``
header to comply with Reddit's API rules.
"""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from signal_dash.models import Post, Source

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default timeout (seconds) for all outbound HTTP requests.
_DEFAULT_TIMEOUT = 15.0

#: Maximum characters of body text to retain per post (avoid massive payloads).
_MAX_BODY_CHARS = 2_000


# ---------------------------------------------------------------------------
# HTML / markup helpers
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities from *text*.

    Parameters
    ----------
    text:
        Raw string that may contain HTML markup.

    Returns
    -------
    str
        Plain text with tags removed and entities decoded.
    """
    no_tags = _HTML_TAG_RE.sub(" ", text)
    decoded = html.unescape(no_tags)
    # Collapse multiple whitespace characters into a single space.
    return " ".join(decoded.split())


def _truncate(text: str, max_chars: int = _MAX_BODY_CHARS) -> str:
    """Truncate *text* to at most *max_chars* characters.

    Parameters
    ----------
    text:
        Input string.
    max_chars:
        Maximum allowed length.

    Returns
    -------
    str
        Original string if it fits, otherwise the first *max_chars*
        characters followed by an ellipsis.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


# ---------------------------------------------------------------------------
# Reddit ingestion
# ---------------------------------------------------------------------------


async def fetch_reddit_posts(
    keyword: str,
    *,
    subreddit: str = "all",
    limit: int = 25,
    base_url: str = "https://www.reddit.com",
    user_agent: str = "signal_dash/0.1.0 (social listening bot)",
    client: Optional[httpx.AsyncClient] = None,
) -> list[Post]:
    """Fetch recent Reddit posts matching *keyword* via the public JSON API.

    Reddit's public JSON API does not require authentication and is accessed
    by appending ``.json`` to a search URL.  The ``User-Agent`` header must
    be descriptive per Reddit's API rules.

    Parameters
    ----------
    keyword:
        Search term to query.
    subreddit:
        Subreddit to search within.  Use ``"all"`` for all subreddits.
    limit:
        Maximum number of posts to return (1–100).
    base_url:
        Base URL for the Reddit API.  Override for testing.
    user_agent:
        ``User-Agent`` header value sent to Reddit.
    client:
        An existing :class:`httpx.AsyncClient` to use.  When ``None`` a
        temporary client is created for this request.

    Returns
    -------
    list[Post]
        Parsed :class:`~signal_dash.models.Post` objects, possibly empty
        if the request fails or returns no results.

    Raises
    ------
    httpx.HTTPStatusError
        If Reddit returns a non-2xx status code.
    httpx.RequestError
        If a network-level error occurs and no ``client`` was supplied
        (the error propagates to the caller).
    """
    url = f"{base_url.rstrip('/')}/r/{subreddit}/search.json"
    params: dict[str, Any] = {
        "q": keyword,
        "sort": "new",
        "limit": min(max(limit, 1), 100),
        "restrict_sr": "false",
        "type": "link",
    }
    headers = {"User-Agent": user_agent}

    _own_client = client is None
    _client: httpx.AsyncClient = (
        httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) if _own_client else client  # type: ignore[assignment]
    )

    try:
        response = await _client.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Reddit API returned HTTP %d for keyword '%s': %s",
            exc.response.status_code,
            keyword,
            exc,
        )
        raise
    except httpx.RequestError as exc:
        logger.warning("Reddit request error for keyword '%s': %s", keyword, exc)
        raise
    finally:
        if _own_client:
            await _client.aclose()

    return _parse_reddit_response(data, keyword=keyword, base_url=base_url)


def _parse_reddit_response(
    data: dict[str, Any],
    keyword: str,
    base_url: str = "https://www.reddit.com",
) -> list[Post]:
    """Parse a raw Reddit search JSON response into a list of Posts.

    Parameters
    ----------
    data:
        Parsed JSON response body from the Reddit search API.
    keyword:
        The keyword used for the search (propagated to each Post).
    base_url:
        Base URL used to construct canonical post URLs.

    Returns
    -------
    list[Post]
        Parsed posts.  Invalid or unparseable items are skipped with a
        warning log.
    """
    posts: list[Post] = []
    try:
        children = data["data"]["children"]
    except (KeyError, TypeError):
        logger.warning("Unexpected Reddit API response structure.")
        return posts

    for child in children:
        try:
            item: dict[str, Any] = child.get("data", {})
            source_id: str = item.get("name") or item.get("id", "")
            if not source_id:
                continue

            # Build the canonical URL.  Use the permalink if available.
            permalink: str = item.get("permalink", "")
            if permalink:
                canonical_url = f"{base_url.rstrip('/')}{permalink}"
            else:
                canonical_url = item.get("url", "")
            if not canonical_url:
                continue

            title: str = _strip_html(item.get("title") or "")
            selftext: str = _strip_html(item.get("selftext") or "")
            body = _truncate(selftext)

            author: Optional[str] = item.get("author") or None
            if author and author.lower() in ("[deleted]", "[removed]", "automoderator"):
                author = None

            score: int = int(item.get("score") or 0)

            created_utc = item.get("created_utc")
            if created_utc is not None:
                fetched_at = datetime.fromtimestamp(
                    float(created_utc), tz=timezone.utc
                )
            else:
                fetched_at = datetime.now(tz=timezone.utc)

            post = Post(
                source_id=source_id,
                source=Source.REDDIT,
                author=author,
                title=title or None,
                body=body,
                url=canonical_url,
                score=score,
                keyword=keyword,
                fetched_at=fetched_at,
            )
            posts.append(post)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed Reddit post: %s", exc)
            continue

    logger.info("Parsed %d Reddit posts for keyword '%s'.", len(posts), keyword)
    return posts


# ---------------------------------------------------------------------------
# Mastodon ingestion
# ---------------------------------------------------------------------------


async def fetch_mastodon_posts(
    keyword: str,
    *,
    base_url: str = "https://mastodon.social",
    limit: int = 20,
    client: Optional[httpx.AsyncClient] = None,
) -> list[Post]:
    """Fetch recent Mastodon statuses matching *keyword* from a public timeline.

    Uses the ``/api/v2/search`` endpoint which is available without
    authentication on most Mastodon-compatible instances.  The endpoint
    returns statuses containing the keyword.

    Parameters
    ----------
    keyword:
        Search term to query.
    base_url:
        Mastodon instance base URL.
    limit:
        Maximum number of statuses to return (1–40).
    client:
        An existing :class:`httpx.AsyncClient` to use.  When ``None`` a
        temporary client is created for this request.

    Returns
    -------
    list[Post]
        Parsed :class:`~signal_dash.models.Post` objects.

    Raises
    ------
    httpx.HTTPStatusError
        If the Mastodon instance returns a non-2xx status code.
    httpx.RequestError
        If a network-level error occurs.
    """
    url = f"{base_url.rstrip('/')}/api/v2/search"
    params: dict[str, Any] = {
        "q": keyword,
        "type": "statuses",
        "limit": min(max(limit, 1), 40),
        "resolve": "false",
    }
    headers = {"User-Agent": "signal_dash/0.1.0 (social listening bot)"}

    _own_client = client is None
    _client: httpx.AsyncClient = (
        httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) if _own_client else client  # type: ignore[assignment]
    )

    try:
        response = await _client.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Mastodon API returned HTTP %d for keyword '%s': %s",
            exc.response.status_code,
            keyword,
            exc,
        )
        raise
    except httpx.RequestError as exc:
        logger.warning("Mastodon request error for keyword '%s': %s", keyword, exc)
        raise
    finally:
        if _own_client:
            await _client.aclose()

    return _parse_mastodon_response(data, keyword=keyword, base_url=base_url)


def _parse_mastodon_response(
    data: dict[str, Any],
    keyword: str,
    base_url: str = "https://mastodon.social",
) -> list[Post]:
    """Parse a raw Mastodon search JSON response into a list of Posts.

    Parameters
    ----------
    data:
        Parsed JSON response body from the Mastodon search API.
    keyword:
        The keyword used for the search.
    base_url:
        Base URL of the Mastodon instance.

    Returns
    -------
    list[Post]
        Parsed posts.  Invalid items are skipped with a warning log.
    """
    posts: list[Post] = []
    try:
        statuses: list[dict[str, Any]] = data.get("statuses", [])
    except (AttributeError, TypeError):
        logger.warning("Unexpected Mastodon API response structure.")
        return posts

    for status in statuses:
        try:
            source_id: str = str(status.get("id") or "")
            if not source_id:
                continue

            canonical_url: str = status.get("url") or status.get("uri") or ""
            if not canonical_url:
                canonical_url = f"{base_url.rstrip('/')}/web/statuses/{source_id}"

            raw_content: str = status.get("content") or ""
            body = _truncate(_strip_html(raw_content))

            # Extract author information from the nested account object.
            account: dict[str, Any] = status.get("account") or {}
            author_str: Optional[str] = (
                account.get("acct") or account.get("username") or None
            )

            # Mastodon uses favourites_count + reblogs_count as engagement.
            favourites: int = int(status.get("favourites_count") or 0)
            reblogs: int = int(status.get("reblogs_count") or 0)
            score: int = favourites + reblogs

            # Parse ISO 8601 creation timestamp.
            created_at_str: str = status.get("created_at") or ""
            if created_at_str:
                try:
                    fetched_at = datetime.fromisoformat(
                        created_at_str.replace("Z", "+00:00")
                    )
                except ValueError:
                    fetched_at = datetime.now(tz=timezone.utc)
            else:
                fetched_at = datetime.now(tz=timezone.utc)

            post = Post(
                source_id=source_id,
                source=Source.MASTODON,
                author=author_str,
                title=None,  # Mastodon statuses have no title field
                body=body,
                url=canonical_url,
                score=score,
                keyword=keyword,
                fetched_at=fetched_at,
            )
            posts.append(post)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed Mastodon status: %s", exc)
            continue

    logger.info(
        "Parsed %d Mastodon statuses for keyword '%s'.", len(posts), keyword
    )
    return posts
