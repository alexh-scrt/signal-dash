"""LLM-backed post classifier with a rule-based offline stub fallback.

This module exposes a single public entry point — :func:`classify_posts` —
that accepts a list of :class:`~signal_dash.models.Post` objects and returns
a list of :class:`~signal_dash.models.Signal` objects enriched with:

- ``sentiment_score``  — float in ``[-1.0, 1.0]``
- ``topics``           — up to 3 short tag strings
- ``signal_strength``  — float in ``[0.0, 1.0]``

Two classification back-ends are provided:

1. **LLM back-end** (:func:`_classify_with_llm`) — batches posts and sends
   them to the OpenAI Chat Completions API (default model: ``gpt-4o-mini``).
   Activated automatically when an OpenAI API key is present in settings.

2. **Stub back-end** (:func:`_classify_stub`) — a deterministic, zero-network
   rule-based classifier that inspects keyword frequency, punctuation, and
   capitalisation to produce plausible-looking scores.  Used when no API key
   is configured, making the app safe for fully offline development and CI.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Optional

from signal_dash.models import Post, Signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentiment word lists used by the stub classifier
# ---------------------------------------------------------------------------

_POSITIVE_WORDS: frozenset[str] = frozenset({
    "great", "awesome", "excellent", "good", "love", "loved", "loving",
    "fantastic", "wonderful", "amazing", "superb", "brilliant", "outstanding",
    "happy", "joy", "joyful", "pleased", "glad", "delighted", "exciting",
    "excited", "thrilled", "best", "perfect", "beautiful", "helpful",
    "useful", "impressive", "innovative", "recommend", "recommended",
    "nice", "cool", "fun", "interesting", "insightful", "powerful",
    "fast", "efficient", "elegant", "clean", "simple", "easy",
    "intuitive", "reliable", "robust", "solid", "stable", "winning",
    "win", "won", "success", "successful", "improved", "improvement",
    "progress", "growing", "growth", "positive",
})

_NEGATIVE_WORDS: frozenset[str] = frozenset({
    "bad", "terrible", "awful", "horrible", "poor", "worst", "hate",
    "hated", "hating", "dislike", "ugly", "broken", "bug", "bugs",
    "buggy", "crash", "crashed", "crashing", "error", "errors",
    "fail", "failed", "failing", "failure", "slow", "sluggish",
    "frustrating", "frustrated", "annoying", "annoyed", "useless",
    "worthless", "disappointing", "disappointed", "disappointment",
    "broken", "issue", "issues", "problem", "problems", "wrong",
    "incorrect", "misleading", "outdated", "deprecated", "dead",
    "scam", "spam", "toxic", "harmful", "dangerous", "security",
    "vulnerable", "vulnerability", "leak", "leaked", "breach",
    "negative", "concern", "worried", "worry", "sad", "unhappy",
    "painful", "struggle", "difficult", "complicated", "complex",
})

# ---------------------------------------------------------------------------
# Topic keyword lists used by the stub classifier
# ---------------------------------------------------------------------------

_TOPIC_PATTERNS: list[tuple[str, list[str]]] = [
    ("ai", ["ai", "artificial intelligence", "machine learning", "ml", "gpt",
             "llm", "neural", "deep learning", "chatgpt", "openai", "gemini",
             "claude", "transformer", "embedding", "fine-tuning"]),
    ("python", ["python", "pypi", "pip", "django", "flask", "fastapi",
                 "pandas", "numpy", "scipy", "matplotlib", "jupyter",
                 "asyncio", "pydantic", "sqlalchemy"]),
    ("security", ["security", "vulnerability", "cve", "exploit", "hack",
                   "breach", "malware", "ransomware", "phishing", "zero-day",
                   "firewall", "encryption", "authentication", "oauth"]),
    ("performance", ["performance", "speed", "latency", "throughput", "benchmark",
                      "optimise", "optimize", "profiling", "cache", "caching",
                      "fast", "slow", "bottleneck", "memory", "cpu", "gpu"]),
    ("web", ["web", "http", "https", "api", "rest", "graphql", "frontend",
              "backend", "javascript", "typescript", "react", "vue", "nextjs",
              "html", "css", "browser", "chrome", "firefox"]),
    ("cloud", ["cloud", "aws", "azure", "gcp", "kubernetes", "docker",
                "container", "serverless", "lambda", "terraform",
                "devops", "ci/cd", "deployment", "infrastructure"]),
    ("data", ["data", "database", "sql", "nosql", "postgres", "mysql",
               "sqlite", "redis", "mongodb", "analytics", "dashboard",
               "visualisation", "visualization", "bigdata", "warehouse"]),
    ("open-source", ["open source", "opensource", "github", "gitlab",
                      "repository", "repo", "fork", "pull request", "pr",
                      "commit", "release", "license", "mit", "apache"]),
    ("community", ["community", "forum", "discussion", "conference",
                    "meetup", "hackathon", "tutorial", "documentation",
                    "docs", "guide", "beginner", "learning"]),
    ("jobs", ["job", "jobs", "hiring", "hire", "career", "salary",
               "remote", "freelance", "internship", "developer",
               "engineer", "programmer"]),
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def classify_posts(
    posts: list[Post],
    *,
    openai_api_key: str = "",
    openai_model: str = "gpt-4o-mini",
    batch_size: int = 10,
) -> list[Signal]:
    """Classify a list of posts and return enriched Signal objects.

    Dispatches to the LLM back-end when *openai_api_key* is non-empty,
    otherwise falls back to the rule-based stub classifier.

    Parameters
    ----------
    posts:
        Raw posts to classify.  May be empty (returns ``[]`` immediately).
    openai_api_key:
        OpenAI API key.  An empty string activates the stub classifier.
    openai_model:
        OpenAI model identifier (used only when key is present).
    batch_size:
        Number of posts to send in a single LLM request.

    Returns
    -------
    list[Signal]
        One :class:`~signal_dash.models.Signal` per input post, in the same
        order as *posts*.  Posts that fail classification are assigned
        neutral stub scores rather than dropped.
    """
    if not posts:
        return []

    if openai_api_key.strip():
        logger.info(
            "Classifying %d posts via OpenAI (%s) in batches of %d.",
            len(posts),
            openai_model,
            batch_size,
        )
        return await _classify_with_llm(
            posts,
            api_key=openai_api_key,
            model=openai_model,
            batch_size=batch_size,
        )
    else:
        logger.info("Classifying %d posts via stub classifier.", len(posts))
        return [_classify_stub(post) for post in posts]


# ---------------------------------------------------------------------------
# Stub (rule-based) classifier
# ---------------------------------------------------------------------------


def _classify_stub(post: Post) -> Signal:
    """Apply a deterministic rule-based classifier to a single post.

    The algorithm:

    1. Tokenises the full text (title + body) into lowercase words.
    2. Counts positive and negative sentiment words from look-up sets.
    3. Applies small adjustments for exclamation marks and ALL-CAPS words.
    4. Computes a normalised sentiment score in ``[-1.0, 1.0]``.
    5. Derives topic tags by matching against ``_TOPIC_PATTERNS``.
    6. Computes signal strength from the platform score and sentiment magnitude.

    Parameters
    ----------
    post:
        The post to classify.

    Returns
    -------
    Signal
        Classified signal with stub-derived sentiment, topics, and strength.
    """
    text = post.full_text.lower()
    words = re.findall(r"[a-z]+", text)
    word_set = set(words)

    positive_hits = len(word_set & _POSITIVE_WORDS)
    negative_hits = len(word_set & _NEGATIVE_WORDS)

    # Boost for exclamation marks (excitement can be positive)
    exclamation_count = post.full_text.count("!")
    positive_hits += min(exclamation_count, 3) * 0.5  # type: ignore[assignment]

    # Small penalty for question marks suggesting uncertainty/frustration
    question_count = post.full_text.count("?")
    negative_hits += min(question_count, 3) * 0.2  # type: ignore[assignment]

    # Compute raw score
    total_hits = positive_hits + negative_hits
    if total_hits == 0:
        raw_score = 0.0
    else:
        raw_score = (positive_hits - negative_hits) / (total_hits + 2.0)

    # Clamp to [-1.0, 1.0]
    sentiment_score = max(-1.0, min(1.0, float(raw_score)))

    # Derive up to 3 topic tags
    topics = _extract_topics(text)

    # Signal strength: combine platform engagement and sentiment magnitude.
    # Use a log-scaled score to dampen very high upvote counts.
    log_score = math.log1p(max(post.score, 0)) / math.log1p(1000)
    sentiment_magnitude = abs(sentiment_score)
    raw_strength = 0.4 * min(log_score, 1.0) + 0.6 * sentiment_magnitude
    signal_strength = max(0.0, min(1.0, raw_strength))

    return Signal.from_post(
        post=post,
        sentiment_score=round(sentiment_score, 4),
        topics=topics,
        signal_strength=round(signal_strength, 4),
    )


def _extract_topics(text: str) -> list[str]:
    """Extract up to 3 topic tags from lowercased *text*.

    Iterates through :data:`_TOPIC_PATTERNS` and returns the labels of the
    first 3 patterns whose keywords appear anywhere in *text*.

    Parameters
    ----------
    text:
        Lowercased, plain-text content to scan.

    Returns
    -------
    list[str]
        Up to 3 matching topic tag labels.
    """
    matched: list[str] = []
    for label, keywords in _TOPIC_PATTERNS:
        if any(kw in text for kw in keywords):
            matched.append(label)
            if len(matched) == 3:
                break
    return matched


# ---------------------------------------------------------------------------
# LLM (OpenAI) classifier
# ---------------------------------------------------------------------------


async def _classify_with_llm(
    posts: list[Post],
    *,
    api_key: str,
    model: str = "gpt-4o-mini",
    batch_size: int = 10,
) -> list[Signal]:
    """Classify posts in batches using the OpenAI Chat Completions API.

    Sends each batch as a single prompt and parses the structured JSON
    response.  Falls back to the stub classifier for any post whose LLM
    result cannot be parsed.

    Parameters
    ----------
    posts:
        Posts to classify.
    api_key:
        OpenAI API key.
    model:
        Model identifier.
    batch_size:
        Number of posts per API call.

    Returns
    -------
    list[Signal]
        Classified signals in the same order as *posts*.
    """
    from openai import AsyncOpenAI

    aclient = AsyncOpenAI(api_key=api_key)
    signals: list[Signal] = []

    for batch_start in range(0, len(posts), batch_size):
        batch = posts[batch_start : batch_start + batch_size]
        batch_signals = await _classify_batch_llm(batch, client=aclient, model=model)
        signals.extend(batch_signals)

    return signals


async def _classify_batch_llm(
    posts: list[Post],
    *,
    client: Any,
    model: str,
) -> list[Signal]:
    """Classify a single batch of posts via the OpenAI API.

    Builds a structured prompt that asks the model to return a JSON array
    where each element corresponds to one input post and contains:
    ``sentiment_score`` (float, −1 to 1), ``topics`` (list of up to 3
    strings), and ``signal_strength`` (float, 0 to 1).

    Parameters
    ----------
    posts:
        A batch of posts (should be <= ``batch_size``).
    client:
        An :class:`openai.AsyncOpenAI` client instance.
    model:
        Model identifier.

    Returns
    -------
    list[Signal]
        One signal per post.  Posts with unparseable LLM output fall back
        to the stub classifier.
    """
    numbered_posts = "\n\n".join(
        f"[{i + 1}] {post.full_text[:500]}" for i, post in enumerate(posts)
    )
    system_prompt = (
        "You are a social media intelligence analyst. "
        "For each numbered post below, return a JSON array where each element "
        "is an object with these exact keys:\n"
        "  - sentiment_score: float between -1.0 (very negative) and 1.0 (very positive)\n"
        "  - topics: list of up to 3 concise lowercase topic tags (e.g. ['ai', 'python'])\n"
        "  - signal_strength: float between 0.0 (weak signal) and 1.0 (strong/trending)\n"
        "Return ONLY valid JSON — no markdown, no explanation. "
        "The array must have exactly as many elements as there are numbered posts."
    )
    user_prompt = f"Posts to classify:\n\n{numbered_posts}"

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        raw_content: str = response.choices[0].message.content or ""
        return _parse_llm_response(raw_content, posts)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "LLM classification failed for batch of %d posts, falling back to stub: %s",
            len(posts),
            exc,
        )
        return [_classify_stub(post) for post in posts]


def _parse_llm_response(raw_content: str, posts: list[Post]) -> list[Signal]:
    """Parse the LLM JSON response and build Signal objects.

    Attempts to extract a JSON array from *raw_content*.  If the array
    length does not match ``len(posts)`` or any element is malformed,
    the corresponding post is classified with the stub instead.

    Parameters
    ----------
    raw_content:
        Raw string returned by the LLM (may contain extra whitespace or
        markdown code fences).
    posts:
        The original posts in the same order as the LLM input.

    Returns
    -------
    list[Signal]
        One signal per post.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_content).strip()
    # Attempt to find the JSON array boundary
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        logger.warning("LLM response did not contain a JSON array; using stub fallback.")
        return [_classify_stub(post) for post in posts]

    try:
        parsed: list[dict[str, Any]] = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM JSON response: %s", exc)
        return [_classify_stub(post) for post in posts]

    if not isinstance(parsed, list) or len(parsed) != len(posts):
        logger.warning(
            "LLM returned %d results for %d posts; using stub fallback.",
            len(parsed) if isinstance(parsed, list) else -1,
            len(posts),
        )
        return [_classify_stub(post) for post in posts]

    signals: list[Signal] = []
    for post, item in zip(posts, parsed):
        try:
            sentiment_score = float(item.get("sentiment_score", 0.0))
            sentiment_score = max(-1.0, min(1.0, sentiment_score))

            raw_topics = item.get("topics", [])
            if isinstance(raw_topics, list):
                topics = [str(t).lower().strip() for t in raw_topics if str(t).strip()][:3]
            else:
                topics = []

            signal_strength = float(item.get("signal_strength", 0.0))
            signal_strength = max(0.0, min(1.0, signal_strength))

            signal = Signal.from_post(
                post=post,
                sentiment_score=round(sentiment_score, 4),
                topics=topics,
                signal_strength=round(signal_strength, 4),
            )
            signals.append(signal)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Malformed LLM result for post %s, using stub: %s",
                post.source_id,
                exc,
            )
            signals.append(_classify_stub(post))

    return signals
