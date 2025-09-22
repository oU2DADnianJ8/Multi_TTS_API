"""Helpers for discovering trending Hugging Face text-to-speech models."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Iterable, List

from huggingface_hub import list_models

try:
    from huggingface_hub import ModelFilter  # type: ignore
except ImportError:  # pragma: no cover - depends on huggingface_hub version
    ModelFilter = None  # type: ignore

try:
    from huggingface_hub import ModelSort  # type: ignore
except ImportError:  # pragma: no cover - depends on huggingface_hub version
    ModelSort = None  # type: ignore

from requests import HTTPError

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 60 * 30  # 30 minutes
_trending_cache: Dict[str, object] = {"timestamp": 0.0, "data": []}


def _sort_candidates() -> Iterable[object]:
    """Return a sequence of preferred sort orders for ``list_models``."""

    candidates: List[object] = []
    if ModelSort is not None:
        for attr in ("TRENDING", "LIKES", "DOWNLOADS"):
            value = getattr(ModelSort, attr, None)
            if value is not None and value not in candidates:
                candidates.append(value)
    else:  # pragma: no cover - fallback for very old ``huggingface_hub`` versions
        candidates = ["trending", "likes", "downloads"]
    return candidates


def _describe_sort(sort_value: object) -> str:
    """Return a human-readable name for a ``list_models`` sort value."""

    return str(getattr(sort_value, "value", sort_value))


def _fetch_trending_models() -> List[Dict[str, object]]:
    """Synchronously query the Hugging Face hub for trending TTS models."""

    logger.info("Fetching trending text-to-speech models from Hugging Face")
    base_kwargs = {
        # Request more models when we cannot rely on server-side filtering
        # so that post-filtering still returns roughly ``limit`` results.
        "limit": 40 if ModelFilter is None else 20,
    }

    if ModelFilter is not None:
        base_kwargs["filter"] = ModelFilter(task="text-to-speech")

    models = []
    last_error: Exception | None = None
    for sort_value in _sort_candidates():
        kwargs = dict(base_kwargs)
        kwargs["sort"] = sort_value

        try:
            models = list(list_models(**kwargs))
        except HTTPError as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 400:
                logger.warning(
                    "Sort '%s' rejected by Hugging Face; falling back to next candidate.",
                    _describe_sort(sort_value),
                )
                last_error = exc
                continue
            raise
        else:
            break
    else:  # pragma: no cover - propagate last encountered 400 error if all candidates fail
        if last_error is not None:
            raise last_error

    results: List[Dict[str, object]] = []

    for model in models:
        if ModelFilter is None and getattr(model, "pipeline_tag", None) != "text-to-speech":
            # ``huggingface_hub`` versions without ``ModelFilter`` support do not
            # provide server-side filtering.  Perform a lightweight client-side
            # filter instead to maintain backwards compatibility.
            continue

        results.append(
            {
                "id": model.modelId,
                "pipeline_tag": model.pipeline_tag,
                "likes": model.likes or 0,
                "downloads": model.downloads or 0,
                "tags": model.tags or [],
                "library_name": model.library_name,
                "last_modified": model.lastModified.isoformat() if model.lastModified else None,
            }
        )

        if len(results) >= 20:
            break

    return results


async def get_trending_tts_models(force_refresh: bool = False) -> List[Dict[str, object]]:
    """Return cached metadata about the top trending TTS models.

    The Hugging Face Hub is polled at most once every ``_CACHE_TTL_SECONDS``
    unless ``force_refresh`` is set to ``True``.
    """

    now = time.time()
    if not force_refresh:
        if _trending_cache["data"] and now - float(_trending_cache["timestamp"]) < _CACHE_TTL_SECONDS:
            return list(_trending_cache["data"])

    data = await asyncio.to_thread(_fetch_trending_models)
    _trending_cache["timestamp"] = now
    _trending_cache["data"] = data
    return list(data)
