"""Helpers for discovering trending Hugging Face text-to-speech models."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Iterable, List

import requests
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

    try:
        return _fetch_trending_via_http()
    except Exception as exc:  # pragma: no cover - network access required
        logger.warning("Trending API request failed: %s; falling back to legacy Hub client.", exc)
        return _fetch_trending_via_list_models()


def _fetch_trending_via_http() -> List[Dict[str, object]]:
    url = "https://huggingface.co/api/models"
    params = {
        "pipeline_tag": "text-to-speech",
        "sort": "trending",
        "limit": 50,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected payload returned by Hugging Face trending endpoint.")

    results: List[Dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        normalised = _normalise_http_model(item)
        if not normalised.get("id"):
            continue
        if normalised.get("pipeline_tag") and normalised.get("pipeline_tag") != "text-to-speech":
            continue
        results.append(normalised)
        if len(results) >= 20:
            break

    return results


def _fetch_trending_via_list_models() -> List[Dict[str, object]]:
    base_kwargs = {
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


def _normalise_http_model(item: Dict[str, object]) -> Dict[str, object]:
    pipeline_tag = item.get("pipeline_tag")
    if not pipeline_tag and isinstance(item.get("tags"), list):
        for tag in item["tags"]:  # type: ignore[index]
            if isinstance(tag, str) and tag.startswith("pipeline:"):
                pipeline_tag = tag.split(":", 1)[1]
                break

    return {
        "id": item.get("id") or item.get("modelId") or item.get("name"),
        "pipeline_tag": pipeline_tag,
        "likes": item.get("likes") or 0,
        "downloads": item.get("downloads") or 0,
        "tags": item.get("tags") or [],
        "library_name": item.get("library_name"),
        "last_modified": item.get("lastModified") or item.get("last_modified"),
    }


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
