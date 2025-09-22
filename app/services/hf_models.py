"""Helpers for discovering trending Hugging Face text-to-speech models."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List

from huggingface_hub import ModelFilter, list_models

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 60 * 30  # 30 minutes
_trending_cache: Dict[str, object] = {"timestamp": 0.0, "data": []}


def _fetch_trending_models() -> List[Dict[str, object]]:
    """Synchronously query the Hugging Face hub for trending TTS models."""

    logger.info("Fetching trending text-to-speech models from Hugging Face")
    models = list_models(
        filter=ModelFilter(task="text-to-speech"),
        sort="trending",
        limit=20,
    )
    results: List[Dict[str, object]] = []

    for model in models:
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
