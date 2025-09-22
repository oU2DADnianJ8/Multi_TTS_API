"""Model management and inference helpers for text-to-speech pipelines."""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Dict, Tuple

import torch
from transformers import pipeline

logger = logging.getLogger(__name__)


class TTSManager:
    """Lazy loader and cache for Hugging Face text-to-speech pipelines."""

    def __init__(self) -> None:
        self._pipelines: Dict[str, object] = {}
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._device = 0 if torch.cuda.is_available() else -1
        self._torch_dtype = torch.float16 if torch.cuda.is_available() else None

    def _load_pipeline(self, model_id: str) -> object:
        load_kwargs = {}
        if self._torch_dtype is not None:
            load_kwargs["torch_dtype"] = self._torch_dtype

        def _attempt_load(**extra_kwargs: object) -> object:
            return pipeline(
                task="text-to-speech",
                model=model_id,
                device=self._device,
                **load_kwargs,
                **extra_kwargs,
            )

        logger.info("Loading text-to-speech model '%s'", model_id)

        try:
            return _attempt_load()
        except ValueError as exc:
            message = str(exc)
            if "Unrecognized model" not in message and "trust_remote_code" not in message:
                logger.exception("Failed to load model '%s'", model_id)
                raise RuntimeError(f"Unable to load model '{model_id}': {exc}") from exc

            logger.warning(
                "Model '%s' requires trusting remote code; retrying with trust_remote_code=True.",
                model_id,
            )

            try:
                return _attempt_load(trust_remote_code=True)
            except Exception as retry_exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to load model '%s' with trusted remote code", model_id)
                raise RuntimeError(
                    f"Unable to load model '{model_id}' even with trusted remote code: {retry_exc}"
                ) from retry_exc
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to load model '%s'", model_id)
            raise RuntimeError(f"Unable to load model '{model_id}': {exc}") from exc

    @property
    def device(self) -> str:
        """Return a human-readable description of the active compute device."""

        return "cuda" if self._device == 0 else "cpu"

    async def get_pipeline(self, model_id: str) -> object:
        if model_id in self._pipelines:
            return self._pipelines[model_id]

        lock = self._locks[model_id]
        async with lock:
            if model_id in self._pipelines:
                return self._pipelines[model_id]
            pipeline_obj = await asyncio.to_thread(self._load_pipeline, model_id)
            self._pipelines[model_id] = pipeline_obj
            return pipeline_obj

    async def synthesize(self, model_id: str, text: str) -> Tuple[object, int]:
        if not text.strip():
            raise ValueError("Input text must not be empty.")

        pipeline_obj = await self.get_pipeline(model_id)

        def _run_inference() -> Tuple[object, int]:
            result = pipeline_obj(text)

            if isinstance(result, list):
                if not result:
                    raise RuntimeError("Pipeline returned an empty result list.")
                result = result[0]

            if not isinstance(result, dict):
                raise RuntimeError("Unexpected output type from pipeline; expected a dictionary response.")

            audio_payload = result.get("audio")
            sampling_rate = result.get("sampling_rate")

            if isinstance(audio_payload, dict):
                sampling_rate = sampling_rate or audio_payload.get("sampling_rate")
                audio_payload = audio_payload.get("array") or audio_payload.get("bytes")

            if sampling_rate is None or audio_payload is None:
                raise RuntimeError(
                    "Unexpected output from pipeline; expected 'audio' content and a 'sampling_rate'."
                )

            return audio_payload, int(sampling_rate)

        audio, sample_rate = await asyncio.to_thread(_run_inference)
        return audio, sample_rate


__all__ = ["TTSManager"]
