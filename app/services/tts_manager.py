"""Model management and inference helpers for text-to-speech runtimes."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from app.services.model_installer import ModelInstaller, PreparedModel

logger = logging.getLogger(__name__)


class _BaseSynthesizer:
    """Common protocol for concrete synthesiser implementations."""

    def synthesize(self, text: str, voice: str | None = None, language: str | None = None) -> Tuple[Any, int]:
        raise NotImplementedError


class _TransformersSynthesizer(_BaseSynthesizer):
    def __init__(self, model_path: Path, device_index: int, torch_dtype: torch.dtype | None) -> None:
        from transformers import pipeline

        load_kwargs: Dict[str, Any] = {"task": "text-to-speech", "model": str(model_path), "device": device_index}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        logger.info("Loading transformers pipeline from %s", model_path)
        self._pipeline = pipeline(trust_remote_code=True, **load_kwargs)

    def synthesize(self, text: str, voice: str | None = None, language: str | None = None) -> Tuple[Any, int]:
        call_kwargs: Dict[str, Any] = {}
        if voice:
            call_kwargs["voice"] = voice
        if language:
            call_kwargs["language"] = language

        result = self._pipeline(text, **call_kwargs)

        if isinstance(result, list):
            if not result:
                raise RuntimeError("Pipeline returned an empty result list.")
            result = result[0]

        if not isinstance(result, dict):
            raise RuntimeError("Unexpected output type from transformers pipeline; expected a mapping response.")

        audio_payload = result.get("audio")
        sampling_rate = result.get("sampling_rate")

        if isinstance(audio_payload, dict):
            sampling_rate = sampling_rate or audio_payload.get("sampling_rate")
            audio_payload = audio_payload.get("array") or audio_payload.get("bytes")

        if sampling_rate is None or audio_payload is None:
            raise RuntimeError("Unexpected output from pipeline; expected 'audio' content and a 'sampling_rate'.")

        return audio_payload, int(sampling_rate)


class _KokoroSynthesizer(_BaseSynthesizer):
    def __init__(self, model_id: str, model_path: Path, device_index: int) -> None:
        from kokoro import Kokoro  # type: ignore

        kwargs: Dict[str, Any] = {"download_root": str(model_path)}
        if device_index == 0:
            kwargs["device"] = "cuda"
        else:
            kwargs["device"] = "cpu"

        logger.info("Loading Kokoro model '%s'", model_id)
        try:
            self._engine = Kokoro.from_pretrained(model_id, **kwargs)  # type: ignore[attr-defined]
        except TypeError:
            # Older versions of kokoro-onnx do not support download_root/device hints.
            self._engine = Kokoro.from_pretrained(model_id)  # type: ignore[attr-defined]

        self._default_voice = self._determine_default_voice()
        self._sample_rate = self._determine_sample_rate()

    def _determine_default_voice(self) -> str | None:
        for attr in ("available_voices", "voices", "speakers"):
            voices = getattr(self._engine, attr, None)
            if isinstance(voices, (list, tuple)) and voices:
                return str(voices[0])
        return None

    def _determine_sample_rate(self) -> int:
        for attr in ("sample_rate", "sampling_rate"):
            value = getattr(self._engine, attr, None)
            if isinstance(value, (int, float)) and value:
                return int(value)
        return 24000

    def synthesize(self, text: str, voice: str | None = None, language: str | None = None) -> Tuple[Any, int]:
        kwargs: Dict[str, Any] = {}
        selected_voice = voice or self._default_voice
        if selected_voice:
            kwargs["voice"] = selected_voice

        if hasattr(self._engine, "generate"):
            result = self._engine.generate(text, **kwargs)
        elif hasattr(self._engine, "tts"):
            if "voice" in kwargs:
                kwargs["speaker"] = kwargs.pop("voice")
            result = self._engine.tts(text, **kwargs)
        elif callable(self._engine):
            result = self._engine(text, **kwargs)
        else:  # pragma: no cover - defensive branch for unexpected API changes
            raise RuntimeError("Kokoro model does not expose a recognised synthesis method.")

        sample_rate = self._sample_rate
        audio = result

        if isinstance(result, tuple):
            if len(result) >= 2:
                audio, sr = result[0], result[1]
                if isinstance(sr, (int, float)) and sr:
                    sample_rate = int(sr)
            elif result:
                audio = result[0]

        return audio, sample_rate


class _CoquiSynthesizer(_BaseSynthesizer):
    def __init__(self, model_id: str, model_path: Path, device_index: int) -> None:
        from TTS.api import TTS  # type: ignore

        logger.info("Loading Coqui TTS model '%s'", model_id)
        model_file = self._find_model_file(model_path)
        config_path = self._find_config_file(model_path)

        init_kwargs: Dict[str, Any] = {"progress_bar": False, "gpu": device_index == 0}

        if model_file and config_path:
            init_kwargs.update({"model_path": str(model_file), "config_path": str(config_path)})
            self._tts = TTS(**init_kwargs)
        else:
            self._tts = TTS(model_name=model_id, **init_kwargs)

        self._sample_rate = self._infer_sample_rate()

    def _find_model_file(self, model_path: Path) -> Path | None:
        candidates = list(model_path.glob("**/*.pth")) + list(model_path.glob("**/*.pt"))
        return candidates[0] if candidates else None

    def _find_config_file(self, model_path: Path) -> Path | None:
        for name in ("config.json", "config.yaml", "config.yml"):
            candidate = next(model_path.glob(f"**/{name}"), None)
            if candidate:
                return candidate
        return None

    def _infer_sample_rate(self) -> int:
        for path in (
            ("output_sample_rate",),
            ("sample_rate",),
            ("tts_config", "audio", "sample_rate"),
            ("synthesizer", "output_sample_rate"),
        ):
            value = self._traverse_attribute(self._tts, path)
            if isinstance(value, (int, float)) and value:
                return int(value)
        return 24000

    def synthesize(self, text: str, voice: str | None = None, language: str | None = None) -> Tuple[Any, int]:
        kwargs: Dict[str, Any] = {}
        if voice:
            kwargs["speaker"] = voice
        if language:
            kwargs["language"] = language
        audio = self._tts.tts(text, **kwargs)
        return audio, self._sample_rate

    @staticmethod
    def _traverse_attribute(root: Any, path: Iterable[str]) -> Any:
        value = root
        for key in path:
            if value is None:
                return None
            if isinstance(value, dict):
                value = value.get(key)
            else:
                value = getattr(value, key, None)
        return value


@dataclass(slots=True)
class _SynthCandidate:
    name: str
    factory: Any


class TTSManager:
    """Lazy loader and cache for Hugging Face text-to-speech models."""

    def __init__(self) -> None:
        self._models: Dict[str, _BaseSynthesizer] = {}
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._installer = ModelInstaller()
        self._device_index = 0 if torch.cuda.is_available() else -1
        self._torch_dtype = torch.float16 if torch.cuda.is_available() else None

    @property
    def device(self) -> str:
        """Return a human-readable description of the active compute device."""

        return "cuda" if self._device_index == 0 else "cpu"

    async def get_synthesizer(self, model_id: str) -> _BaseSynthesizer:
        if model_id in self._models:
            return self._models[model_id]

        lock = self._locks[model_id]
        async with lock:
            if model_id in self._models:
                return self._models[model_id]

            prepared = await self._installer.prepare_model(model_id)
            synthesizer = await asyncio.to_thread(self._build_synthesizer, prepared)
            self._models[model_id] = synthesizer
            return synthesizer

    def _build_synthesizer(self, prepared: PreparedModel) -> _BaseSynthesizer:
        candidates = self._candidate_factories(prepared)
        errors: List[str] = []

        for candidate in candidates:
            try:
                logger.debug("Attempting to initialise %s backend for '%s'", candidate.name, prepared.model_id)
                synthesizer: _BaseSynthesizer = candidate.factory()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Initialising %s backend failed for '%s': %s", candidate.name, prepared.model_id, exc)
                errors.append(f"{candidate.name}: {exc}")
                continue
            else:
                logger.info("Initialised %s backend for '%s'", candidate.name, prepared.model_id)
                return synthesizer

        error_message = ", ".join(errors) if errors else "no compatible backend found"
        raise RuntimeError(f"Unable to load model '{prepared.model_id}': {error_message}")

    def _candidate_factories(self, prepared: PreparedModel) -> List[_SynthCandidate]:
        info = prepared.info
        library = (info.library_name or "").lower()
        tags = {tag.lower() for tag in info.tags or []}

        candidates: List[_SynthCandidate] = []

        if library in {"kokoro", "kokoro-onnx"} or "kokoro" in tags:
            candidates.append(
                _SynthCandidate(
                    name="kokoro",
                    factory=lambda: _KokoroSynthesizer(info.modelId, prepared.local_path, self._device_index),
                )
            )

        if library in {"tts", "coqui", "coqui-tts"} or any(tag in tags for tag in {"xtts", "coqui"}):
            candidates.append(
                _SynthCandidate(
                    name="coqui",
                    factory=lambda: _CoquiSynthesizer(info.modelId, prepared.local_path, self._device_index),
                )
            )

        candidates.append(
            _SynthCandidate(
                name="transformers",
                factory=lambda: _TransformersSynthesizer(prepared.local_path, self._device_index, self._torch_dtype),
            )
        )

        return candidates

    async def synthesize(
        self,
        model_id: str,
        text: str,
        voice: str | None = None,
        language: str | None = None,
    ) -> Tuple[Any, int]:
        if not text.strip():
            raise ValueError("Input text must not be empty.")

        synthesizer = await self.get_synthesizer(model_id)

        def _run_inference() -> Tuple[Any, int]:
            audio, sample_rate = synthesizer.synthesize(text, voice=voice, language=language)
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            if isinstance(audio, np.ndarray):
                return audio, int(sample_rate)
            if isinstance(audio, (list, tuple)):
                return np.asarray(audio, dtype=np.float32), int(sample_rate)
            return audio, int(sample_rate)

        return await asyncio.to_thread(_run_inference)


__all__ = ["TTSManager"]

