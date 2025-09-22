"""Utilities for downloading Hugging Face models and their dependencies on-demand."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set

from huggingface_hub import HfApi, ModelInfo, snapshot_download

logger = logging.getLogger(__name__)


_DEFAULT_CACHE_DIR = Path(os.environ.get("MULTI_TTS_MODEL_CACHE", "./.cache/models")).resolve()


def _any_keyword_in_string(value: str | None, keywords: Iterable[str]) -> bool:
    if not value:
        return False
    value_lower = value.lower()
    return any(keyword in value_lower for keyword in keywords)


def _iter_card_strings(card_data: dict | None) -> Iterable[str]:
    if not isinstance(card_data, dict):
        return []

    stack: List[object] = [card_data]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            stack.extend(current.values())
            stack.extend(current.keys())
        elif isinstance(current, (list, tuple, set)):
            stack.extend(current)
        elif isinstance(current, str):
            yield current


def _info_matches_keywords(info: ModelInfo, keywords: Iterable[str]) -> bool:
    keyword_set = {keyword.lower() for keyword in keywords}
    if _any_keyword_in_string(info.modelId, keyword_set):
        return True
    if _any_keyword_in_string(info.library_name, keyword_set):
        return True
    if _any_keyword_in_string(getattr(info, "pipeline_tag", None), keyword_set):
        return True

    for tag in info.tags or []:
        if _any_keyword_in_string(tag, keyword_set):
            return True

    for sibling in getattr(info, "siblings", []) or []:
        if _any_keyword_in_string(getattr(sibling, "rfilename", None), keyword_set):
            return True

    for entry in _iter_card_strings(info.cardData):
        if _any_keyword_in_string(entry, keyword_set):
            return True

    return False


def is_kokoro_model(info: ModelInfo) -> bool:
    """Return ``True`` if *info* appears to describe a Kokoro model."""

    return _info_matches_keywords(info, {"kokoro", "kokoro-onnx"})


def is_coqui_model(info: ModelInfo) -> bool:
    """Return ``True`` if *info* appears to describe a Coqui TTS/XTTS model."""

    return _info_matches_keywords(info, {"coqui", "xtts"})


@dataclass(slots=True)
class PreparedModel:
    """Metadata returned after ensuring a model has been downloaded locally."""

    model_id: str
    local_path: Path
    info: ModelInfo


class ModelInstaller:
    """Download Hugging Face model repositories and install runtime dependencies lazily."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = (cache_dir or _DEFAULT_CACHE_DIR).resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._api = HfApi()
        self._locks: dict[str, asyncio.Lock] = {}
        self._installed_requirements: Set[str] = set()

    async def prepare_model(self, model_id: str) -> PreparedModel:
        """Ensure *model_id* is downloaded locally and runtime dependencies are installed."""

        lock = self._locks.setdefault(model_id, asyncio.Lock())
        async with lock:
            return await asyncio.to_thread(self._prepare_model_sync, model_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_model_sync(self, model_id: str) -> PreparedModel:
        logger.info("Ensuring model '%s' is available locally", model_id)
        info = self._api.model_info(model_id)
        local_path = self._download_model(model_id)
        self._install_dependencies(local_path, info)
        return PreparedModel(model_id=model_id, local_path=local_path, info=info)

    def _download_model(self, model_id: str) -> Path:
        safe_id = self._safe_model_dir(model_id)
        target_dir = self._cache_dir / safe_id
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        return target_dir

    def _install_dependencies(self, local_dir: Path, info: ModelInfo) -> None:
        requirements: Set[str] = set()
        requirements.update(self._requirements_from_library(info))
        requirements.update(self._requirements_from_card(info))
        requirements.update(self._requirements_from_files(local_dir))

        packages = [req for req in sorted(requirements) if req and req not in self._installed_requirements]
        if not packages:
            return

        logger.info("Installing %d requirement(s) for model '%s'", len(packages), info.modelId)
        try:
            self._run_pip_install(packages)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to install dependencies for model '%s'", info.modelId)
            raise RuntimeError(f"Unable to install dependencies for model '{info.modelId}': {exc}") from exc
        else:
            self._installed_requirements.update(packages)

    # ------------------------------------------------------------------
    # Requirement gathering helpers
    # ------------------------------------------------------------------
    def _requirements_from_library(self, info: ModelInfo) -> Set[str]:
        library = (info.library_name or "").lower()
        tags = {tag.lower() for tag in info.tags or [] if isinstance(tag, str)}

        requirements: Set[str] = set()

        if (
            library in {"tts", "coqui", "coqui-tts"}
            or any(tag in tags for tag in {"xtts", "coqui"})
            or is_coqui_model(info)
        ):
            requirements.add("TTS==0.22.0")

        if (
            library in {"kokoro", "kokoro-onnx"}
            or "kokoro" in tags
            or is_kokoro_model(info)
        ):
            requirements.add("kokoro-onnx>=0.1.0")

        if library in {"piper", "piper-tts"} or "piper" in tags:
            requirements.add("piper-tts>=1.2.0")

        return requirements

    def _requirements_from_card(self, info: ModelInfo) -> Set[str]:
        requirements: Set[str] = set()
        card_data = info.cardData or {}
        potential_keys = [
            "pip_requirements",
            "pip_requirements.txt",
            "requirements",
            "dependencies",
        ]

        for key in potential_keys:
            value = card_data.get(key)
            if isinstance(value, str):
                requirements.add(value.strip())
            elif isinstance(value, Iterable):
                for item in value:
                    if isinstance(item, str):
                        requirements.add(item.strip())

        return {req for req in requirements if req}

    def _requirements_from_files(self, local_dir: Path) -> Set[str]:
        requirements: Set[str] = set()
        candidate_files = [
            local_dir / "requirements.txt",
            local_dir / "requirements.in",
            local_dir / "requirements-dev.txt",
            local_dir / "pip_requirements.txt",
        ]

        for path in candidate_files:
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:  # pragma: no cover - best effort only
                continue
            requirements.update(self._parse_requirements_text(text))

        return requirements

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_model_dir(model_id: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id)
        return sanitized.strip("-")

    @staticmethod
    def _parse_requirements_text(text: str) -> Set[str]:
        results: Set[str] = set()
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            results.add(stripped)
        return results

    @staticmethod
    def _run_pip_install(packages: List[str]) -> None:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "--no-input", *packages]
        logger.debug("Running pip install: %s", " ".join(cmd))
        subprocess.check_call(cmd)


__all__ = ["ModelInstaller", "PreparedModel", "is_coqui_model", "is_kokoro_model"]

