"""Utility helpers for working with generated audio."""
from __future__ import annotations

import io
from typing import Iterable, Union

import numpy as np
from scipy.io import wavfile

try:  # pragma: no cover - torch is an optional runtime dependency for tests
    import torch
except ImportError:  # pragma: no cover - allows running without torch installed
    torch = None  # type: ignore

__all__ = ["audio_array_to_wav_bytes", "chunk_bytes"]


def audio_array_to_wav_bytes(audio: Union[Iterable[float], bytes, bytearray, memoryview], sample_rate: int) -> bytes:
    """Convert audio data into an in-memory WAV byte buffer.

    The Hugging Face text-to-speech pipelines may return either a NumPy array,
    a PyTorch tensor or pre-encoded audio bytes.  This helper normalises all
    supported formats into WAV-encoded bytes so that responses can be streamed
    to API clients.
    """

    if isinstance(audio, (bytes, bytearray, memoryview)):
        return bytes(audio)

    if torch is not None and isinstance(audio, torch.Tensor):
        audio_array = audio.detach().cpu().numpy()
    else:
        audio_array = np.asarray(audio)

    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)

    # Ensure we stay within the valid floating point PCM range.
    if audio_array.size:
        audio_array = np.clip(audio_array, -1.0, 1.0)

    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio_array)
    buffer.seek(0)
    return buffer.read()


def chunk_bytes(data: bytes, chunk_size: int = 32 * 1024) -> Iterable[bytes]:
    """Yield the provided bytes object in streaming-sized chunks."""

    view = memoryview(data)
    for start in range(0, len(view), chunk_size):
        yield view[start : start + chunk_size].tobytes()
