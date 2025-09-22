"""Utility helpers for working with generated audio."""
from __future__ import annotations

import io
from typing import Iterable

import numpy as np
from scipy.io import wavfile

__all__ = ["audio_array_to_wav_bytes", "chunk_bytes"]


def audio_array_to_wav_bytes(audio: Iterable[float], sample_rate: int) -> bytes:
    """Convert a NumPy audio array into an in-memory WAV byte buffer.

    The Hugging Face text-to-speech pipelines generally return audio as a
    floating-point NumPy array with values in the ``[-1, 1]`` range.  This
    helper normalises the array, converts it to ``float32`` (the most widely
    supported PCM encoding for WAV files) and serialises it into an in-memory
    bytes object that can be streamed to API clients.
    """

    audio_array = np.asarray(audio)

    if audio_array.dtype not in (np.float32, np.float64):
        audio_array = audio_array.astype(np.float32)
    else:
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
