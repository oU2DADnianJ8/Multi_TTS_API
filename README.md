# Multi-Model Streaming TTS API

A self-hosted, OpenAI-compatible text-to-speech (TTS) server that streams audio
responses using the latest trending models from the Hugging Face Hub. The
project bundles a FastAPI backend, intelligent model management, and an
interactive single-page web console for effortless experimentation.

## Features

- **OpenAI-compatible endpoint** – `/v1/audio/speech` accepts OpenAI-style JSON
  payloads and returns streamed WAV audio for minimal latency.
- **Automatic model discovery** – `/api/models` lists the top 20 trending
  `text-to-speech` models directly from Hugging Face.
- **Dynamic model loading & caching** – models are downloaded and cached on
  demand, with automatic GPU acceleration when CUDA is available.
- **Modern web GUI** – a responsive SPA served from the backend allows you to
  browse models, submit prompts, and play the generated speech instantly.
- **Production-ready FastAPI stack** – CORS, health checks, async streaming, and
  structured error handling included out of the box.

## Prerequisites

- Python 3.10 or newer.
- A recent version of `pip`.
- (Optional but recommended) A CUDA-capable GPU for faster inference.
- Sufficient disk space for caching large TTS models downloaded from Hugging
  Face.
- Some models may require access to gated repositories; if so, export a
  `HUGGINGFACE_HUB_TOKEN` environment variable before starting the server.

## Installation

Clone the repository and install the dependencies inside a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the server

Launch the API with Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The application exposes:

- `GET /` – interactive single-page UI for browsing and testing models.
- `GET /api/models` – JSON catalogue of the top 20 trending Hugging Face
  text-to-speech models (cached for 30 minutes).
- `POST /v1/audio/speech` – OpenAI-style streaming TTS synthesis endpoint.
- `OPTIONS /v1/audio/speech` – CORS pre-flight handler.
- `GET /healthz` – simple health probe including the active compute device.

## Using the API

### Retrieve available models

```bash
curl http://localhost:8000/api/models | jq
```

Example response:

```json
{
  "count": 20,
  "models": [
    {
      "id": "espnet/kan-bayashi_ljspeech_vits",
      "pipeline_tag": "text-to-speech",
      "likes": 632,
      "downloads": 138214,
      "tags": ["text-to-speech", "vits"],
      "library_name": "espnet",
      "last_modified": "2024-05-06T11:24:25"
    }
  ]
}
```

### Synthesize speech (streaming)

```bash
curl \
  -X POST \
  -H "Content-Type: application/json" \
  -o output.wav \
  http://localhost:8000/v1/audio/speech \
  -d '{"model": "espnet/kan-bayashi_ljspeech_vits", "input": "Hello from the Multi-Model TTS API!", "format": "wav"}'
```

The response body is streamed, so playback can start before the full file is
downloaded. The default format is WAV; other formats can be layered in by
extending `app/utils/audio.py`.

### Web console

Navigate to <http://localhost:8000/>. The page automatically fetches the model
catalogue, lets you pick a model, submit text, and plays the streamed audio once
available. The UI is ideal for smoke-testing models before integrating the API
into your own tooling.

## Caching & performance notes

- Models are cached locally using the standard Hugging Face cache (typically
  `~/.cache/huggingface/`). Subsequent requests reuse the loaded model instance,
  avoiding redundant downloads.
- GPU acceleration is automatically enabled when CUDA is detected. On CPU-only
  environments, expect slower first-time inference.
- Cold starts may take several seconds while a model downloads and initialises.
  Monitor server logs for progress feedback.

## Project structure

```
.
├── app
│   ├── main.py               # FastAPI application and routes
│   ├── services
│   │   ├── hf_models.py      # Trending model discovery helpers
│   │   └── tts_manager.py    # Lazy-loading TTS pipeline manager
│   ├── static
│   │   └── index.html        # Single-page UI for model selection/testing
│   └── utils
│       └── audio.py          # Audio serialisation utilities
├── requirements.txt
└── README.md
```

## Next steps

- Configure a process manager (systemd, supervisor, Docker, etc.) for
  production deployments.
- Add authentication in front of the API if exposing it publicly.
- Extend the `SpeechRequest` model to surface additional generation controls
  specific to your favourite Hugging Face TTS models.

Enjoy building with fully self-hosted, streaming text-to-speech! 🚀
