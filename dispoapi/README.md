# DispoAPI

FastAPI server for the Dispo Camera -- a disposable camera that takes photos, transforms them with AI, and prints them on a thermal receipt printer.

Built at TreeHacks 2025.

## Architecture

The Raspberry Pi Zero 2W captures a photo and POSTs it to this server with a mode and provider. The server routes the image to the chosen AI provider (OpenAI, Gemini, Modal, or Perplexity), gets the transformed result back, and returns it to the Pi as base64. The Pi then dithers and prints it.

```
Camera (Pi Zero 2W)  --->  DispoAPI server  --->  AI provider
       <--- base64 image + text response <---
       |
       v
  Thermal printer
```

## Available Modes

| Mode | What it does |
|------|-------------|
| `ghibli` | Studio Ghibli anime style |
| `greek` | Classical Greek marble statues |
| `duck` | Replace all people with ducks |
| `gpu` | PS2-era low-poly 3D graphics |
| `thinker` | Rodin's "The Thinker" sculpture |
| `business_card` | Extract info into a business card layout |
| `pricing` | Estimate retail prices of visible items |
| `1846` | Time-travel to 1846 |
| `1922` | Time-travel to 1922 |
| `1955` | Time-travel to 1955 |
| `1984` | Time-travel to 1984 |
| `1999` | Time-travel to 1999 |

Any mode can be used with any provider. The caller picks the provider per-request.

## Available Providers

| Provider | Description |
|----------|-------------|
| `openai` | OpenAI gpt-4o Responses API (default) |
| `gemini` | Google Gemini 2.5 Flash image generation |
| `modal` | Flux Kontext on Modal (H100) |
| `perplexity` | Perplexity Sonar API (returns text, not an image) |

## Setup

```bash
# Install dependencies
uv sync

# Copy and fill in your API keys in .env
```

### Required API Keys

| Variable | Required for |
|----------|-------------|
| `OPENAI_API_KEY` | `openai` provider (gpt-image-1) |
| `GEMINI_API_KEY` | `gemini` provider |
| `PERPLEXITY_API_KEY` | `perplexity` provider (Sonar API) |
| `MODAL_ENDPOINT_URL` | `modal` provider |

You only need the keys for the providers you're actually using.

## Modal Setup

The `modal` provider runs Flux Kontext (image-to-image diffusion) on a serverless H100 GPU.

```bash
# 1. Accept the model license
#    https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev

# 2. Create a Modal secret named "huggingface-secret" with your HF_TOKEN
#    https://modal.com/secrets

# 3. Auth with Modal
uv run modal setup

# 4. Deploy
uv run modal deploy modal_app.py
```

Set the printed endpoint URL as `MODAL_ENDPOINT_URL` in your `.env`.

The container auto-scales to zero when idle (5 min timeout). First request cold-starts in ~30-60s, subsequent requests take ~5-10s. Cost is ~$0.01-0.03 per image.

## Running

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API

### `GET /health`

Returns `{"status": "ok"}`.

### `POST /filter`

Multipart form data:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | string | (required) | One of the available modes above |
| `provider` | string | `openai` | AI provider: `openai`, `gemini`, `modal`, `perplexity` |
| `image` | file | (required) | Source image (JPEG, PNG, etc.) |

Response:

```json
{
  "status": "ok",
  "mode": "ghibli",
  "provider": "openai",
  "text": "",
  "image_b64": "<base64 encoded image>"
}
```

For `perplexity`, `text` contains the analysis and `image_b64` echoes back the original image.

### Example (curl)

```bash
# Default provider (openai)
curl -X POST http://localhost:8000/filter \
  -F "mode=ghibli" \
  -F "image=@photo.jpg"

# Explicit provider
curl -X POST http://localhost:8000/filter \
  -F "mode=ghibli" \
  -F "provider=gemini" \
  -F "image=@photo.jpg"
```

## Testing

Place a test image at `test.jpeg`, then:

```bash
# Test all modes with openai (default)
uv run python test_api.py

# Test all modes with gemini
uv run python test_api.py --provider gemini

# Test specific modes
uv run python test_api.py --provider openai ghibli duck 1984
```

Results are saved to `test_outputs/<provider>_<mode>.png`.

Edit `ALL_MODES` in `test_api.py` to comment out modes you don't want to run.

## Project Structure

```
main.py          # FastAPI app, endpoints, AI provider calls
modal_app.py     # Modal deployment — Flux Kontext on H100
prompts.py       # Mode-to-prompt mapping (filter vs search)
test_api.py      # Test script for hitting every mode
pyproject.toml   # Dependencies and project metadata
```
