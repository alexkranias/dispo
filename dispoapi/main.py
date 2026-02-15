"""
DispoAPI — FastAPI server for Dispo Camera image filters.

Stateless: image in -> AI transform -> image out.
"""

import base64
import io
import logging
import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from PIL import Image

from prompts import FILTER_MODES, MODE_PROMPTS, MODE_PROVIDERS, SEARCH_MODES, VALID_MODES

load_dotenv()

log = logging.getLogger("dispoapi")

app = FastAPI(
    title="DispoAPI",
    description="Image filter API for Dispo Camera",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Clients (initialized lazily from env vars)
# ---------------------------------------------------------------------------

_openai_client: OpenAI | None = None
_gemini_client: genai.Client | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def encode_image_b64(image_bytes: bytes) -> str:
    """Encode raw image bytes to a base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def decode_image_b64(data: str) -> bytes:
    """Decode a base64 string back to raw image bytes."""
    return base64.b64decode(data)


def get_prompt_for_mode(mode: str) -> str:
    """Look up the prompt for a given mode. Raises 400 if invalid."""
    if mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode '{mode}'. Valid modes: {sorted(VALID_MODES)}",
        )
    return MODE_PROMPTS[mode]


# ---------------------------------------------------------------------------
# AI provider calls
# ---------------------------------------------------------------------------


async def call_openai(image_bytes: bytes, prompt: str) -> bytes:
    """
    Send image + prompt to OpenAI via the Responses API with the
    image_generation tool (gpt-4o). Returns transformed image bytes.
    """
    client = _get_openai_client()
    b64_input = encode_image_b64(image_bytes)

    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{b64_input}",
                    },
                ],
            }
        ],
        tools=[{"type": "image_generation", "size": "1024x1024", "quality": "low"}],
    )

    for output in response.output:
        if output.type == "image_generation_call":
            return base64.b64decode(output.result)

    raise HTTPException(
        status_code=502, detail="OpenAI did not return an image in its response."
    )


async def call_openai_image_edit(image_bytes: bytes, prompt: str) -> bytes:
    """
    Send image + prompt to OpenAI via the Images Edit API (gpt-image-1.5).
    Returns transformed image bytes.
    """
    client = _get_openai_client()

    response = client.images.edits(
        image=io.BytesIO(image_bytes),
        prompt=prompt,
        model="gpt-image-1.5",
        n=1,
        size="1024x1024",
        quality="auto",
        background="auto",
        moderation="auto",
        input_fidelity="high",
    )

    # The response contains a list of image objects with b64_json or url
    if response.data and len(response.data) > 0:
        img_data = response.data[0]
        if img_data.b64_json:
            return base64.b64decode(img_data.b64_json)
        elif img_data.url:
            # Download from URL if b64 not available
            async with httpx.AsyncClient(timeout=60.0) as http:
                resp = await http.get(img_data.url)
                return resp.content

    raise HTTPException(
        status_code=502, detail="OpenAI image edit did not return an image."
    )


async def call_gemini(image_bytes: bytes, prompt: str) -> bytes:
    """
    Send image + prompt to Gemini for image transformation.
    Returns the transformed image as raw bytes.
    """
    client = _get_gemini_client()

    input_image = Image.open(io.BytesIO(image_bytes))

    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[prompt, input_image],
    )

    if not response.parts:
        raise HTTPException(status_code=502, detail="Gemini returned no response.")

    for part in response.parts:
        if part.inline_data is not None:
            return part.inline_data.data

    raise HTTPException(
        status_code=502, detail="Gemini did not return an image in its response."
    )


async def call_modal(image_bytes: bytes, prompt: str) -> bytes:
    """
    Send image + prompt to a model hosted on Modal for inference.
    Expects a Modal web endpoint that accepts JSON with base64 image + prompt
    and returns JSON with a base64 image.
    """
    modal_url = os.environ.get("MODAL_ENDPOINT_URL")
    if not modal_url:
        raise HTTPException(status_code=500, detail="MODAL_ENDPOINT_URL not set")

    payload = {
        "image_b64": encode_image_b64(image_bytes),
        "prompt": prompt,
    }

    async with httpx.AsyncClient(timeout=600.0) as http:
        resp = await http.post(modal_url, json=payload)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Modal endpoint returned {resp.status_code}: {resp.text[:200]}",
        )

    data = resp.json()
    return decode_image_b64(data["image_b64"])


async def call_perplexity(image_bytes: bytes, prompt: str) -> str:
    """
    Send image to Perplexity Sonar API for analysis (e.g. pricing mode).
    Returns a text response (not an image).
    """
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="PERPLEXITY_API_KEY not set")

    # Resize large images to keep the request payload reasonable
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = encode_image_b64(buf.getvalue())
    mime = "image/jpeg"

    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                        },
                    },
                ],
            }
        ],
    }

    resp = httpx.post(
        "https://api.perplexity.ai/chat/completions",
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=60.0,
    )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Perplexity returned {resp.status_code}: {resp.text[:200]}",
        )

    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------

PROVIDER_CALL = {
    "openai": call_openai,
    "openai_image_edit": call_openai_image_edit,
    "gemini": call_gemini,
    "modal": call_modal,
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Liveness check."""
    return {"status": "ok"}


@app.get("/modes")
async def get_modes():
    """Return all available modes for the camera to consume."""
    return {
        "modes": sorted(VALID_MODES),
        "filter_modes": sorted(FILTER_MODES),
        "search_modes": sorted(SEARCH_MODES),
    }


@app.post("/filter")
async def apply_filter(
    mode: str = Form(..., description="Filter mode to apply"),
    image: UploadFile = File(..., description="Source image to transform"),
):
    """
    Apply an AI image filter/transformation based on the given mode.

    The provider is determined automatically per mode (see MODE_PROVIDERS).
    """
    prompt = get_prompt_for_mode(mode)
    image_bytes = await image.read()

    # Validate upload
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a valid image."
        )

    text = ""

    # Determine provider: search modes -> perplexity, filter modes -> MODE_PROVIDERS
    if mode in SEARCH_MODES:
        provider = "perplexity"
    else:
        provider = MODE_PROVIDERS.get(mode, "gemini")

    try:
        if mode in SEARCH_MODES:
            text = await call_perplexity(image_bytes, prompt)
            result_b64 = encode_image_b64(image_bytes)  # echo original image back
        else:
            result_bytes = await PROVIDER_CALL[provider](image_bytes, prompt)
            result_b64 = encode_image_b64(result_bytes)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"{provider} error: {e}")

    return JSONResponse(
        content={
            "status": "ok",
            "mode": mode,
            "provider": provider,
            "text": text,
            "image_b64": result_b64,
        }
    )
