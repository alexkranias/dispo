"""
DispoAPI — FastAPI server for Dispo Camera image filters.

Stateless: image in -> AI transform -> image out.
"""

import base64
import io
import logging
import os
import re

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont

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


MAX_DIMENSION = 480


def downscale_image(image_bytes: bytes, max_dim: int = MAX_DIMENSION) -> bytes:
    """Downscale an image so its largest side is at most *max_dim* pixels.

    Preserves aspect ratio. Returns JPEG bytes. If the image is already
    within bounds it is re-encoded without resizing.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    # Convert to RGB so we can always save as JPEG (handles RGBA, palette, etc.)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def render_price_image(title: str, price: str, width: int = 1280, height: int = 720) -> bytes:
    """Render a clean white image with item name and price text using Pillow."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Try to load a nice font, fall back to default
    try:
        font_price = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 280)
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 80)
    except OSError:
        try:
            font_price = ImageFont.truetype("arial.ttf", 280)
            font_title = ImageFont.truetype("arial.ttf", 80)
        except OSError:
            font_price = ImageFont.load_default(size=280)
            font_title = ImageFont.load_default(size=80)

    # Draw price centred
    price_bbox = draw.textbbox((0, 0), price, font=font_price)
    price_w = price_bbox[2] - price_bbox[0]
    price_h = price_bbox[3] - price_bbox[1]
    price_x = (width - price_w) // 2
    price_y = (height - price_h) // 2

    draw.text((price_x, price_y), price, fill="black", font=font_price)

    # Draw item title centred above the price
    if title and title != "N/A":
        title_bbox = draw.textbbox((0, 0), title, font=font_title)
        title_w = title_bbox[2] - title_bbox[0]
        # Truncate if too wide
        while title_w > width - 80 and len(title) > 10:
            title = title[: len(title) - 4] + "…"
            title_bbox = draw.textbbox((0, 0), title, font=font_title)
            title_w = title_bbox[2] - title_bbox[0]
        title_x = (width - title_w) // 2
        title_y = price_y - 90
        draw.text((title_x, title_y), title, fill="#555555", font=font_title)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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
    Send image + prompt to OpenAI Images Edit API (gpt-image-1.5).
    Uses the REST API directly since the SDK may not support the new
    JSON-based edit endpoint yet.
    Returns transformed image bytes.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    b64_input = encode_image_b64(image_bytes)

    payload = {
        "images": [
            {"image_url": f"data:image/jpeg;base64,{b64_input}"}
        ],
        "prompt": prompt,
        "model": "gpt-image-1.5",
        "n": 1,
        "size": "auto",
        "quality": "auto",
        "background": "auto",
        "moderation": "auto",
        "input_fidelity": "high",
    }

    async with httpx.AsyncClient(timeout=300.0) as http:
        resp = await http.post(
            "https://api.openai.com/v1/images/edits",
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI image edit returned {resp.status_code}: {resp.text[:300]}",
        )

    data = resp.json()
    images = data.get("data", [])
    if images and images[0].get("b64_json"):
        return base64.b64decode(images[0]["b64_json"])

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


# ---------------------------------------------------------------------------
# Business card structured OCR model
# ---------------------------------------------------------------------------


class BusinessCardInfo(BaseModel):
    """Structured contact info extracted from an image via OCR."""
    name: str = ""
    title: str = ""
    company: str = ""
    contact: str = ""  # email, website, phone, social handle — whatever is found


def _make_blank_card(width: int = 1050, height: int = 600) -> Image.Image:
    """Create a plain white image at 3.5:2 aspect ratio (standard business card)."""
    return Image.new("RGB", (width, height), "white")


async def _ocr_business_card(client: genai.Client, image: Image.Image) -> BusinessCardInfo:
    """Use Gemini to extract structured contact info from an image."""
    ocr_prompt = (
        "Look at this image. Extract any information useful for a business card. "
        "Return JSON with these fields (empty string if not found): "
        "name, title, company, contact (email/website/phone/social)."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[ocr_prompt, image],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BusinessCardInfo,
        ),
    )

    text = response.text.strip() if response.text else "{}"
    log.info("business_card OCR: %s", text[:300])

    try:
        return BusinessCardInfo.model_validate_json(text)
    except Exception:
        return BusinessCardInfo()


async def call_business_card(image_bytes: bytes, prompt: str) -> bytes:
    """
    Business card generation:
    1. Structured OCR to extract name/title/company/contact from the photo
    2. Send blank white 3.5:2 card canvas + photo to Gemini Flash with the
       extracted info to compose the card.
    """
    client = _get_gemini_client()
    input_image = Image.open(io.BytesIO(image_bytes))

    # -- Step 1: Structured OCR ---------------------------------------------
    info = await _ocr_business_card(client, input_image)

    # Build a text block from whatever fields were found
    info_lines = []
    if info.name:
        info_lines.append(f"Name: {info.name}")
    if info.title:
        info_lines.append(f"Title: {info.title}")
    if info.company:
        info_lines.append(f"Company: {info.company}")
    if info.contact:
        info_lines.append(f"Contact: {info.contact}")

    if info_lines:
        info_block = "\n".join(info_lines)
    else:
        info_block = "No contact info found — use placeholder lines."

    log.info("business_card info block:\n%s", info_block)

    # -- Step 2: Generate the card image ------------------------------------
    blank_card = _make_blank_card()

    gen_prompt = (
        "The first image is a blank white business card. The second image is a photo.\n"
        "Edit the blank card:\n"
        "- Place a nice circular headshot of the main subject on the left. Make them look professional, looking at the camera head on.\n"
        f"- On the right, typeset this info:\n{info_block}\n\n"
        "ONLY include the info listed above. Do NOT invent or add any text, names, "
        "or details not explicitly provided. Leave blank any area where info is missing.\n"
        "Keep it minimal, clean, modern sans-serif. White background, no borders."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[gen_prompt, blank_card, input_image],
        config=genai_types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        ),
    )

    if not response.parts:
        raise HTTPException(
            status_code=502, detail="Gemini returned no response for business card."
        )

    raw_bytes: bytes | None = None
    for part in response.parts:
        if part.inline_data is not None:
            raw_bytes = part.inline_data.data
            break

    if raw_bytes is None:
        raise HTTPException(
            status_code=502,
            detail="Gemini did not return an image for the business card.",
        )

    # Post-process: ensure exact 3.5:2 landscape ratio
    card = Image.open(io.BytesIO(raw_bytes))

    if card.height > card.width:
        card = card.rotate(90, expand=True)

    target_ratio = 3.5 / 2.0  # 1.75
    current_ratio = card.width / card.height

    if current_ratio > target_ratio:
        new_width = int(card.height * target_ratio)
        left = (card.width - new_width) // 2
        card = card.crop((left, 0, left + new_width, card.height))
    elif current_ratio < target_ratio:
        new_height = int(card.width / target_ratio)
        top = (card.height - new_height) // 2
        card = card.crop((0, top, card.width, top + new_height))

    buf = io.BytesIO()
    card.save(buf, format="PNG")
    return buf.getvalue()


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
    "business_card": call_business_card,
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

    # Validate upload (verify() consumes the object, so use a fresh open)
    try:
        img_check = Image.open(io.BytesIO(image_bytes))
        img_check.verify()
    except Exception:
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a valid image."
        )

    text = ""
    title = ""
    price = ""

    # Determine provider: search modes -> perplexity, filter modes -> MODE_PROVIDERS
    if mode in SEARCH_MODES:
        provider = "perplexity"
    else:
        provider = MODE_PROVIDERS.get(mode, "gemini")

    log.info("mode=%s provider=%s image_size=%d", mode, provider, len(image_bytes))

    try:
        if mode in SEARCH_MODES:
            text = await call_perplexity(image_bytes, prompt)

            # Strip Perplexity citation references like [1], [3], etc.
            text = re.sub(r"\[\d+\]", "", text).strip()

            # Parse "ITEM NAME — $PRICE" into structured title/price fields
            if "—" in text:
                parts = text.split("—", 1)
                title = parts[0].strip()
                price = parts[1].strip()
            else:
                title = text.strip()
                price = "N/A"

            # If price wasn't detected, return early with no image
            if price == "N/A" or title == "N/A":
                return JSONResponse(
                    content={
                        "status": "not_found",
                        "mode": mode,
                        "provider": provider,
                        "title": title,
                        "price": price,
                        "text": text,
                        "image_b64": "",
                    }
                )

            # Render a clean white image with the price text
            price_img_bytes = render_price_image(title, price)
            result_b64 = encode_image_b64(price_img_bytes)
        else:
            # Downscale before sending to diffusion models
            scaled_bytes = downscale_image(image_bytes)
            result_bytes = await PROVIDER_CALL[provider](scaled_bytes, prompt)
            result_b64 = encode_image_b64(result_bytes)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Provider %s failed for mode %s", provider, mode)
        raise HTTPException(status_code=502, detail=f"{provider} error: {e}")

    return JSONResponse(
        content={
            "status": "ok",
            "mode": mode,
            "provider": provider,
            "title": title,
            "price": price,
            "text": text,
            "image_b64": result_b64,
        }
    )
