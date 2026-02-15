"""
Mode-to-prompt mapping for the /filter endpoint.

Modes fall into two categories:
  - filter: uses diffusion (OpenAI / Gemini / Modal) to transform the image
  - search: uses Perplexity Sonar API to analyze the image and return text
"""

# -- Filter modes (diffusion — image in, transformed image out) ------------

FILTER_PROMPTS: dict[str, str] = {
    "greek": "Make everyone waer ancient greek style robes. Make surrounding architecture look like ancient greece. Do not change the framing or composition of the image.",
    "ghibli": "Imagine the scene in the style of Studio Ghibli.",
    "duck": "Replace every human in the image with human-sized duck in the same pose.",
    "GPUMODE": "Replace every human in the image with a vertically oriented H100 GPU. It's like the 2001 space odyssey monolith.",
    "thinker": "Make the main subjects of the image pose in the style of the thinker. Do not change the scene, just change their pose.",
    "business_card": "Take the main person in the image and design a business card with their face in it.",
    "1846": "Reimagine the image as if it were the year 1846. Do NOT change the framing or composition of the image, and keep the main subjects the same. Replace elements that are not appropriate for the time period.",
    "1929": "Reimagine the image as if it were the 1929. Do NOT change the framing or composition of the image, and keep the main subjects the same. Replace elements that are not appropriate for the time period.",
    "1955": "Reimagine the image as if it were the 1955. Do NOT change the framing or composition of the image, and keep the main subjects the same. Replace elements that are not appropriate for the time period.",
    "1984": "Reimagine the image as if it were the 1984. Do NOT change the framing or composition of the image, and keep the main subjects the same. Replace elements that are not appropriate for the time period.",
    "1999": "Reimagine the image as if it were the 1999. Do NOT change the framing or composition of the image, and keep the main subjects the same. Replace elements that are not appropriate for the time period.",
}

# -- Search modes (Perplexity — image in, text out) -------------------------

SEARCH_PROMPTS: dict[str, str] = {
    "pricing": (
        "Look at this image carefully. Identify the single main subject or product in the image. "
        "Search the web to find its current approximate retail price in USD. "
        "Be specific about the item (brand, model, size if visible). "
        "If you cannot confidently identify the main subject or find a price, respond with exactly:\n"
        "N/A — N/A\n\n"
        "Otherwise, respond with ONLY one line in this exact format (no extra text):\n"
        "ITEM NAME — $PRICE"
    ),
}

# -- Provider routing per mode -----------------------------------------------
# Every filter mode is "gemini" except "ghibli" which uses OpenAI gpt-image-1.5.
# Search modes are routed to "perplexity" in the endpoint logic.

MODE_PROVIDERS: dict[str, str] = {
    "greek": "gemini",
    "ghibli": "openai_image_edit",
    "duck": "gemini",
    "GPUMODE": "gemini",
    "thinker": "gemini",
    "business_card": "gemini",
    "1846": "gemini",
    "1929": "gemini",
    "1955": "gemini",
    "1984": "gemini",
    "1999": "gemini",
}

# -- Combined lookups -------------------------------------------------------

MODE_PROMPTS: dict[str, str] = {**FILTER_PROMPTS, **SEARCH_PROMPTS}
VALID_MODES: set[str] = set(MODE_PROMPTS.keys())
FILTER_MODES: set[str] = set(FILTER_PROMPTS.keys())
SEARCH_MODES: set[str] = set(SEARCH_PROMPTS.keys())
