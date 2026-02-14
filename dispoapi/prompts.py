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
        "Look at this image carefully. Identify every distinct product, item, or object visible. "
        "For each item, search the web to find its current approximate retail price in USD. "
        "Return a clean list with one line per item in this exact format:\n"
        "ITEM NAME — $PRICE\n\n"
        "Be specific about the item (brand, model, size if visible). "
        "If you can't determine the exact product, estimate based on the closest match you can find. "
        "Only list items that are clearly visible. Keep it concise."
    ),
}

# -- Combined lookups -------------------------------------------------------

MODE_PROMPTS: dict[str, str] = {**FILTER_PROMPTS, **SEARCH_PROMPTS}
VALID_MODES: set[str] = set(MODE_PROMPTS.keys())
FILTER_MODES: set[str] = set(FILTER_PROMPTS.keys())
SEARCH_MODES: set[str] = set(SEARCH_PROMPTS.keys())
