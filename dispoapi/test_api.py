"""
Test script — calls the /filter endpoint for every mode using test.jpeg.
Saves each result to test_outputs/<provider>_<mode>.png.

Usage:
    # Start the server first:
    #   uv run uvicorn main:app --reload --port 8000
    #
    # Then run this script (defaults to openai provider):
    #   uv run python test_api.py
    #
    # Choose a provider:
    #   uv run python test_api.py --provider gemini
    #
    # Test specific modes:
    #   uv run python test_api.py --provider openai ghibli duck 1984
"""

import base64
import os
import sys
import time

import httpx
from dotenv import load_dotenv

from prompts import SEARCH_MODES

load_dotenv()

SERVER = os.environ.get("TEST_SERVER", "http://localhost:8000")
IMAGE_PATH = "data/demo.jpg"
OUTPUT_DIR = "data"

# Comment out any modes you don't want to run
ALL_MODES = [
    "greek",
    "ghibli",
    "duck",
    "gpu",
    "thinker",
    "business_card",
    "pricing",
    "1846",
    "1922",
    "1955",
    "1984",
    "1999",
]


def test_mode(mode: str, provider: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Mode: {mode}  |  Provider: {provider}")
    print(f"{'=' * 60}")

    with open(IMAGE_PATH, "rb") as f:
        image_data = f.read()

    start = time.time()

    with httpx.Client(timeout=600.0) as client:
        resp = client.post(
            f"{SERVER}/filter",
            data={"mode": mode, "provider": provider},
            files={"image": ("test.jpeg", image_data, "image/jpeg")},
        )

    elapsed = time.time() - start

    if resp.status_code != 200:
        print(f"  FAILED ({resp.status_code}): {resp.text[:300]}")
        return

    data = resp.json()
    print(f"  Status:   {data['status']}")
    print(f"  Time:     {elapsed:.1f}s")

    if mode in SEARCH_MODES:
        # Save text response for search modes
        text = data.get("text", "")
        print(f"  Text:\n{text}")
        out_path = os.path.join(OUTPUT_DIR, f"{provider}_{mode}.txt")
        with open(out_path, "w") as f:
            f.write(text)
        print(f"  Saved:    {out_path}")
    else:
        # Save image for filter modes
        img_bytes = base64.b64decode(data["image_b64"])
        out_path = os.path.join(OUTPUT_DIR, f"{provider}_{mode}.png")
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        print(f"  Saved:    {out_path} ({len(img_bytes) / 1024:.0f} KB)")


def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: {IMAGE_PATH} not found. Place a test image at {IMAGE_PATH}.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Parse --provider flag
    args = sys.argv[1:]
    provider = "openai"
    if "--provider" in args:
        idx = args.index("--provider")
        provider = args[idx + 1]
        args = args[:idx] + args[idx + 2 :]

    # Check server is up
    try:
        resp = httpx.get(f"{SERVER}/health", timeout=5.0)
        resp.raise_for_status()
    except Exception:
        print(f"Error: server not reachable at {SERVER}. Start it first:")
        print(f"  uv run uvicorn main:app --reload --port 8000")
        sys.exit(1)

    modes = args if args else ALL_MODES

    print(f"  Provider: {provider}")
    print(f"  Modes:    {', '.join(modes)}")

    for mode in modes:
        test_mode(mode, provider)

    print(f"\n{'=' * 60}")
    print(f"  Done. Results in {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
