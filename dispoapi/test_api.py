"""
Test script — calls the /filter endpoint for every mode using demo.jpg.
Saves each result to data/<mode>.png.

Provider is now determined server-side per mode.

Usage:
    # Start the server first:
    #   uv run uvicorn main:app --reload --port 8000
    #
    # Then run this script (all modes):
    #   uv run python test_api.py
    #
    # Test specific modes:
    #   uv run python test_api.py ghibli duck 1984
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
IMAGE_PATH = "data/demo2.jpeg"
OUTPUT_DIR = "data"

# Comment out any modes you don't want to run
ALL_MODES = [
    "greek",
    "ghibli",
    "duck",
    "GPUMODE",
    "thinker",
    "business_card",
    "pricing",
    "1846",
    "1929",
    "1955",
    "1984",
    "1999",
]


def test_mode(mode: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Mode: {mode}")
    print(f"{'=' * 60}")

    with open(IMAGE_PATH, "rb") as f:
        image_data = f.read()

    start = time.time()

    with httpx.Client(timeout=600.0) as client:
        resp = client.post(
            f"{SERVER}/filter",
            data={"mode": mode},
            files={"image": ("demo.jpg", image_data, "image/jpeg")},
        )

    elapsed = time.time() - start

    if resp.status_code != 200:
        print(f"  FAILED ({resp.status_code}): {resp.text[:300]}")
        return

    data = resp.json()
    provider = data.get("provider", "unknown")
    print(f"  Status:   {data['status']}")
    print(f"  Provider: {provider}")
    print(f"  Time:     {elapsed:.1f}s")

    if mode in SEARCH_MODES:
        # Save text response for search modes
        text = data.get("text", "")
        print(f"  Text:\n{text}")
        out_path = os.path.join(OUTPUT_DIR, f"{provider}_{mode}.txt")
        with open(out_path, "w") as f:
            f.write(text)
        print(f"  Saved:    {out_path}")

        # Also save the image if one was returned
        img_b64 = data.get("image_b64", "")
        if img_b64:
            img_bytes = base64.b64decode(img_b64)
            img_path = os.path.join(OUTPUT_DIR, f"{provider}_{mode}.png")
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            print(f"  Saved:    {img_path} ({len(img_bytes) / 1024:.0f} KB)")
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

    args = sys.argv[1:]

    # Check server is up
    try:
        resp = httpx.get(f"{SERVER}/health", timeout=5.0)
        resp.raise_for_status()
    except Exception:
        print(f"Error: server not reachable at {SERVER}. Start it first:")
        print(f"  uv run uvicorn main:app --reload --port 8000")
        sys.exit(1)

    modes = args if args else ALL_MODES

    print(f"  Modes: {', '.join(modes)}")

    for mode in modes:
        test_mode(mode)

    print(f"\n{'=' * 60}")
    print(f"  Done. Results in {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
