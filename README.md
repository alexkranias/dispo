# Diffuji

A diffusion-powered instant camera that turns half-real, half-dreamed moments into physical prints.

Built at **TreeHacks 2026**.

## Inspiration

The world needs a bit more silliness. Instant cameras are fun, and there's something special about memorializing a moment through physical media. But what if these moments were part dream, part hallucination of reality?

**Diffuji** captures a photo and reimagines it through image-to-image diffusion models, then prints the result on thermal sticker paper — no ink required.

## What It Does

Take a picture and choose from a range of filters. These aren't your typical color grades — they can completely reimagine the context the photo was taken in:

| Mode | Description |
|------|-------------|
| `ghibli` | Studio Ghibli anime style |
| `greek` | Classical Greek marble statues |
| `duck` | Replace all people with ducks |
| `gpu` | PS2-era low-poly 3D graphics |
| `thinker` | Rodin's "The Thinker" sculpture |
| `1846` | Time-travel to 1846 |
| `1922` | Time-travel to 1922 |
| `1955` | Time-travel to 1955 |
| `1984` | Time-travel to 1984 |
| `1999` | Time-travel to 1999 |
| `business_card` | Extract info into a business card layout |
| `pricing` | Estimate retail prices of visible items |

The **pricing** mode was inspired by a teammate's experience working at a thrift store — snap a pic of an object and use Perplexity to search the internet for competitive prices, then print a reference-backed price tag.

## How We Built It

### Hardware

| Component | Purpose |
|-----------|---------|
| Raspberry Pi Zero 2W | Low power draw, WiFi, 512 MB RAM, $15 |
| Arducam camera module | Image capture |
| TTL thermal printer | Inkless printing onto sticker paper |
| Rotary encoder | Switching between modes |
| Push button | Shutter |
| Tactile switch | Power |
| I2C OLED display | Displaying modes and settings |
| 2x 18650 batteries + 3A 5V UPS | Power supply with voltage regulation |
| 1000 µF capacitor | Guards against current spikes from the printer |

The shell was designed in Fusion 360 and 3D-printed at the TreeHacks makerspace. Components were soldered together and mounted with hot glue and screws.

### Software

#### On-Device (`camera/`)

A Python script running on the Pi that:

- Manages hardware inputs (rotary encoder, shutter button, power switch)
- Displays animations and state on the OLED screen
- Captures images and either processes them locally or sends them to the cloud
- Dithers images for thermal printing using **ordered Bayer dithering** with custom gamma and contrast adjustments
- Prints the final result

#### Cloud API (`dispoapi/`)

A FastAPI server hosted on [Railway](https://railway.app) that serves as the camera's brain:

- Provides the camera with its available modes
- Routes images to AI providers for diffusion-based transformation
- Returns processed images as base64

**AI Providers:**

| Provider | Description |
|----------|-------------|
| OpenAI | GPT-image-1 via Responses API |
| Gemini | Google Gemini 2.5 Flash image generation |
| Modal | Flux Kontext on serverless H100 GPU |
| Perplexity | Sonar API for text-based search and pricing |

### Architecture

```
┌─────────────────────┐         ┌──────────────┐         ┌─────────────┐
│  Raspberry Pi Zero  │  POST   │   DispoAPI   │  route  │ AI Provider │
│  - Arducam          │ ──────> │  (Railway)   │ ──────> │ OpenAI /    │
│  - OLED display     │ <────── │              │ <────── │ Gemini /    │
│  - Rotary encoder   │  base64 │              │  image  │ Modal /     │
│  - Thermal printer  │         └──────────────┘         │ Perplexity  │
└─────────────────────┘                                  └─────────────┘
         │
         v
   ┌───────────┐
   │  Thermal  │
   │  Sticker  │
   │  Print    │
   └───────────┘
```

## Challenges We Ran Into

- **Paper jamming** — the sticker paper would bunch up inside the printer. We found that peeling back some of the sticker backing before feeding it helped.
- **Inconsistent style transfer** — prompting the diffusion model to consistently apply a style was tricky. We also discovered that feeding images upside-down produces terrible results (too far out of distribution).
- **On-device diffusion** — running diffusion on a device with only 512 MB of RAM is painfully slow, pushing us toward the cloud API approach.

## Accomplishments We're Proud Of

We built a sleek, functional product that was genuinely fun to use. We hope everyone at the hackathon enjoyed getting custom sticker prints!

## What We Learned

- Dithering algorithms and how to make images look aesthetic on thermal paper
- Inference strategies for extremely low-RAM systems
- Hardware integration with the Raspberry Pi ecosystem (I2C, TTL serial, GPIO)
- Prompting diffusion models for consistent style transfer

## What's Next

**On-device diffusion.** With emerging lightweight diffusion architectures and quantization techniques, the dream is to run the full pipeline on the Pi itself — no cloud required.

## Project Structure

```
dispo/
├── camera/             # On-device Python scripts (Pi Zero 2W)
│   └── main1.py        # Hardware control, capture, dither, print
├── dispoapi/           # Cloud API server
│   ├── main.py         # FastAPI app, endpoints, AI provider routing
│   ├── modal_app.py    # Modal deployment — Flux Kontext on H100
│   ├── prompts.py      # Mode-to-prompt mapping
│   ├── test_api.py     # Test script
│   └── pyproject.toml  # Dependencies
└── README.md
```

## Getting Started

See [`dispoapi/README.md`](dispoapi/README.md) for API setup, configuration, and usage instructions.
