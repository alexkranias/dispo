"""
Modal app — Flux Kontext image-to-image endpoint.

Deploys a web endpoint that accepts {image_b64, prompt} and returns {image_b64}.
Matches the request/response format expected by call_modal() in main.py.

Setup:
  1. Accept the Flux Kontext license: https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
  2. Create a Modal secret named "huggingface-secret" with your HF_TOKEN
  3. modal deploy modal_app.py

Usage:
  POST https://<your-workspace>--dispoapi-modal-inference.modal.run
  Body: {"image_b64": "<base64>", "prompt": "..."}
  Response: {"image_b64": "<base64>"}
"""

import base64
from io import BytesIO
from pathlib import Path

import modal
from pydantic import BaseModel


class InferenceRequest(BaseModel):
    image_b64: str
    prompt: str


app = modal.App("dispoapi-modal")

diffusers_commit_sha = "00f95b9755718aabb65456e791b8408526ae6e76"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(
        "Pillow~=11.2.1",
        "accelerate~=1.8.1",
        f"git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha}",
        "huggingface-hub==0.36.0",
        "optimum-quanto==0.2.7",
        "safetensors==0.5.3",
        "sentencepiece==0.2.0",
        "torch==2.7.1",
        "transformers~=4.53.0",
        "fastapi[standard]",
        "pydantic",
        extra_options="--index-strategy unsafe-best-match",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
)

MODEL_NAME = "black-forest-labs/FLUX.1-Kontext-dev"
MODEL_REVISION = "f9fdd1a95e0dfd7653cb0966cda2486745122695"

CACHE_DIR = Path("/cache")
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

image = image.env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": str(CACHE_DIR)})

with image.imports():
    import torch
    from diffusers import FluxKontextPipeline
    from diffusers.utils import load_image
    from PIL import Image


@app.cls(
    image=image,
    gpu="H100",
    max_containers=1,
    volumes={CACHE_DIR: cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
)
class Model:
    @modal.enter()
    def load_model(self):
        self.pipe = FluxKontextPipeline.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        ).to("cuda")

    @modal.fastapi_endpoint(method="POST")
    def inference(self, body: InferenceRequest):
        image_bytes = base64.b64decode(body.image_b64)
        prompt = body.prompt

        init_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        init_image.thumbnail((1024, 1024))

        result = self.pipe(
            image=init_image,
            prompt=prompt,
            guidance_scale=3.5,
            num_inference_steps=20,
            output_type="pil",
        ).images[0]

        buf = BytesIO()
        result.save(buf, format="PNG")
        result_b64 = base64.b64encode(buf.getvalue()).decode()

        return {"image_b64": result_b64}
