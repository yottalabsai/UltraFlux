from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ultraflux.autoencoder_kl import AutoencoderKL
from ultraflux.pipeline_flux import FluxPipeline
from ultraflux.transformer_flux_visionyarn import FluxTransformer2DModel


# --------------------------
# Environment / config
# --------------------------
MODEL_ID = os.getenv("ULTRAFLUX_MODEL_ID", "Owen777/UltraFlux-v1")
TRANSFORMER_SUBFOLDER = os.getenv("ULTRAFLUX_TRANSFORMER_SUBFOLDER", "transformer")
VAE_SUBFOLDER = os.getenv("ULTRAFLUX_VAE_SUBFOLDER", "vae")
DEVICE = os.getenv("ULTRAFLUX_DEVICE", "cuda")
RESULTS_DIR = Path(os.getenv("ULTRAFLUX_RESULTS_DIR", "results"))
MAX_SEQ_LEN = int(os.getenv("ULTRAFLUX_MAX_SEQUENCE_LENGTH", "512"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_pipeline: Optional[FluxPipeline] = None
_pipeline_lock = threading.Lock()
_inference_lock = threading.Lock()



# ============================================================================
#  PIPELINE LOADER (Lazy, thread-safe)
# ============================================================================
def _load_pipeline() -> FluxPipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        # --- Load Models ---
        vae = AutoencoderKL.from_pretrained(
            MODEL_ID,
            subfolder=VAE_SUBFOLDER,
            torch_dtype=torch.bfloat16,
        )

        transformer = FluxTransformer2DModel.from_pretrained(
            MODEL_ID,
            subfolder=TRANSFORMER_SUBFOLDER,
            torch_dtype=torch.bfloat16,
        )


        # --- Load Pipeline ---
        pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            vae=vae,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
        )

        pipe.scheduler.config.use_dynamic_shifting = False
        pipe.scheduler.config.time_shift = 4

        pipe = pipe.to(DEVICE)
        pipe.set_progress_bar_config(disable=True)

        _pipeline = pipe
        return _pipeline


# ============================================================================
#  Request / Response Models
# ============================================================================
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1600)
    height: int = Field(4096, ge=256, le=4096)
    width: int = Field(4096, ge=256, le=4096)
    num_inference_steps: int = Field(50, ge=10, le=200)
    guidance_scale: float = Field(4.0, ge=0.0, le=20.0)
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    prompt: str
    seed: int
    image_path: str


# ============================================================================
#  FastAPI Service
# ============================================================================
app = FastAPI(
    title="UltraFlux Inference Service",
    version="1.0.0",
    description="HTTP service for UltraFlux image generation",
)


@app.on_event("startup")
def _startup():
    _load_pipeline()


@app.get("/healthz")
def healthz():
    return {
        "status": "ok" if _pipeline is not None else "initializing",
        "device": DEVICE,
    }


@app.post("/generate")
def generate_image(req: GenerateRequest):
    pipeline = _load_pipeline()

    # Seed generator
    generator = torch.Generator("cpu")
    seed = req.seed if req.seed is not None else torch.randint(0, 2**31 - 1, (1,)).item()
    generator = generator.manual_seed(seed)

    # SERIALIZE inference to avoid GPU contention
    with _inference_lock:
        try:
            out = pipeline(
                req.prompt,
                height=req.height,
                width=req.width,
                guidance_scale=req.guidance_scale,
                num_inference_steps=req.num_inference_steps,
                max_sequence_length=MAX_SEQ_LEN,
                generator=generator,
            )
        except torch.cuda.OutOfMemoryError:
            raise HTTPException(status_code=503, detail="CUDA out of memory")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    image = out.images[0]

    timestamp = int(time.time() * 1000)
    out_path = RESULTS_DIR / f"ultraflux_{timestamp}.jpeg"
    image.save(out_path)

    # ⬇️ Return the actual image file, not a JSON path
    return FileResponse(
        path=out_path,
        media_type="image/jpeg",
        filename=out_path.name
    )


@app.get("/")
def root():
    return {
        "message": "UltraFlux inference service is running.",
        "model_id": MODEL_ID,
        "device": DEVICE,
    }
