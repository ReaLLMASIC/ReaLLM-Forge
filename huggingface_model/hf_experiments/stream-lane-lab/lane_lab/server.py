"""Local streaming viewer and a small, causal HTTP forecasting interface.

The synthetic signal endpoint is a demo data source. Forecasting accepts only
the supplied history; signal seeds and future truth are never model features.
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .data import sine_series
from .model import LaneModel


class ForecastRequest(BaseModel):
    model_id: str
    history: list[list[float]]
    horizon: int = Field(default=24, ge=1, le=128)
    mode: Literal["mean", "greedy", "sample"] = "mean"
    temperature: float = Field(default=1.0, gt=0, le=10)
    top_k: int = Field(default=0, ge=0, le=4096)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


def create_app(runs: str | Path, device: str = "cpu", threads: int = 1) -> FastAPI:
    """Build an app; checkpoint paths are resolved exclusively below ``runs``.

    One cached model and a lock bound memory use and serialize model loading and
    inference. The default host in :func:`serve` is loopback. This intentionally
    small development server is not a multi-tenant deployment service.
    """
    root = Path(runs).expanduser().resolve()
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    app = FastAPI(title="Stream Lane Lab", version="0.1.0")
    lock = threading.RLock()
    cache: dict = {"id": None, "model": None}

    def inventory() -> dict[str, dict]:
        found = {}
        if not root.exists():
            return found
        for config_file in sorted(root.rglob("config.json")):
            directory = config_file.parent.resolve()
            if not directory.is_relative_to(root):
                continue
            if not any((directory / name).is_file() for name in
                       ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")):
                continue
            try:
                config = json.loads(config_file.read_text())
                columns = config["columns"]
                if not isinstance(columns, list) or not columns:
                    continue
            except (OSError, ValueError, KeyError):
                continue
            key = str(directory.relative_to(root)) or "."
            found[key] = {
                "id": key, "name": key, "channels": len(columns),
                "columns": columns, "context_length": config.get("context_length", 64),
                "d_model": config.get("d_model"), "n_layers": config.get("n_layers"),
                "path": directory,
            }
        return found

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/api/models")
    def models():
        return {"models": [{k: v for k, v in item.items() if k != "path"}
                           for item in inventory().values()],
                "device": device, "threads": threads,
                "message": "" if root.exists() else "Checkpoint directory does not exist yet."}

    @app.get("/api/signal")
    def signal(seed: int = Query(default=2026, ge=0, le=2**31 - 1),
               frames: int = Query(default=4096, ge=1, le=16384),
               channels: int = Query(default=1, ge=1, le=3),
               start: int = Query(default=0, ge=0, le=10**9)):
        values = sine_series(seed=seed, frames=frames, channels=channels, start=start)
        return {"values": np.asarray(values).tolist(), "start": start,
                "channels": channels, "seed": seed}

    @app.post("/api/forecast")
    def forecast(request: ForecastRequest):
        options = inventory()
        spec = options.get(request.model_id)
        if spec is None:
            raise HTTPException(404, "Checkpoint not found; refresh the model list.")
        context_length = spec["context_length"]
        if not 1 <= len(request.history) <= 16384:
            raise HTTPException(422, "Provide between 1 and 16384 observed frames.")
        channels = spec["channels"]
        if any(len(row) != channels for row in request.history):
            raise HTTPException(422, f"Every history row must contain {channels} columns.")
        if any(not math.isfinite(v) for row in request.history for v in row):
            raise HTTPException(422, "History must contain finite numbers.")
        with np.errstate(over="ignore"):
            values = np.asarray(request.history[-context_length:], dtype=np.float32)
        if not np.isfinite(values).all():
            raise HTTPException(422, "History must fit finite float32 values.")
        clipped = []
        for c, column in enumerate(spec["columns"]):
            minimum, maximum = column.get("minimum", -1), column.get("maximum", 1)
            clipped.append(int(((values[:, c] < minimum) | (values[:, c] > maximum)).sum()))
        with lock:
            loading = cache["id"] != request.model_id
            if loading:
                try:
                    model = LaneModel.from_pretrained(spec["path"]).to(device).eval()
                except Exception as exc:
                    raise HTTPException(500, f"Could not load checkpoint: {exc}") from exc
                cache.update(id=request.model_id, model=model)
            model = cache["model"]
            history = torch.as_tensor(values, device=device).unsqueeze(0)
            generator = torch.Generator(device=device).manual_seed(request.seed)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    predictions = model.forecast(
                        history, request.horizon, mode=request.mode,
                        temperature=request.temperature, top_k=request.top_k,
                        generator=generator,
                    )
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize(device)
            except (ValueError, RuntimeError) as exc:
                raise HTTPException(400, str(exc)) from exc
            latency_ms = (time.perf_counter() - started) * 1000
            return {"predictions": predictions[0].float().cpu().tolist(),
                    "latency_ms": latency_ms, "model_id": request.model_id,
                    "history_frames_used": len(values), "horizon": request.horizon,
                    "mode": request.mode, "model_loaded": loading,
                    "out_of_range_counts": clipped,
                    "note": "Forecast uses supplied history only."}

    return app


def serve(runs: str | Path, host: str = "127.0.0.1", port: int = 8765,
          device: str = "cpu", threads: int = 1) -> None:
    import uvicorn
    uvicorn.run(create_app(runs, device=device, threads=threads), host=host, port=port)
