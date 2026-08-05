from __future__ import annotations

import os
from html import escape as html_escape
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .model_service import (
    DEFAULT_DEVICE,
    DEFAULT_MODEL_NAME,
    angle_degrees,
    common_close_tokens,
    get_current_assets,
    get_model_status,
    iter_neighborhood_csv,
    linear_transform_neighbors,
    list_local_models,
    load_active_model,
    minimum_angular_distances,
    nearest_neighbors,
    pairwise_angle_bin_tokens,
    pairwise_angle_distribution,
    recursive_angle_group,
    search_tokens,
)
from .schemas import (
    AngleResponse,
    CommonCloseTokensResponse,
    LocalModelRecord,
    LinearTransformNeighborsRequest,
    LinearTransformNeighborsResponse,
    LocalModelsResponse,
    MinAngularDistancesResponse,
    ModelLoadRequest,
    NeighborhoodResponse,
    PairwiseAngleBinTokensResponse,
    PairwiseAngleDistributionResponse,
    RecursiveAngleGroupResponse,
    StatusResponse,
    TokenRecord,
    TokenSearchResponse,
)

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="LM-Head Angle Explorer")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
INDEX_TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"


def _status_from_assets(assets) -> StatusResponse:
    return StatusResponse(
        loaded=True,
        model_name=assets.model_name,
        requested_device=assets.requested_device,
        effective_device=assets.effective_device,
        vocab_size=assets.vocab_size,
        hidden_dim=assets.hidden_dim,
    )


def _load_assets_or_500():
    try:
        return get_current_assets()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - depends on runtime/model availability
        raise HTTPException(status_code=500, detail=f"Failed to load model assets: {exc}") from exc


def _token_record(token_id: int) -> TokenRecord:
    assets = _load_assets_or_500()
    try:
        info = assets.token(token_id)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return TokenRecord(token_id=info.token_id, raw=info.raw, display=info.display)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the browser UI as plain HTML.

    Some mixed FastAPI/Starlette installations route older templating call
    signatures incorrectly, which makes GET / fail before any front-end buttons
    can be wired.  The page only needs two escaped defaults, so a plain HTML
    response keeps the homepage independent of Starlette's template API version.
    """
    html = INDEX_TEMPLATE_PATH.read_text(encoding="utf-8")
    html = html.replace("{{ model_name }}", html_escape(DEFAULT_MODEL_NAME, quote=True))
    html = html.replace("{{ requested_device }}", html_escape(DEFAULT_DEVICE, quote=True))
    return HTMLResponse(html)


@app.get("/api/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return StatusResponse(**get_model_status())


@app.post("/api/model/load", response_model=StatusResponse)
def load_model(request: ModelLoadRequest) -> StatusResponse:
    model_name = request.model_name.strip()
    device = (request.device or DEFAULT_DEVICE).strip() or DEFAULT_DEVICE
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name cannot be blank.")

    try:
        assets = load_active_model(
            model_name,
            device,
            force_reload=request.force_reload,
            allow_download=request.allow_download,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - depends on runtime/model availability
        raise HTTPException(status_code=500, detail=f"Failed to load {model_name!r}: {exc}") from exc

    return _status_from_assets(assets)


@app.get("/api/models/available", response_model=LocalModelsResponse)
def available_models() -> LocalModelsResponse:
    """Return Hugging Face model repos already present in the local cache."""
    models = [
        LocalModelRecord(
            model_name=item.model_name,
            cache_path=item.cache_path,
            size_bytes=item.size_bytes,
            last_modified=item.last_modified,
        )
        for item in list_local_models()
    ]
    return LocalModelsResponse(models=models)


@app.get("/api/tokens/search", response_model=TokenSearchResponse)
def tokens_search(
    q: str = Query("", description="Case-insensitive token text or substring."),
) -> TokenSearchResponse:
    """Search token raw/display text and return every literal match.

    This literal route must be registered before any dynamic token-id route.
    Otherwise /api/tokens/search may be treated as token_id="search".
    """
    assets = _load_assets_or_500()
    matches = search_tokens(assets, q)
    results = [
        TokenRecord(token_id=info.token_id, raw=info.raw, display=info.display)
        for info in matches
    ]
    return TokenSearchResponse(query=q, results=results)


@app.get("/api/tokens/id/{token_id:int}", response_model=TokenRecord)
def token_by_id_explicit(token_id: int) -> TokenRecord:
    return _token_record(token_id)


@app.get("/api/tokens/{token_id:int}", response_model=TokenRecord)
def token_by_id(token_id: int) -> TokenRecord:
    """Backward-compatible token lookup route.

    The :int converter plus route ordering prevents static paths like
    /api/tokens/search from being parsed as token IDs.
    """
    return _token_record(token_id)


@app.get("/api/angle", response_model=AngleResponse)
def pairwise_angle(
    token_a: int = Query(..., ge=0),
    token_b: int = Query(..., ge=0),
) -> AngleResponse:
    assets = _load_assets_or_500()
    try:
        info_a = assets.token(token_a)
        info_b = assets.token(token_b)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    angle = angle_degrees(assets, token_a, token_b)
    return AngleResponse(
        token_a_id=token_a,
        token_a_raw=info_a.raw,
        token_a_display=info_a.display,
        token_a_magnitude=float(assets.magnitudes[token_a].item()),
        token_b_id=token_b,
        token_b_raw=info_b.raw,
        token_b_display=info_b.display,
        token_b_magnitude=float(assets.magnitudes[token_b].item()),
        angle_deg=angle,
    )


@app.get("/api/common-close-tokens", response_model=CommonCloseTokensResponse)
def tokens_close_to_both_pairwise_anchors(
    token_a: int = Query(..., ge=0),
    token_b: int = Query(..., ge=0),
    max_angle_deg: float = Query(35.0, ge=0, le=180),
) -> CommonCloseTokensResponse:
    """Return tokens within max_angle_deg of both selected pairwise tokens."""
    assets = _load_assets_or_500()
    try:
        info_a = assets.token(token_a)
        info_b = assets.token(token_b)
        rows = common_close_tokens(assets, token_a, token_b, max_angle_deg)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return CommonCloseTokensResponse(
        token_a_id=token_a,
        token_a_raw=info_a.raw,
        token_a_display=info_a.display,
        token_a_magnitude=float(assets.magnitudes[token_a].item()),
        token_b_id=token_b,
        token_b_raw=info_b.raw,
        token_b_display=info_b.display,
        token_b_magnitude=float(assets.magnitudes[token_b].item()),
        threshold_deg=float(max_angle_deg),
        match_count=len(rows),
        rows=rows,
    )


def _linear_transform_response(
    *,
    examples: list[tuple[int, int]],
    input_token_id: int,
    limit: int,
    transform_type: str,
    ridge_lambda: float = 1e-3,
    transform_scale: float = 1.0,
) -> LinearTransformNeighborsResponse:
    assets = _load_assets_or_500()
    try:
        data = linear_transform_neighbors(
            assets,
            input_token_id=input_token_id,
            examples=examples,
            limit=limit,
            transform_type=transform_type,
            ridge_lambda=ridge_lambda,
            transform_scale=transform_scale,
        )
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return LinearTransformNeighborsResponse(**data)


@app.get("/api/linear-transform-neighbors", response_model=LinearTransformNeighborsResponse)
def transformed_token_neighbors(
    source_token_id: int = Query(..., ge=0),
    target_token_id: int = Query(..., ge=0),
    input_token_id: int = Query(..., ge=0),
    limit: int = Query(200, ge=1, le=5000),
    transform_type: str = Query("closest_identity", description="closest_identity, orthogonal, least_squares, ridge, or offset"),
    ridge_lambda: float = Query(1e-3, ge=0),
    transform_scale: float = Query(1.0, ge=-1000.0, le=1000.0, description="Scale the fitted transform effect; 0 keeps input, 1 is normal, >1 extrapolates."),
) -> LinearTransformNeighborsResponse:
    """Backward-compatible single-example transform endpoint."""
    return _linear_transform_response(
        examples=[(source_token_id, target_token_id)],
        input_token_id=input_token_id,
        limit=limit,
        transform_type=transform_type,
        ridge_lambda=ridge_lambda,
        transform_scale=transform_scale,
    )


@app.post("/api/linear-transform-neighbors", response_model=LinearTransformNeighborsResponse)
def transformed_token_neighbors_multi(request: LinearTransformNeighborsRequest) -> LinearTransformNeighborsResponse:
    """Apply a transform learned from one or more token-pair examples."""
    examples = [(item.source_token_id, item.target_token_id) for item in request.examples]
    return _linear_transform_response(
        examples=examples,
        input_token_id=request.input_token_id,
        limit=request.limit,
        transform_type=request.transform_type,
        ridge_lambda=request.ridge_lambda,
        transform_scale=request.transform_scale,
    )


@app.get("/api/neighborhood", response_model=NeighborhoodResponse)
def token_neighborhood(
    anchor_id: int = Query(..., ge=0),
    limit: int = Query(500, ge=1, le=5000),
    include_self: bool = Query(True),
) -> NeighborhoodResponse:
    assets = _load_assets_or_500()
    try:
        anchor = assets.token(anchor_id)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    rows = nearest_neighbors(assets, anchor_id, limit, include_self=include_self)
    return NeighborhoodResponse(
        anchor_id=anchor_id,
        anchor_raw=anchor.raw,
        anchor_display=anchor.display,
        anchor_magnitude=float(assets.magnitudes[anchor_id].item()),
        limit=limit,
        rows=rows,
    )


@app.get("/api/neighborhood.csv")
def token_neighborhood_csv(anchor_id: int = Query(..., ge=0)) -> StreamingResponse:
    assets = _load_assets_or_500()
    try:
        assets.token(anchor_id)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    headers = {"Content-Disposition": f'attachment; filename="token_{anchor_id}_neighborhood.csv"'}
    return StreamingResponse(
        iter_neighborhood_csv(assets, anchor_id),
        media_type="text/csv; charset=utf-8",
        headers=headers,
    )


@app.get("/api/pairwise-angle-bins", response_model=PairwiseAngleDistributionResponse)
def pairwise_angle_bins(
    block_size: int = Query(2048, ge=1, le=16384),
    compute_device: str = Query("auto", description="auto, cpu, cuda:0, etc."),
    include_self: bool = Query(False, description="Include i == j self-pairs in the 0-degree bin."),
) -> PairwiseAngleDistributionResponse:
    assets = _load_assets_or_500()
    try:
        data = pairwise_angle_distribution(
            assets,
            block_size=block_size,
            compute_device=compute_device,
            include_self=include_self,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PairwiseAngleDistributionResponse(**data)


@app.get("/api/pairwise-angle-bins/{bin_index:int}/tokens", response_model=PairwiseAngleBinTokensResponse)
def pairwise_angle_bin_token_list(bin_index: int) -> PairwiseAngleBinTokensResponse:
    assets = _load_assets_or_500()
    try:
        data = pairwise_angle_bin_tokens(assets, bin_index)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PairwiseAngleBinTokensResponse(**data)


@app.get("/api/min-angular-distances", response_model=MinAngularDistancesResponse)
def min_angular_distances(
    block_size: int = Query(2048, ge=1, le=16384),
    compute_device: str = Query("auto", description="auto, cpu, cuda:0, etc."),
) -> MinAngularDistancesResponse:
    """Return the closest non-self token for every token in the active model."""
    assets = _load_assets_or_500()
    try:
        data = minimum_angular_distances(
            assets,
            block_size=block_size,
            compute_device=compute_device,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return MinAngularDistancesResponse(**data)


@app.get("/api/recursive-angle-group", response_model=RecursiveAngleGroupResponse)
def recursive_token_angle_group(
    seed_id: int = Query(..., ge=0),
    max_angle_deg: float = Query(35.0, ge=0, le=180),
    group_size_limit: int = Query(100, ge=1, le=1000),
    block_size: int = Query(2048, ge=1, le=16384),
    compute_device: str = Query("auto", description="auto, cpu, cuda:0, etc."),
) -> RecursiveAngleGroupResponse:
    """Return a recursively expanded within-angle token group and graph edges."""
    assets = _load_assets_or_500()
    try:
        data = recursive_angle_group(
            assets,
            seed_id,
            max_angle_deg=max_angle_deg,
            group_size_limit=group_size_limit,
            block_size=block_size,
            compute_device=compute_device,
        )
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RecursiveAngleGroupResponse(**data)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").casefold() == "true",
    )
