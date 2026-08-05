from __future__ import annotations

import os
from html import escape as html_escape
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .model_store import (
    DEFAULT_COMPUTE_DEVICE,
    DEFAULT_MODEL_NAME,
    current_assets,
    id_window,
    list_local_models,
    load_model,
    model_status,
    nearest_neighbors,
    search_tokens,
    selected_token_rows,
    selected_vectors,
    token_record,
    unload_model,
)
from .projections import nearest_neighbor_edges, project_vectors, projection_catalog
from .schemas import (
    LocalModelRecord,
    LocalModelsResponse,
    ModelLoadRequest,
    NeighborhoodResponse,
    ProjectionCompareRequest,
    ProjectionCompareResponse,
    ProjectionComparisonRow,
    ProjectionMethodsResponse,
    ProjectionPoint,
    ProjectionRequest,
    ProjectionResponse,
    StatusResponse,
    TokenRecord,
    TokenSearchResponse,
    TokenWindowResponse,
)

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "templates" / "index.html"

app = FastAPI(
    title="Hugging Face Vocabulary Sphere",
    description="Regex token search and fidelity-aware 3-D spherical projections of Hugging Face vocabulary vectors.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


def _assets_or_http():
    try:
        return current_assets()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _raise_client_error(exc: Exception, status: int = 400):
    raise HTTPException(status_code=status, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = INDEX_PATH.read_text(encoding="utf-8")
    html = html.replace("{{ default_model }}", html_escape(DEFAULT_MODEL_NAME, quote=True))
    html = html.replace("{{ default_device }}", html_escape(DEFAULT_COMPUTE_DEVICE, quote=True))
    return HTMLResponse(html)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return StatusResponse(**model_status())


@app.post("/api/model/load", response_model=StatusResponse)
def model_load(request: ModelLoadRequest) -> StatusResponse:
    try:
        assets = load_model(
            request.model_name,
            revision=request.revision,
            matrix_source=request.matrix_source,
            compute_device=request.compute_device,
            allow_download=request.allow_download,
            force_reload=request.force_reload,
        )
    except ValueError as exc:
        _raise_client_error(exc)
    except Exception as exc:  # model/network/runtime dependent
        raise HTTPException(status_code=500, detail=f"Model loading failed: {exc}") from exc
    return StatusResponse(
        loaded=True,
        model_name=assets.model_name,
        revision=assets.revision,
        matrix_source=assets.matrix_source,
        tensor_name=assets.tensor_name,
        vocab_size=assets.vocab_size,
        hidden_dim=assets.hidden_dim,
        dtype=assets.dtype,
        compute_device=assets.compute_device,
        memory_bytes=assets.memory_bytes,
        load_strategy=assets.load_strategy,
    )


@app.delete("/api/model", response_model=StatusResponse)
def model_unload() -> StatusResponse:
    unload_model()
    return StatusResponse(**model_status())


@app.get("/api/models/local", response_model=LocalModelsResponse)
def local_models() -> LocalModelsResponse:
    return LocalModelsResponse(
        models=[
            LocalModelRecord(
                model_name=item.model_name,
                cache_path=item.cache_path,
                size_bytes=item.size_bytes,
                last_modified=item.last_modified,
            )
            for item in list_local_models()
        ]
    )


@app.get("/api/tokens/search", response_model=TokenSearchResponse)
def token_search(
    pattern: str = Query("", max_length=256),
    mode: str = Query("regex"),
    case_sensitive: bool = Query(False),
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
) -> TokenSearchResponse:
    assets = _assets_or_http()
    try:
        return TokenSearchResponse(
            **search_tokens(
                assets,
                pattern,
                mode=mode,
                case_sensitive=case_sensitive,
                limit=limit,
                offset=offset,
            )
        )
    except ValueError as exc:
        _raise_client_error(exc)


@app.get("/api/tokens/{token_id:int}", response_model=TokenRecord)
def token_by_id(token_id: int) -> TokenRecord:
    assets = _assets_or_http()
    try:
        return TokenRecord(**token_record(assets, token_id))
    except IndexError as exc:
        _raise_client_error(exc, 404)


@app.get("/api/tokens/neighbors", response_model=NeighborhoodResponse)
def token_neighbors(
    anchor_id: int = Query(..., ge=0),
    limit: int = Query(100, ge=1, le=5000),
    include_anchor: bool = Query(True),
) -> NeighborhoodResponse:
    assets = _assets_or_http()
    try:
        rows = nearest_neighbors(assets, anchor_id, limit=limit, include_anchor=include_anchor)
    except (IndexError, ValueError) as exc:
        _raise_client_error(exc)
    return NeighborhoodResponse(
        anchor_id=anchor_id,
        limit=limit,
        include_anchor=include_anchor,
        rows=[TokenRecord(**row) for row in rows],
    )


@app.get("/api/tokens/window", response_model=TokenWindowResponse)
def token_id_window(
    center_id: int = Query(..., ge=0),
    count: int = Query(100, ge=1, le=5000),
) -> TokenWindowResponse:
    assets = _assets_or_http()
    try:
        rows = id_window(assets, center_id, count)
    except (IndexError, ValueError) as exc:
        _raise_client_error(exc)
    return TokenWindowResponse(center_id=center_id, count=len(rows), rows=[TokenRecord(**row) for row in rows])


@app.get("/api/projection/methods", response_model=ProjectionMethodsResponse)
def projection_methods() -> ProjectionMethodsResponse:
    return ProjectionMethodsResponse(methods=projection_catalog())


def _prepare_projection_selection(token_ids: list[int], anchor_id: int | None) -> tuple[object, list[int], object, int | None]:
    assets = _assets_or_http()
    requested_ids = list(token_ids)
    if anchor_id is not None and anchor_id not in requested_ids:
        requested_ids.insert(0, anchor_id)
    ids, vectors = selected_vectors(assets, requested_ids)
    anchor_index = ids.index(anchor_id) if anchor_id is not None else None
    return assets, ids, vectors, anchor_index


@app.post("/api/projection", response_model=ProjectionResponse)
def projection(request: ProjectionRequest) -> ProjectionResponse:
    try:
        assets, token_ids, vectors, anchor_index = _prepare_projection_selection(request.token_ids, request.anchor_id)
        output = project_vectors(
            vectors,
            method=request.method,
            seed=request.seed,
            anchor_index=anchor_index,
            center_mode=request.center_mode,
            manifold_neighbors=request.manifold_neighbors,
            tsne_perplexity=request.tsne_perplexity,
            umap_min_dist=request.umap_min_dist,
            align_anchor=request.align_anchor,
        )
        token_rows = selected_token_rows(assets, token_ids, anchor_id=request.anchor_id)
        points = [
            ProjectionPoint(
                **row,
                index=index,
                x=float(output.coordinates[index, 0]),
                y=float(output.coordinates[index, 1]),
                z=float(output.coordinates[index, 2]),
                is_anchor=request.anchor_id is not None and row["token_id"] == request.anchor_id,
            )
            for index, row in enumerate(token_rows)
        ]
        edges = nearest_neighbor_edges(vectors, k=request.edge_k, max_edges=request.max_edges)
    except (ValueError, IndexError, RuntimeError) as exc:
        _raise_client_error(exc)

    return ProjectionResponse(
        model_name=assets.model_name,
        revision=assets.revision,
        matrix_source=assets.matrix_source,
        tensor_name=assets.tensor_name,
        requested_method=output.requested_method,
        actual_method=output.actual_method,
        anchor_id=request.anchor_id,
        points=points,
        edges=edges,
        metrics=output.metrics,
        details=output.details,
        warnings=output.warnings,
    )


@app.post("/api/projection/compare", response_model=ProjectionCompareResponse)
def projection_compare(request: ProjectionCompareRequest) -> ProjectionCompareResponse:
    try:
        assets, token_ids, vectors, anchor_index = _prepare_projection_selection(request.token_ids, request.anchor_id)
    except (ValueError, IndexError) as exc:
        _raise_client_error(exc)

    rows: list[ProjectionComparisonRow] = []
    seen: set[str] = set()
    for method in request.methods:
        key = method.strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        try:
            output = project_vectors(
                vectors,
                method=key,
                seed=request.seed,
                anchor_index=anchor_index,
                center_mode=request.center_mode,
                manifold_neighbors=request.manifold_neighbors,
                tsne_perplexity=request.tsne_perplexity,
                umap_min_dist=request.umap_min_dist,
                align_anchor=request.align_anchor,
            )
            rows.append(
                ProjectionComparisonRow(
                    requested_method=key,
                    actual_method=output.actual_method,
                    success=True,
                    metrics=output.metrics,
                    details=output.details,
                    warnings=output.warnings,
                )
            )
        except Exception as exc:
            rows.append(
                ProjectionComparisonRow(
                    requested_method=key,
                    success=False,
                    error=str(exc),
                )
            )
    return ProjectionCompareResponse(model_name=assets.model_name, token_count=len(token_ids), rows=rows)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").casefold() == "true",
    )
