from __future__ import annotations

import math
import os
from html import escape as html_escape
from pathlib import Path

import numpy as np
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
    model_vector_function,
    nearest_neighbors,
    search_tokens,
    selected_token_rows,
    selected_vectors,
    token_record,
    tokenize_text,
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
    TextTokenizeRequest,
    TextTokenizeResponse,
    TokenRecord,
    TokenSearchResponse,
    TokenWindowResponse,
)
from .vector_math import VectorExpressionResult, alias_for_index, evaluate_vector_expressions, vector_dimension_metrics

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "templates" / "index.html"

app = FastAPI(
    title="Hugging Face Vocabulary Geometry Studio",
    description="Regex/ID search, text tokenization, vector arithmetic with SLERP, and fidelity-aware spherical or free 3-D projections of Hugging Face vocabulary vectors.",
    version="1.3.0",
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


@app.post("/api/tokens/tokenize", response_model=TextTokenizeResponse)
def tokenize_supplied_text(request: TextTokenizeRequest) -> TextTokenizeResponse:
    assets = _assets_or_http()
    try:
        return TextTokenizeResponse(
            **tokenize_text(
                assets,
                request.text,
                add_special_tokens=request.add_special_tokens,
                max_tokens=request.max_tokens,
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


def _prepare_projection_selection(
    token_ids: list[int],
    anchor_id: int | None,
    arithmetic_expressions: list[object] | None = None,
) -> tuple[object, list[int], np.ndarray, np.ndarray, int | None, list[VectorExpressionResult]]:
    assets = _assets_or_http()
    requested_ids = list(token_ids)
    if anchor_id is not None and anchor_id not in requested_ids:
        requested_ids.insert(0, anchor_id)
    ids, base_vectors = selected_vectors(assets, requested_ids)
    anchor_index = ids.index(anchor_id) if anchor_id is not None else None
    resultants = evaluate_vector_expressions(
        base_vectors,
        arithmetic_expressions or [],
        model_function=lambda name, args: model_vector_function(assets, name, args),
    )
    if resultants:
        vectors = np.vstack([base_vectors, *(item.vector[None, :] for item in resultants)])
    else:
        vectors = base_vectors
    return assets, ids, base_vectors, vectors, anchor_index, resultants


def _projection_rows(
    assets,
    token_ids: list[int],
    base_vectors: np.ndarray,
    anchor_index: int | None,
    resultants: list[VectorExpressionResult],
    *,
    anchor_id: int | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, row in enumerate(selected_token_rows(assets, token_ids, anchor_id=anchor_id)):
        rows.append(
            {
                **row,
                "kind": "token",
                "alias": alias_for_index(index),
                "label": row["display"],
                **vector_dimension_metrics(base_vectors[index]),
                "expression": None,
                "referenced_aliases": [],
            }
        )

    anchor_vector = base_vectors[anchor_index] if anchor_index is not None else None
    anchor_norm = float(np.linalg.norm(anchor_vector)) if anchor_vector is not None else 0.0
    for item in resultants:
        cosine: float | None = None
        angle: float | None = None
        if anchor_vector is not None and anchor_norm > 1e-12 and item.magnitude > 1e-12:
            cosine = float(np.clip(np.dot(item.vector, anchor_vector) / (item.magnitude * anchor_norm), -1.0, 1.0))
            angle = float(math.degrees(math.acos(cosine)))
        rows.append(
            {
                "token_id": None,
                "raw": item.expression,
                "display": item.label,
                "special": False,
                "present_in_tokenizer": False,
                "magnitude": item.magnitude,
                **vector_dimension_metrics(item.vector),
                "rank": None,
                "cosine_similarity": None,
                "angle_deg": None,
                "cosine_to_anchor": cosine,
                "angle_to_anchor_deg": angle,
                "kind": "resultant",
                "alias": item.alias,
                "label": item.label,
                "expression": item.expression,
                "referenced_aliases": list(item.referenced_aliases),
            }
        )
    return rows


@app.post("/api/projection", response_model=ProjectionResponse)
def projection(request: ProjectionRequest) -> ProjectionResponse:
    try:
        assets, token_ids, base_vectors, vectors, anchor_index, resultants = _prepare_projection_selection(
            request.token_ids,
            request.anchor_id,
            request.arithmetic_expressions,
        )
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
            geometry_mode=request.geometry_mode,
        )
        token_rows = _projection_rows(
            assets,
            token_ids,
            base_vectors,
            anchor_index,
            resultants,
            anchor_id=request.anchor_id,
        )
        points = [
            ProjectionPoint(
                **row,
                index=index,
                x=float(output.coordinates[index, 0]),
                y=float(output.coordinates[index, 1]),
                z=float(output.coordinates[index, 2]),
                is_anchor=(
                    row.get("kind") == "token"
                    and request.anchor_id is not None
                    and row.get("token_id") == request.anchor_id
                ),
            )
            for index, row in enumerate(token_rows)
        ]
        edges = nearest_neighbor_edges(vectors, k=request.edge_k, max_edges=request.max_edges)
        output.metrics.update(
            {
                "base_token_count": len(token_ids),
                "resultant_count": len(resultants),
                "vector_count": len(vectors),
            }
        )
        output.details.update(
            {
                "base_token_count": len(token_ids),
                "resultant_count": len(resultants),
                "vector_count": len(vectors),
            }
        )
    except (ValueError, IndexError, RuntimeError) as exc:
        _raise_client_error(exc)

    return ProjectionResponse(
        model_name=assets.model_name,
        revision=assets.revision,
        matrix_source=assets.matrix_source,
        tensor_name=assets.tensor_name,
        requested_method=output.requested_method,
        actual_method=output.actual_method,
        geometry_mode=request.geometry_mode,
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
        assets, token_ids, _base_vectors, vectors, anchor_index, resultants = _prepare_projection_selection(
            request.token_ids,
            request.anchor_id,
            request.arithmetic_expressions,
        )
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
                geometry_mode=request.geometry_mode,
            )
            output.metrics.update(
                {
                    "base_token_count": len(token_ids),
                    "resultant_count": len(resultants),
                    "vector_count": len(vectors),
                }
            )
            output.details.update(
                {
                    "base_token_count": len(token_ids),
                    "resultant_count": len(resultants),
                    "vector_count": len(vectors),
                }
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
    return ProjectionCompareResponse(model_name=assets.model_name, token_count=len(vectors), rows=rows)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").casefold() == "true",
    )
