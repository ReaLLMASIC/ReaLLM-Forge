from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class StatusResponse(BaseModel):
    loaded: bool
    model_name: str
    revision: str
    matrix_source: str
    tensor_name: str | None = None
    vocab_size: int = Field(ge=0)
    hidden_dim: int = Field(ge=0)
    dtype: str | None = None
    compute_device: str
    memory_bytes: int = Field(ge=0)
    load_strategy: str | None = None


class ModelLoadRequest(BaseModel):
    model_name: str = Field(min_length=1, max_length=500)
    revision: str = Field(default="main", min_length=1, max_length=200)
    matrix_source: Literal["auto", "input", "output"] = "auto"
    compute_device: str = Field(default="auto", min_length=1, max_length=50)
    allow_download: bool = True
    force_reload: bool = False


class LocalModelRecord(BaseModel):
    model_name: str
    cache_path: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    last_modified: float | None = None


class LocalModelsResponse(BaseModel):
    models: list[LocalModelRecord]


class TokenRecord(BaseModel):
    token_id: int = Field(ge=0)
    raw: str
    display: str
    special: bool = False
    present_in_tokenizer: bool = True
    magnitude: float | None = None
    rank: int | None = Field(default=None, ge=1)
    cosine_similarity: float | None = None
    angle_deg: float | None = None
    cosine_to_anchor: float | None = None
    angle_to_anchor_deg: float | None = None


class TokenSearchResponse(BaseModel):
    query: str
    mode: str
    case_sensitive: bool
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)
    total_matches: int = Field(ge=0)
    truncated: bool
    results: list[TokenRecord]


class NeighborhoodResponse(BaseModel):
    anchor_id: int = Field(ge=0)
    limit: int = Field(ge=1)
    include_anchor: bool
    rows: list[TokenRecord]


class TokenWindowResponse(BaseModel):
    center_id: int = Field(ge=0)
    count: int = Field(ge=1)
    rows: list[TokenRecord]


class ProjectionRequest(BaseModel):
    token_ids: list[int] = Field(min_length=2, max_length=5000)
    anchor_id: int | None = Field(default=None, ge=0)
    method: str = Field(default="auto", min_length=1, max_length=80)
    seed: int = 42
    center_mode: Literal["mean", "anchor", "none"] = "mean"
    manifold_neighbors: int = Field(default=15, ge=2, le=250)
    tsne_perplexity: float = Field(default=30.0, ge=2.0, le=250.0)
    umap_min_dist: float = Field(default=0.1, ge=0.0, le=0.99)
    align_anchor: bool = True
    edge_k: int = Field(default=2, ge=0, le=20)
    max_edges: int = Field(default=4000, ge=0, le=20000)

    @field_validator("token_ids")
    @classmethod
    def validate_distinct_ids(cls, values: list[int]) -> list[int]:
        if any(value < 0 for value in values):
            raise ValueError("token_ids must be non-negative.")
        if len(set(values)) < 2:
            raise ValueError("At least two distinct token IDs are required.")
        return values


class ProjectionPoint(TokenRecord):
    index: int = Field(ge=0)
    x: float
    y: float
    z: float
    is_anchor: bool = False


class ProjectionEdge(BaseModel):
    source_index: int = Field(ge=0)
    target_index: int = Field(ge=0)
    cosine_similarity: float
    angle_deg: float


class ProjectionResponse(BaseModel):
    model_name: str
    revision: str
    matrix_source: str
    tensor_name: str
    requested_method: str
    actual_method: str
    anchor_id: int | None = None
    points: list[ProjectionPoint]
    edges: list[ProjectionEdge]
    metrics: dict[str, Any]
    details: dict[str, Any]
    warnings: list[str]


class ProjectionCompareRequest(BaseModel):
    token_ids: list[int] = Field(min_length=2, max_length=2000)
    anchor_id: int | None = Field(default=None, ge=0)
    methods: list[str] = Field(
        default_factory=lambda: ["spherical_pca", "tangent_pca", "cosine_kernel", "angular_mds", "random"],
        min_length=1,
        max_length=9,
    )
    seed: int = 42
    center_mode: Literal["mean", "anchor", "none"] = "mean"
    manifold_neighbors: int = Field(default=15, ge=2, le=250)
    tsne_perplexity: float = Field(default=30.0, ge=2.0, le=250.0)
    umap_min_dist: float = Field(default=0.1, ge=0.0, le=0.99)
    align_anchor: bool = True


class ProjectionComparisonRow(BaseModel):
    requested_method: str
    actual_method: str | None = None
    success: bool
    metrics: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class ProjectionCompareResponse(BaseModel):
    model_name: str
    token_count: int = Field(ge=2)
    rows: list[ProjectionComparisonRow]


class ProjectionMethodRecord(BaseModel):
    key: str
    label: str
    family: str
    complexity: str
    best_for: str
    caveat: str
    max_points: int = Field(ge=2)
    stochastic: bool
    available: bool
    availability_note: str | None = None


class ProjectionMethodsResponse(BaseModel):
    methods: list[ProjectionMethodRecord]
