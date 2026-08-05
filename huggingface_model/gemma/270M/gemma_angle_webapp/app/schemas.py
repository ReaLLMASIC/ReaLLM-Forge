from __future__ import annotations

from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    loaded: bool = False
    model_name: str
    requested_device: str
    effective_device: str
    vocab_size: int = Field(ge=0)
    hidden_dim: int = Field(ge=0)


class ModelLoadRequest(BaseModel):
    model_name: str = Field(..., min_length=1, description="Hugging Face model repo ID, e.g. Qwen/Qwen3.5-0.8B-Base")
    device: str | None = Field(default=None, description="Torch device string such as cpu, cuda:0, or auto")
    force_reload: bool = False
    allow_download: bool = Field(
        default=True,
        description="When true, missing files may be downloaded from the Hugging Face Hub. When false, only the local cache is used.",
    )


class LocalModelRecord(BaseModel):
    model_name: str
    cache_path: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    last_modified: float | None = None


class LocalModelsResponse(BaseModel):
    models: list[LocalModelRecord]


class TokenRecord(BaseModel):
    token_id: int
    raw: str
    display: str


class TokenSearchResponse(BaseModel):
    query: str
    results: list[TokenRecord]


class LinearTransformExampleRequest(BaseModel):
    source_token_id: int = Field(ge=0)
    target_token_id: int = Field(ge=0)


class LinearTransformNeighborsRequest(BaseModel):
    examples: list[LinearTransformExampleRequest] = Field(
        default_factory=list,
        description="One or more source→target token pairs used to define the transform.",
    )
    input_token_id: int = Field(ge=0, description="Token ID whose vector is transformed before nearest-neighbor search.")
    limit: int = Field(default=200, ge=1, le=5000)
    transform_type: str = Field(default="closest_identity")
    ridge_lambda: float = Field(default=1e-3, ge=0)
    transform_scale: float = Field(
        default=1.0,
        ge=-1000.0,
        le=1000.0,
        description="Scale applied to the learned transform effect: 0 keeps the input, 1 applies the fitted transform, values >1 extrapolate.",
    )


class AngleResponse(BaseModel):
    token_a_id: int
    token_a_raw: str
    token_a_display: str
    token_a_magnitude: float
    token_b_id: int
    token_b_raw: str
    token_b_display: str
    token_b_magnitude: float
    angle_deg: float


class CommonCloseTokenRow(BaseModel):
    rank: int = Field(ge=1)
    token_id: int
    token_raw: str
    token_display: str
    angle_to_token_a_deg: float
    angle_to_token_b_deg: float
    magnitude: float


class CommonCloseTokensResponse(BaseModel):
    token_a_id: int
    token_a_raw: str
    token_a_display: str
    token_a_magnitude: float
    token_b_id: int
    token_b_raw: str
    token_b_display: str
    token_b_magnitude: float
    threshold_deg: float = Field(ge=0)
    match_count: int = Field(ge=0)
    rows: list[CommonCloseTokenRow]


class TransformNeighborRow(BaseModel):
    rank: int = Field(ge=1)
    token_id: int
    token_raw: str
    token_display: str
    angle_deg: float
    angle_to_original_input_deg: float
    cosine_similarity: float
    magnitude: float




class TransformExampleRecord(BaseModel):
    example_index: int = Field(ge=1)
    source_token_id: int
    source_token_raw: str
    source_token_display: str
    source_token_magnitude: float
    target_token_id: int
    target_token_raw: str
    target_token_display: str
    target_token_magnitude: float
    source_to_target_angle_deg: float


class LinearTransformNeighborsResponse(BaseModel):
    model_name: str
    vocab_size: int = Field(ge=0)
    hidden_dim: int = Field(ge=0)
    pair_count: int = Field(default=1, ge=1)
    examples: list[TransformExampleRecord] = Field(default_factory=list)
    effective_source_rank: int = Field(default=1, ge=0)
    ridge_lambda: float = Field(default=1e-3, ge=0)
    source_token_id: int
    source_token_raw: str
    source_token_display: str
    source_token_magnitude: float
    target_token_id: int
    target_token_raw: str
    target_token_display: str
    target_token_magnitude: float
    input_token_id: int
    input_token_raw: str
    input_token_display: str
    input_token_magnitude: float
    transform_type: str
    transform_description: str
    source_to_target_angle_deg: float
    input_to_transformed_angle_deg: float
    transformed_vector_magnitude: float
    coefficient: float
    transform_scale: float = 1.0
    transform_parameter_label: str = "Transform parameter"
    limit: int = Field(ge=1)
    example_fit_rmse: float = Field(default=0.0, ge=0)
    example_fit_max_angle_deg: float = Field(default=0.0, ge=0)
    rows: list[TransformNeighborRow]


class NeighborhoodRow(BaseModel):
    rank: int
    token_id: int
    token_raw: str
    token_display: str
    angle_deg: float
    magnitude: float


class NeighborhoodResponse(BaseModel):
    anchor_id: int
    anchor_raw: str
    anchor_display: str
    anchor_magnitude: float
    limit: int = Field(ge=1)
    rows: list[NeighborhoodRow]



class PairwiseAngleBin(BaseModel):
    rank: int = Field(ge=1)
    bin_index: int = Field(ge=0)
    angle_min_deg: float = Field(ge=0)
    angle_max_deg: float = Field(ge=0)
    label: str
    count: int = Field(ge=0)
    token_count: int = Field(default=0, ge=0)


class PairwiseAngleDistributionResponse(BaseModel):
    model_name: str
    vocab_size: int = Field(ge=0)
    hidden_dim: int = Field(ge=0)
    total_pairs: int = Field(ge=0)
    bin_degrees: float = Field(gt=0)
    angle_min_deg: float = Field(ge=0)
    angle_max_deg: float = Field(ge=0)
    block_size: int = Field(ge=1)
    compute_device: str
    include_self: bool
    acute_angle: bool = True
    elapsed_seconds: float = Field(ge=0)
    bins: list[PairwiseAngleBin]


class PairwiseAngleBinTokensResponse(BaseModel):
    model_name: str
    vocab_size: int = Field(ge=0)
    bin_index: int = Field(ge=0)
    angle_min_deg: float = Field(ge=0)
    angle_max_deg: float = Field(ge=0)
    label: str
    token_count: int = Field(ge=0)
    tokens: list[TokenRecord]

class MinAngularDistanceRow(BaseModel):
    min_angle_rank: int = Field(ge=1)
    token_id: int
    token_raw: str
    token_display: str
    magnitude: float
    min_angle_deg: float
    other_token_id: int
    other_token_raw: str
    other_token_display: str
    other_magnitude: float


class MinAngularDistancesResponse(BaseModel):
    model_name: str
    vocab_size: int = Field(ge=0)
    hidden_dim: int = Field(ge=0)
    total_pairs: int = Field(ge=0)
    block_size: int = Field(ge=1)
    compute_device: str
    elapsed_seconds: float = Field(ge=0)
    rows: list[MinAngularDistanceRow]

class RecursiveGroupNode(BaseModel):
    token_id: int
    token_raw: str
    token_display: str
    magnitude: float
    connected_count: int = Field(ge=0)


class RecursiveGroupEdge(BaseModel):
    source_token_id: int
    target_token_id: int
    angle_deg: float


class RecursiveAngleGroupResponse(BaseModel):
    model_name: str
    vocab_size: int = Field(ge=0)
    hidden_dim: int = Field(ge=0)
    seed_token_id: int
    seed_token_raw: str
    seed_token_display: str
    max_angle_deg: float = Field(ge=0)
    group_size_limit: int = Field(ge=1)
    block_size: int = Field(ge=1)
    compute_device: str
    elapsed_seconds: float = Field(ge=0)
    scanned_count: int = Field(ge=0)
    truncated: bool
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    nodes: list[RecursiveGroupNode]
    edges: list[RecursiveGroupEdge]
    dictionary: dict[str, RecursiveGroupNode]

