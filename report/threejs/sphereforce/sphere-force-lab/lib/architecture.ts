export type ArchitectureConfig = {
  modelDim: number;
  layers: number;
  mlpDim: number;
  heads: number;
  qkHeadDim: number;
  valueHeadDim: number;
  attentionDim: number;
  blockMode: "full" | "attention" | "mlp";
  positionEncoding: "absolute" | "rope";
  activation: "gelu" | "relu" | "relu2";
  maxContextLength: number;
};

export const DEFAULT_ARCHITECTURE: ArchitectureConfig = {
  modelDim: 3, layers: 1, mlpDim: 12, heads: 1, qkHeadDim: 3,
  valueHeadDim: 3, attentionDim: 3, blockMode: "full",
  positionEncoding: "absolute", activation: "gelu", maxContextLength: 10,
};

export function architectureParameterCount(a: ArchitectureConfig, embeddingRows: number) {
  const d = a.modelDim;
  const attention = 2 * d * a.heads * a.qkHeadDim + 2 * d * a.attentionDim + d;
  const mlp = 2 * d * a.mlpDim + a.mlpDim + 2 * d;
  return embeddingRows * d + (a.positionEncoding === "absolute" ? a.maxContextLength * d : 0) + d +
    a.layers * ((a.blockMode !== "mlp" ? attention : 0) + (a.blockMode !== "attention" ? mlp : 0));
}

export function validateArchitecture(input: Partial<ArchitectureConfig> = {}, batchSize = 10, embeddingRows = 200): ArchitectureConfig {
  const a = { ...DEFAULT_ARCHITECTURE, ...input };
  if (input.attentionDim === undefined) a.attentionDim = a.heads * a.valueHeadDim;
  const limits = { modelDim: [2, 128], layers: [1, 8], mlpDim: [1, 1024], heads: [1, 16], qkHeadDim: [1, 128], valueHeadDim: [1, 128], attentionDim: [1, 2048], maxContextLength: [1, 256] };
  const labels = { modelDim: "Model dimension", layers: "Decoder blocks", mlpDim: "MLP hidden dimension", heads: "Attention heads", qkHeadDim: "Q/K width per head", valueHeadDim: "V/O width per head", attentionDim: "Attention hidden dimension", maxContextLength: "Maximum context length" };
  for (const key of Object.keys(limits) as (keyof typeof limits)[]) {
    const [min, max] = limits[key];
    if (!Number.isInteger(a[key]) || a[key] < min || a[key] > max) throw new Error(`${labels[key]} must be an integer from ${min} to ${max}.`);
  }
  if (!["full", "attention", "mlp"].includes(a.blockMode) || !["absolute", "rope"].includes(a.positionEncoding) || !["gelu", "relu", "relu2"].includes(a.activation)) throw new Error("Invalid architecture option.");
  if (a.attentionDim !== a.heads * a.valueHeadDim) throw new Error("Attention hidden dimension must equal heads × V/O width per head. Choose a total divisible by the head count.");
  if (a.positionEncoding === "rope" && a.blockMode === "mlp") throw new Error("RoPE rotates attention queries and keys. Use absolute positions for MLP-only blocks.");
  if (a.positionEncoding === "rope" && a.qkHeadDim % 2 !== 0) throw new Error("RoPE requires an even Q/K width per head (for example 4). The model dimension may still be odd.");
  if (architectureParameterCount(a, embeddingRows) > 2_000_000) throw new Error("This configuration exceeds 2 million parameters. Reduce widths or decoder blocks for the browser simulation.");
  const t = a.maxContextLength, d = a.modelDim;
  const attentionCells = a.blockMode === "mlp" ? 0 : batchSize * a.heads * t * t * a.layers;
  const work = batchSize * t * a.layers * ((a.blockMode !== "mlp" ? 2 * d * a.heads * a.qkHeadDim + 2 * d * a.attentionDim + t * a.heads * (a.qkHeadDim + a.valueHeadDim) : 0) + (a.blockMode !== "attention" ? 2 * d * a.mlpDim : 0));
  if (attentionCells > 4_000_000 || work > 100_000_000) throw new Error("This configuration is too large for interactive CPU training. Reduce batch size, context length, widths, or decoder blocks.");
  return a;
}
