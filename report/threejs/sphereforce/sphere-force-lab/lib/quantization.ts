export type QuantizationFormat = "fp32" | "ternary" | "int3" | "int4" | "int5" | "sym-int3" | "sym-int4" | "sym-int5";
export type BlendSchedule = "immediate" | "linear" | "cosine";
export type QatConfig = { format: QuantizationFormat; schedule: BlendSchedule; start: number; duration: number };

export const QUANTIZATION_FORMATS: { value: QuantizationFormat; label: string; min: number; max: number }[] = [
  { value: "fp32", label: "Full precision", min: 0, max: 0 },
  { value: "ternary", label: "Ternary · −1, 0, 1", min: -1, max: 1 },
  { value: "int3", label: "Int3 · −4…3", min: -4, max: 3 },
  { value: "int4", label: "Int4 · −8…7", min: -8, max: 7 },
  { value: "int5", label: "Int5 · −16…15", min: -16, max: 15 },
  { value: "sym-int3", label: "Symmetric int3 · −3…3", min: -3, max: 3 },
  { value: "sym-int4", label: "Symmetric int4 · −7…7", min: -7, max: 7 },
  { value: "sym-int5", label: "Symmetric int5 · −15…15", min: -15, max: 15 },
];

export const DEFAULT_QAT: QatConfig = { format: "fp32", schedule: "linear", start: 0, duration: 200 };

export function validateQat(config: QatConfig): QatConfig {
  if (!QUANTIZATION_FORMATS.some((format) => format.value === config.format)) throw new Error("Choose a supported quantization format.");
  if (!["immediate", "linear", "cosine"].includes(config.schedule)) throw new Error("Choose a supported blend schedule.");
  if (!Number.isInteger(config.start) || config.start < 0 || config.start > 50_000) throw new Error("QAT start must be an iteration from 0 to 50,000.");
  if (!Number.isInteger(config.duration) || config.duration < 1 || config.duration > 50_000) throw new Error("QAT duration must be from 1 to 50,000 iterations.");
  if (config.format !== "fp32" && config.schedule !== "immediate" && config.start + config.duration > 50_000) throw new Error("This blend would finish beyond 50,000 iterations. Choose an earlier start or shorter duration.");
  return { ...config };
}

export function blendAt(config: QatConfig, step: number) {
  if (config.format === "fp32" || step < config.start) return 0;
  if (config.schedule === "immediate") return 1;
  const progress = Math.max(0, Math.min(1, (step - config.start) / config.duration));
  return config.schedule === "cosine" ? (1 - Math.cos(Math.PI * progress)) / 2 : progress;
}

// Zero point 0; detached, per-tensor scale, recomputed from the current master weights.
export function quantizationScale(values: ArrayLike<number>, format: QuantizationFormat) {
  const range = QUANTIZATION_FORMATS.find((item) => item.value === format)!;
  if (format === "fp32") return 1;
  let scale = 0;
  for (let i = 0; i < values.length; i += 1) {
    scale = Math.max(scale, values[i] >= 0 ? values[i] / range.max : values[i] / range.min);
  }
  return scale > 1e-12 ? scale : 1;
}

export function quantizeValue(value: number, format: QuantizationFormat, scale: number) {
  if (format === "fp32") return value;
  const range = QUANTIZATION_FORMATS.find((item) => item.value === format)!;
  // Nearest integer, ties away from zero, including negative half steps.
  const code = Math.sign(value) * Math.floor(Math.abs(value / scale) + 0.5);
  return Math.max(range.min, Math.min(range.max, code)) * scale;
}
