"""Tokens <-> integer CSV values <-> real units for csv_mc_int models (pure Python; numpy/torch optional).

    proc = CsvMcIntProcessor.from_pretrained("out/ecg_acc_6L_384D_b256/hf")
    ids = proc.encode_rows([[123, -5, 980, 12], ...])      # (K, T) token ids, as nested lists
    rows = proc.decode_tokens(ids)                          # back to integer rows [[123, -5, 980, 12], ...]
    proc.to_real(123, 0) -> 0.123 (mV)                      # lane 0 = ecg_uv, scale 1000

token_id = value - int_min (values outside [int_min, int_max] are clipped first), exactly the
encoding of ReaLLM-Forge's data/csv_mc_int (prepare_csv_integer_multicontext.py).
A real value is value / scale when the lane has a scale, else value * step (unit as the lane says).
"""

import json
import os
from typing import Any, Dict, List, Optional, Sequence


class CsvMcIntProcessor:
    def __init__(self, lanes: List[Dict[str, Any]]):
        if not lanes:
            raise ValueError("no lanes")
        self.lanes = [dict(x) for x in lanes]
        for x in self.lanes:
            x["int_min"], x["int_max"] = int(x["int_min"]), int(x["int_max"])

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, config) -> "CsvMcIntProcessor":
        lanes = getattr(config, "lanes", None) if not isinstance(config, dict) else config.get("lanes")
        if not lanes:
            raise ValueError("the model config has no lanes (convert with --columns / --dataset)")
        return cls(lanes)

    @classmethod
    def from_pretrained(cls, path: str, **_unused) -> "CsvMcIntProcessor":
        with open(os.path.join(path, "config.json")) as f:
            return cls.from_config(json.load(f))

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "columns.json"), "w") as f:
            json.dump({"columns": self.lanes}, f, indent=2)

    # ------------------------------------------------------------------ properties
    @property
    def names(self) -> List[str]:
        return [str(x.get("column", f"lane_{i}")) for i, x in enumerate(self.lanes)]

    @property
    def units(self) -> List[str]:
        return [str(x.get("unit", "")) for x in self.lanes]

    def __len__(self) -> int:
        return len(self.lanes)

    # ------------------------------------------------------------------ conversions
    def encode_value(self, value: int, lane: int) -> int:
        x = self.lanes[lane]
        return min(max(int(value), x["int_min"]), x["int_max"]) - x["int_min"]

    def decode_token(self, token: int, lane: int) -> int:
        x = self.lanes[lane]
        return min(max(int(token), 0), x["int_max"] - x["int_min"]) + x["int_min"]

    def to_real(self, value: float, lane: int) -> float:
        x = self.lanes[lane]
        if x.get("scale"):
            return float(value) / float(x["scale"])
        return float(value) * float(x.get("step", 1))

    def from_real(self, real: float, lane: int) -> int:
        """Real units -> the nearest integer value (half away from zero is fine here: live inputs
        come pre-quantised by the same code that built the training CSV)."""
        x = self.lanes[lane]
        v = real * float(x["scale"]) if x.get("scale") else real / float(x.get("step", 1))
        return int(round(v))

    def encode_rows(self, rows: Sequence[Sequence[int]]) -> List[List[int]]:
        """Integer CSV rows (T rows of K values) -> (K, T) token ids."""
        k = len(self.lanes)
        return [[self.encode_value(r[i], i) for r in rows] for i in range(k)]

    def decode_tokens(self, ids) -> List[List[int]]:
        """(K, T) token ids (lists, numpy or a tensor) -> T integer rows of K values."""
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        k = len(self.lanes)
        if len(ids) != k:
            raise ValueError(f"expected {k} lanes, got {len(ids)}")
        return [[self.decode_token(ids[i][t], i) for i in range(k)] for t in range(len(ids[0]))]

    def tokens_to_real(self, ids) -> List[List[float]]:
        """(K, T) token ids -> (K, T) real values (e.g. mV, mg)."""
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return [[self.to_real(self.decode_token(t, i), i) for t in lane] for i, lane in enumerate(ids)]

    def to_json(self) -> Dict[str, Any]:
        return {"columns": self.lanes}

    @staticmethod
    def lanes_from_files(manifest: Optional[Dict[str, Any]] = None, columns: Optional[List[Dict[str, Any]]] = None
                         ) -> List[Dict[str, Any]]:
        """Lane specs in model order from a csv_mc_int manifest.json (ranges) and, optionally, the
        exporter's column list (units / scale / step / decode), matched by column name."""
        info = {c["column"]: c for c in (columns or [])}
        lanes = []
        if manifest:
            for c in manifest["columns"]:
                name = c.get("source_column") or c.get("context_name")
                lane = {"column": name, "int_min": int(c["int_min"]), "int_max": int(c["int_max"]),
                        "vocab_size": int(c["vocab_size"])}
                extra = info.get(name, {})
                for key in ("unit", "scale", "step", "decode", "signal"):
                    if key in extra:
                        lane[key] = extra[key]
                if (extra.get("int_min", lane["int_min"]), extra.get("int_max", lane["int_max"])) != \
                        (lane["int_min"], lane["int_max"]):
                    raise ValueError(f"{name}: range in the column list differs from the manifest")
                lanes.append(lane)
        else:
            for c in columns or []:
                lanes.append({k: c[k] for k in ("column", "int_min", "int_max", "vocab_size", "unit", "scale",
                                                "step", "decode", "signal") if k in c})
        return lanes
