# utils/bit_allocation_logger.py
import os
import csv
import re
import time
from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple

import torch


_LAYER_RE = re.compile(r"(?:^|\.)(?:transformer\.)?h\.(\d+)(?:\.|$)")


def _safe_tb_name(name: str) -> str:
    return name.replace(".", "/")


def _parse_module_name(name: str) -> Tuple[int, str, str]:
    """
    return (layer_id, family, role)
    family: attn / mlp / other
    role:  attn_q / attn_k / attn_v / attn_proj / mlp_fc / mlp_proj / other
    """
    m = _LAYER_RE.search(name)
    layer_id = int(m.group(1)) if m else -1

    family = "other"
    role = "other"

    if ".attn." in name:
        family = "attn"
        if "c_attn_q" in name:
            role = "attn_q"
        elif "c_attn_k" in name:
            role = "attn_k"
        elif "c_attn_v" in name:
            role = "attn_v"
        elif "c_proj" in name:
            role = "attn_proj"
        else:
            role = "attn_other"
    elif ".mlp." in name:
        family = "mlp"
        if "c_fc" in name:
            role = "mlp_fc"
        elif "c_proj" in name:
            role = "mlp_proj"
        else:
            role = "mlp_other"

    return layer_id, family, role


class BitAllocationLogger:
    """
    Log bit choice of AdaptiveBitLinear
    - bit_param, b_cont, b_int in each module
    - attn vs mlp weighted average mean
    - bit bounding per tensor
    """

    def __init__(
        self,
        model: torch.nn.Module,
        out_dir: str,
        writer: Optional[Any] = None,
        log_per_module_tb: bool = True,
        log_hist_tb: bool = True,
    ):
        self.model = model
        self.writer = writer
        self.log_per_module_tb = log_per_module_tb
        self.log_hist_tb = log_hist_tb

        self.out_dir = out_dir
        self.save_dir = os.path.join(out_dir, "bit_alloc")
        os.makedirs(self.save_dir, exist_ok=True)

        self.modules_csv = os.path.join(self.save_dir, "bit_modules.csv")
        self.layers_csv = os.path.join(self.save_dir, "bit_layers.csv")
        self.types_csv = os.path.join(self.save_dir, "bit_types.csv")
        self.events_csv = os.path.join(self.save_dir, "bit_events.csv")
        self.grads_csv  = os.path.join(self.save_dir, "bit_grads.csv")


        self._init_csv_files()

        # Detecting bit dropping
        self._prev_b_int: Dict[str, float] = {}

    def _init_csv_files(self):
        if not os.path.exists(self.modules_csv):
            with open(self.modules_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "time",
                    "iter",
                    "tokens_trained",
                    "dataset",
                    "module_name",
                    "layer",
                    "family",
                    "role",
                    "in_features",
                    "out_features",
                    "weight_numel",
                    "bias_numel",
                    "bit_param",
                    "b_cont",
                    "b_int",
                    "total_bits_int",
                    "total_bits_cont",
                ])

        if not os.path.exists(self.layers_csv):
            with open(self.layers_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "time",
                    "iter",
                    "tokens_trained",
                    "dataset",
                    "layer",
                    "avg_b_int_weighted",
                    "avg_b_cont_weighted",
                    "param_count",
                    "total_bits_int",
                    "total_bits_cont",
                ])

        if not os.path.exists(self.types_csv):
            with open(self.types_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "time",
                    "iter",
                    "tokens_trained",
                    "dataset",
                    "family",
                    "avg_b_int_weighted",
                    "avg_b_cont_weighted",
                    "param_count",
                    "total_bits_int",
                    "total_bits_cont",
                ])

        if not os.path.exists(self.events_csv):
            with open(self.events_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "time",
                    "iter",
                    "tokens_trained",
                    "dataset",
                    "module_name",
                    "layer",
                    "family",
                    "role",
                    "old_b_int",
                    "new_b_int",
                ])
                
        if not os.path.exists(self.grads_csv):
            with open(self.grads_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "time",
                    "iter",
                    "tokens_trained",
                    "dataset",
                    "module_name",
                    "layer",
                    "family",
                    "role",
                    "bit_param",
                    "b_cont",
                    "b_int",
                    "grad",
                    "abs_grad",
                ])


    def _iter_adaptive_bit_modules(self):
        """
        According to following attributes
        - bit_param
        - min_bits/max_bits
        - weight
        """
        for name, m in self.model.named_modules():
            if (
                hasattr(m, "bit_param")
                and hasattr(m, "min_bits")
                and hasattr(m, "max_bits")
                and hasattr(m, "weight")
            ):
                yield name, m

    @torch.no_grad()
    def _get_bits(self, m) -> Tuple[float, float, float]:
        """
          b_cont = clamp(bit_param)
          b_int  = round(b_cont)
        """
        bit_param = float(m.bit_param.detach().float().item())
        b_cont_t = torch.clamp(m.bit_param.detach().float(), float(m.min_bits), float(m.max_bits))
        b_cont = float(b_cont_t.item())
        b_int = float(torch.round(b_cont_t).item())
        return bit_param, b_cont, b_int

    @torch.no_grad()
    def snapshot(self, iter_num: int, tokens_trained: Optional[float], dataset: str):
        """
        Log snapshot after some iter
        """
        ts = time.time()
        tokens_val = "" if tokens_trained is None else tokens_trained

        # Aggregative stats, per layer, attn/mlp
        layer_acc = defaultdict(lambda: {"p": 0, "b_int_bits": 0.0, "b_cont_bits": 0.0})
        family_acc = defaultdict(lambda: {"p": 0, "b_int_bits": 0.0, "b_cont_bits": 0.0})

        b_int_list = []
        b_cont_list = []

        module_rows: List[List[Any]] = []
        event_rows: List[List[Any]] = []

        for name, m in self._iter_adaptive_bit_modules():
            layer_id, family, role = _parse_module_name(name)
            bit_param, b_cont, b_int = self._get_bits(m)

            weight_numel = int(m.weight.numel())
            bias_numel = int(m.bias.numel()) if getattr(m, "bias", None) is not None else 0
            param_count = weight_numel + bias_numel

            total_bits_int = b_int * param_count
            total_bits_cont = b_cont * param_count

            module_rows.append([
                ts,
                iter_num,
                tokens_val,
                dataset,
                name,
                layer_id,
                family,
                role,
                int(getattr(m, "in_features", -1)),
                int(getattr(m, "out_features", -1)),
                weight_numel,
                bias_numel,
                bit_param,
                b_cont,
                b_int,
                total_bits_int,
                total_bits_cont,
            ])

            # Mean square distribution
            b_int_list.append(b_int)
            b_cont_list.append(b_cont)

            # layer clustering
            layer_acc[layer_id]["p"] += param_count
            layer_acc[layer_id]["b_int_bits"] += total_bits_int
            layer_acc[layer_id]["b_cont_bits"] += total_bits_cont

            family_acc[family]["p"] += param_count
            family_acc[family]["b_int_bits"] += total_bits_int
            family_acc[family]["b_cont_bits"] += total_bits_cont

            # b_int shifting event detection
            prev = self._prev_b_int.get(name, None)
            if prev is None:
                self._prev_b_int[name] = b_int
            else:
                if b_int != prev:
                    event_rows.append([
                        ts,
                        iter_num,
                        tokens_val,
                        dataset,
                        name,
                        layer_id,
                        family,
                        role,
                        prev,
                        b_int,
                    ])
                    self._prev_b_int[name] = b_int

            # TensorBoard per module
            if self.writer is not None and self.log_per_module_tb:
                tag = _safe_tb_name(name)
                self.writer.add_scalar(f"bit_alloc/modules/b_int/{tag}", b_int, iter_num)
                self.writer.add_scalar(f"bit_alloc/modules/b_cont/{tag}", b_cont, iter_num)
                self.writer.add_scalar(f"bit_alloc/modules/bit_param/{tag}", bit_param, iter_num)

        # Write modules.csv
        with open(self.modules_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerows(module_rows)

        # Write events.csv
        if event_rows:
            with open(self.events_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerows(event_rows)

        layer_rows = []
        for layer_id in sorted(layer_acc.keys()):
            p = layer_acc[layer_id]["p"]
            if p <= 0:
                continue
            total_int = layer_acc[layer_id]["b_int_bits"]
            total_cont = layer_acc[layer_id]["b_cont_bits"]
            avg_int = total_int / p
            avg_cont = total_cont / p
            layer_rows.append([ts, iter_num, tokens_val, dataset, layer_id, avg_int, avg_cont, p, total_int, total_cont])

        with open(self.layers_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerows(layer_rows)

        type_rows = []
        for fam in sorted(family_acc.keys()):
            p = family_acc[fam]["p"]
            if p <= 0:
                continue
            total_int = family_acc[fam]["b_int_bits"]
            total_cont = family_acc[fam]["b_cont_bits"]
            avg_int = total_int / p
            avg_cont = total_cont / p
            type_rows.append([ts, iter_num, tokens_val, dataset, fam, avg_int, avg_cont, p, total_int, total_cont])

        with open(self.types_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerows(type_rows)

        # TensorBoard：Aggregative Curve
        if self.writer is not None:
            # attn vs mlp average bitwidth
            for fam, row in [(r[4], r) for r in type_rows]:
                avg_int = float(row[5])
                avg_cont = float(row[6])
                self.writer.add_scalar(f"bit_alloc/type/{fam}_avg_b_int", avg_int, iter_num)
                self.writer.add_scalar(f"bit_alloc/type/{fam}_avg_b_cont", avg_cont, iter_num)

            # average bitwidth per layer
            for r in layer_rows:
                layer_id = int(r[4])
                avg_int = float(r[5])
                avg_cont = float(r[6])
                self.writer.add_scalar(f"bit_alloc/layer_{layer_id:02d}/avg_b_int", avg_int, iter_num)
                self.writer.add_scalar(f"bit_alloc/layer_{layer_id:02d}/avg_b_cont", avg_cont, iter_num)

            # Bit distribution histogram
            if self.log_hist_tb and len(b_int_list) > 0:
                self.writer.add_histogram(
                    "bit_alloc/hist/b_int",
                    torch.tensor(b_int_list, dtype=torch.float32),
                    iter_num
                )
                self.writer.add_histogram(
                    "bit_alloc/hist/b_cont",
                    torch.tensor(b_cont_list, dtype=torch.float32),
                    iter_num
                )

    @torch.no_grad()
    def snapshot_grads(self, iter_num: int, tokens_trained: Optional[float], dataset: str, topk: int = 10):
        """
        Logging gradient of bit_param layerwise (Require to call after unscaling gradient in train.py)
        Deliverables
          - CSV: out_dir/bit_alloc/bit_grads.csv
          - TB:  bit_grad/modules/abs/<module> + layer/type Aggregation + histogram
        """
        ts = time.time()
        tokens_val = "" if tokens_trained is None else tokens_trained

        rows = []
        abs_grads = []

        layer_acc = defaultdict(lambda: {"sum_abs": 0.0, "cnt": 0})
        fam_acc   = defaultdict(lambda: {"sum_abs": 0.0, "cnt": 0})

        per_module = []  # for ranking

        for name, m in self._iter_adaptive_bit_modules():
            g = m.bit_param.grad
            if g is None:
                continue

            grad = float(g.detach().float().item())
            abs_grad = abs(grad)

            bit_param, b_cont, b_int = self._get_bits(m)
            layer_id, family, role = _parse_module_name(name)

            rows.append([
                ts,
                iter_num,
                tokens_val,
                dataset,
                name,
                layer_id,
                family,
                role,
                bit_param,
                b_cont,
                b_int,
                grad,
                abs_grad,
            ])

            abs_grads.append(abs_grad)
            per_module.append((abs_grad, name, layer_id, family, role, b_int, b_cont, bit_param))

            layer_acc[layer_id]["sum_abs"] += abs_grad
            layer_acc[layer_id]["cnt"] += 1
            fam_acc[family]["sum_abs"] += abs_grad
            fam_acc[family]["cnt"] += 1

            # TensorBoard per module
            if self.writer is not None and self.log_per_module_tb:
                tag = _safe_tb_name(name)
                self.writer.add_scalar(f"bit_grad/modules/abs/{tag}", abs_grad, iter_num)
                self.writer.add_scalar(f"bit_grad/modules/raw/{tag}", grad, iter_num)

        # CSV
        if rows:
            with open(self.grads_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerows(rows)

        if self.writer is not None and abs_grads:
            # histogram
            self.writer.add_histogram(
                "bit_grad/hist/abs_grad",
                torch.tensor(abs_grads, dtype=torch.float32),
                iter_num,
            )

            # family avg
            for fam, v in fam_acc.items():
                if v["cnt"] > 0:
                    self.writer.add_scalar(
                        f"bit_grad/type/{fam}_mean_abs_grad",
                        v["sum_abs"] / v["cnt"],
                        iter_num,
                    )

            # per-layer avg
            for layer_id, v in layer_acc.items():
                if v["cnt"] > 0 and layer_id >= 0:
                    self.writer.add_scalar(
                        f"bit_grad/layer_{layer_id:02d}/mean_abs_grad",
                        v["sum_abs"] / v["cnt"],
                        iter_num,
                    )

        per_module.sort(reverse=True, key=lambda t: t[0])
        return per_module[:topk]
