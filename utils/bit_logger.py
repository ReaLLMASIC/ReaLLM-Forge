# utils/bit_logger.py
import os
import re
import csv
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch


_LAYER_RE = re.compile(r"\.h\.(\d+)\.")


def _parse_layer_idx(module_name: str) -> Optional[int]:
    m = _LAYER_RE.search(module_name)
    return int(m.group(1)) if m else None


def _parse_family(module_name: str) -> str:
    # transformer.h.{i}.attn.xxx / transformer.h.{i}.mlp.xxx
    if ".attn." in module_name:
        return "attn"
    if ".mlp." in module_name:
        return "mlp"
    return "other"


def _parse_role(module_name: str) -> str:
    # e.g. transformer.h.0.attn.c_attn_q -> c_attn_q
    return module_name.split(".")[-1]


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _init_csv(path: str, fieldnames: List[str], overwrite: bool) -> None:
    if overwrite or (not os.path.exists(path)):
        _mkdir(os.path.dirname(path))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()


def _append_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    _mkdir(os.path.dirname(path))
    need_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            w.writeheader()
        w.writerows(rows)


@dataclass
class BitModuleInfo:
    name: str
    module: torch.nn.Module
    layer: Optional[int]
    family: str
    role: str
    param_count: int


class BitLogger:

    MODULE_FIELDS = [
        "time",
        "run_id",
        "phase",
        "iter",
        "tokens_trained",
        "dataset",
        "module_name",
        "layer",
        "family",
        "role",
        "param_count",
        "bit_param",
        "b_cont",
        "b_int",
        "grad",
        "abs_grad",
    ]

    GRAD_FIELDS = [
        "time",
        "run_id",
        "phase",
        "iter",
        "tokens_trained",
        "dataset",
        "module_name",
        "layer",
        "family",
        "role",
        "bit_param",
        "b_int",
        "grad",
        "abs_grad",
    ]

    LAYER_FIELDS = [
        "time",
        "run_id",
        "phase",
        "iter",
        "tokens_trained",
        "dataset",
        "layer",
        "avg_b_int_weighted",
        "avg_b_cont_weighted",
        "avg_abs_grad_weighted",
        "param_count",
        "total_bits_int",
        "total_bits_cont",
    ]

    TYPE_FIELDS = [
        "time",
        "run_id",
        "phase",
        "iter",
        "tokens_trained",
        "dataset",
        "family",
        "avg_b_int_weighted",
        "avg_b_cont_weighted",
        "avg_abs_grad_weighted",
        "param_count",
        "total_bits_int",
        "total_bits_cont",
    ]

    EVENT_FIELDS = [
        "time",
        "run_id",
        "phase",
        "iter",
        "tokens_trained",
        "dataset",
        "module_name",
        "layer",
        "family",
        "role",
        "old_b_int",
        "new_b_int",
        "bit_param",
        "b_cont",
    ]

    def __init__(
        self,
        model: torch.nn.Module,
        out_dir: str,
        run_id: Optional[str] = None,
        dataset_name: str = "",
        log_dir: Optional[str] = None,
        log_every: int = 20,
        overwrite: bool = False,
        is_master: bool = True,
        enabled: bool = True,
    ):
        self.enabled = bool(enabled and is_master)
        self.dataset_name = dataset_name
        self.log_every = int(log_every)
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")

        base = log_dir if log_dir is not None else os.path.join(out_dir, "bit_logs")

        self.run_dir = os.path.join(base, self.run_id)

        _mkdir(self.run_dir)

        self.paths = {
            "modules": os.path.join(self.run_dir, "bit_modules.csv"),
            "grads": os.path.join(self.run_dir, "bit_grads.csv"),
            "layers": os.path.join(self.run_dir, "bit_layers.csv"),
            "types": os.path.join(self.run_dir, "bit_types.csv"),
            "events": os.path.join(self.run_dir, "bit_events.csv"),
        }

        # Delete old files if overwrite=True 
        _init_csv(self.paths["modules"], self.MODULE_FIELDS, overwrite=overwrite)
        _init_csv(self.paths["grads"], self.GRAD_FIELDS, overwrite=overwrite)
        _init_csv(self.paths["layers"], self.LAYER_FIELDS, overwrite=overwrite)
        _init_csv(self.paths["types"], self.TYPE_FIELDS, overwrite=overwrite)
        _init_csv(self.paths["events"], self.EVENT_FIELDS, overwrite=overwrite)

        self.modules: List[BitModuleInfo] = self._collect_bit_modules(model)

        # b_int leaping events
        self._prev_b_int: Dict[str, Optional[float]] = {m.name: None for m in self.modules}

        # In case iter/phase repeatly write in each run
        self._logged_keys: set[tuple[str, int]] = set()

    @staticmethod
    def _is_bit_module(m: torch.nn.Module) -> bool:
        return hasattr(m, "bit_param") and isinstance(getattr(m, "bit_param"), torch.nn.Parameter)

    def _collect_bit_modules(self, model: torch.nn.Module) -> List[BitModuleInfo]:
        out: List[BitModuleInfo] = []
        for name, m in model.named_modules():
            if not self._is_bit_module(m):
                continue

            layer = _parse_layer_idx(name)
            family = _parse_family(name)
            role = _parse_role(name)

            # Reserve for bias
            param_count = 0
            if hasattr(m, "weight") and m.weight is not None:
                param_count += int(m.weight.numel())
            if hasattr(m, "bias") and m.bias is not None:
                param_count += int(m.bias.numel())

            out.append(
                BitModuleInfo(
                    name=name,
                    module=m,
                    layer=layer,
                    family=family,
                    role=role,
                    param_count=param_count,
                )
            )
        return out

    def _read_bits_detached(self, m: torch.nn.Module) -> Tuple[float, float, float]:
        """
        Rerurning:
        - bit_param(Continuous, no clipping)
        - b_cont(Continuous after clip)
        - b_int(Discrete atfer round)
        """
        # Compatible to AdaptiveBitLinear/min_bits(max_bits)
        min_bits = float(getattr(m, "min_bits", 1.0))
        max_bits = float(getattr(m, "max_bits", 8.0))

        with torch.no_grad():
            bit_param = float(m.bit_param.detach().float().cpu().item())
            b_cont = float(torch.clamp(m.bit_param.detach().float(), min_bits, max_bits).cpu().item())
            b_int = float(torch.round(torch.clamp(m.bit_param.detach().float(), min_bits, max_bits)).cpu().item())
        return bit_param, b_cont, b_int

    def maybe_log(
        self,
        iter_num: int,
        tokens_trained: int,
        phase: str = "train",
        dataset_name: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        if self.log_every > 0 and (iter_num % self.log_every != 0):
            return

        key = (phase, int(iter_num))
        if key in self._logged_keys:
            return
        self._logged_keys.add(key)

        t = time.time()
        dataset = dataset_name or self.dataset_name

        module_rows: List[Dict[str, Any]] = []
        grad_rows: List[Dict[str, Any]] = []
        event_rows: List[Dict[str, Any]] = []

        layer_acc: Dict[int, Dict[str, float]] = {}
        type_acc: Dict[str, Dict[str, float]] = {}

        def _acc_init():
            return {
                "param_count": 0.0,
                "sum_bits_int": 0.0,
                "sum_bits_cont": 0.0,
                "sum_abs_grad_weighted": 0.0,
            }

        for info in self.modules:
            name = info.name
            m = info.module

            bit_param, b_cont, b_int = self._read_bits_detached(m)

            g = m.bit_param.grad
            grad = float(g.detach().float().cpu().item()) if g is not None else float("nan")
            abs_grad = abs(grad) if g is not None else float("nan")

            n = float(info.param_count)
            total_bits_int = b_int * n
            total_bits_cont = b_cont * n

            module_rows.append(
                {
                    "time": t,
                    "run_id": self.run_id,
                    "phase": phase,
                    "iter": int(iter_num),
                    "tokens_trained": int(tokens_trained),
                    "dataset": dataset,
                    "module_name": name,
                    "layer": -1 if info.layer is None else int(info.layer),
                    "family": info.family,
                    "role": info.role,
                    "param_count": int(info.param_count),
                    "bit_param": bit_param,
                    "b_cont": b_cont,
                    "b_int": b_int,
                    "grad": grad,
                    "abs_grad": abs_grad,
                }
            )

            grad_rows.append(
                {
                    "time": t,
                    "run_id": self.run_id,
                    "phase": phase,
                    "iter": int(iter_num),
                    "tokens_trained": int(tokens_trained),
                    "dataset": dataset,
                    "module_name": name,
                    "layer": -1 if info.layer is None else int(info.layer),
                    "family": info.family,
                    "role": info.role,
                    "bit_param": bit_param,
                    "b_int": b_int,
                    "grad": grad,
                    "abs_grad": abs_grad,
                }
            )

            # event when b_int change
            prev = self._prev_b_int.get(name, None)
            if prev is None:
                self._prev_b_int[name] = b_int
            elif float(prev) != float(b_int):
                event_rows.append(
                    {
                        "time": t,
                        "run_id": self.run_id,
                        "phase": phase,
                        "iter": int(iter_num),
                        "tokens_trained": int(tokens_trained),
                        "dataset": dataset,
                        "module_name": name,
                        "layer": -1 if info.layer is None else int(info.layer),
                        "family": info.family,
                        "role": info.role,
                        "old_b_int": float(prev),
                        "new_b_int": float(b_int),
                        "bit_param": bit_param,
                        "b_cont": b_cont,
                    }
                )
                self._prev_b_int[name] = b_int

            # layer aggregate
            if info.layer is not None:
                if info.layer not in layer_acc:
                    layer_acc[info.layer] = _acc_init()
                la = layer_acc[info.layer]
                la["param_count"] += n
                la["sum_bits_int"] += total_bits_int
                la["sum_bits_cont"] += total_bits_cont
                if g is not None:
                    la["sum_abs_grad_weighted"] += abs_grad * n

            # type/family
            fam = info.family
            if fam not in type_acc:
                type_acc[fam] = _acc_init()
            ta = type_acc[fam]
            ta["param_count"] += n
            ta["sum_bits_int"] += total_bits_int
            ta["sum_bits_cont"] += total_bits_cont
            if g is not None:
                ta["sum_abs_grad_weighted"] += abs_grad * n

        layer_rows: List[Dict[str, Any]] = []
        for layer, la in sorted(layer_acc.items(), key=lambda kv: kv[0]):
            pc = la["param_count"]
            avg_b_int = la["sum_bits_int"] / pc if pc > 0 else float("nan")
            avg_b_cont = la["sum_bits_cont"] / pc if pc > 0 else float("nan")
            avg_abs_grad = la["sum_abs_grad_weighted"] / pc if pc > 0 else float("nan")

            layer_rows.append(
                {
                    "time": t,
                    "run_id": self.run_id,
                    "phase": phase,
                    "iter": int(iter_num),
                    "tokens_trained": int(tokens_trained),
                    "dataset": dataset,
                    "layer": int(layer),
                    "avg_b_int_weighted": avg_b_int,
                    "avg_b_cont_weighted": avg_b_cont,
                    "avg_abs_grad_weighted": avg_abs_grad,
                    "param_count": int(pc),
                    "total_bits_int": la["sum_bits_int"],
                    "total_bits_cont": la["sum_bits_cont"],
                }
            )

        type_rows: List[Dict[str, Any]] = []
        for fam, ta in sorted(type_acc.items(), key=lambda kv: kv[0]):
            pc = ta["param_count"]
            avg_b_int = ta["sum_bits_int"] / pc if pc > 0 else float("nan")
            avg_b_cont = ta["sum_bits_cont"] / pc if pc > 0 else float("nan")
            avg_abs_grad = ta["sum_abs_grad_weighted"] / pc if pc > 0 else float("nan")

            type_rows.append(
                {
                    "time": t,
                    "run_id": self.run_id,
                    "phase": phase,
                    "iter": int(iter_num),
                    "tokens_trained": int(tokens_trained),
                    "dataset": dataset,
                    "family": fam,
                    "avg_b_int_weighted": avg_b_int,
                    "avg_b_cont_weighted": avg_b_cont,
                    "avg_abs_grad_weighted": avg_abs_grad,
                    "param_count": int(pc),
                    "total_bits_int": ta["sum_bits_int"],
                    "total_bits_cont": ta["sum_bits_cont"],
                }
            )

        # Write sheets
        _append_csv(self.paths["modules"], self.MODULE_FIELDS, module_rows)
        _append_csv(self.paths["grads"], self.GRAD_FIELDS, grad_rows)
        _append_csv(self.paths["layers"], self.LAYER_FIELDS, layer_rows)
        _append_csv(self.paths["types"], self.TYPE_FIELDS, type_rows)
        _append_csv(self.paths["events"], self.EVENT_FIELDS, event_rows)

    def get_run_dir(self) -> str:
        return self.run_dir

    def list_bit_params(self) -> List[torch.nn.Parameter]:
        return [info.module.bit_param for info in self.modules]
