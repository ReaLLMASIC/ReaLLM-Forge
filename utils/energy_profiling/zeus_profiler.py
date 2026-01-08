"""Reusable helpers for optional Zeus energy profiling."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Iterable, Optional


_ZEUS_AVAILABLE = find_spec("zeus.monitor") is not None
if _ZEUS_AVAILABLE:
    from zeus.monitor import ZeusMonitor
else:
    ZeusMonitor = None

class ZeusWindow:
    def __init__(self, monitor: ZeusMonitor | None, name: str, enabled: bool) -> None:
        self._monitor = monitor
        self._name = name
        self._enabled = enabled
        self._measurement = None
        self._total_energy_joules = None

    def __enter__(self) -> "ZeusWindow":
        if self._enabled and self._monitor is not None:
            self._monitor.begin_window(self._name)
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if not self._enabled or self._monitor is None:
            return
        measurement = self._monitor.end_window(self._name)
        total_energy = _extract_total_energy_joules(measurement)
        self._measurement = measurement
        self._total_energy_joules = total_energy

    @property
    def total_energy_joules(self) -> float | None:
        return self._total_energy_joules

    @property
    def measurement(self) -> object | None:
        return self._measurement


class ZeusProfiler:
    def __init__(
        self,
        enabled: bool,
        gpu_indices: Optional[Iterable[int]] = None,
        cpu_indices: Optional[Iterable[int]] = None,
    ) -> None:
        self._enabled = enabled and _ZEUS_AVAILABLE
        self._gpu_indices = list(gpu_indices) if gpu_indices is not None else None
        self._cpu_indices = list(cpu_indices) if cpu_indices is not None else None
        self._monitor = None
        if self._enabled:
            self._monitor = ZeusMonitor(
                gpu_indices=self._gpu_indices,
                cpu_indices=self._cpu_indices,
            )
        elif enabled and not _ZEUS_AVAILABLE:
            print("Zeus profiling requested, but zeus.monitor is unavailable.")

    @classmethod
    def from_args(cls, args, device: str) -> "ZeusProfiler":
        if not getattr(args, "zeus_profile", False):
            return cls(enabled=False)

        use_gpu = bool(getattr(args, "zeus_profile_gpu", True)) and "cuda" in device
        use_cpu = bool(getattr(args, "zeus_profile_cpu", False))

        gpu_indices = None
        cpu_indices = None

        if use_gpu:
            gpu_indices = getattr(args, "zeus_gpu_indices", None)
            if gpu_indices is None:
                import torch

                gpu_indices = [torch.cuda.current_device()]

        if use_cpu:
            cpu_indices = getattr(args, "zeus_cpu_indices", None)

        return cls(enabled=use_gpu or use_cpu, gpu_indices=gpu_indices, cpu_indices=cpu_indices)

    def window(self, name: str) -> ZeusWindow:
        return ZeusWindow(self._monitor, name, self._enabled)

    @property
    def enabled(self) -> bool:
        return self._enabled


def _extract_total_energy_joules(measurement: object) -> float | None:
    if measurement is None:
        return None

    if hasattr(measurement, "total_energy"):
        total_energy = getattr(measurement, "total_energy")
        if isinstance(total_energy, dict):
            return float(sum(total_energy.values()))
        return float(total_energy)

    if hasattr(measurement, "energy"):
        energy = getattr(measurement, "energy")
        if isinstance(energy, dict):
            return float(sum(energy.values()))
        return float(energy)

    return None
