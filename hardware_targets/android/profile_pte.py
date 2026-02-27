"""Utility for profiling ExecuTorch `.pte` programs on Android devices via `adb`."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

METRICS_BEGIN = "EXECUTORCH_METRICS_BEGIN"
METRICS_END = "EXECUTORCH_METRICS_END"


@dataclass(slots=True)
class MetricsSummary:
    phase: str
    tokens: int
    latency_ms: float
    energy_mj: float

    @property
    def latency_per_token_ms(self) -> float:
        return self.latency_ms / max(self.tokens, 1)

    @property
    def energy_per_token_mj(self) -> float:
        return self.energy_mj / max(self.tokens, 1)


def _adb_cmd(args: list[str], serial: Optional[str] = None, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    base = ["adb"]
    if serial:
        base += ["-s", serial]
    result = subprocess.run(base + args, check=True, capture_output=True, text=True, **kwargs)
    return result


def _extract_metrics(stdout: str) -> Dict[str, MetricsSummary]:
    pattern = re.compile(rf"{METRICS_BEGIN}(.*?){METRICS_END}", re.DOTALL)
    match = pattern.search(stdout)
    if not match:
        return {}
    payload = match.group(1).strip()
    data = json.loads(payload)
    summaries: Dict[str, MetricsSummary] = {}
    for phase, values in data.items():
        summaries[phase] = MetricsSummary(
            phase=phase,
            tokens=int(values.get("tokens", 0)),
            latency_ms=float(values.get("latency_ms", 0.0)),
            energy_mj=float(values.get("energy_mj", 0.0)),
        )
    return summaries


def _format_summary(summary: MetricsSummary) -> str:
    return (
        f"{summary.phase}: tokens={summary.tokens} "
        f"latency={summary.latency_ms:.2f}ms (per token {summary.latency_per_token_ms:.2f}ms) "
        f"energy={summary.energy_mj:.3f}mJ (per token {summary.energy_per_token_mj:.3f}mJ)"
    )


def profile(args: argparse.Namespace) -> None:
    remote_dir = Path(args.remote_dir)
    remote_dir_str = str(remote_dir)
    remote_runner = remote_dir / Path(args.runner).name
    remote_pte = remote_dir / Path(args.pte).name

    print(f"[INFO] Pushing runner to {remote_runner}")
    _adb_cmd(["push", args.runner, str(remote_runner)], serial=args.serial)
    print(f"[INFO] Pushing PTE to {remote_pte}")
    _adb_cmd(["push", args.pte, str(remote_pte)], serial=args.serial)

    prompt = args.prompt or "Hello world!"
    runner_invocation = (
        f"cd {shlex.quote(remote_dir_str)} && "
        f"chmod +x {shlex.quote(remote_runner.name)} && "
        f"echo {shlex.quote(prompt)} | "
        f"{shlex.quote('./' + remote_runner.name)}"
    )

    print(f"[INFO] Launching runner via adb shell: {runner_invocation}")
    result = _adb_cmd(["shell", runner_invocation], serial=args.serial)
    stdout = result.stdout
    if stdout:
        print("[DEVICE OUTPUT]")
        print(stdout)

    summaries = _extract_metrics(stdout)
    if not summaries:
        print(
            "[WARN] No ExecuTorch metrics detected. Ensure the runner prints JSON between "
            f"{METRICS_BEGIN} and {METRICS_END}."
        )
        return

    print("[INFO] Parsed metrics:")
    for summary in summaries.values():
        print("  " + _format_summary(summary))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, help="Path to the compiled ExecuTorch runner binary.")
    parser.add_argument("--pte", required=True, help="Path to the exported ExecuTorch .pte program.")
    parser.add_argument(
        "--remote-dir",
        default="/data/local/tmp/nanogpt",
        help="Directory on the device where artifacts will be staged.",
    )
    parser.add_argument(
        "--prompt",
        help="Prompt text to feed into the runner. Defaults to 'Hello world!'.",
    )
    parser.add_argument(
        "--serial",
        help="Optional adb serial number when multiple devices are connected.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        profile(args)
    except FileNotFoundError as exc:
        print(f"[ERROR] Failed to invoke external tool: {exc}")
    except subprocess.CalledProcessError as exc:
        print("[ERROR] adb command failed:")
        print(exc.stderr)


if __name__ == "__main__":
    main()
