#!/usr/bin/env python3
"""
Convert audio into self-describing bounded mel-state CSV and PNG containers.

The CPU path requires only NumPy and FFmpeg.  If PyTorch with CUDA is installed,
--device auto can use a GPU for the batched FFT and mel projection.  Audio is
decoded to a temporary memory-mapped float32 file, so long inputs do not need
to fit in RAM.

Each CSV embeds compact reconstruction metadata in a comment after its header.  Each
QMel-PNG stores the exact same integer states in an 8-bit indexed or 16-bit
grayscale raster and embeds the metadata both as iTXt and as a checksummed
pixel header.  Either file can
therefore be passed to token_mel_to_audio.py without a JSON sidecar.

Examples
--------
    python audio_to_token_mel.py speech.m4a --preset medium
    python audio_to_token_mel.py interview.mp4 --preset all --device auto
    python audio_to_token_mel.py speech.wav --preset small --output-format csv
    python audio_to_token_mel.py speech.flac --preset medium \
        --columns-per-timestep 32 --states-per-column 32 --top-db 56
    python audio_to_token_mel.py speech.wav --preset medium --agc
    python audio_to_token_mel.py speech.wav --preset medium \
        --timestep-ms 8 --columns-per-timestep 32 --states-per-column 48

Every emitted mel state is an integer in [0, levels - 1].  Hard saturation
prevents out-of-range token IDs.  It does not guarantee that every valid state,
or every cross-band pattern, was sufficiently represented in training.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


VERSION = "2.4.0"
EPS_POWER = 1.0e-20
CSV_METADATA_PREFIX = "#TOKEN_MEL_CSV_V2 "
PNG_METADATA_KEY = "qmel.v2"
PNG_PIXEL_MAGIC = b"QMELPNG2"
MAX_HISTOGRAM_CELLS = 16_777_216


class StoreConsistentValue(argparse.Action):
    """Store an aliased option, rejecting contradictory repeated values."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        previous = getattr(namespace, self.dest, None)
        if previous is not None and previous != values:
            parser.error(
                f"{option_string}={values} conflicts with an earlier value "
                f"of {previous} for the same option."
            )
        setattr(namespace, self.dest, values)


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    sample_rate: int
    n_fft: int
    win_ms: float
    hop_ms: float
    n_mels: int
    fmin: float
    fmax: float
    top_db: float
    levels: int

    @property
    def win_length(self) -> int:
        return int(round(self.sample_rate * self.win_ms / 1000.0))

    @property
    def hop_length(self) -> int:
        return int(round(self.sample_rate * self.hop_ms / 1000.0))

    @property
    def db_min(self) -> float:
        return -self.top_db

    @property
    def db_max(self) -> float:
        return 0.0

    @property
    def quantization_step_db(self) -> float:
        return self.top_db / (self.levels - 1)

    @property
    def frames_per_second(self) -> float:
        return self.sample_rate / self.hop_length


@dataclass(frozen=True)
class AgcConfig:
    """Broadband waveform automatic-gain-control parameters."""

    profile: str
    target_dbfs: float
    attack_ms: float
    release_ms: float
    max_gain_db: float
    max_attenuation_db: float
    gate_dbfs: float
    peak_dbfs: float


PRESETS: dict[str, Preset] = {
    "small": Preset(
        name="small",
        description=(
            "Lossy speech-content representation: 0-6 kHz, coarse spectral "
            "and amplitude resolution, and a 20 ms hop."
        ),
        sample_rate=12_000,
        n_fft=512,
        win_ms=40.0,
        hop_ms=20.0,
        n_mels=20,
        fmin=50.0,
        fmax=6_000.0,
        top_db=48.0,
        levels=16,
    ),
    "medium": Preset(
        name="medium",
        description=(
            "General speech/ASR baseline: 0-8 kHz, 40 mel bands, a 10 ms "
            "hop, and approximately 1 dB quantization."
        ),
        sample_rate=16_000,
        n_fft=512,
        win_ms=25.0,
        hop_ms=10.0,
        n_mels=40,
        fmin=50.0,
        fmax=8_000.0,
        top_db=64.0,
        levels=64,
    ),
    "high": Preset(
        name="high",
        description=(
            "High-quality speech feature representation: 0-12 kHz, 80 mel "
            "bands, a 10 ms hop, and 8-bit amplitude states."
        ),
        sample_rate=24_000,
        n_fft=1024,
        win_ms=25.0,
        hop_ms=10.0,
        n_mels=80,
        fmin=20.0,
        fmax=12_000.0,
        top_db=80.0,
        levels=256,
    ),
    "ultra": Preset(
        name="ultra",
        description=(
            "Ultra-quality wideband representation: 0-16 kHz, 160 mel bands, "
            "an 8 ms hop, a 4096-point FFT, and 10-bit amplitude states."
        ),
        sample_rate=32_000,
        n_fft=4096,
        win_ms=32.0,
        hop_ms=8.0,
        n_mels=160,
        fmin=20.0,
        fmax=16_000.0,
        top_db=96.0,
        levels=1024,
    ),
    "max": Preset(
        name="max",
        description=(
            "Maximum built-in fidelity: full 24 kHz audio bandwidth, 320 mel "
            "bands, a 5 ms hop, an 8192-point FFT, and 12-bit states."
        ),
        sample_rate=48_000,
        n_fft=8192,
        win_ms=32.0,
        hop_ms=5.0,
        n_mels=320,
        fmin=20.0,
        fmax=24_000.0,
        top_db=112.0,
        levels=4096,
    ),
}

AGC_PROFILES: dict[str, AgcConfig] = {
    "small": AgcConfig(
        profile="small",
        target_dbfs=-18.0,
        attack_ms=20.0,
        release_ms=400.0,
        max_gain_db=24.0,
        max_attenuation_db=18.0,
        gate_dbfs=-55.0,
        peak_dbfs=-1.0,
    ),
    "medium": AgcConfig(
        profile="medium",
        target_dbfs=-20.0,
        attack_ms=10.0,
        release_ms=500.0,
        max_gain_db=24.0,
        max_attenuation_db=24.0,
        gate_dbfs=-60.0,
        peak_dbfs=-1.0,
    ),
    "high": AgcConfig(
        profile="high",
        target_dbfs=-22.0,
        attack_ms=10.0,
        release_ms=650.0,
        max_gain_db=18.0,
        max_attenuation_db=24.0,
        gate_dbfs=-65.0,
        peak_dbfs=-1.0,
    ),
    "ultra": AgcConfig(
        profile="ultra",
        target_dbfs=-23.0,
        attack_ms=8.0,
        release_ms=800.0,
        max_gain_db=15.0,
        max_attenuation_db=24.0,
        gate_dbfs=-70.0,
        peak_dbfs=-1.0,
    ),
    "max": AgcConfig(
        profile="max",
        target_dbfs=-24.0,
        attack_ms=5.0,
        release_ms=1000.0,
        max_gain_db=12.0,
        max_attenuation_db=24.0,
        gate_dbfs=-75.0,
        peak_dbfs=-1.0,
    ),
}


def periodic_hann(length: int) -> np.ndarray:
    """Return a periodic Hann window, matching common STFT frontends."""
    if length < 2:
        return np.ones((length,), dtype=np.float32)
    return np.hanning(length + 1)[:-1].astype(np.float32)


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    """HTK mel conversion."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    """Inverse HTK mel conversion."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def make_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build unit-sum triangular mel filters.

    Unit-sum filters make each band a weighted mean of FFT-bin power rather than
    giving wider high-frequency filters automatically larger magnitudes.  This
    is useful for a fixed, portable quantization frontend.
    """
    nyquist = sample_rate / 2.0
    if not (0.0 <= fmin < fmax <= nyquist + 1.0e-9):
        raise ValueError(
            f"Expected 0 <= fmin < fmax <= Nyquist ({nyquist:g} Hz); "
            f"received fmin={fmin:g}, fmax={fmax:g}."
        )
    if n_mels < 1:
        raise ValueError("n_mels must be at least 1.")

    fft_hz = np.linspace(0.0, nyquist, n_fft // 2 + 1, dtype=np.float64)
    mel_edges = np.linspace(
        float(hz_to_mel(fmin)),
        float(hz_to_mel(fmax)),
        n_mels + 2,
        dtype=np.float64,
    )
    hz_edges = mel_to_hz(mel_edges)
    filters = np.zeros((n_mels, fft_hz.size), dtype=np.float64)

    for band in range(n_mels):
        lower, center, upper = hz_edges[band : band + 3]
        left = (fft_hz - lower) / max(center - lower, np.finfo(float).eps)
        right = (upper - fft_hz) / max(upper - center, np.finfo(float).eps)
        triangle = np.maximum(0.0, np.minimum(left, right))
        weight_sum = float(triangle.sum())
        if weight_sum <= 0.0:
            raise ValueError(
                f"Mel band {band} is empty. Increase n_fft, reduce n_mels, "
                "or narrow the frequency range."
            )
        filters[band] = triangle / weight_sum

    centers_hz = hz_edges[1:-1]
    return (
        filters.astype(np.float32),
        centers_hz.astype(np.float64),
        hz_edges.astype(np.float64),
    )


def validate_preset(preset: Preset) -> None:
    for name in ("win_ms", "hop_ms", "fmin", "fmax", "top_db"):
        if not math.isfinite(float(getattr(preset, name))):
            raise ValueError(f"{name} must be finite.")
    if preset.sample_rate < 1:
        raise ValueError("sample_rate must be positive.")
    if preset.win_length < 2 or preset.hop_length < 1:
        raise ValueError("The window and hop must contain at least 2 and 1 samples.")
    if preset.hop_length > preset.win_length // 2:
        minimum_window_samples = 2 * preset.hop_length
        minimum_window_ms = (
            1000.0 * minimum_window_samples / preset.sample_rate
        )
        raise ValueError(
            f"--timestep-ms {preset.hop_ms:g} gives a {preset.hop_length}-sample "
            f"hop at {preset.sample_rate} samples/s, while --win-ms "
            f"{preset.win_ms:g} gives a {preset.win_length}-sample window. "
            "For complete centered-frame coverage the window must be at least "
            f"twice the hop: use --win-ms {minimum_window_ms:g} or larger."
        )
    if preset.n_fft < preset.win_length:
        suggested_n_fft = 1 << (preset.win_length - 1).bit_length()
        raise ValueError(
            f"n_fft ({preset.n_fft}) must be >= win_length "
            f"({preset.win_length} samples at {preset.sample_rate} samples/s). "
            f"Use --n-fft {suggested_n_fft} or larger."
        )
    if preset.n_mels < 1 or preset.n_mels > 4096:
        raise ValueError("columns per timestep must be in [1, 4096].")
    if preset.levels < 2 or preset.levels > 65_536:
        raise ValueError("states per column must be in [2, 65536].")
    histogram_cells = preset.n_mels * preset.levels
    if histogram_cells > MAX_HISTOGRAM_CELLS:
        raise ValueError(
            f"columns Ã— states ({histogram_cells:,}) exceeds the "
            f"{MAX_HISTOGRAM_CELLS:,}-cell diagnostics-memory limit. "
            "Reduce --columns-per-timestep or --states-per-column."
        )
    if not math.isfinite(preset.top_db) or preset.top_db <= 0.0:
        raise ValueError("top_db must be positive.")
    make_mel_filterbank(
        preset.sample_rate,
        preset.n_fft,
        preset.n_mels,
        preset.fmin,
        preset.fmax,
    )


def validate_agc_config(config: AgcConfig) -> None:
    """Reject configurations that are unstable or have ambiguous semantics."""
    numeric_fields = (
        "target_dbfs",
        "attack_ms",
        "release_ms",
        "max_gain_db",
        "max_attenuation_db",
        "gate_dbfs",
        "peak_dbfs",
    )
    for name in numeric_fields:
        if not math.isfinite(float(getattr(config, name))):
            raise ValueError(f"AGC {name} must be finite.")
    if not -120.0 <= config.gate_dbfs < config.target_dbfs:
        raise ValueError(
            "AGC gate_dbfs must be in [-120, target_dbfs)."
        )
    if not config.target_dbfs < config.peak_dbfs <= 0.0:
        raise ValueError(
            "AGC levels must satisfy target_dbfs < peak_dbfs <= 0."
        )
    if not 0.1 <= config.attack_ms <= 10_000.0:
        raise ValueError("AGC attack_ms must be in [0.1, 10000].")
    if not config.attack_ms <= config.release_ms <= 60_000.0:
        raise ValueError(
            "AGC release_ms must be in [attack_ms, 60000]."
        )
    if not 0.0 <= config.max_gain_db <= 60.0:
        raise ValueError("AGC max_gain_db must be in [0, 60].")
    if not 0.0 <= config.max_attenuation_db <= 60.0:
        raise ValueError(
            "AGC max_attenuation_db must be in [0, 60]."
        )


def get_ffmpeg_version(ffmpeg: str) -> str:
    try:
        result = subprocess.run(
            [ffmpeg, "-version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.splitlines()[0] if result.stdout else "unknown"


def decode_to_f32_mono(
    input_path: Path,
    raw_path: Path,
    sample_rate: int,
    ffmpeg: str,
) -> None:
    """Decode the first audio stream to mono little-endian float32 PCM."""
    if shutil.which(ffmpeg) is None:
        raise RuntimeError(
            f"Could not find '{ffmpeg}'. Install FFmpeg or pass --ffmpeg PATH."
        )

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    ]
    with raw_path.open("wb") as raw_file:
        result = subprocess.run(
            command,
            stdout=raw_file,
            stderr=subprocess.PIPE,
            text=False,
        )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg could not decode the input audio:\n{message}")

    size = raw_path.stat().st_size
    if size == 0:
        raise RuntimeError("FFmpeg produced no audio samples.")
    if size % np.dtype("<f4").itemsize:
        raise RuntimeError("Decoded PCM byte count is not a multiple of float32.")


def mean_memmap(audio: np.memmap, chunk_samples: int = 4_000_000) -> float:
    """Compute a float64 mean without materializing the full memory map."""
    total = 0.0
    count = 0
    for start in range(0, audio.size, chunk_samples):
        chunk = np.asarray(
            audio[start : start + chunk_samples], dtype=np.float32
        )
        chunk = np.nan_to_num(
            chunk, copy=True, nan=0.0, posinf=1.0, neginf=-1.0
        )
        np.clip(chunk, -1.0, 1.0, out=chunk)
        total += float(chunk.sum(dtype=np.float64))
        count += int(chunk.size)
    return total / count if count else 0.0


def _finite_dbfs(amplitude: float) -> float | None:
    """Convert a nonnegative linear amplitude to finite dBFS or None."""
    if not math.isfinite(amplitude) or amplitude <= 0.0:
        return None
    return 20.0 * math.log10(amplitude)


def build_agc_metadata(
    config: AgcConfig,
    sample_rate: int,
    control_frame_samples: int,
) -> dict[str, Any]:
    """Describe the lossy preprocessing needed to interpret reconstruction."""
    return {
        "enabled": True,
        "algorithm": "causal_block_rms_db_v1",
        "profile": config.profile,
        "target_dbfs": config.target_dbfs,
        "control_frame_samples": control_frame_samples,
        "control_frame_ms": (
            1000.0 * control_frame_samples / sample_rate
        ),
        "attack_ms": config.attack_ms,
        "release_ms": config.release_ms,
        "max_gain_db": config.max_gain_db,
        "max_attenuation_db": config.max_attenuation_db,
        "gate_dbfs": config.gate_dbfs,
        "peak_ceiling_dbfs": config.peak_dbfs,
        "limiter": "hard_peak_ceiling",
        "initial_gain_db": 0.0,
        "gain_envelope_stored": False,
        "reversible": False,
        "reconstruction_domain": "agc_normalized",
    }


def apply_waveform_agc(
    audio: np.memmap,
    output_path: Path,
    sample_rate: int,
    control_frame_samples: int,
    config: AgcConfig,
    dc_offset: float,
    quiet: bool,
) -> dict[str, Any]:
    """
    Apply bounded broadband AGC to decoded mono samples in a streaming pass.

    The detector sees one non-overlapping control frame (the actual mel hop)
    before emitting it. Gain is smoothed in dB and interpolated over the
    frame. Below the gate, positive gain is forbidden. A final hard ceiling
    contains transients while fixed mel/quantizer bin meanings remain intact.
    The gain envelope is intentionally not serialized, so this preprocessing
    is lossy with respect to the source's absolute loudness.
    """
    validate_agc_config(config)
    if sample_rate < 1 or control_frame_samples < 1:
        raise ValueError("AGC sample rate and control frame must be positive.")

    peak_linear = 10.0 ** (config.peak_dbfs / 20.0)
    current_gain_db = 0.0
    minimum_gain_db = 0.0
    maximum_gain_db = 0.0
    weighted_gain_db = 0.0
    input_sum_squares = 0.0
    output_sum_squares = 0.0
    input_peak = 0.0
    output_peak = 0.0
    limited_samples = 0
    gated_frames = 0
    frame_count = 0
    last_percent = -10

    with output_path.open("wb") as output_file:
        for start in range(0, audio.size, control_frame_samples):
            block = np.asarray(
                audio[start : start + control_frame_samples],
                dtype=np.float32,
            )
            block = np.nan_to_num(
                block, copy=True, nan=0.0, posinf=1.0, neginf=-1.0
            )
            np.clip(block, -1.0, 1.0, out=block)
            if dc_offset:
                block -= np.float32(dc_offset)
                np.clip(block, -1.0, 1.0, out=block)

            block64 = block.astype(np.float64, copy=False)
            block_peak = (
                float(np.max(np.abs(block64))) if block.size else 0.0
            )
            block_sum_squares = float(np.dot(block64, block64))
            block_rms = (
                math.sqrt(block_sum_squares / block.size)
                if block.size
                else 0.0
            )
            rms_dbfs = _finite_dbfs(block_rms)
            peak_dbfs = _finite_dbfs(block_peak)

            if rms_dbfs is None or rms_dbfs < config.gate_dbfs:
                gated_frames += 1
                desired_gain_db = 0.0
                if peak_dbfs is not None:
                    desired_gain_db = min(
                        desired_gain_db,
                        config.peak_dbfs - peak_dbfs,
                    )
            else:
                desired_gain_db = config.target_dbfs - rms_dbfs
                if peak_dbfs is not None:
                    desired_gain_db = min(
                        desired_gain_db,
                        config.peak_dbfs - peak_dbfs,
                    )
            desired_gain_db = min(
                config.max_gain_db,
                max(-config.max_attenuation_db, desired_gain_db),
            )

            time_constant_ms = (
                config.attack_ms
                if desired_gain_db < current_gain_db
                else config.release_ms
            )
            block_ms = 1000.0 * block.size / sample_rate
            smoothing = math.exp(-block_ms / time_constant_ms)
            next_gain_db = (
                desired_gain_db
                + smoothing * (current_gain_db - desired_gain_db)
            )
            gains_db = np.linspace(
                current_gain_db,
                next_gain_db,
                num=block.size,
                endpoint=True,
                dtype=np.float64,
            )
            gains_linear = np.power(10.0, gains_db / 20.0)
            processed64 = block64 * gains_linear
            over_ceiling = np.abs(processed64) > peak_linear
            limited_samples += int(np.count_nonzero(over_ceiling))
            np.clip(
                processed64, -peak_linear, peak_linear, out=processed64
            )
            processed = processed64.astype("<f4")
            output_file.write(processed.tobytes(order="C"))

            input_sum_squares += block_sum_squares
            output_sum_squares += float(np.dot(processed64, processed64))
            input_peak = max(input_peak, block_peak)
            if processed64.size:
                output_peak = max(
                    output_peak, float(np.max(np.abs(processed64)))
                )
            if gains_db.size:
                minimum_gain_db = min(
                    minimum_gain_db, float(np.min(gains_db))
                )
                maximum_gain_db = max(
                    maximum_gain_db, float(np.max(gains_db))
                )
                weighted_gain_db += float(gains_db.sum(dtype=np.float64))
            current_gain_db = next_gain_db
            frame_count += 1
            last_percent = print_progress(
                quiet,
                "agc",
                min(start + block.size, audio.size),
                int(audio.size),
                last_percent,
            )

    sample_count = int(audio.size)
    input_rms = (
        math.sqrt(input_sum_squares / sample_count) if sample_count else 0.0
    )
    output_rms = (
        math.sqrt(output_sum_squares / sample_count) if sample_count else 0.0
    )
    description = build_agc_metadata(
        config, sample_rate, control_frame_samples
    )
    return {
        **description,
        "control_frames": frame_count,
        "samples_processed": sample_count,
        "gated_control_frames": gated_frames,
        "gated_control_frame_fraction": (
            gated_frames / frame_count if frame_count else 0.0
        ),
        "minimum_controller_gain_db": minimum_gain_db,
        "maximum_controller_gain_db": maximum_gain_db,
        "mean_controller_gain_db": (
            weighted_gain_db / sample_count if sample_count else 0.0
        ),
        "final_controller_gain_db": current_gain_db,
        "hard_limited_samples": limited_samples,
        "hard_limited_sample_fraction": (
            limited_samples / sample_count if sample_count else 0.0
        ),
        "input_rms_dbfs": _finite_dbfs(input_rms),
        "output_rms_dbfs": _finite_dbfs(output_rms),
        "input_peak_dbfs": _finite_dbfs(input_peak),
        "output_peak_dbfs": _finite_dbfs(output_peak),
        "dc_offset_removed_before_agc": dc_offset,
    }


def iter_centered_frames(
    audio: np.memmap,
    n_frames: int,
    win_length: int,
    hop_length: int,
    batch_frames: int,
    dc_offset: float,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yield centered, zero-padded frame batches.

    Frame t is centered on sample t * hop_length.  There are
    ceil(num_samples / hop_length) frames, so timestamps are always inside the
    decoded signal.  Values outside the signal at either edge are zero.
    """
    half_window = win_length // 2
    for first_frame in range(0, n_frames, batch_frames):
        count = min(batch_frames, n_frames - first_frame)
        virtual_start = first_frame * hop_length - half_window
        virtual_length = (count - 1) * hop_length + win_length
        segment = np.zeros((virtual_length,), dtype=np.float32)

        source_start = max(0, virtual_start)
        source_end = min(audio.size, virtual_start + virtual_length)
        if source_end > source_start:
            destination_start = source_start - virtual_start
            values = np.asarray(audio[source_start:source_end], dtype=np.float32)
            values = np.nan_to_num(
                values, copy=True, nan=0.0, posinf=1.0, neginf=-1.0
            )
            # Float PCM full scale is conventionally [-1, 1]. Saturating
            # pathological finite values keeps CPU and CUDA power finite.
            np.clip(values, -1.0, 1.0, out=values)
            segment[
                destination_start : destination_start + values.size
            ] = values

        if dc_offset:
            # Apply DC removal only where real samples exist; zero padding stays 0.
            real_start = max(0, -virtual_start)
            real_end = min(virtual_length, audio.size - virtual_start)
            if real_end > real_start:
                segment[real_start:real_end] -= np.float32(dc_offset)
                np.clip(
                    segment[real_start:real_end],
                    -1.0,
                    1.0,
                    out=segment[real_start:real_end],
                )

        stride = segment.strides[0]
        frames = np.lib.stride_tricks.as_strided(
            segment,
            shape=(count, win_length),
            strides=(hop_length * stride, stride),
            writeable=False,
        ).copy()
        yield first_frame, frames


class MelBackend:
    """NumPy CPU or PyTorch CUDA implementation of power-mel projection."""

    def __init__(
        self,
        device_request: str,
        window: np.ndarray,
        n_fft: int,
        filterbank: np.ndarray,
    ) -> None:
        self.n_fft = n_fft
        self.window_np = window.astype(np.float32, copy=False)
        self.filterbank_np = filterbank.astype(np.float32, copy=False)
        self.power_normalizer = max(float(window.sum()) / 2.0, 1.0e-12) ** 2
        self.kind = "numpy"
        self.device = "cpu"
        self.torch: Any | None = None
        self.window_t: Any | None = None
        self.filterbank_t: Any | None = None

        requested = device_request.lower()
        if requested == "cpu":
            return
        if requested != "auto" and not requested.startswith("cuda"):
            raise ValueError("--device must be auto, cpu, cuda, or cuda:N.")

        try:
            import torch  # type: ignore
        except ImportError as error:
            if requested.startswith("cuda"):
                raise RuntimeError(
                    "CUDA was requested, but PyTorch is not installed. Install "
                    "a CUDA-enabled PyTorch build or use --device cpu."
                ) from error
            return

        if not torch.cuda.is_available():
            if requested.startswith("cuda"):
                raise RuntimeError(
                    "CUDA was requested, but torch.cuda.is_available() is false."
                )
            return

        device = "cuda:0" if requested in {"auto", "cuda"} else requested
        try:
            torch.empty((1,), device=device)
        except Exception as error:
            raise RuntimeError(f"Could not initialize CUDA device '{device}'.") from error

        self.kind = "torch"
        self.device = device
        self.torch = torch
        self.window_t = torch.as_tensor(self.window_np, device=device)
        self.filterbank_t = torch.as_tensor(self.filterbank_np, device=device)

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.device}"

    def transform(self, frames: np.ndarray) -> np.ndarray:
        if self.kind == "numpy":
            windowed = frames * self.window_np[None, :]
            spectrum = np.fft.rfft(windowed, n=self.n_fft, axis=1)
            power = (
                spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
            ) / self.power_normalizer
            mel = power @ self.filterbank_np.T
            return np.maximum(mel, 0.0).astype(np.float32, copy=False)

        assert self.torch is not None
        assert self.window_t is not None
        assert self.filterbank_t is not None
        torch = self.torch
        with torch.inference_mode():
            frame_t = torch.as_tensor(frames, device=self.device)
            spectrum_t = torch.fft.rfft(
                frame_t * self.window_t[None, :],
                n=self.n_fft,
                dim=1,
            )
            power_t = spectrum_t.abs().square() / self.power_normalizer
            mel_t = power_t @ self.filterbank_t.T
            return mel_t.clamp_min_(0.0).float().cpu().numpy()


def choose_reference(
    mode: str,
    max_power: float,
    sampled_power: np.ndarray,
    percentile: float,
    fixed_reference_power: float | None,
) -> tuple[float, str]:
    if mode == "fixed":
        if (
            fixed_reference_power is None
            or not math.isfinite(fixed_reference_power)
            or fixed_reference_power <= 0.0
        ):
            raise ValueError(
                "--reference-power must be positive when --reference-mode fixed."
            )
        return fixed_reference_power, "fixed"
    if max_power <= EPS_POWER:
        return 1.0, "silent_fallback"
    if mode == "file_peak":
        return max_power, "file_peak"
    if mode == "file_percentile":
        if sampled_power.size:
            reference = float(np.percentile(sampled_power, percentile))
        else:
            reference = max_power
        if not math.isfinite(reference) or reference <= EPS_POWER:
            reference = max_power
            return reference, "file_peak_fallback"
        return reference, f"file_percentile_{percentile:g}"
    raise ValueError(f"Unknown reference mode: {mode}")


def mel_to_quantized(
    mel_power: np.ndarray,
    reference_power: float,
    preset: Preset,
    silent: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if silent:
        db_raw = np.full(mel_power.shape, preset.db_min, dtype=np.float32)
    else:
        safe = np.maximum(mel_power, np.float32(EPS_POWER))
        db_raw = (
            10.0
            * (
                np.log10(safe, dtype=np.float32)
                - np.float32(math.log10(reference_power))
            )
        ).astype(np.float32, copy=False)
    db_clipped = np.clip(db_raw, preset.db_min, preset.db_max)
    scaled = (db_clipped - preset.db_min) * (
        (preset.levels - 1) / preset.top_db
    )
    quantized = np.floor(scaled + 0.5)
    quantized = np.clip(quantized, 0, preset.levels - 1)
    if preset.levels <= 256:
        quantized = quantized.astype(np.uint8)
    elif preset.levels <= 65_536:
        quantized = quantized.astype(np.uint16)
    else:
        quantized = quantized.astype(np.uint32)
    return quantized, db_raw


def dequantize_db(quantized: np.ndarray, preset: Preset) -> np.ndarray:
    return (
        preset.db_min
        + quantized.astype(np.float32)
        * np.float32(preset.top_db / (preset.levels - 1))
    )


def histogram_stats(
    histograms: np.ndarray,
    low_clip_count: np.ndarray,
    high_clip_count: np.ndarray,
    frame_count: int,
    centers_hz: np.ndarray,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for band, counts_u64 in enumerate(histograms):
        counts = counts_u64.astype(np.float64)
        total = float(counts.sum())
        probabilities = counts / total if total else np.zeros_like(counts)
        nonzero = probabilities > 0.0
        entropy_bits = float(
            -(probabilities[nonzero] * np.log2(probabilities[nonzero])).sum()
        )
        effective_states = float(2.0**entropy_bits)
        occupied = int(np.count_nonzero(counts_u64))
        output.append(
            {
                "band": band,
                "center_hz": float(centers_hz[band]),
                "occupied_states": occupied,
                "total_states": int(counts_u64.size),
                "effective_states": effective_states,
                "normalized_effective_state_fraction": (
                    effective_states / counts_u64.size
                ),
                "entropy_bits": entropy_bits,
                "minimum_state_count": int(counts_u64.min()),
                "maximum_state_count": int(counts_u64.max()),
                "floor_state_fraction": (
                    float(counts_u64[0]) / frame_count if frame_count else 0.0
                ),
                "ceiling_state_fraction": (
                    float(counts_u64[-1]) / frame_count if frame_count else 0.0
                ),
                "preclip_below_fraction": (
                    float(low_clip_count[band]) / frame_count
                    if frame_count
                    else 0.0
                ),
                "preclip_above_fraction": (
                    float(high_clip_count[band]) / frame_count
                    if frame_count
                    else 0.0
                ),
                "state_counts": [int(value) for value in counts_u64],
            }
        )
    return output


def compact_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def build_roundtrip_metadata(
    preset: Preset,
    sample_count: int,
    frame_count: int,
    include_time_columns: bool,
    reference_power: float,
    reference_mode: str,
    silent: bool,
    fft_power_normalizer: float,
    state_crc32: int,
    agc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the compact transform description embedded in CSV and PNG."""
    metadata: dict[str, Any] = {
        "format": "bounded_quantized_mel_v2",
        "version": 2,
        "shape": {
            "timesteps": frame_count,
            "bands": preset.n_mels,
            "axis_order": "TC",
        },
        "waveform": {
            "sample_rate": preset.sample_rate,
            "decoded_sample_count": sample_count,
            "channels": 1,
        },
        "stft": {
            "n_fft": preset.n_fft,
            "win_length": preset.win_length,
            "hop_length": preset.hop_length,
            "window": "periodic_hann",
            "frame_center": "t*hop",
            "padding": "constant_zero",
            "fft_zero_padding": "right",
            "power_normalizer": fft_power_normalizer,
        },
        "mel": {
            "n_mels": preset.n_mels,
            "fmin": preset.fmin,
            "fmax": preset.fmax,
            "scale": "HTK",
            "filter_shape": "triangular",
            "filter_normalization": "unit_sum",
        },
        "quantizer": {
            "kind": "uniform_db",
            "levels": preset.levels,
            "integer_min": 0,
            "integer_max": preset.levels - 1,
            "db_min": preset.db_min,
            "db_max": preset.db_max,
            "clip": "saturate",
            "zero_state_semantics": "lower_censored_or_silence",
            "reference_power": reference_power,
            "reference_mode": reference_mode,
            "silent": silent,
        },
        "csv": {
            "time_columns_included": include_time_columns,
            "metadata_prefix": CSV_METADATA_PREFIX.strip(),
        },
        "integrity": {
            "state_order": "row_major_TC",
            "state_dtype": "uint8" if preset.levels <= 256 else "uint16_le",
            "state_crc32": f"{state_crc32:08x}",
        },
        "profile": preset.name,
    }
    if agc is not None:
        metadata["preprocessing"] = {"agc": agc}
    return metadata


def _palette_for_levels(levels: int) -> list[int]:
    """Build a compact inferno-like palette whose first K entries span it."""
    stops = (
        (0.00, (0, 0, 4)),
        (0.20, (59, 15, 112)),
        (0.40, (140, 41, 129)),
        (0.60, (221, 73, 104)),
        (0.80, (253, 159, 108)),
        (1.00, (252, 253, 191)),
    )

    def sample(position: float) -> tuple[int, int, int]:
        for index in range(len(stops) - 1):
            left_x, left_rgb = stops[index]
            right_x, right_rgb = stops[index + 1]
            if position <= right_x:
                fraction = (position - left_x) / max(right_x - left_x, 1.0e-12)
                return tuple(
                    int(round(a + fraction * (b - a)))
                    for a, b in zip(left_rgb, right_rgb)
                )
        return stops[-1][1]

    palette: list[int] = []
    for index in range(256):
        if index < levels:
            position = index / max(levels - 1, 1)
            rgb = sample(position)
        else:
            rgb = (index, index, index)
        palette.extend(rgb)
    return palette


def choose_png_tile_width(
    frame_count: int,
    n_mels: int,
    requested_width: int | None,
    maximum_width: int,
) -> int:
    if requested_width is not None:
        if requested_width < 1 or requested_width > maximum_width:
            raise ValueError(
                f"--png-tile-width must be in [1, {maximum_width}]."
            )
        return requested_width
    target = int(math.ceil(math.sqrt(frame_count * n_mels)))
    rounded = max(128, ((target + 63) // 64) * 64)
    return min(maximum_width, rounded)


def encode_scaled_u16_states(
    states: np.ndarray,
    levels: int,
) -> np.ndarray:
    """Map q in [0,K-1] bijectively onto a visible 16-bit grayscale lattice."""
    values = np.asarray(states, dtype=np.uint64)
    encoded = (
        values * np.uint64(65_535) + np.uint64((levels - 1) // 2)
    ) // np.uint64(levels - 1)
    return encoded.astype(np.uint16)


def write_state_png(
    output_path: Path,
    state_path: Path,
    roundtrip_metadata: dict[str, Any],
    frame_count: int,
    n_mels: int,
    levels: int,
    tile_width: int | None,
    maximum_width: int,
    maximum_pixels: int,
) -> dict[str, Any]:
    """
    Write exact q states as an 8-bit indexed or 16-bit grayscale PNG.

    The canonical JSON is stored in a compressed iTXt chunk and redundantly in
    a paired-byte pixel header. Removing ancillary PNG metadata therefore does
    not prevent decoding as long as the integer state pixels remain unchanged.
    """
    try:
        from PIL import Image, PngImagePlugin  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "PNG output requires Pillow. Install it with "
            "'python -m pip install pillow'."
        ) from error

    width = choose_png_tile_width(
        frame_count, n_mels, tile_width, maximum_width
    )
    sixteen_bit = levels > 256
    png_mode = "I;16" if sixteen_bit else "P"
    state_pixel_codec = (
        "scaled_u16_state_lattice_v1"
        if sixteen_bit
        else "palette_index_equals_state"
    )
    png_metadata = json.loads(compact_json(roundtrip_metadata))
    png_metadata["png"] = {
        "format": "QMel-PNG",
        "version": 2,
        "mode": png_mode,
        "layout": "frequency_flipped_strip_stack",
        "tile_width": width,
        "state_pixel_codec": state_pixel_codec,
        "pixel_header_codec": "paired_zlib_json_v1",
        "metadata_key": PNG_METADATA_KEY,
    }
    canonical = compact_json(png_metadata)
    compressed = zlib.compress(canonical.encode("utf-8"), level=9)
    compressed_crc = zlib.crc32(compressed) & 0xFFFFFFFF
    raw_header = (
        PNG_PIXEL_MAGIC
        + struct.pack(">II", len(compressed), compressed_crc)
        + compressed
    )
    paired_header = np.empty((2 * len(raw_header),), dtype=np.uint8)
    header_bytes = np.frombuffer(raw_header, dtype=np.uint8)
    paired_header[0::2] = header_bytes
    paired_header[1::2] = 255 - header_bytes
    header_rows = int(math.ceil(paired_header.size / width))
    strips = int(math.ceil(frame_count / width))
    height = header_rows + strips * n_mels
    pixel_count = width * height
    if pixel_count > maximum_pixels:
        raise ValueError(
            f"PNG would require {pixel_count:,} pixels, exceeding "
            f"--png-max-pixels={maximum_pixels:,}. Use CSV or raise the limit."
        )

    canvas_dtype = np.uint16 if sixteen_bit else np.uint8
    state_dtype = "<u2" if sixteen_bit else np.uint8
    canvas = np.zeros((height, width), dtype=canvas_dtype)
    canvas.reshape(-1)[: paired_header.size] = paired_header
    states = np.memmap(
        state_path,
        dtype=state_dtype,
        mode="r",
        shape=(frame_count, n_mels),
    )
    for strip_index in range(strips):
        start = strip_index * width
        end = min(frame_count, start + width)
        rows = slice(
            header_rows + strip_index * n_mels,
            header_rows + (strip_index + 1) * n_mels,
        )
        payload = states[start:end].T[::-1, :]
        if sixteen_bit:
            payload = encode_scaled_u16_states(payload, levels)
        canvas[rows, : end - start] = payload

    if sixteen_bit:
        image = Image.fromarray(canvas)
    else:
        image = Image.frombytes(
            "P", (width, height), canvas.tobytes(order="C")
        )
        image.putpalette(_palette_for_levels(levels))
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_itxt(PNG_METADATA_KEY, canonical, zip=True)
    image.save(
        output_path,
        format="PNG",
        pnginfo=pnginfo,
        compress_level=9,
        optimize=False,
    )
    del states
    return {
        "kind": "exact_state_raster",
        "mode": png_mode,
        "state_pixel_codec": state_pixel_codec,
        "width": width,
        "height": height,
        "header_rows": header_rows,
        "strips": strips,
        "payload_cells": frame_count * n_mels,
        "container_pixels": pixel_count,
        "frequency_flipped_for_display": True,
        "time_continues_in_strips_top_to_bottom": True,
    }


def write_self_describing_csv(
    output_path: Path,
    body_path: Path,
    roundtrip_metadata: dict[str, Any],
) -> None:
    with body_path.open("rb") as body_file, output_path.open("wb") as output_file:
        header = body_file.readline()
        if not header:
            raise RuntimeError("CSV body is missing its header.")
        output_file.write(header)
        metadata_line = (
            CSV_METADATA_PREFIX + compact_json(roundtrip_metadata) + "\n"
        ).encode("utf-8")
        output_file.write(metadata_line)
        shutil.copyfileobj(body_file, output_file, length=1024 * 1024)


def output_paths(
    input_path: Path,
    output_dir: Path,
    output_prefix: str | None,
    preset_name: str,
) -> dict[str, Path]:
    prefix = output_prefix if output_prefix else input_path.stem
    base = f"{prefix}.{preset_name}.mel"
    return {
        "csv": output_dir / f"{base}.csv",
        "png": output_dir / f"{base}.png",
        "json": output_dir / f"{base}.json",
    }


def check_outputs(
    paths: dict[str, Path],
    make_csv: bool,
    make_png: bool,
    make_json: bool,
    force: bool,
) -> None:
    selected: list[Path] = []
    if make_csv:
        selected.append(paths["csv"])
    if make_png:
        selected.append(paths["png"])
    if make_json:
        selected.append(paths["json"])
    existing = [path for path in selected if path.exists()]
    if existing and not force:
        joined = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing output(s); pass --force:\n" + joined
        )


def print_progress(
    quiet: bool,
    stage: str,
    completed: int,
    total: int,
    last_percent: int,
) -> int:
    if quiet or total <= 0:
        return last_percent
    percent = int(100 * completed / total)
    if percent >= last_percent + 10 or completed == total:
        print(f"[{stage}] {min(percent, 100):3d}%", file=sys.stderr)
        return percent
    return last_percent


def convert_one(
    input_path: Path,
    output_dir: Path,
    output_prefix: str | None,
    preset: Preset,
    device: str,
    ffmpeg: str,
    batch_frames: int,
    reference_mode: str,
    reference_percentile: float,
    fixed_reference_power: float | None,
    max_reference_values: int,
    include_time_columns: bool,
    output_format: str,
    write_json: bool,
    png_tile_width: int | None,
    png_max_width: int,
    png_max_pixels: int,
    remove_dc: bool,
    agc_config: AgcConfig | None,
    force: bool,
    quiet: bool,
) -> dict[str, Any]:
    """Convert one file/preset to standalone CSV and/or exact-state PNG."""
    validate_preset(preset)
    if max_reference_values < preset.n_mels:
        raise ValueError(
            "--max-reference-values must be at least the number of mel bands "
            f"({preset.n_mels}) so every sampled reference frame is complete."
        )
    make_csv = output_format in {"csv", "both"}
    make_png = output_format in {"png", "both"}
    if not make_csv and not make_png:
        raise ValueError("--output-format must be csv, png, or both.")
    paths = output_paths(input_path, output_dir, output_prefix, preset.name)
    check_outputs(
        paths,
        make_csv=make_csv,
        make_png=make_png,
        make_json=write_json,
        force=force,
    )

    filterbank, centers_hz, mel_edges_hz = make_mel_filterbank(
        preset.sample_rate,
        preset.n_fft,
        preset.n_mels,
        preset.fmin,
        preset.fmax,
    )
    window = periodic_hann(preset.win_length)
    backend = MelBackend(device, window, preset.n_fft, filterbank)
    if batch_frames < 1:
        raise ValueError("--batch-frames must be positive.")

    temp_csv = paths["csv"].with_suffix(paths["csv"].suffix + ".part")
    temp_json = paths["json"].with_suffix(paths["json"].suffix + ".part")
    temp_png = paths["png"].with_suffix(paths["png"].suffix + ".part")
    temp_files = [temp_csv, temp_json, temp_png]

    try:
        with tempfile.TemporaryDirectory(prefix="audio_to_token_mel_") as temp_dir:
            temp_directory = Path(temp_dir)
            raw_path = temp_directory / "decoded.f32le"
            agc_path = temp_directory / "agc.f32le"
            csv_body_path = temp_directory / "states_body.csv"
            state_path = temp_directory / "states.bin"

            if not quiet:
                print(
                    f"[decode] {input_path.name} -> mono {preset.sample_rate} Hz",
                    file=sys.stderr,
                )
            decode_to_f32_mono(input_path, raw_path, preset.sample_rate, ffmpeg)
            sample_count = raw_path.stat().st_size // np.dtype("<f4").itemsize
            decoded_audio = np.memmap(
                raw_path, dtype="<f4", mode="r", shape=(sample_count,)
            )
            duration_seconds = sample_count / preset.sample_rate
            n_frames = max(1, math.ceil(sample_count / preset.hop_length))
            dc_offset = mean_memmap(decoded_audio) if remove_dc else 0.0
            agc_diagnostics: dict[str, Any] | None = None
            if agc_config is not None:
                agc_diagnostics = apply_waveform_agc(
                    audio=decoded_audio,
                    output_path=agc_path,
                    sample_rate=preset.sample_rate,
                    control_frame_samples=preset.hop_length,
                    config=agc_config,
                    dc_offset=dc_offset,
                    quiet=quiet,
                )
                del decoded_audio
                audio = np.memmap(
                    agc_path,
                    dtype="<f4",
                    mode="r",
                    shape=(sample_count,),
                )
                frame_dc_offset = 0.0
            else:
                audio = decoded_audio
                frame_dc_offset = dc_offset

            max_sample_frames = max(1, max_reference_values // preset.n_mels)
            reference_frame_stride = max(
                1, math.ceil(n_frames / max_sample_frames)
            )
            sampled_batches: list[np.ndarray] = []
            max_power = 0.0
            last_percent = -10

            for first, frames in iter_centered_frames(
                audio,
                n_frames,
                preset.win_length,
                preset.hop_length,
                batch_frames,
                frame_dc_offset,
            ):
                mel_power = backend.transform(frames)
                batch_max = float(np.max(mel_power))
                if math.isfinite(batch_max):
                    max_power = max(max_power, batch_max)

                if reference_mode == "file_percentile":
                    global_indices = np.arange(
                        first, first + mel_power.shape[0], dtype=np.int64
                    )
                    selected = global_indices % reference_frame_stride == 0
                    if selected.any():
                        sampled_batches.append(
                            mel_power[selected]
                            .astype(np.float32, copy=True)
                            .reshape(-1)
                        )
                last_percent = print_progress(
                    quiet,
                    "reference",
                    first + frames.shape[0],
                    n_frames,
                    last_percent,
                )

            sampled_power = (
                np.concatenate(sampled_batches)
                if sampled_batches
                else np.empty((0,), dtype=np.float32)
            )
            reference_power, reference_label = choose_reference(
                reference_mode,
                max_power,
                sampled_power,
                reference_percentile,
                fixed_reference_power,
            )
            silent = max_power <= EPS_POWER
            del sampled_batches, sampled_power

            histograms = np.zeros(
                (preset.n_mels, preset.levels), dtype=np.uint64
            )
            low_clip_count = np.zeros((preset.n_mels,), dtype=np.uint64)
            high_clip_count = np.zeros((preset.n_mels,), dtype=np.uint64)
            aggregate_low = 0
            aggregate_high = 0
            state_crc32 = 0

            with ExitStack() as stack:
                csv_file = (
                    stack.enter_context(
                        csv_body_path.open("w", encoding="utf-8", newline="")
                    )
                    if make_csv
                    else None
                )
                state_file = (
                    stack.enter_context(state_path.open("wb"))
                    if make_png
                    else None
                )
                feature_names = [
                    f"mel_{band:03d}_q" for band in range(preset.n_mels)
                ]
                if csv_file is not None:
                    header = (
                        ["frame_index", "time_s"] + feature_names
                        if include_time_columns
                        else feature_names
                    )
                    csv_file.write(",".join(header) + "\n")

                last_percent = -10
                for first, frames in iter_centered_frames(
                    audio,
                    n_frames,
                    preset.win_length,
                    preset.hop_length,
                    batch_frames,
                    frame_dc_offset,
                ):
                    mel_power = backend.transform(frames)
                    quantized, db_raw = mel_to_quantized(
                        mel_power, reference_power, preset, silent
                    )
                    count = quantized.shape[0]

                    if csv_file is not None:
                        if include_time_columns:
                            indices = np.arange(
                                first, first + count, dtype=np.int64
                            )
                            times = (
                                indices.astype(np.float64)
                                * preset.hop_length
                                / preset.sample_rate
                            )
                            table = np.column_stack((indices, times, quantized))
                            formats: Sequence[str] = (
                                ["%d", "%.9f"] + ["%d"] * preset.n_mels
                            )
                            np.savetxt(
                                csv_file, table, delimiter=",", fmt=formats
                            )
                        else:
                            np.savetxt(
                                csv_file, quantized, delimiter=",", fmt="%d"
                            )

                    if preset.levels <= 256:
                        canonical_states = np.ascontiguousarray(
                            quantized, dtype=np.uint8
                        )
                    else:
                        canonical_states = np.ascontiguousarray(
                            quantized, dtype="<u2"
                        )
                    state_crc32 = zlib.crc32(
                        canonical_states.tobytes(order="C"), state_crc32
                    )
                    if state_file is not None:
                        state_file.write(canonical_states.tobytes(order="C"))

                    for band in range(preset.n_mels):
                        histograms[band] += np.bincount(
                            quantized[:, band].astype(np.int64),
                            minlength=preset.levels,
                        ).astype(np.uint64)

                    low = db_raw < preset.db_min
                    high = db_raw > preset.db_max
                    low_clip_count += low.sum(axis=0, dtype=np.uint64)
                    high_clip_count += high.sum(axis=0, dtype=np.uint64)
                    aggregate_low += int(low.sum())
                    aggregate_high += int(high.sum())

                    last_percent = print_progress(
                        quiet,
                        "write",
                        first + count,
                        n_frames,
                        last_percent,
                    )

            state_crc32 &= 0xFFFFFFFF
            roundtrip_metadata = build_roundtrip_metadata(
                preset=preset,
                sample_count=sample_count,
                frame_count=n_frames,
                include_time_columns=include_time_columns,
                reference_power=reference_power,
                reference_mode=reference_label,
                silent=silent,
                fft_power_normalizer=backend.power_normalizer,
                state_crc32=state_crc32,
                agc=(
                    build_agc_metadata(
                        agc_config,
                        preset.sample_rate,
                        preset.hop_length,
                    )
                    if agc_config is not None
                    else None
                ),
            )

            if make_csv:
                write_self_describing_csv(
                    temp_csv, csv_body_path, roundtrip_metadata
                )

            png_description: dict[str, Any] | None = None
            if make_png:
                png_description = write_state_png(
                    output_path=temp_png,
                    state_path=state_path,
                    roundtrip_metadata=roundtrip_metadata,
                    frame_count=n_frames,
                    n_mels=preset.n_mels,
                    levels=preset.levels,
                    tile_width=png_tile_width,
                    maximum_width=png_max_width,
                    maximum_pixels=png_max_pixels,
                )

            per_band_stats = histogram_stats(
                histograms,
                low_clip_count,
                high_clip_count,
                n_frames,
                centers_hz,
            )
            aggregate_histogram = histograms.sum(axis=0, dtype=np.uint64)
            total_cells = n_frames * preset.n_mels
            physical_columns = preset.n_mels + (
                2 if include_time_columns else 0
            )
            metadata: dict[str, Any] = {
                "format": "bounded_quantized_mel_v2",
                "generator": {
                    "name": Path(__file__).name,
                    "version": VERSION,
                },
                "input": {
                    "file_name": input_path.name,
                    "file_size_bytes": input_path.stat().st_size,
                    "decoded_sample_count": sample_count,
                    "decoded_duration_seconds": duration_seconds,
                    "decoded_channels": 1,
                    "decoded_waveform_saturation_range": [-1.0, 1.0],
                    "dc_offset_removed": dc_offset if remove_dc else 0.0,
                    "dc_removal_uses_whole_file_mean": remove_dc,
                },
                "preprocessing": {
                    "agc": (
                        agc_diagnostics
                        if agc_diagnostics is not None
                        else {"enabled": False}
                    )
                },
                "outputs": {
                    "csv": paths["csv"].name if make_csv else None,
                    "png": paths["png"].name if make_png else None,
                    "metadata_json": paths["json"].name if write_json else None,
                },
                "roundtrip_metadata": roundtrip_metadata,
                "visualization": png_description,
                "preset": {
                    **asdict(preset),
                    "win_length_samples": preset.win_length,
                    "hop_length_samples": preset.hop_length,
                    "timestep_ms": (
                        1000.0
                        * preset.hop_length
                        / preset.sample_rate
                    ),
                    "timesteps_per_second": preset.frames_per_second,
                    "frames_per_second": preset.frames_per_second,
                    "db_min": preset.db_min,
                    "db_max": preset.db_max,
                    "quantization_step_db": preset.quantization_step_db,
                },
                "framing": {
                    "frame_count": n_frames,
                    "centered": True,
                    "padding": "constant_zero",
                    "frame_time_seconds": (
                        "frame_index * hop_length / sample_rate"
                    ),
                    "frame_count_formula": "ceil(decoded_samples / hop_length)",
                    "timestep_ms": (
                        1000.0
                        * preset.hop_length
                        / preset.sample_rate
                    ),
                    "timesteps_per_second": preset.frames_per_second,
                    "window": "periodic_hann",
                    "lookahead_samples": preset.win_length // 2,
                },
                "mel": {
                    "scale": "HTK",
                    "filter_shape": "triangular",
                    "filter_normalization": "unit_sum",
                    "power_spectrogram": True,
                    "fft_power_normalizer": backend.power_normalizer,
                    "fft_power_formula": (
                        "|RFFT(window * frame)|^2 / (sum(window)/2)^2"
                    ),
                    "one_sided_power_convention": (
                        "rfft bins are not doubled at interior frequencies"
                    ),
                    "minimum_power_before_log10": EPS_POWER,
                    "band_centers_hz": [float(value) for value in centers_hz],
                    "band_edges_hz": [float(value) for value in mel_edges_hz],
                },
                "reference": {
                    "requested_mode": reference_mode,
                    "effective_mode": reference_label,
                    "reference_power": reference_power,
                    "reference_percentile": (
                        reference_percentile
                        if reference_mode == "file_percentile"
                        else None
                    ),
                    "maximum_observed_mel_power": max_power,
                    "silent": silent,
                    "future_dependent": reference_mode
                    in {"file_percentile", "file_peak"},
                },
                "causality": {
                    "zero_lookahead": False,
                    "future_dependent": True,
                    "reasons": [
                        reason
                        for condition, reason in (
                            (
                                reference_mode
                                in {"file_percentile", "file_peak"},
                                "whole-file spectral reference",
                            ),
                            (remove_dc, "whole-file waveform mean removal"),
                            (
                                agc_config is not None,
                                "AGC buffers one control frame",
                            ),
                            (True, "centered window uses half-window lookahead"),
                        )
                        if condition
                    ],
                },
                "quantization": {
                    "formula": (
                        "q=floor(clip((dB-db_min)*(levels-1)/"
                        "(db_max-db_min),0,levels-1)+0.5)"
                    ),
                    "inverse_formula": (
                        "dB_hat=db_min+q*(db_max-db_min)/(levels-1)"
                    ),
                    "integer_min": 0,
                    "integer_max": preset.levels - 1,
                    "states_per_mel_cell": preset.levels,
                    "state_crc32": f"{state_crc32:08x}",
                    "aggregate_state_counts": [
                        int(value) for value in aggregate_histogram
                    ],
                    "aggregate_preclip_below_fraction": (
                        aggregate_low / total_cells if total_cells else 0.0
                    ),
                    "aggregate_preclip_above_fraction": (
                        aggregate_high / total_cells if total_cells else 0.0
                    ),
                    "per_band": per_band_stats,
                },
                "csv_schema": {
                    "rows": n_frames,
                    "mel_feature_columns": preset.n_mels,
                    "physical_columns": physical_columns,
                    "time_columns_included": include_time_columns,
                    "self_describing": True,
                    "metadata_position": "comment_after_header",
                    "tokenize_only_columns_matching": "mel_*_q",
                },
                "state_space": {
                    "states_per_cell": preset.levels,
                    "independent_band_state_embeddings": (
                        preset.n_mels * preset.levels
                    ),
                    "factorized_band_plus_level_embeddings": (
                        preset.n_mels + preset.levels
                    ),
                    "nominal_frame_state_space": (
                        f"{preset.levels}^{preset.n_mels}"
                    ),
                    "nominal_frame_state_bits": (
                        preset.n_mels * math.log2(preset.levels)
                    ),
                    "serialized_scalar_cells_per_second": (
                        preset.n_mels * preset.frames_per_second
                    ),
                },
                "runtime": {
                    "backend": backend.label,
                    "batch_frames": batch_frames,
                    "ffmpeg": get_ffmpeg_version(ffmpeg),
                },
            }

            if write_json:
                with temp_json.open("w", encoding="utf-8") as metadata_file:
                    json.dump(metadata, metadata_file, indent=2, sort_keys=True)
                    metadata_file.write("\n")

            del audio

        if make_csv:
            os.replace(temp_csv, paths["csv"])
        if make_png:
            os.replace(temp_png, paths["png"])
        if write_json:
            os.replace(temp_json, paths["json"])
        return metadata
    except Exception:
        for path in temp_files:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def apply_overrides(preset: Preset, args: argparse.Namespace) -> Preset:
    replacements: dict[str, Any] = {}
    mapping = {
        "sample_rate": args.sample_rate,
        "n_fft": args.n_fft,
        "win_ms": args.win_ms,
        "hop_ms": args.hop_ms,
        "n_mels": args.n_mels,
        "fmin": args.fmin,
        "fmax": args.fmax,
        "top_db": args.top_db,
        "levels": args.levels,
    }
    for key, value in mapping.items():
        if value is not None:
            replacements[key] = value
    return dataclasses.replace(preset, **replacements)


def resolve_agc_config(
    preset_name: str,
    args: argparse.Namespace,
) -> AgcConfig | None:
    """Apply optional expert overrides to one ladder's AGC profile."""
    if not args.agc:
        return None
    replacements: dict[str, float] = {}
    mapping = {
        "target_dbfs": args.agc_target_dbfs,
        "attack_ms": args.agc_attack_ms,
        "release_ms": args.agc_release_ms,
        "max_gain_db": args.agc_max_gain_db,
        "max_attenuation_db": args.agc_max_attenuation_db,
        "gate_dbfs": args.agc_gate_dbfs,
        "peak_dbfs": args.agc_peak_dbfs,
    }
    for key, value in mapping.items():
        if value is not None:
            replacements[key] = value
    config = dataclasses.replace(AGC_PROFILES[preset_name], **replacements)
    validate_agc_config(config)
    return config


def describe_presets() -> None:
    print(
        "preset  sr(Hz)  win/step(ms) mels  dB range    levels/range  "
        "dB step  cells/s"
    )
    for preset in PRESETS.values():
        print(
            f"{preset.name:<7} "
            f"{preset.sample_rate:>6}  "
            f"{preset.win_ms:g}/{preset.hop_ms:g}".ljust(14)
            + f"{preset.n_mels:>4}  "
            f"[{-preset.top_db:g},0]".ljust(12)
            + f"{preset.levels:>5} / 0..{preset.levels - 1:<4}  "
            f"{preset.quantization_step_db:>7.4f}  "
            f"{preset.n_mels * preset.frames_per_second:>7.0f}"
        )


def describe_agc_profiles() -> None:
    print(
        "preset   target  attack  release  max boost/atten  gate   peak  "
        "control interval"
    )
    for preset_name, config in AGC_PROFILES.items():
        preset = PRESETS[preset_name]
        print(
            f"{preset_name:<7} "
            f"{config.target_dbfs:>6g}  "
            f"{config.attack_ms:>6g}  "
            f"{config.release_ms:>7g}  "
            f"{config.max_gain_db:>6g}/{config.max_attenuation_db:<6g}  "
            f"{config.gate_dbfs:>5g}  "
            f"{config.peak_dbfs:>5g}  "
            f"{preset.hop_ms:g} ms (selected timestep)"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decode any FFmpeg-supported audio file and write self-describing "
            "bounded mel-state CSV and/or exact-state PNG containers."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {VERSION}"
    )
    parser.add_argument("input", nargs="?", type=Path, help="Input audio/video file.")
    parser.add_argument(
        "--preset",
        choices=[*PRESETS, "all"],
        default="medium",
        help="Quality/state-space preset.",
    )
    parser.add_argument(
        "--describe-presets",
        action="store_true",
        help="Print the built-in preset table and exit.",
    )
    parser.add_argument(
        "--describe-agc",
        action="store_true",
        help="Print the five built-in AGC profiles and exit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for generated outputs.",
    )
    parser.add_argument(
        "--output-prefix",
        help="Output basename prefix; defaults to the input filename stem.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or cuda:N. CPU uses NumPy.",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="FFmpeg executable name or path.",
    )
    parser.add_argument(
        "--batch-frames",
        type=int,
        default=512,
        help="Frames per FFT batch; lower this to reduce peak memory.",
    )
    parser.add_argument(
        "--reference-mode",
        choices=["file_percentile", "file_peak", "fixed"],
        default="file_percentile",
        help=(
            "Spectral gain reference. file_* is gain-invariant but "
            "future-dependent; fixed is corpus/deployment comparable."
        ),
    )
    parser.add_argument(
        "--reference-percentile",
        type=float,
        default=99.5,
        help="Percentile used by file_percentile reference mode.",
    )
    parser.add_argument(
        "--reference-power",
        type=float,
        help="Positive frozen reference used by --reference-mode fixed.",
    )
    parser.add_argument(
        "--max-reference-values",
        type=int,
        default=262_144,
        help="Bounded deterministic sample size for percentile estimation.",
    )
    parser.add_argument(
        "--include-time-columns",
        action="store_true",
        help="Prepend frame_index,time_s to the CSV (do not tokenize them).",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "png", "both"],
        default="both",
        help="Generate a standalone CSV, standalone PNG, or both.",
    )
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Also write the verbose diagnostics JSON sidecar.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Deprecated alias for --output-format csv.",
    )
    parser.add_argument(
        "--png-tile-width",
        type=int,
        help="Frames per horizontal PNG strip; automatic when omitted.",
    )
    parser.add_argument(
        "--png-max-width",
        type=int,
        default=8192,
        help="Maximum automatic or explicit PNG strip width.",
    )
    parser.add_argument(
        "--png-max-pixels",
        type=int,
        default=80_000_000,
        help="Safety limit for the exact-state PNG raster.",
    )
    parser.add_argument(
        "--keep-dc",
        action="store_true",
        help="Do not remove the decoded file's global DC offset.",
    )
    agc = parser.add_argument_group(
        "optional broadband waveform AGC (fixed mel token bins are unchanged)"
    )
    agc.add_argument(
        "--agc",
        action="store_true",
        help=(
            "Enable the selected quality ladder's lossy waveform AGC. "
            "With --preset all, each ladder uses its own profile."
        ),
    )
    agc.add_argument(
        "--agc-target-dbfs",
        type=float,
        help="Override the ladder's RMS target in dBFS.",
    )
    agc.add_argument(
        "--agc-attack-ms",
        type=float,
        help="Override the gain-reduction time constant in milliseconds.",
    )
    agc.add_argument(
        "--agc-release-ms",
        type=float,
        help="Override the gain-increase time constant in milliseconds.",
    )
    agc.add_argument(
        "--agc-max-gain-db",
        type=float,
        help="Override the maximum positive controller gain.",
    )
    agc.add_argument(
        "--agc-max-attenuation-db",
        type=float,
        help="Override the maximum controller attenuation.",
    )
    agc.add_argument(
        "--agc-gate-dbfs",
        type=float,
        help="Override the RMS gate; gated frames receive no positive boost.",
    )
    agc.add_argument(
        "--agc-peak-dbfs",
        type=float,
        help="Override the final hard peak ceiling in dBFS.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages.",
    )

    custom = parser.add_argument_group(
        "preset overrides (use one built-in preset as the starting point)"
    )
    custom.add_argument(
        "--sample-rate",
        "--samples-per-second",
        dest="sample_rate",
        type=int,
        action=StoreConsistentValue,
        metavar="HZ",
        help=(
            "Decoded waveform samples per second. This sets audio bandwidth "
            "and converts --win-ms/--timestep-ms into integer sample counts; "
            "fmax may not exceed half this value."
        ),
    )
    custom.add_argument("--n-fft", type=int)
    custom.add_argument("--win-ms", type=float)
    custom.add_argument(
        "--timestep-ms",
        "--hop-ms",
        dest="hop_ms",
        type=float,
        action=StoreConsistentValue,
        metavar="MS",
        help=(
            "Time between mel rows in milliseconds. --hop-ms is an exact "
            "alias; the rounded sample hop is embedded for reconstruction."
        ),
    )
    custom.add_argument(
        "--columns-per-timestep",
        "--n-mels",
        dest="n_mels",
        type=int,
        action=StoreConsistentValue,
        metavar="C",
        help=(
            "Tokenized mel columns in every row. --n-mels is an exact alias."
        ),
    )
    custom.add_argument("--fmin", type=float)
    custom.add_argument("--fmax", type=float)
    custom.add_argument("--top-db", type=float)
    custom.add_argument(
        "--states-per-column",
        "--levels",
        dest="levels",
        type=int,
        action=StoreConsistentValue,
        metavar="K",
        help=(
            "Legal integer states per mel column; need not be a power of 2. "
            "--levels is an exact alias."
        ),
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.describe_presets or args.describe_agc:
        return
    if args.input is None:
        parser.error(
            "input is required unless --describe-presets or --describe-agc "
            "is used."
        )
    if not args.input.is_file():
        parser.error(f"input is not a readable file: {args.input}")
    if not (0.0 < args.reference_percentile <= 100.0):
        parser.error("--reference-percentile must be in (0, 100].")
    if args.max_reference_values < 1:
        parser.error("--max-reference-values must be positive.")
    if args.png_tile_width is not None and args.png_tile_width < 1:
        parser.error("--png-tile-width must be positive.")
    if args.png_max_width < 1:
        parser.error("--png-max-width must be positive.")
    if args.png_max_pixels < 1:
        parser.error("--png-max-pixels must be positive.")
    if args.n_mels is not None and not 1 <= args.n_mels <= 4096:
        parser.error("--columns-per-timestep must be in [1, 4096].")
    if args.levels is not None and not 2 <= args.levels <= 65_536:
        parser.error("--states-per-column must be in [2, 65536].")
    if args.sample_rate is not None and args.sample_rate < 1:
        parser.error("--samples-per-second must be positive.")
    if args.hop_ms is not None and (
        not math.isfinite(args.hop_ms) or args.hop_ms <= 0.0
    ):
        parser.error("--timestep-ms must be finite and positive.")
    if (
        args.png_tile_width is not None
        and args.png_tile_width > args.png_max_width
    ):
        parser.error("--png-tile-width cannot exceed --png-max-width.")
    if args.reference_mode == "fixed" and (
        args.reference_power is None
        or not math.isfinite(args.reference_power)
        or args.reference_power <= 0.0
    ):
        parser.error(
            "--reference-mode fixed requires a positive --reference-power."
        )
    agc_override_names = (
        "agc_target_dbfs",
        "agc_attack_ms",
        "agc_release_ms",
        "agc_max_gain_db",
        "agc_max_attenuation_db",
        "agc_gate_dbfs",
        "agc_peak_dbfs",
    )
    if not args.agc and any(
        getattr(args, name) is not None for name in agc_override_names
    ):
        parser.error("AGC parameter overrides require --agc.")
    override_names = (
        "sample_rate",
        "n_fft",
        "win_ms",
        "hop_ms",
        "n_mels",
        "fmin",
        "fmax",
        "top_db",
        "levels",
    )
    if args.preset == "all" and any(
        getattr(args, name) is not None for name in override_names
    ):
        parser.error("Preset overrides cannot be combined with --preset all.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if args.describe_presets:
        describe_presets()
    if args.describe_agc:
        describe_agc_profiles()
    if args.describe_presets or args.describe_agc:
        return 0

    assert args.input is not None
    input_path = args.input.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir.resolve()
    preset_names = list(PRESETS) if args.preset == "all" else [args.preset]
    summaries: list[dict[str, Any]] = []
    output_format = "csv" if args.no_png else args.output_format

    try:
        for preset_name in preset_names:
            preset = apply_overrides(PRESETS[preset_name], args)
            agc_config = resolve_agc_config(preset_name, args)
            metadata = convert_one(
                input_path=input_path,
                output_dir=output_dir,
                output_prefix=args.output_prefix,
                preset=preset,
                device=args.device,
                ffmpeg=args.ffmpeg,
                batch_frames=args.batch_frames,
                reference_mode=args.reference_mode,
                reference_percentile=args.reference_percentile,
                fixed_reference_power=args.reference_power,
                max_reference_values=args.max_reference_values,
                include_time_columns=args.include_time_columns,
                output_format=output_format,
                write_json=args.write_json,
                png_tile_width=args.png_tile_width,
                png_max_width=args.png_max_width,
                png_max_pixels=args.png_max_pixels,
                remove_dc=not args.keep_dc,
                agc_config=agc_config,
                force=args.force,
                quiet=args.quiet,
            )
            summaries.append(
                {
                    "preset": preset.name,
                    "csv": (
                        str(
                            output_paths(
                                input_path,
                                output_dir,
                                args.output_prefix,
                                preset.name,
                            )["csv"]
                        )
                        if output_format in {"csv", "both"}
                        else None
                    ),
                    "png": (
                        str(
                            output_paths(
                                input_path,
                                output_dir,
                                args.output_prefix,
                                preset.name,
                            )["png"]
                        )
                        if output_format in {"png", "both"}
                        else None
                    ),
                    "rows": metadata["csv_schema"]["rows"],
                    "mel_columns": metadata["csv_schema"]["mel_feature_columns"],
                    "columns_per_timestep": metadata["csv_schema"][
                        "mel_feature_columns"
                    ],
                    "states_per_cell": metadata["quantization"][
                        "states_per_mel_cell"
                    ],
                    "states_per_column": metadata["quantization"][
                        "states_per_mel_cell"
                    ],
                    "sample_rate": preset.sample_rate,
                    "samples_per_second": preset.sample_rate,
                    "nyquist_hz": preset.sample_rate / 2.0,
                    "n_fft": preset.n_fft,
                    "window_ms": preset.win_ms,
                    "window_samples": preset.win_length,
                    "hop_samples": preset.hop_length,
                    "timestep_ms": metadata["framing"]["timestep_ms"],
                    "timesteps_per_second": metadata["framing"][
                        "timesteps_per_second"
                    ],
                    "agc_enabled": metadata["preprocessing"]["agc"][
                        "enabled"
                    ],
                    "agc_profile": (
                        metadata["preprocessing"]["agc"].get("profile")
                    ),
                    "backend": metadata["runtime"]["backend"],
                }
            )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps({"outputs": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

