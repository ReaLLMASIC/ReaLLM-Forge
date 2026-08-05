#!/usr/bin/env python3
"""
Reconstruct best-effort WAV audio from one self-describing mel CSV or PNG.

No JSON sidecar is required.  Version-2 CSV files contain compact metadata in a
comment after the header.  QMel-PNG version 2 stores exact state IDs as palette
indices or on an exact 16-bit grayscale lattice and carries the same metadata
in both iTXt and a checksummed pixel header.

The result cannot equal the source waveform: mel projection, saturation,
quantization, mono downmixing, and discarded phase are intrinsically lossy.
This script uses nonnegative mel inversion followed by a custom Griffin-Lim
implementation matching audio_to_token_mel.py's framing convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import struct
import sys
import tempfile
import wave
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from audio_to_token_mel import (
    CSV_METADATA_PREFIX,
    PNG_METADATA_KEY,
    PNG_PIXEL_MAGIC,
    PRESETS,
    Preset,
    StoreConsistentValue,
    compact_json,
    encode_scaled_u16_states,
    make_mel_filterbank,
    periodic_hann,
    validate_preset,
)


VERSION = "2.4.0"
MAX_EMBEDDED_METADATA_BYTES = 1_000_000
AGC_LOSS_WARNING = (
    "Source was normalized with lossy AGC; original absolute level and "
    "loudness envelope cannot be recovered because the gain envelope was "
    "not stored."
)


@dataclass
class MelContainer:
    states: np.ndarray
    metadata: dict[str, Any]
    source_kind: str
    metadata_origin: str
    warnings: list[str]


def canonical_state_crc(states: np.ndarray, levels: int) -> int:
    if levels <= 256:
        canonical = np.ascontiguousarray(states, dtype=np.uint8)
    else:
        canonical = np.ascontiguousarray(states, dtype="<u2")
    return zlib.crc32(canonical.tobytes(order="C")) & 0xFFFFFFFF


def parse_crc(value: str | int) -> int:
    if isinstance(value, int):
        return value
    return int(value, 16)


def bounded_zlib_decompress(
    compressed: bytes,
    maximum_output_bytes: int = MAX_EMBEDDED_METADATA_BYTES,
) -> bytes:
    decoder = zlib.decompressobj()
    output = decoder.decompress(compressed, maximum_output_bytes + 1)
    if (
        len(output) > maximum_output_bytes
        or decoder.unconsumed_tail
        or not decoder.eof
    ):
        raise ValueError("QMel metadata exceeds the decompression safety limit.")
    output += decoder.flush()
    if len(output) > maximum_output_bytes:
        raise ValueError("QMel metadata exceeds the decompression safety limit.")
    return output


def validate_roundtrip_metadata(metadata: dict[str, Any]) -> None:
    """Reject metadata that this inverse cannot safely or faithfully apply."""
    format_name = metadata.get("format")
    version = int(metadata.get("version", -1))
    if not (
        (format_name == "bounded_quantized_mel_v2" and version == 2)
        or (format_name == "metadata_free_best_effort" and version == 0)
    ):
        raise ValueError(
            f"Unsupported mel container format/version: {format_name!r}/{version}."
        )
    shape = metadata["shape"]
    waveform = metadata["waveform"]
    stft = metadata["stft"]
    mel = metadata["mel"]
    quantizer = metadata["quantizer"]

    timesteps = int(shape["timesteps"])
    bands = int(shape["bands"])
    sample_rate = int(waveform["sample_rate"])
    sample_count = int(waveform["decoded_sample_count"])
    n_fft = int(stft["n_fft"])
    win_length = int(stft["win_length"])
    hop_length = int(stft["hop_length"])
    n_mels = int(mel["n_mels"])
    levels = int(quantizer["levels"])
    integer_min = int(quantizer.get("integer_min", 0))
    integer_max = int(quantizer.get("integer_max", levels - 1))

    if shape.get("axis_order") != "TC":
        raise ValueError("Only timesteps-by-columns (TC) metadata is supported.")
    if timesteps < 1 or bands < 1 or bands > 4096:
        raise ValueError("Metadata shape dimensions must be positive.")
    if sample_rate < 1 or sample_rate > 768_000 or sample_count < 1:
        raise ValueError("Metadata sample rate and sample count must be positive.")
    if (
        win_length < 2
        or hop_length < 1
        or n_fft < win_length
        or n_fft > 262_144
    ):
        raise ValueError("Metadata contains an invalid STFT size or hop.")
    if hop_length > win_length // 2:
        raise ValueError(
            "Metadata hop length exceeds half the centered window."
        )
    if math.ceil(sample_count / hop_length) != timesteps:
        raise ValueError(
            "Metadata sample count, hop length, and timestep count disagree."
        )
    if n_mels != bands:
        raise ValueError("Metadata mel-band count disagrees with its shape.")
    fmin = float(mel["fmin"])
    fmax = float(mel["fmax"])
    if not (
        math.isfinite(fmin)
        and math.isfinite(fmax)
        and 0.0 <= fmin < fmax <= sample_rate / 2.0 + 1.0e-9
    ):
        raise ValueError("Metadata contains an invalid mel frequency range.")
    if levels < 2 or levels > 65_536:
        raise ValueError("Metadata quantizer levels must be in [2,65536].")
    if integer_min != 0 or integer_max != levels - 1:
        raise ValueError("Only contiguous zero-based quantizer states are supported.")
    db_min = float(quantizer["db_min"])
    db_max = float(quantizer["db_max"])
    reference = float(quantizer["reference_power"])
    normalizer = float(stft["power_normalizer"])
    if not (
        math.isfinite(db_min)
        and math.isfinite(db_max)
        and db_min < db_max
    ):
        raise ValueError("Metadata contains an invalid dB interval.")
    if not math.isfinite(reference) or reference <= 0.0:
        raise ValueError("Metadata reference power must be positive and finite.")
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("Metadata STFT power normalizer must be positive.")
    expected_conventions = {
        "window": "periodic_hann",
        "frame_center": "t*hop",
        "padding": "constant_zero",
        "fft_zero_padding": "right",
    }
    for key, expected in expected_conventions.items():
        if stft.get(key) != expected:
            raise ValueError(
                f"Unsupported STFT {key}={stft.get(key)!r}; expected {expected!r}."
            )
    if (
        mel.get("scale") != "HTK"
        or mel.get("filter_shape") != "triangular"
        or mel.get("filter_normalization") != "unit_sum"
    ):
        raise ValueError("Only HTK, unit-sum mel metadata is supported.")

    preprocessing = metadata.get("preprocessing")
    if preprocessing is not None:
        if not isinstance(preprocessing, dict):
            raise ValueError("Metadata preprocessing must be an object.")
        agc = preprocessing.get("agc")
        if agc is not None:
            validate_agc_metadata(agc)
            if (
                agc.get("enabled") is True
                and int(agc["control_frame_samples"]) != hop_length
            ):
                raise ValueError(
                    "Metadata AGC control frame must equal the STFT hop length."
                )
            if agc.get("enabled") is True:
                expected_control_ms = 1000.0 * hop_length / sample_rate
                if not math.isclose(
                    float(agc["control_frame_ms"]),
                    expected_control_ms,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-9,
                ):
                    raise ValueError(
                        "Metadata AGC control-frame milliseconds disagree "
                        "with its sample rate and hop length."
                    )


def validate_agc_metadata(agc: Any) -> None:
    """Validate optional v2.3 AGC provenance without requiring it for v2.2."""
    if not isinstance(agc, dict):
        raise ValueError("Metadata preprocessing.agc must be an object.")
    enabled = agc.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("Metadata preprocessing.agc.enabled must be boolean.")
    if not enabled:
        return

    if agc.get("algorithm") != "causal_block_rms_db_v1":
        raise ValueError("Metadata contains an unsupported AGC algorithm.")
    profile = agc.get("profile")
    if not isinstance(profile, str) or not profile:
        raise ValueError("Metadata AGC profile must be a nonempty string.")
    if agc.get("gain_envelope_stored") is not False:
        raise ValueError("AGC metadata must declare that no gain envelope is stored.")
    if agc.get("reversible") is not False:
        raise ValueError("AGC metadata must declare the preprocessing irreversible.")
    if agc.get("reconstruction_domain") != "agc_normalized":
        raise ValueError("Metadata contains an unsupported AGC reconstruction domain.")
    if agc.get("limiter") != "hard_peak_ceiling":
        raise ValueError("Metadata contains an unsupported AGC limiter.")

    numeric_fields = (
        "target_dbfs",
        "control_frame_ms",
        "attack_ms",
        "release_ms",
        "max_gain_db",
        "max_attenuation_db",
        "gate_dbfs",
        "peak_ceiling_dbfs",
        "initial_gain_db",
    )
    values: dict[str, float] = {}
    for field in numeric_fields:
        value = agc.get(field)
        if isinstance(value, bool):
            raise ValueError(f"Metadata AGC {field} must be finite.")
        try:
            values[field] = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Metadata AGC {field} must be finite.") from error
        if not math.isfinite(values[field]):
            raise ValueError(f"Metadata AGC {field} must be finite.")

    control_frame_samples = agc.get("control_frame_samples")
    if (
        isinstance(control_frame_samples, bool)
        or not isinstance(control_frame_samples, int)
        or control_frame_samples < 1
    ):
        raise ValueError(
            "Metadata AGC control_frame_samples must be a positive integer."
        )
    if values["control_frame_ms"] <= 0.0:
        raise ValueError("Metadata AGC control_frame_ms must be positive.")
    if not 0.1 <= values["attack_ms"] <= 10_000.0:
        raise ValueError("Metadata AGC attack must be in [0.1,10000] ms.")
    if not values["attack_ms"] <= values["release_ms"] <= 60_000.0:
        raise ValueError(
            "Metadata AGC release must be in [attack_ms,60000] ms."
        )
    if not (
        0.0 <= values["max_gain_db"] <= 60.0
        and 0.0 <= values["max_attenuation_db"] <= 60.0
    ):
        raise ValueError("Metadata AGC gain limits must be in [0,60] dB.")
    if not (
        -120.0
        <= values["gate_dbfs"]
        < values["target_dbfs"]
        < values["peak_ceiling_dbfs"]
        <= 0.0
    ):
        raise ValueError(
            "Metadata AGC gate, target, and peak ceiling are inconsistent."
        )


def verify_states(
    states: np.ndarray,
    metadata: dict[str, Any],
    *,
    verify_crc: bool = True,
) -> np.ndarray:
    validate_roundtrip_metadata(metadata)
    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError("State matrix must have shape timesteps x mel_bands.")
    if not np.isfinite(states).all():
        raise ValueError("State matrix contains NaN or infinity.")
    if np.issubdtype(states.dtype, np.integer):
        states_i64 = states.astype(np.int64, copy=False)
    else:
        rounded = np.rint(states)
        if not np.allclose(states, rounded, rtol=0.0, atol=1.0e-7):
            raise ValueError("State matrix contains non-integer values.")
        states_i64 = rounded.astype(np.int64)

    shape = metadata["shape"]
    levels = int(metadata["quantizer"]["levels"])
    expected = (int(shape["timesteps"]), int(shape["bands"]))
    if states_i64.shape != expected:
        raise ValueError(
            f"State shape {states_i64.shape} does not match metadata {expected}."
        )
    if int(states_i64.min()) < 0 or int(states_i64.max()) >= levels:
        raise ValueError(
            f"State IDs must be in [0, {levels - 1}], received "
            f"[{states_i64.min()}, {states_i64.max()}]."
        )

    integrity = metadata.get("integrity", {})
    expected_crc = integrity.get("state_crc32")
    if verify_crc and expected_crc is not None:
        actual_crc = canonical_state_crc(states_i64, levels)
        if actual_crc != parse_crc(expected_crc):
            raise ValueError(
                f"State CRC mismatch: expected {expected_crc}, "
                f"calculated {actual_crc:08x}."
            )
    compact_dtype = np.uint8 if levels <= 256 else np.uint16
    return states_i64.astype(compact_dtype, copy=False)


def read_csv_container(path: Path) -> MelContainer:
    metadata: dict[str, Any] | None = None
    header_line: str | None = None
    header_index: int | None = None

    with path.open("r", encoding="utf-8", newline="") as input_file:
        for line_index, line in enumerate(input_file):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(CSV_METADATA_PREFIX):
                metadata = json.loads(stripped[len(CSV_METADATA_PREFIX) :])
                if header_line is not None:
                    break
                continue
            if stripped.startswith("#"):
                continue
            if header_line is None:
                header_line = line
                header_index = line_index
            else:
                # In v2 the metadata comment precedes every data row. Reaching
                # data here establishes the metadata-free fallback promptly.
                break

    if header_line is None or header_index is None:
        raise ValueError("CSV has no header.")
    header = next(csv.reader([header_line]))
    feature_indices = [
        index
        for index, name in enumerate(header)
        if re.fullmatch(r"mel_\d+_q", name.strip())
    ]
    warnings: list[str] = []
    if metadata is None:
        try:
            states = np.loadtxt(
                path,
                delimiter=",",
                comments="#",
                skiprows=header_index + 1,
                usecols=feature_indices or None,
                ndmin=2,
                dtype=np.int64,
            )
        except ValueError as error:
            raise ValueError(
                f"Could not parse rectangular numeric CSV data: {error}"
            ) from error
        warnings.append(
            "CSV has no embedded v2 metadata; preset/configuration must be inferred."
        )
        return MelContainer(
            states=states,
            metadata={},
            source_kind="csv",
            metadata_origin="none",
            warnings=warnings,
        )

    validate_roundtrip_metadata(metadata)
    expected_rows = int(metadata["shape"]["timesteps"])
    expected_columns = int(metadata["shape"]["bands"])
    levels = int(metadata["quantizer"]["levels"])
    if len(feature_indices) != expected_columns:
        raise ValueError(
            f"CSV header contains {len(feature_indices)} mel columns; "
            f"metadata declares {expected_columns}."
        )
    if expected_rows * expected_columns > path.stat().st_size:
        raise ValueError("CSV metadata declares more state cells than the file can hold.")
    compact_dtype = np.uint8 if levels <= 256 else np.uint16
    states = np.empty(
        (expected_rows, expected_columns), dtype=compact_dtype
    )
    row_count = 0
    pending_lines: list[str] = []

    def consume_pending() -> None:
        nonlocal row_count
        if not pending_lines:
            return
        try:
            values = np.loadtxt(
                pending_lines,
                delimiter=",",
                usecols=feature_indices,
                ndmin=2,
                dtype=np.int64,
            )
        except ValueError as error:
            raise ValueError(
                f"Could not parse rectangular integer CSV data: {error}"
            ) from error
        end = row_count + values.shape[0]
        if end > expected_rows:
            raise ValueError("CSV contains more data rows than its metadata declares.")
        if (
            values.shape[1] != expected_columns
            or int(values.min()) < 0
            or int(values.max()) >= levels
        ):
            raise ValueError(
                f"CSV state rows must have {expected_columns} integers in "
                f"[0,{levels - 1}]."
            )
        states[row_count:end] = values
        row_count = end
        pending_lines.clear()

    with path.open("r", encoding="utf-8", newline="") as input_file:
        for line_index, line in enumerate(input_file):
            if line_index <= header_index:
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            pending_lines.append(line)
            if len(pending_lines) >= 4096:
                consume_pending()
    consume_pending()
    if row_count != expected_rows:
        raise ValueError(
            f"CSV contains {row_count} data rows; metadata declares "
            f"{expected_rows}."
        )
    states = verify_states(states, metadata)
    return MelContainer(
        states=states,
        metadata=metadata,
        source_kind="csv",
        metadata_origin="csv_comment",
        warnings=warnings,
    )


def decode_paired_pixel_header(
    pixels: np.ndarray,
) -> tuple[dict[str, Any], int]:
    flat = np.asarray(pixels).reshape(-1)
    minimum_pixels = 2 * (len(PNG_PIXEL_MAGIC) + 8)
    if flat.size < minimum_pixels:
        raise ValueError("PNG is too small to contain a QMel pixel header.")

    first_pairs = flat[:minimum_pixels].reshape(-1, 2)
    if np.any(first_pairs > 255) or np.any(first_pairs < 0):
        raise ValueError("QMel pixel header contains non-byte values.")
    if not np.all(
        first_pairs[:, 0].astype(np.uint32)
        + first_pairs[:, 1].astype(np.uint32)
        == 255
    ):
        raise ValueError("QMel pixel-header complement check failed.")
    fixed = first_pairs[:, 0].astype(np.uint8).tobytes()
    if fixed[: len(PNG_PIXEL_MAGIC)] != PNG_PIXEL_MAGIC:
        raise ValueError("QMel pixel-header magic is absent.")
    compressed_length, compressed_crc = struct.unpack(
        ">II", fixed[len(PNG_PIXEL_MAGIC) : len(PNG_PIXEL_MAGIC) + 8]
    )
    if compressed_length < 1 or compressed_length > MAX_EMBEDDED_METADATA_BYTES:
        raise ValueError("QMel compressed metadata length is unreasonable.")

    raw_length = len(PNG_PIXEL_MAGIC) + 8 + compressed_length
    encoded_length = 2 * raw_length
    if encoded_length > flat.size:
        raise ValueError("PNG is truncated inside its QMel pixel header.")
    pairs = flat[:encoded_length].reshape(-1, 2)
    if np.any(pairs > 255) or np.any(pairs < 0):
        raise ValueError("QMel metadata header contains non-byte values.")
    if not np.all(
        pairs[:, 0].astype(np.uint32)
        + pairs[:, 1].astype(np.uint32)
        == 255
    ):
        raise ValueError("QMel metadata complement check failed.")
    raw = pairs[:, 0].astype(np.uint8).tobytes()
    compressed = raw[len(PNG_PIXEL_MAGIC) + 8 :]
    if zlib.crc32(compressed) & 0xFFFFFFFF != compressed_crc:
        raise ValueError("QMel compressed-metadata CRC mismatch.")
    try:
        metadata = json.loads(
            bounded_zlib_decompress(compressed).decode("utf-8")
        )
    except (zlib.error, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("QMel pixel metadata cannot be decoded.") from error
    header_rows = int(math.ceil(encoded_length / pixels.shape[1]))
    return metadata, header_rows


def decode_scaled_u16_states(
    encoded_pixels: np.ndarray,
    levels: int,
) -> np.ndarray:
    values = np.asarray(encoded_pixels)
    if np.any(values < 0) or np.any(values > 65_535):
        raise ValueError("16-bit QMel payload contains out-of-range pixels.")
    decoded = (
        values.astype(np.uint64) * np.uint64(levels - 1)
        + np.uint64(32_767)
    ) // np.uint64(65_535)
    compact_dtype = np.uint8 if levels <= 256 else np.uint16
    states = decoded.astype(compact_dtype)
    if not np.array_equal(
        encode_scaled_u16_states(states, levels),
        values.astype(np.uint16),
    ):
        raise ValueError(
            "16-bit QMel payload contains pixels outside the state lattice."
        )
    return states


def extract_png_states(
    pixels: np.ndarray,
    metadata: dict[str, Any],
    data_y0: int,
) -> np.ndarray:
    shape = metadata["shape"]
    frame_count = int(shape["timesteps"])
    n_mels = int(shape["bands"])
    width = int(metadata["png"]["tile_width"])
    if width != pixels.shape[1]:
        raise ValueError(
            f"PNG width {pixels.shape[1]} does not match metadata width {width}."
        )
    strips = int(math.ceil(frame_count / width))
    required_height = data_y0 + strips * n_mels
    if pixels.shape[0] < required_height:
        raise ValueError(
            f"PNG height {pixels.shape[0]} is below required {required_height}."
        )

    levels = int(metadata["quantizer"]["levels"])
    pixel_codec = metadata["png"]["state_pixel_codec"]
    compact_dtype = np.uint8 if levels <= 256 else np.uint16
    states = np.empty((frame_count, n_mels), dtype=compact_dtype)
    for strip_index in range(strips):
        start = strip_index * width
        end = min(frame_count, start + width)
        rows = slice(
            data_y0 + strip_index * n_mels,
            data_y0 + (strip_index + 1) * n_mels,
        )
        payload = pixels[rows, : end - start][::-1, :].T
        if pixel_codec == "palette_index_equals_state":
            if np.any(payload < 0) or np.any(payload >= levels):
                raise ValueError("Indexed QMel payload has an invalid state ID.")
            decoded = payload.astype(compact_dtype)
        elif pixel_codec == "scaled_u16_state_lattice_v1":
            decoded = decode_scaled_u16_states(payload, levels)
        else:
            raise ValueError(f"Unsupported QMel state pixel codec: {pixel_codec!r}.")
        states[start:end] = decoded
    return states


def read_png_container(
    path: Path,
    maximum_pixels: int = 80_000_000,
) -> MelContainer:
    try:
        from PIL import Image  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "PNG input requires Pillow: python -m pip install pillow"
        ) from error

    with path.open("rb") as raw_file:
        header = raw_file.read(24)
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise ValueError("Input does not contain a valid PNG IHDR.")
    width, height = struct.unpack(">II", header[16:24])
    pixel_count = width * height
    if width < 1 or height < 1 or pixel_count > maximum_pixels:
        raise ValueError(
            f"PNG dimensions {width}x{height} exceed "
            f"--png-max-pixels={maximum_pixels:,}."
        )

    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = maximum_pixels
    try:
        with Image.open(path) as image:
            if image.size != (width, height):
                raise ValueError("PNG dimensions disagree with its IHDR.")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Animated PNG is unsupported.")
            ancillary_text = image.info.get(PNG_METADATA_KEY)
            mode = image.mode
            pixels = np.asarray(image)
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit

    warnings: list[str] = []
    exact_modes = {"P", "I;16", "I;16L", "I;16B", "I"}
    if mode not in exact_modes:
        warnings.append(
            f"PNG mode {mode!r} is not an exact QMel integer mode."
        )
        return MelContainer(
            states=np.asarray(pixels),
            metadata={},
            source_kind="png",
            metadata_origin="none",
            warnings=warnings,
        )

    pixel_metadata: dict[str, Any] | None = None
    header_rows: int | None = None
    try:
        pixel_metadata, header_rows = decode_paired_pixel_header(pixels)
    except ValueError as error:
        warnings.append(str(error))

    text_metadata: dict[str, Any] | None = None
    if ancillary_text:
        try:
            text_metadata = json.loads(ancillary_text)
        except json.JSONDecodeError as error:
            warnings.append(f"Invalid {PNG_METADATA_KEY} iTXt JSON: {error}")

    if pixel_metadata is not None and text_metadata is not None:
        if compact_json(pixel_metadata) != compact_json(text_metadata):
            raise ValueError("PNG iTXt metadata disagrees with pixel metadata.")
    metadata = pixel_metadata or text_metadata
    if metadata is None:
        return MelContainer(
            states=np.asarray(pixels),
            metadata={},
            source_kind="png",
            metadata_origin="none",
            warnings=warnings,
        )
    validate_roundtrip_metadata(metadata)
    declared_mode = metadata["png"].get("mode")
    pixel_codec = metadata["png"].get("state_pixel_codec")
    if pixel_codec == "palette_index_equals_state" and mode != "P":
        raise ValueError(
            f"QMel metadata declares indexed states but PNG mode is {mode!r}."
        )
    if pixel_codec == "scaled_u16_state_lattice_v1" and mode == "P":
        raise ValueError("QMel metadata declares 16-bit states in a palette PNG.")
    levels = int(metadata["quantizer"]["levels"])
    if pixel_codec == "palette_index_equals_state" and levels > 256:
        raise ValueError("Palette QMel cannot represent more than 256 states.")
    if pixel_codec == "scaled_u16_state_lattice_v1" and levels <= 256:
        raise ValueError("16-bit QMel codec is inconsistent with its state count.")
    if declared_mode == "P" and mode != "P":
        raise ValueError("QMel PNG mode disagrees with its embedded metadata.")

    if header_rows is None:
        n_mels = int(metadata["shape"]["bands"])
        frames = int(metadata["shape"]["timesteps"])
        strips = int(math.ceil(frames / pixels.shape[1]))
        payload_rows = strips * n_mels
        header_rows = pixels.shape[0] - payload_rows
        if header_rows < 0:
            raise ValueError("PNG is shorter than its declared state payload.")
        if header_rows == 0:
            header_rows = 0
            warnings.append("Pixel header was stripped; using surviving iTXt.")
        else:
            warnings.append(
                "Pixel header is damaged; locating the payload from image "
                "height and surviving iTXt."
            )

    states = extract_png_states(np.asarray(pixels), metadata, header_rows)
    states = verify_states(states, metadata)
    return MelContainer(
        states=states,
        metadata=metadata,
        source_kind="png",
        metadata_origin=(
            "pixel_header+iTXt"
            if pixel_metadata is not None and text_metadata is not None
            else "pixel_header"
            if pixel_metadata is not None
            else "iTXt"
        ),
        warnings=warnings,
    )


def preset_from_name_or_columns(
    input_path: Path,
    columns: int,
    requested: str | None,
) -> Preset:
    if requested:
        preset = PRESETS[requested]
        if preset.n_mels != columns:
            raise ValueError(
                f"--preset {requested} expects {preset.n_mels} columns, "
                f"but the input has {columns}."
            )
        return preset
    profile_pattern = "|".join(re.escape(name) for name in PRESETS)
    filename_match = re.search(
        rf"\.({profile_pattern})\.", input_path.name
    )
    if filename_match:
        preset = PRESETS[filename_match.group(1)]
        if preset.n_mels == columns:
            return preset
    candidates = [preset for preset in PRESETS.values() if preset.n_mels == columns]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        "Metadata-free input is ambiguous. Pass --preset or use a v2 "
        "self-describing CSV/PNG."
    )


def metadata_from_preset(
    preset: Preset,
    frame_count: int,
    reference_power: float,
    sample_count: int | None,
    duration: float | None,
) -> dict[str, Any]:
    if duration is not None:
        inferred_samples = int(round(duration * preset.sample_rate))
    elif sample_count is not None:
        inferred_samples = sample_count
    else:
        inferred_samples = frame_count * preset.hop_length
    window = periodic_hann(preset.win_length)
    return {
        "format": "metadata_free_best_effort",
        "version": 0,
        "profile": preset.name,
        "shape": {
            "timesteps": frame_count,
            "bands": preset.n_mels,
            "axis_order": "TC",
        },
        "waveform": {
            "sample_rate": preset.sample_rate,
            "decoded_sample_count": inferred_samples,
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
            "power_normalizer": max(float(window.sum()) / 2.0, 1.0e-12) ** 2,
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
            "reference_mode": "metadata_free_assumption",
            "silent": False,
        },
        "integrity": {},
    }


def fallback_preset_from_options(
    preset_name: str | None,
    columns_per_timestep: int | None,
    states_per_column: int | None,
    timestep_ms: float | None,
) -> Preset | None:
    manual_override = (
        columns_per_timestep is not None
        or states_per_column is not None
        or timestep_ms is not None
    )
    if preset_name is None:
        if manual_override:
            raise ValueError(
                "Metadata-free manual geometry/timestep overrides require "
                "--preset to provide "
                "the remaining sample-rate, STFT, mel-range, and dB settings."
            )
        return None

    replacements: dict[str, Any] = {}
    if columns_per_timestep is not None:
        replacements["n_mels"] = columns_per_timestep
    if states_per_column is not None:
        replacements["levels"] = states_per_column
    if timestep_ms is not None:
        replacements["hop_ms"] = timestep_ms
    preset = replace(PRESETS[preset_name], **replacements)
    validate_preset(preset)
    return preset


def resolve_legacy_container(
    container: MelContainer,
    input_path: Path,
    preset_name: str | None,
    columns_per_timestep: int | None,
    states_per_column: int | None,
    timestep_ms: float | None,
    reference_power: float,
    frame_count: int | None,
    sample_count: int | None,
    duration: float | None,
) -> MelContainer:
    if container.metadata:
        if preset_name and container.metadata.get("profile") != preset_name:
            raise ValueError(
                "--preset conflicts with embedded metadata; remove the override."
            )
        embedded_columns = int(container.metadata["shape"]["bands"])
        embedded_states = int(container.metadata["quantizer"]["levels"])
        if (
            columns_per_timestep is not None
            and columns_per_timestep != embedded_columns
        ):
            raise ValueError(
                "--columns-per-timestep conflicts with embedded metadata "
                f"({columns_per_timestep} requested, {embedded_columns} stored)."
            )
        if (
            states_per_column is not None
            and states_per_column != embedded_states
        ):
            raise ValueError(
                "--states-per-column conflicts with embedded metadata "
                f"({states_per_column} requested, {embedded_states} stored)."
            )
        if timestep_ms is not None:
            sample_rate = int(container.metadata["waveform"]["sample_rate"])
            embedded_hop = int(container.metadata["stft"]["hop_length"])
            requested_hop = int(round(sample_rate * timestep_ms / 1000.0))
            if requested_hop < 1:
                raise ValueError(
                    "--timestep-ms rounds to fewer than one sample at the "
                    f"embedded sample rate ({sample_rate} Hz)."
                )
            if requested_hop != embedded_hop:
                embedded_timestep_ms = 1000.0 * embedded_hop / sample_rate
                raise ValueError(
                    "--timestep-ms conflicts with embedded metadata "
                    f"({timestep_ms:g} ms requested -> {requested_hop} samples, "
                    f"{embedded_timestep_ms:g} ms / {embedded_hop} samples stored)."
                )
        return container

    fallback_preset = fallback_preset_from_options(
        preset_name,
        columns_per_timestep,
        states_per_column,
        timestep_ms,
    )
    states = np.asarray(container.states)
    if container.source_kind == "png":
        if states.ndim != 2:
            raise ValueError("Metadata-free PNG must be a 2-D integer raster.")
        if fallback_preset is None:
            raise ValueError(
                "A non-QMel PNG requires --preset and should be an unlabelled "
                "raw integer state raster in QMel strip orientation, not a "
                "plotted screenshot."
            )
        preset = fallback_preset
        if states.shape[0] % preset.n_mels != 0:
            raise ValueError(
                "Metadata-free PNG height is not a multiple of preset mel bands."
            )
        strips = states.shape[0] // preset.n_mels
        inferred_frames = frame_count or strips * states.shape[1]
        if inferred_frames > strips * states.shape[1]:
            raise ValueError("--frame-count exceeds PNG payload capacity.")
        compact_dtype = np.uint8 if preset.levels <= 256 else np.uint16
        recovered = np.empty(
            (inferred_frames, preset.n_mels), dtype=compact_dtype
        )
        for strip_index in range(strips):
            start = strip_index * states.shape[1]
            end = min(inferred_frames, start + states.shape[1])
            if end <= start:
                break
            rows = slice(
                strip_index * preset.n_mels,
                (strip_index + 1) * preset.n_mels,
            )
            payload = states[rows, : end - start][::-1, :].T
            if np.any(payload < 0) or np.any(payload >= preset.levels):
                raise ValueError(
                    "Metadata-free PNG contains a direct state ID outside "
                    f"[0, {preset.levels - 1}]."
                )
            recovered[start:end] = payload
        states = recovered
    preset = fallback_preset or preset_from_name_or_columns(
        input_path, int(states.shape[1]), None
    )
    if int(states.shape[1]) != preset.n_mels:
        raise ValueError(
            f"The input has {states.shape[1]} columns, but the requested "
            f"geometry expects {preset.n_mels}."
        )
    metadata = metadata_from_preset(
        preset,
        int(states.shape[0]),
        reference_power,
        sample_count,
        duration,
    )
    states = verify_states(states, metadata, verify_crc=False)
    container.states = states
    container.metadata = metadata
    container.metadata_origin = "inferred"
    container.warnings.append(
        "Reconstruction settings were inferred; amplitude and duration may differ."
    )
    return container


def dequantize_mel_power(
    states: np.ndarray,
    metadata: dict[str, Any],
    floor_mode: str,
) -> np.ndarray:
    quantizer = metadata["quantizer"]
    levels = int(quantizer["levels"])
    db_min = float(quantizer["db_min"])
    db_max = float(quantizer["db_max"])
    reference = float(quantizer["reference_power"])
    db = db_min + states.astype(np.float64) * (
        (db_max - db_min) / (levels - 1)
    )
    mel_power = reference * np.power(10.0, db / 10.0)
    if floor_mode == "zero":
        mel_power[states == 0] = 0.0
    if bool(quantizer.get("silent", False)):
        mel_power.fill(0.0)
    return mel_power.astype(np.float32)


def multiplicative_mel_inverse_numpy(
    mel_power: np.ndarray,
    filterbank: np.ndarray,
    iterations: int,
    batch_frames: int,
) -> np.ndarray:
    frame_count = mel_power.shape[0]
    frequency_bins = filterbank.shape[1]
    output = np.empty((frame_count, frequency_bins), dtype=np.float32)
    h = filterbank.astype(np.float64)
    column_mass = h.sum(axis=0)[None, :]
    epsilon = 1.0e-18
    for start in range(0, frame_count, batch_frames):
        end = min(frame_count, start + batch_frames)
        target = mel_power[start:end].astype(np.float64)
        numerator = target @ h
        estimate = np.where(
            column_mass > epsilon,
            numerator / np.maximum(column_mass, epsilon),
            0.0,
        )
        estimate = np.maximum(estimate, epsilon)
        for _ in range(iterations):
            denominator = ((estimate @ h.T) @ h) + epsilon
            estimate *= numerator / denominator
        output[start:end] = np.maximum(estimate, 0.0).astype(np.float32)
    return output


def pinv_mel_inverse(
    mel_power: np.ndarray,
    filterbank: np.ndarray,
    batch_frames: int,
) -> np.ndarray:
    inverse = np.linalg.pinv(filterbank.T, rcond=1.0e-5)
    output = np.empty(
        (mel_power.shape[0], filterbank.shape[1]), dtype=np.float32
    )
    for start in range(0, mel_power.shape[0], batch_frames):
        end = min(mel_power.shape[0], start + batch_frames)
        estimate = mel_power[start:end] @ inverse
        output[start:end] = np.maximum(estimate, 0.0).astype(np.float32)
    return output


def stft_exact_numpy(
    audio: np.ndarray,
    frame_count: int,
    win_length: int,
    hop_length: int,
    n_fft: int,
    window: np.ndarray,
) -> np.ndarray:
    half = win_length // 2
    total_length = (frame_count - 1) * hop_length + win_length
    padded = np.zeros((total_length,), dtype=np.float32)
    available = max(0, min(audio.size, total_length - half))
    if available:
        padded[half : half + available] = audio[:available]
    stride = padded.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(frame_count, win_length),
        strides=(hop_length * stride, stride),
        writeable=False,
    )
    return np.fft.rfft(
        frames * window[None, :], n=n_fft, axis=1
    ).astype(np.complex64)


def overlap_add_plan_numpy(
    frame_count: int,
    win_length: int,
    hop_length: int,
    window: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame_offsets = (
        np.arange(frame_count, dtype=np.int64)[:, None] * hop_length
    )
    window_offsets = np.arange(win_length, dtype=np.int64)[None, :]
    indices = (frame_offsets + window_offsets).reshape(-1)
    total_length = (frame_count - 1) * hop_length + win_length
    denominator = np.bincount(
        indices,
        weights=np.broadcast_to(
            window.astype(np.float64) ** 2,
            (frame_count, win_length),
        ).reshape(-1),
        minlength=total_length,
    )
    return indices, denominator


def istft_exact_numpy(
    spectrum: np.ndarray,
    sample_count: int,
    win_length: int,
    hop_length: int,
    n_fft: int,
    window: np.ndarray,
    overlap_indices: np.ndarray | None = None,
    denominator: np.ndarray | None = None,
) -> np.ndarray:
    frame_count = spectrum.shape[0]
    total_length = (frame_count - 1) * hop_length + win_length
    frames = np.fft.irfft(spectrum, n=n_fft, axis=1)[:, :win_length]
    window64 = window.astype(np.float64)
    if overlap_indices is None or denominator is None:
        overlap_indices, denominator = overlap_add_plan_numpy(
            frame_count, win_length, hop_length, window
        )
    output = np.bincount(
        overlap_indices,
        weights=(frames * window64[None, :]).reshape(-1),
        minlength=total_length,
    )
    valid = denominator > 1.0e-12
    output[valid] /= denominator[valid]
    output[~valid] = 0.0
    half = win_length // 2
    cropped = np.zeros((sample_count,), dtype=np.float32)
    available = max(0, min(sample_count, total_length - half))
    if available:
        cropped[:available] = output[half : half + available].astype(np.float32)
    return cropped


def griffin_lim_numpy(
    magnitude: np.ndarray,
    sample_count: int,
    win_length: int,
    hop_length: int,
    n_fft: int,
    window: np.ndarray,
    iterations: int,
    seed: int,
    initial_phase: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    phase = np.exp(
        2j * np.pi * rng.random(magnitude.shape, dtype=np.float32)
    ).astype(np.complex64)
    if initial_phase is not None:
        count = min(initial_phase.shape[0], phase.shape[0])
        if initial_phase.shape[1] != phase.shape[1]:
            raise ValueError("Initial phase frequency-bin count is inconsistent.")
        phase[:count] = initial_phase[-count:]
    if not np.any(magnitude > 0.0):
        return np.zeros((sample_count,), dtype=np.float32), phase
    spectrum = magnitude.astype(np.float32) * phase
    audio = np.zeros((sample_count,), dtype=np.float32)
    overlap_indices, denominator = overlap_add_plan_numpy(
        magnitude.shape[0], win_length, hop_length, window
    )
    for _ in range(iterations):
        audio = istft_exact_numpy(
            spectrum,
            sample_count,
            win_length,
            hop_length,
            n_fft,
            window,
            overlap_indices,
            denominator,
        )
        rebuilt = stft_exact_numpy(
            audio,
            magnitude.shape[0],
            win_length,
            hop_length,
            n_fft,
            window,
        )
        phase = rebuilt / np.maximum(np.abs(rebuilt), 1.0e-12)
        spectrum = magnitude * phase
    return (
        istft_exact_numpy(
            spectrum,
            sample_count,
            win_length,
            hop_length,
            n_fft,
            window,
            overlap_indices,
            denominator,
        ),
        phase,
    )


def resolve_torch_device(request: str) -> tuple[Any | None, str]:
    if request == "cpu":
        return None, "numpy:cpu"
    if request != "auto" and not request.startswith("cuda"):
        raise ValueError("--device must be auto, cpu, cuda, or cuda:N.")
    try:
        import torch  # type: ignore
    except ImportError as error:
        if request.startswith("cuda"):
            raise RuntimeError("CUDA requested but PyTorch is not installed.") from error
        return None, "numpy:cpu"
    if not torch.cuda.is_available():
        if request.startswith("cuda"):
            raise RuntimeError("CUDA requested but no CUDA device is available.")
        return None, "numpy:cpu"
    device = "cuda:0" if request in {"auto", "cuda"} else request
    torch.empty((1,), device=device)
    return torch, f"torch:{device}"


def multiplicative_mel_inverse_torch(
    torch: Any,
    device: str,
    mel_power: np.ndarray,
    filterbank: np.ndarray,
    iterations: int,
    batch_frames: int,
) -> np.ndarray:
    h = torch.as_tensor(filterbank, dtype=torch.float32, device=device)
    column_mass = h.sum(dim=0, keepdim=True)
    output = np.empty(
        (mel_power.shape[0], filterbank.shape[1]), dtype=np.float32
    )
    with torch.inference_mode():
        for start in range(0, mel_power.shape[0], batch_frames):
            end = min(mel_power.shape[0], start + batch_frames)
            target = torch.as_tensor(
                mel_power[start:end], dtype=torch.float32, device=device
            )
            numerator = target @ h
            estimate = torch.where(
                column_mass > 1.0e-12,
                numerator / column_mass.clamp_min(1.0e-12),
                torch.zeros_like(numerator),
            ).clamp_min(1.0e-12)
            for _ in range(iterations):
                denominator = ((estimate @ h.T) @ h).clamp_min(1.0e-12)
                estimate.mul_(numerator / denominator)
            output[start:end] = estimate.clamp_min_(0).cpu().numpy()
    return output


def griffin_lim_torch(
    torch: Any,
    device: str,
    magnitude_numpy: np.ndarray,
    sample_count: int,
    win_length: int,
    hop_length: int,
    n_fft: int,
    window_numpy: np.ndarray,
    iterations: int,
    seed: int,
    initial_phase: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    import torch.nn.functional as functional  # type: ignore

    magnitude = torch.as_tensor(
        magnitude_numpy, dtype=torch.float32, device=device
    )
    frame_count = magnitude.shape[0]
    window = torch.as_tensor(window_numpy, dtype=torch.float32, device=device)
    total_length = (frame_count - 1) * hop_length + win_length
    half = win_length // 2
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    def inverse(spectrum: Any) -> Any:
        frames = torch.fft.irfft(spectrum, n=n_fft, dim=1)[:, :win_length]
        columns = (frames * window[None, :]).T.unsqueeze(0)
        output = functional.fold(
            columns,
            output_size=(1, total_length),
            kernel_size=(1, win_length),
            stride=(1, hop_length),
        ).reshape(-1)
        norm_columns = (
            window.square()[:, None]
            .expand(win_length, frame_count)
            .unsqueeze(0)
        )
        norm = functional.fold(
            norm_columns,
            output_size=(1, total_length),
            kernel_size=(1, win_length),
            stride=(1, hop_length),
        ).reshape(-1)
        output = output / norm.clamp_min(1.0e-12)
        result = torch.zeros((sample_count,), dtype=torch.float32, device=device)
        available = max(0, min(sample_count, total_length - half))
        if available:
            result[:available] = output[half : half + available]
        return result

    def analysis(audio: Any) -> Any:
        padded = torch.zeros(
            (total_length,), dtype=torch.float32, device=device
        )
        available = max(0, min(sample_count, total_length - half))
        if available:
            padded[half : half + available] = audio[:available]
        frames = padded.unfold(0, win_length, hop_length)
        return torch.fft.rfft(
            frames * window[None, :], n=n_fft, dim=1
        )

    angles = 2.0 * math.pi * torch.rand(
        magnitude.shape, generator=generator, device=device
    )
    phase = torch.polar(torch.ones_like(angles), angles)
    if initial_phase is not None:
        if initial_phase.shape[1] != phase.shape[1]:
            raise ValueError("Initial phase frequency-bin count is inconsistent.")
        count = min(initial_phase.shape[0], phase.shape[0])
        phase[:count] = torch.as_tensor(
            initial_phase[-count:], dtype=phase.dtype, device=device
        )
    if not bool(torch.any(magnitude > 0)):
        return (
            np.zeros((sample_count,), dtype=np.float32),
            phase.cpu().numpy().astype(np.complex64),
        )
    spectrum = magnitude * phase
    with torch.inference_mode():
        for _ in range(iterations):
            audio = inverse(spectrum)
            rebuilt = analysis(audio)
            phase = rebuilt / rebuilt.abs().clamp_min(1.0e-12)
            spectrum = magnitude * phase
        return (
            inverse(spectrum).cpu().numpy().astype(np.float32),
            phase.cpu().numpy().astype(np.complex64),
        )


def write_pcm16_wav(
    output_path: Path,
    audio: np.ndarray,
    sample_rate: int,
    peak_target: float,
    block_samples: int = 4_000_000,
) -> float:
    original_peak = 0.0
    for start in range(0, int(audio.size), block_samples):
        block = np.nan_to_num(
            np.asarray(audio[start : start + block_samples], dtype=np.float32),
            copy=True,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if block.size:
            original_peak = max(original_peak, float(np.max(np.abs(block))))
    scale = (
        peak_target / original_peak
        if original_peak > 0.0 and peak_target > 0.0
        else 1.0
    )
    with wave.open(str(output_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        for start in range(0, int(audio.size), block_samples):
            block = np.nan_to_num(
                np.asarray(
                    audio[start : start + block_samples], dtype=np.float32
                ),
                copy=True,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            np.multiply(block, scale, out=block)
            np.clip(block, -1.0, 1.0, out=block)
            pcm = np.rint(block * 32767.0).astype("<i2")
            output.writeframes(pcm.tobytes(order="C"))
    return original_peak


def invert_state_chunk(
    states: np.ndarray,
    metadata: dict[str, Any],
    filterbank: np.ndarray,
    torch: Any | None,
    device: str,
    mel_inverse: str,
    mel_iterations: int,
    mel_batch_frames: int,
    griffin_lim_iterations: int,
    floor_mode: str,
    sample_count: int,
    seed: int,
    initial_phase: np.ndarray | None,
    error_frames: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    stft = metadata["stft"]
    mel_power = dequantize_mel_power(states, metadata, floor_mode)
    if mel_inverse == "pinv":
        linear_power = pinv_mel_inverse(
            mel_power, filterbank, mel_batch_frames
        )
    elif torch is not None:
        linear_power = multiplicative_mel_inverse_torch(
            torch,
            device,
            mel_power,
            filterbank,
            mel_iterations,
            mel_batch_frames,
        )
    else:
        linear_power = multiplicative_mel_inverse_numpy(
            mel_power,
            filterbank,
            mel_iterations,
            mel_batch_frames,
        )

    magnitude = np.sqrt(
        np.maximum(linear_power, 0.0)
        * float(stft["power_normalizer"])
    ).astype(np.float32)
    window = periodic_hann(int(stft["win_length"]))
    if torch is not None:
        audio, phase = griffin_lim_torch(
            torch,
            device,
            magnitude,
            sample_count,
            int(stft["win_length"]),
            int(stft["hop_length"]),
            int(stft["n_fft"]),
            window,
            griffin_lim_iterations,
            seed,
            initial_phase,
        )
    else:
        audio, phase = griffin_lim_numpy(
            magnitude,
            sample_count,
            int(stft["win_length"]),
            int(stft["hop_length"]),
            int(stft["n_fft"]),
            window,
            griffin_lim_iterations,
            seed,
            initial_phase,
        )
    error_frames = max(0, min(error_frames, mel_power.shape[0]))
    target = mel_power[:error_frames]
    residual = (linear_power[:error_frames] @ filterbank.T) - target
    error_square = float(
        np.sum(residual.astype(np.float64) ** 2, dtype=np.float64)
    )
    target_square = float(
        np.sum(target.astype(np.float64) ** 2, dtype=np.float64)
    )
    return audio, phase, error_square, target_square


def reconstruct(
    container: MelContainer,
    output_path: Path,
    device_request: str,
    mel_inverse: str,
    mel_iterations: int,
    mel_batch_frames: int,
    griffin_lim_iterations: int,
    griffin_lim_chunk_frames: int,
    griffin_lim_overlap_frames: int,
    floor_mode: str,
    seed: int,
    peak: float,
) -> dict[str, Any]:
    metadata = container.metadata
    states = container.states
    mel = metadata["mel"]
    stft = metadata["stft"]
    waveform = metadata["waveform"]
    sample_rate = int(waveform["sample_rate"])
    sample_count = int(waveform["decoded_sample_count"])
    hop_length = int(stft["hop_length"])
    win_length = int(stft["win_length"])
    frame_count = int(states.shape[0])
    preprocessing = metadata.get("preprocessing", {})
    agc_metadata = (
        preprocessing.get("agc")
        if isinstance(preprocessing, dict)
        else None
    )
    agc_applied = bool(
        isinstance(agc_metadata, dict)
        and agc_metadata.get("enabled") is True
    )
    agc_profile = (
        str(agc_metadata.get("profile"))
        if agc_applied and isinstance(agc_metadata, dict)
        else None
    )
    if agc_applied and AGC_LOSS_WARNING not in container.warnings:
        container.warnings.append(AGC_LOSS_WARNING)
    filterbank, _, _ = make_mel_filterbank(
        sample_rate,
        int(stft["n_fft"]),
        int(mel["n_mels"]),
        float(mel["fmin"]),
        float(mel["fmax"]),
    )
    torch, backend = resolve_torch_device(device_request)
    device = backend.split(":", 1)[1] if torch is not None else "cpu"

    use_chunks = (
        griffin_lim_chunk_frames > 0
        and frame_count > griffin_lim_chunk_frames
    )
    error_square = 0.0
    target_square = 0.0
    chunk_count = 1

    if not use_chunks:
        audio, _, error_square, target_square = invert_state_chunk(
            states=states,
            metadata=metadata,
            filterbank=filterbank,
            torch=torch,
            device=device,
            mel_inverse=mel_inverse,
            mel_iterations=mel_iterations,
            mel_batch_frames=mel_batch_frames,
            griffin_lim_iterations=griffin_lim_iterations,
            floor_mode=floor_mode,
            sample_count=sample_count,
            seed=seed,
            initial_phase=None,
            error_frames=frame_count,
        )
        original_peak = write_pcm16_wav(
            output_path, audio, sample_rate, peak
        )
    else:
        minimum_overlap = int(math.ceil(win_length / hop_length))
        if griffin_lim_overlap_frames < minimum_overlap:
            raise ValueError(
                "--griffin-lim-overlap-frames must be at least "
                f"{minimum_overlap} for this window and hop."
            )
        if griffin_lim_chunk_frames <= 2 * griffin_lim_overlap_frames:
            raise ValueError(
                "--griffin-lim-chunk-frames must exceed twice the overlap."
            )
        stride = (
            griffin_lim_chunk_frames - griffin_lim_overlap_frames
        )
        with tempfile.TemporaryDirectory(
            prefix="token_mel_to_audio_"
        ) as temp_dir:
            raw_output = Path(temp_dir) / "reconstructed.f32le"
            synthesized = np.memmap(
                raw_output,
                dtype=np.float32,
                mode="w+",
                shape=(sample_count,),
            )
            synthesized[:] = 0.0
            start = 0
            previous_end_sample = 0
            phase_carry: np.ndarray | None = None
            chunk_index = 0
            while start < frame_count:
                end = min(
                    frame_count, start + griffin_lim_chunk_frames
                )
                start_sample = start * hop_length
                local_samples = min(
                    sample_count - start_sample,
                    (end - start) * hop_length,
                )
                unique_frames = (
                    frame_count - start
                    if end == frame_count
                    else stride
                )
                audio, phase, chunk_error, chunk_target = invert_state_chunk(
                    states=states[start:end],
                    metadata=metadata,
                    filterbank=filterbank,
                    torch=torch,
                    device=device,
                    mel_inverse=mel_inverse,
                    mel_iterations=mel_iterations,
                    mel_batch_frames=mel_batch_frames,
                    griffin_lim_iterations=griffin_lim_iterations,
                    floor_mode=floor_mode,
                    sample_count=local_samples,
                    seed=seed + chunk_index,
                    initial_phase=phase_carry,
                    error_frames=unique_frames,
                )
                write_end = start_sample + audio.size
                overlap_samples = max(
                    0, min(previous_end_sample - start_sample, audio.size)
                )
                if overlap_samples:
                    position = (
                        np.arange(overlap_samples, dtype=np.float32) + 1.0
                    ) / (overlap_samples + 1.0)
                    fade_in = 0.5 - 0.5 * np.cos(np.pi * position)
                    previous = np.asarray(
                        synthesized[
                            start_sample : start_sample + overlap_samples
                        ],
                        dtype=np.float32,
                    )
                    synthesized[
                        start_sample : start_sample + overlap_samples
                    ] = (
                        previous * (1.0 - fade_in)
                        + audio[:overlap_samples] * fade_in
                    )
                if overlap_samples < audio.size:
                    synthesized[
                        start_sample + overlap_samples : write_end
                    ] = audio[overlap_samples:]
                previous_end_sample = max(previous_end_sample, write_end)
                error_square += chunk_error
                target_square += chunk_target
                carry_count = min(
                    griffin_lim_overlap_frames, phase.shape[0]
                )
                phase_carry = phase[-carry_count:].copy()
                chunk_index += 1
                print(
                    f"[inverse] frames {end}/{frame_count}",
                    file=sys.stderr,
                )
                if end == frame_count:
                    break
                start += stride
            chunk_count = chunk_index
            synthesized.flush()
            original_peak = write_pcm16_wav(
                output_path, synthesized, sample_rate, peak
            )
            del synthesized

    relative_mel_error = math.sqrt(
        error_square / max(target_square, 1.0e-24)
    )
    return {
        "output": str(output_path),
        "source_kind": container.source_kind,
        "metadata_origin": container.metadata_origin,
        "timesteps": int(states.shape[0]),
        "mel_columns": int(states.shape[1]),
        "columns_per_timestep": int(states.shape[1]),
        "states_per_column": int(metadata["quantizer"]["levels"]),
        "timestep_ms": 1000.0 * hop_length / sample_rate,
        "timesteps_per_second": sample_rate / hop_length,
        "sample_rate": sample_rate,
        "samples": sample_count,
        "duration_seconds": (
            sample_count / sample_rate
        ),
        "backend": backend,
        "mel_inverse": mel_inverse,
        "mel_inverse_iterations": (
            mel_iterations if mel_inverse == "multiplicative" else None
        ),
        "griffin_lim_iterations": griffin_lim_iterations,
        "griffin_lim_chunk_frames": (
            griffin_lim_chunk_frames if use_chunks else None
        ),
        "griffin_lim_overlap_frames": (
            griffin_lim_overlap_frames if use_chunks else None
        ),
        "reconstruction_chunks": chunk_count,
        "floor_mode": floor_mode,
        "relative_mel_reprojection_error": relative_mel_error,
        "pre_normalization_waveform_peak": original_peak,
        "agc_applied": agc_applied,
        "agc_profile": agc_profile,
        "warnings": container.warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct best-effort WAV audio from one self-describing "
            "token-mel CSV or QMel-PNG."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {VERSION}"
    )
    parser.add_argument("input", type=Path, help="Input .csv or .png file.")
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Output PCM16 WAV."
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or cuda:N."
    )
    parser.add_argument(
        "--mel-inverse",
        choices=["multiplicative", "pinv"],
        default="multiplicative",
        help="Nonnegative iterative inversion or faster pseudoinverse.",
    )
    parser.add_argument(
        "--mel-inverse-iters",
        type=int,
        default=32,
        help="Multiplicative mel-inversion iterations.",
    )
    parser.add_argument(
        "--mel-batch-frames",
        type=int,
        default=512,
        help="Frames per mel-inversion batch.",
    )
    parser.add_argument(
        "--griffin-lim-iters",
        type=int,
        default=64,
        help="Phase-recovery iterations.",
    )
    parser.add_argument(
        "--griffin-lim-chunk-frames",
        type=int,
        default=2048,
        help="Bound long phase recovery to this many frames; 0 uses the full file.",
    )
    parser.add_argument(
        "--griffin-lim-overlap-frames",
        type=int,
        default=32,
        help="Overlapped frames used for phase handoff and seam crossfading.",
    )
    parser.add_argument(
        "--floor-mode",
        choices=["zero", "db-floor"],
        default="zero",
        help="Decode q=0 as zero power or as the finite dB floor.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--peak",
        type=float,
        default=0.0,
        help="Normalize reconstructed waveform to this peak; 0 disables.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output WAV.",
    )
    parser.add_argument(
        "--png-max-pixels",
        type=int,
        default=80_000_000,
        help="Safety limit for a trusted standalone PNG raster.",
    )
    legacy = parser.add_argument_group("metadata-free best-effort fallback")
    legacy.add_argument(
        "--preset", choices=list(PRESETS)
    )
    legacy.add_argument(
        "--columns-per-timestep",
        "--n-mels",
        dest="columns_per_timestep",
        type=int,
        action=StoreConsistentValue,
        metavar="C",
        help=(
            "Expected tokenized columns; with metadata-free input this "
            "overrides the selected preset. --n-mels is an exact alias."
        ),
    )
    legacy.add_argument(
        "--states-per-column",
        "--levels",
        dest="states_per_column",
        type=int,
        action=StoreConsistentValue,
        metavar="K",
        help=(
            "Expected legal states per column; with metadata-free input this "
            "overrides the selected preset. --levels is an exact alias."
        ),
    )
    legacy.add_argument(
        "--timestep-ms",
        "--hop-ms",
        dest="timestep_ms",
        type=float,
        action=StoreConsistentValue,
        metavar="MS",
        help=(
            "Duration represented by each row; with metadata-free input this "
            "overrides the selected preset. Equality with embedded metadata is "
            "checked after rounding to integer samples. --hop-ms is an exact "
            "alias."
        ),
    )
    legacy.add_argument(
        "--reference-power",
        type=float,
        default=1.0,
        help="Assumed reference when metadata is absent.",
    )
    legacy.add_argument(
        "--frame-count",
        type=int,
        help="Valid frame count for a metadata-free tiled PNG.",
    )
    legacy.add_argument(
        "--sample-count",
        type=int,
        help="Original decoded sample count when metadata is absent.",
    )
    legacy.add_argument(
        "--duration",
        type=float,
        help="Original duration in seconds when metadata is absent.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input does not exist: {args.input}")
    if args.output.suffix.lower() != ".wav":
        parser.error("Output must use a .wav suffix.")
    if args.output.exists() and not args.force:
        parser.error("Output already exists; pass --force to overwrite it.")
    if args.mel_inverse_iters < 0:
        parser.error("--mel-inverse-iters must be nonnegative.")
    if args.mel_batch_frames < 1:
        parser.error("--mel-batch-frames must be positive.")
    if args.griffin_lim_iters < 0:
        parser.error("--griffin-lim-iters must be nonnegative.")
    if args.griffin_lim_chunk_frames < 0:
        parser.error("--griffin-lim-chunk-frames must be nonnegative.")
    if args.griffin_lim_overlap_frames < 0:
        parser.error("--griffin-lim-overlap-frames must be nonnegative.")
    if args.png_max_pixels < 1:
        parser.error("--png-max-pixels must be positive.")
    if (
        args.columns_per_timestep is not None
        and not 1 <= args.columns_per_timestep <= 4096
    ):
        parser.error("--columns-per-timestep must be in [1, 4096].")
    if (
        args.states_per_column is not None
        and not 2 <= args.states_per_column <= 65_536
    ):
        parser.error("--states-per-column must be in [2, 65536].")
    if args.timestep_ms is not None and (
        not math.isfinite(args.timestep_ms) or args.timestep_ms <= 0.0
    ):
        parser.error("--timestep-ms must be positive and finite.")
    if not math.isfinite(args.peak) or not 0.0 <= args.peak <= 1.0:
        parser.error("--peak must be finite and in [0,1].")
    if (
        not math.isfinite(args.reference_power)
        or args.reference_power <= 0.0
    ):
        parser.error("--reference-power must be positive and finite.")
    if args.frame_count is not None and args.frame_count < 1:
        parser.error("--frame-count must be positive.")
    if args.sample_count is not None and args.sample_count < 1:
        parser.error("--sample-count must be positive.")
    if args.duration is not None and (
        not math.isfinite(args.duration) or args.duration <= 0.0
    ):
        parser.error("--duration must be positive and finite.")

    temporary_output: Path | None = None
    try:
        suffix = args.input.suffix.lower()
        if suffix == ".csv":
            container = read_csv_container(args.input)
        elif suffix == ".png":
            container = read_png_container(
                args.input, maximum_pixels=args.png_max_pixels
            )
        else:
            raise ValueError("Input must be .csv or .png.")
        container = resolve_legacy_container(
            container,
            args.input,
            args.preset,
            args.columns_per_timestep,
            args.states_per_column,
            args.timestep_ms,
            args.reference_power,
            args.frame_count,
            args.sample_count,
            args.duration,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{args.output.stem}.",
            suffix=".tmp.wav",
            dir=args.output.parent,
            delete=False,
        ) as temporary_file:
            temporary_output = Path(temporary_file.name)
        result = reconstruct(
            container=container,
            output_path=temporary_output,
            device_request=args.device,
            mel_inverse=args.mel_inverse,
            mel_iterations=args.mel_inverse_iters,
            mel_batch_frames=args.mel_batch_frames,
            griffin_lim_iterations=args.griffin_lim_iters,
            griffin_lim_chunk_frames=args.griffin_lim_chunk_frames,
            griffin_lim_overlap_frames=args.griffin_lim_overlap_frames,
            floor_mode=args.floor_mode,
            seed=args.seed,
            peak=args.peak,
        )
        os.replace(temporary_output, args.output)
        temporary_output = None
        result["output"] = str(args.output)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        if temporary_output is not None:
            try:
                temporary_output.unlink()
            except FileNotFoundError:
                pass
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
