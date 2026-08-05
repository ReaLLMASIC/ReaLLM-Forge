#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: bash demo.sh INPUT [PRESET] [DEVICE] [OUTPUT_DIR] [OPTIONS]"
    echo
    echo "PRESET: small, medium, high, ultra, or max"
    echo "DEVICE: auto, cpu, cuda, or cuda:N"
    echo "OPTIONS:"
    echo "  --columns-per-timestep C   Alias: --n-mels"
    echo "  --states-per-column K      Alias: --levels"
    echo "  --timestep-ms MS           Alias: --hop-ms"
    echo "  --agc                      Enable the selected preset's AGC profile"
    echo "  --agc-target-dbfs DB"
    echo "  --agc-attack-ms MS"
    echo "  --agc-release-ms MS"
    echo "  --agc-max-gain-db DB"
    echo "  --agc-max-attenuation-db DB"
    echo "  --agc-gate-dbfs DB"
    echo "  --agc-peak-dbfs DB"
}

if [[ $# -lt 1 || "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    [[ $# -ge 1 ]] && exit 0
    exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
input="$1"
shift

preset="medium"
device="auto"
output_dir="mel_out"
if [[ $# -gt 0 && "$1" != --* ]]; then
    preset="$1"
    shift
fi
if [[ $# -gt 0 && "$1" != --* ]]; then
    device="$1"
    shift
fi
if [[ $# -gt 0 && "$1" != --* ]]; then
    output_dir="$1"
    shift
fi

columns=""
states=""
timestep_ms=""
agc_enabled=0
agc_override_seen=0
agc_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --columns-per-timestep|--n-mels)
            [[ $# -ge 2 ]] || {
                echo "$1 requires an integer value." >&2
                exit 1
            }
            if [[ -n "$columns" && "$columns" != "$2" ]]; then
                echo "$1=$2 conflicts with the earlier column value $columns." >&2
                exit 1
            fi
            columns="$2"
            shift 2
            ;;
        --states-per-column|--levels)
            [[ $# -ge 2 ]] || {
                echo "$1 requires an integer value." >&2
                exit 1
            }
            if [[ -n "$states" && "$states" != "$2" ]]; then
                echo "$1=$2 conflicts with the earlier state value $states." >&2
                exit 1
            fi
            states="$2"
            shift 2
            ;;
        --timestep-ms|--hop-ms)
            [[ $# -ge 2 ]] || {
                echo "$1 requires a positive millisecond value." >&2
                exit 1
            }
            if [[ -n "$timestep_ms" && "$timestep_ms" != "$2" ]]; then
                echo "$1=$2 conflicts with the earlier timestep value $timestep_ms." >&2
                exit 1
            fi
            timestep_ms="$2"
            shift 2
            ;;
        --agc)
            agc_enabled=1
            shift
            ;;
        --agc-target-dbfs|--agc-attack-ms|--agc-release-ms|\
        --agc-max-gain-db|--agc-max-attenuation-db|\
        --agc-gate-dbfs|--agc-peak-dbfs)
            [[ $# -ge 2 ]] || {
                echo "$1 requires a numeric value." >&2
                exit 1
            }
            agc_override_seen=1
            agc_args+=("$1" "$2")
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "$input" ]]; then
    echo "Input does not exist: $input" >&2
    exit 1
fi

case "$preset" in
    small|medium|high|ultra|max) ;;
    *)
        echo "Preset must be small, medium, high, ultra, or max." >&2
        exit 1
        ;;
esac

case "$preset" in
    small) base_columns=20; base_states=16 ;;
    medium) base_columns=40; base_states=64 ;;
    high) base_columns=80; base_states=256 ;;
    ultra) base_columns=160; base_states=1024 ;;
    max) base_columns=320; base_states=4096 ;;
esac

if [[ ! "$device" =~ ^(auto|cpu|cuda|cuda:[0-9]+)$ ]]; then
    echo "Device must be auto, cpu, cuda, or cuda:N." >&2
    exit 1
fi

if [[ -n "$columns" ]]; then
    if [[ ! "$columns" =~ ^[0-9]{1,4}$ ]] \
        || (( 10#$columns < 1 || 10#$columns > 4096 )); then
        echo "Columns per timestep must be an integer in [1, 4096]." >&2
        exit 1
    fi
fi
if [[ -n "$states" ]]; then
    if [[ ! "$states" =~ ^[0-9]{1,5}$ ]] \
        || (( 10#$states < 2 || 10#$states > 65536 )); then
        echo "States per column must be an integer in [2, 65536]." >&2
        exit 1
    fi
fi
if [[ -n "$timestep_ms" ]] \
    && [[ ! "$timestep_ms" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]; then
    echo "Timestep must be a positive finite number of milliseconds." >&2
    exit 1
fi
if (( agc_override_seen == 1 && agc_enabled == 0 )); then
    echo "AGC expert overrides require --agc." >&2
    exit 1
fi

filename="${input##*/}"
stem="${filename%.*}"
if [[ -z "$stem" ]]; then
    stem="$filename"
fi

output_prefix="$stem"
forward_args=()
if [[ -n "$columns" || -n "$states" ]]; then
    effective_columns="${columns:-$base_columns}"
    effective_states="${states:-$base_states}"
    output_prefix="$stem.c$effective_columns.k$effective_states"
fi
if [[ -n "$columns" ]]; then
    forward_args+=(--columns-per-timestep "$columns")
fi
if [[ -n "$states" ]]; then
    forward_args+=(--states-per-column "$states")
fi
if [[ -n "$timestep_ms" ]]; then
    timestep_label="${timestep_ms//E/e}"
    output_prefix="$output_prefix.dt${timestep_label}ms"
    forward_args+=(--timestep-ms "$timestep_ms")
fi
if (( agc_enabled == 1 )); then
    output_prefix="$output_prefix.agc"
    forward_args+=(--agc)
    forward_args+=("${agc_args[@]}")
fi

csv_path="$output_dir/$output_prefix.$preset.mel.csv"
png_path="$output_dir/$output_prefix.$preset.mel.png"
csv_audio="$output_dir/$output_prefix.$preset.from_csv.wav"
png_audio="$output_dir/$output_prefix.$preset.from_png.wav"

python3 "$script_dir/audio_to_token_mel.py" "$input" \
    --preset "$preset" \
    --output-format both \
    --output-dir "$output_dir" \
    --output-prefix "$output_prefix" \
    --device "$device" \
    "${forward_args[@]}" \
    --force

python3 "$script_dir/token_mel_to_audio.py" "$csv_path" \
    --output "$csv_audio" \
    --device "$device" \
    --force

python3 "$script_dir/token_mel_to_audio.py" "$png_path" \
    --output "$png_audio" \
    --device "$device" \
    --force

echo
echo "Created:"
echo "  CSV:       $csv_path"
echo "  PNG:       $png_path"
echo "  CSV audio: $csv_audio"
echo "  PNG audio: $png_audio"
