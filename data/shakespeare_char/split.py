#!/usr/bin/env python3
"""Split audio into FLAC segments using an approximate uncompressed-size budget."""
import argparse
import json
import math
from pathlib import Path
import subprocess


def split_and_convert_wav(input_file, target_size_mb=10):
    source = Path(input_file).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not math.isfinite(target_size_mb) or target_size_mb <= 0:
        raise ValueError('target-size-mib must be positive and finite')
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                             '-show_entries', 'stream=sample_rate,channels,bits_per_sample',
                             '-of', 'json', str(source)], check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)['streams'][0]
    bits = int(stream.get('bits_per_sample') or 16)
    bitrate = int(stream['sample_rate']) * int(stream['channels']) * bits
    seconds = target_size_mb * 1024 * 1024 * 8 / bitrate
    pattern = source.with_name(source.stem + '_part_%03d.flac')
    # Existing output segments are not silently overwritten.
    if list(source.parent.glob(source.stem + '_part_*.flac')):
        raise ValueError('Output segments already exist; choose a clean input/output location')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'warning', '-n', '-i', str(source),
                    '-f', 'segment', '-segment_time', str(seconds), '-c:a', 'flac', str(pattern)], check=True)
    outputs = sorted(source.parent.glob(source.stem + '_part_*.flac'))
    limit = target_size_mb * 1024 * 1024
    oversized = [path.name for path in outputs if path.stat().st_size > limit]
    if oversized:
        raise ValueError(f'Approximate bitrate estimate exceeded the size target: {oversized}. Use a smaller duration budget.')
    print(f'Created {len(outputs)} segments; largest is {max(path.stat().st_size for path in outputs)} bytes')
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file')
    parser.add_argument('--target-size-mib', type=float, default=10.)
    args = parser.parse_args()
    try:
        split_and_convert_wav(args.input_file, args.target_size_mib)
    except (OSError, ValueError, KeyError, IndexError, subprocess.CalledProcessError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
