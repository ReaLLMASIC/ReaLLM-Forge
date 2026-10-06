#!/usr/bin/env python3
"""Validated, bounded-memory adapters for the v2 mel container format."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import sys
import tempfile
import wave
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'data' / 'mel_spectrogram'))
from audio_to_token_mel import CSV_METADATA_PREFIX, compact_json
from token_mel_to_audio import canonical_state_crc, read_csv_container, validate_roundtrip_metadata


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def dtype_for_vocab(vocab):
    if not 2 <= vocab <= 65536:
        raise ValueError('Mel levels must be in [2, 65536]')
    # Match the existing train.py/sample.py storage convention, including 65536.
    return ('I', 'uint32') if vocab >= 65536 else ('H', 'uint16')


def mel_columns(header):
    names = [name.strip() for name in header]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate CSV column names')
    columns = [(i, name) for i, name in enumerate(names) if re.fullmatch(r'mel_\d+_q', name)]
    if not columns or [name for _, name in columns] != [f'mel_{i:03d}_q' for i in range(len(columns))]:
        raise ValueError('Mel columns must be unique, contiguous, and in canonical band order')
    if any(name not in {'frame_index', 'time_s'} and not re.fullmatch(r'mel_\d+_q', name) for name in names):
        raise ValueError('Unknown column in mel CSV')
    return columns


def inspect_mel_csv(path):
    metadata = header = None
    with Path(path).open(newline='', encoding='utf-8') as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(CSV_METADATA_PREFIX):
                if metadata is not None:
                    raise ValueError(f'Duplicate metadata: {path}')
                metadata = json.loads(stripped[len(CSV_METADATA_PREFIX):])
            elif stripped.startswith('#'):
                continue
            elif header is None:
                header = next(csv.reader([line]))
            else:
                break
    if header is None or metadata is None:
        raise ValueError(f'A self-describing v2 mel CSV is required: {path}')
    if metadata.get('format') != 'bounded_quantized_mel_v2' or metadata.get('version') != 2:
        raise ValueError(f'Unsupported mel format: {path}')
    validate_roundtrip_metadata(metadata)
    cols = mel_columns(header)
    if len(cols) != metadata['shape']['bands']:
        raise ValueError(f'Header and metadata band counts disagree: {path}')
    dtype_for_vocab(int(metadata['quantizer']['levels']))
    return header, cols, metadata


def iter_state_batches(path, batch_rows=4096):
    if batch_rows < 1:
        raise ValueError('batch_rows must be positive')
    header, columns, metadata = inspect_mel_csv(path)
    levels = int(metadata['quantizer']['levels'])
    buffer = np.empty((batch_rows, len(columns)), dtype=np.uint16)
    count = 0
    header_seen = False
    with Path(path).open(newline='', encoding='utf-8') as stream:
        for row in csv.reader(stream):
            if not row or not any(value.strip() for value in row) or row[0].lstrip().startswith('#'):
                continue
            if not header_seen:
                header_seen = True
                continue
            if len(row) != len(header):
                raise ValueError(f'Wrong number of fields in {path}')
            values = [int(row[i]) for i, _ in columns]
            if min(values) < 0 or max(values) >= levels:
                raise ValueError(f'States outside [0,{levels - 1}] in {path}')
            buffer[count] = values
            count += 1
            if count == batch_rows:
                yield buffer
                count = 0
        if count:
            yield buffer[:count]


def validate_csv(path):
    header, columns, metadata = inspect_mel_csv(path)
    levels = int(metadata['quantizer']['levels'])
    dtype = np.uint8 if levels <= 256 else np.dtype('<u2')
    crc = rows = 0
    nonzero = False
    for batch in iter_state_batches(path):
        crc = zlib.crc32(batch.astype(dtype).tobytes(), crc)
        rows += len(batch)
        nonzero |= bool(np.any(batch))
    if rows != metadata['shape']['timesteps']:
        raise ValueError(f'Declared frame count does not match data: {path}')
    if f'{crc & 0xffffffff:08x}' != metadata.get('integrity', {}).get('state_crc32'):
        raise ValueError(f'State CRC mismatch: {path}')
    if metadata['quantizer'].get('silent') and nonzero:
        raise ValueError(f'Silent container contains nonzero states: {path}')
    return {'path': str(Path(path).resolve()), 'rows': rows, 'metadata': metadata,
            'columns': [name for _, name in columns], 'crc': crc, 'nonzero': nonzero}


def metadata_signature(metadata):
    quantizer = {key: value for key, value in metadata['quantizer'].items() if key != 'silent'}
    return compact_json({'format': metadata['format'], 'version': metadata['version'],
                         'bands': metadata['shape']['bands'], 'axis_order': metadata['shape']['axis_order'],
                         'sample_rate': metadata['waveform']['sample_rate'],
                         'channels': metadata['waveform']['channels'], 'stft': metadata['stft'],
                         'mel': metadata['mel'], 'quantizer': quantizer,
                         'preprocessing': metadata.get('preprocessing')})


def output_metadata(reference, rows, crc, nonzero):
    metadata = copy.deepcopy(reference)
    metadata.pop('mel_mc_int', None)
    metadata['shape']['timesteps'] = rows
    metadata['waveform']['decoded_sample_count'] = rows * int(metadata['stft']['hop_length'])
    metadata['csv']['time_columns_included'] = False
    metadata['integrity']['state_crc32'] = f'{crc & 0xffffffff:08x}'
    # A prompt/first track's silence flag never overrides generated states.
    metadata['quantizer']['silent'] = not nonzero
    validate_roundtrip_metadata(metadata)
    return metadata


def metadata_for_states(reference, states):
    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] != reference['shape']['bands']:
        raise ValueError('Output state dimensions disagree with the reference')
    levels = int(reference['quantizer']['levels'])
    if int(states.min()) < 0 or int(states.max()) >= levels:
        raise ValueError('Output state outside the reference vocabulary')
    return output_metadata(reference, len(states), canonical_state_crc(states, levels), bool(np.any(states)))


def write_mel_state_csv(path, columns, states, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        stream.write(CSV_METADATA_PREFIX + compact_json(metadata) + '\n')
        writer.writerows(states)


def read_mel_csv(path):
    header, cols, _ = inspect_mel_csv(path)
    container = read_csv_container(Path(path))
    return header, cols, container.states, container.metadata


def cmd_concat_csv(args):
    values = list(args.input_csvs)
    if args.input_list:
        values.extend(Path(args.input_list).read_text().splitlines())
    if args.input_json:
        values.extend(json.loads(Path(args.input_json).read_text()))
    inputs = [Path(value).resolve() for value in values if value]
    if not inputs or len(inputs) != len(set(inputs)):
        raise ValueError('Provide distinct input CSVs')
    output = Path(args.output_csv).resolve()
    if output in inputs:
        raise ValueError('Output must not overwrite an input CSV')
    infos = [validate_csv(path) for path in inputs]
    first = infos[0]
    if any(metadata_signature(info['metadata']) != metadata_signature(first['metadata']) for info in infos):
        raise ValueError('Incompatible mel transforms/reference powers; encode with one fixed reference')
    levels = first['metadata']['quantizer']['levels']
    dtype = np.uint8 if levels <= 256 else np.dtype('<u2')
    crc = 0
    for path in inputs:
        for batch in iter_state_batches(path):
            crc = zlib.crc32(batch.astype(dtype).tobytes(), crc)
    metadata = output_metadata(first['metadata'], sum(info['rows'] for info in infos), crc,
                               any(info['nonzero'] for info in infos))
    offset = 0
    records = []
    for info in infos:
        records.append({'path': info['path'], 'start': offset, 'stop': offset + info['rows']})
        offset += info['rows']
    metadata['mel_mc_int'] = {'recordings': records}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', newline='', encoding='utf-8', dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.writer(stream)
            writer.writerow(first['columns'])
            stream.write(CSV_METADATA_PREFIX + compact_json(metadata) + '\n')
            for path in inputs:
                for batch in iter_state_batches(path):
                    writer.writerows(batch)
            stream.close()
            validate_csv(temporary)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    if args.manifest_json:
        atomic_json(args.manifest_json, {'output_csv': str(output), 'total_frames': offset, 'inputs': records})
    print(output)


def check_ratio(ratio):
    if not math.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError('train_ratio must be finite and in (0, 1)')


def dataset_root(output_root):
    relative = Path(output_root)
    if relative.is_absolute() or '..' in relative.parts or relative == Path('.'):
        raise ValueError('output_root must be a relative folder under data/')
    root = (REPO_ROOT / 'data' / relative).resolve()
    if not root.is_relative_to((REPO_ROOT / 'data').resolve()):
        raise ValueError('output_root escapes data/')
    return root


def prepare_segments(splits, output_root, block_size, buffer_rows=4096, encoder=None):
    if block_size < 1 or buffer_rows < 1:
        raise ValueError('block_size and buffer_rows must be positive')
    paths = sorted({str(Path(segment['csv']).resolve()) for segments in splits.values() for segment in segments})
    infos = {path: validate_csv(path) for path in paths}
    if not infos or not splits.get('train') or not splits.get('val'):
        raise ValueError('Training and validation recordings are both required')
    reference = infos[paths[0]]['metadata']
    signature = metadata_signature(reference)
    if any(metadata_signature(info['metadata']) != signature for info in infos.values()):
        raise ValueError('All recordings must use identical mel transforms and a fixed shared reference')
    if reference['quantizer']['reference_mode'] != 'fixed':
        raise ValueError('Training preparation requires fixed reference mode; re-encode the audio')
    normalized = {}
    for split, segments in splits.items():
        normalized[split] = []
        for segment in segments:
            path = str(Path(segment['csv']).resolve())
            start, stop = int(segment.get('start', 0)), int(segment.get('stop', infos[path]['rows']))
            if not 0 <= start < stop <= infos[path]['rows'] or stop - start < block_size + 1:
                raise ValueError(f'{split}: {path} has {stop-start} usable frames; need at least {block_size+1}')
            normalized[split].append({'csv': path, 'start': start, 'stop': stop})
    # A single-recording temporal holdout needs nonoverlapping waveform support.
    guard = math.ceil(reference['stft']['win_length'] / reference['stft']['hop_length'])
    for left in normalized['train']:
        for right in normalized['val']:
            if left['csv'] == right['csv'] and not (left['stop'] + guard <= right['start'] or right['stop'] + guard <= left['start']):
                raise ValueError('Train/validation segments overlap or lack the STFT guard interval')
    identity = {'schema_version': 1, 'splits': normalized, 'signature': signature, 'encoder': encoder,
                'source_sha256': {path: file_sha256(path) for path in paths}}
    identifier = hashlib.sha256(compact_json(identity).encode()).hexdigest()[:24]
    out = dataset_root(output_root)
    generations = out / 'datasets'
    generations.mkdir(parents=True, exist_ok=True)
    destination = generations / identifier
    if destination.exists():
        prepared = json.loads((destination / 'prepared.json').read_text())
        if prepared['identity'] != identity or any(file_sha256(destination / name) != value for name, value in prepared['file_sha256'].items()):
            raise ValueError('An immutable prepared dataset has changed; use a new output root')
        manifest = prepared['manifest']
        atomic_json(out / 'manifest.json', manifest)
        return manifest
    temporary = Path(tempfile.mkdtemp(prefix='.preparing-', dir=generations))
    columns = infos[paths[0]]['columns']
    levels = int(reference['quantizer']['levels'])
    _, dtype_name = dtype_for_vocab(levels)
    dtype = np.dtype('<u4' if dtype_name == 'uint32' else '<u2')
    ranges, counts = {}, {}
    try:
        for name in columns:
            (temporary / name).mkdir()
        for split, segments in normalized.items():
            ranges[split] = []
            position = 0
            from contextlib import ExitStack
            with ExitStack() as stack:
                handles = [stack.enter_context((temporary / name / f'{split}.bin').open('wb')) for name in columns]
                for segment in segments:
                    start = position
                    offset = 0
                    for batch in iter_state_batches(segment['csv'], buffer_rows):
                        lo = max(0, segment['start'] - offset)
                        hi = min(len(batch), segment['stop'] - offset)
                        if hi > lo:
                            selected = batch[lo:hi].astype(dtype)
                            for i, handle in enumerate(handles):
                                handle.write(selected[:, i].tobytes())
                            position += hi-lo
                        offset += len(batch)
                        if offset >= segment['stop']:
                            break
                    ranges[split].append([start, position])
            counts[split] = position
        atomic_json(temporary / 'sequence_ranges.json', {'version': 1, 'ranges': ranges})
        names = []
        for i, name in enumerate(columns):
            metadata = {'tokenizer': 'csv_integer_range', 'vocab_size': levels, 'source_column': name,
                        'context_name': name, 'column_index': i, 'int_min': 0, 'int_max': levels-1,
                        'dtype': dtype_name, 'mel_mc_int': True, 'sequence_ranges_file': '../sequence_ranges.json'}
            with (temporary / name / 'meta.pkl').open('wb') as stream:
                pickle.dump(metadata, stream)
            names.append((destination / name).relative_to(REPO_ROOT / 'data').as_posix())
        manifest = {'schema_version': 1, 'tokenizer': 'mel_integer_range_multicontext_manifest',
                    'output_root': output_root, 'multicontext_datasets': names,
                    'train_rows': counts['train'], 'val_rows': counts['val'], 'rows': sum(counts.values()),
                    'columns': columns, 'vocab_size': levels, 'dtype': dtype_name, 'block_size': block_size,
                    'roundtrip_metadata': reference, 'encoder': encoder, 'splits': normalized,
                    'sequence_ranges': ranges, 'dataset_id': identifier}
        hashes = {path.relative_to(temporary).as_posix(): file_sha256(path) for path in temporary.rglob('*') if path.is_file()}
        atomic_json(temporary / 'prepared.json', {'identity': identity, 'manifest': manifest, 'file_sha256': hashes})
        os.rename(temporary, destination)
        atomic_json(out / 'manifest.json', manifest)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def cmd_prepare(args):
    check_ratio(args.train_ratio)
    info = validate_csv(args.input_csv)
    if args.states_per_column is not None and args.states_per_column != info['metadata']['quantizer']['levels']:
        raise ValueError('states_per_column disagrees with the source metadata')
    records = info['metadata'].get('mel_mc_int', {}).get('recordings')
    if records:
        raise ValueError('Use the folder pipeline to split recordings before concatenation/calibration')
    cut = int(info['rows'] * args.train_ratio)
    guard = math.ceil(info['metadata']['stft']['win_length'] / info['metadata']['stft']['hop_length'])
    splits = {'train': [{'csv': info['path'], 'start': 0, 'stop': cut}],
              'val': [{'csv': info['path'], 'start': cut + guard, 'stop': info['rows']}]}
    prepare_segments(splits, args.output_root, args.block_size, args.buffer_rows)
    print(dataset_root(args.output_root) / 'manifest.json')


def cmd_cut_prompt(args):
    if not math.isfinite(args.cutoff_s) or args.cutoff_s <= 0:
        raise ValueError('cutoff_s must be positive and finite')
    _, cols, states, metadata = read_mel_csv(Path(args.mel_csv))
    hop = int(metadata['stft']['hop_length'])
    sr = int(metadata['waveform']['sample_rate'])
    available = min(int(math.floor(args.cutoff_s * sr)), metadata['waveform']['decoded_sample_count'])
    if args.fully_observed:
        right = metadata['stft']['win_length'] - metadata['stft']['win_length']//2
        n = max(0, (available-right)//hop + 1)
    else:
        n = math.ceil(available / hop)
    n = min(len(states), n)
    if n < 1:
        raise ValueError('Prefix is too short for one fully observed mel frame')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    columns = [name for _, name in cols]
    with (out / 'prompt.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(states[:n])
    write_mel_state_csv(out / 'prompt.mel.csv', columns, states[:n], metadata_for_states(metadata, states[:n]))
    _, dtype = dtype_for_vocab(metadata['quantizer']['levels'])
    (out / 'bins').mkdir(exist_ok=True)
    files = []
    for i, name in enumerate(columns):
        path = out / 'bins' / f'{name}.bin'
        states[:n, i].astype('<u4' if dtype == 'uint32' else '<u2').tofile(path)
        files.append(str(path))
    atomic_json(out / 'prompt_manifest.json', {'csv': str(out/'prompt.csv'), 'start_files': files,
                'dtype': dtype, 'cutoff_s': args.cutoff_s, 'available_seconds': available/sr,
                'prompt_frames': n, 'generation_start_s': n*hop/sr,
                'lookahead_s': metadata['stft']['win_length']/2/sr})
    print(out / 'prompt.csv')


def cmd_wrap_csv(args):
    reference = read_csv_container(Path(args.reference_mel_csv)).metadata
    with Path(args.input_csv).open(newline='', encoding='utf-8') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        cols = mel_columns(header)
        states = np.asarray([[int(row[i]) for i, _ in cols] for row in reader if row], dtype=np.int64)
    metadata = metadata_for_states(reference, states)
    write_mel_state_csv(Path(args.output_csv), [name for _, name in cols], states, metadata)
    validate_csv(args.output_csv)
    print(args.output_csv)


def cmd_stitch(args):
    def read(path):
        with wave.open(str(path), 'rb') as stream:
            return stream.getparams(), stream.readframes(stream.getnframes())
    a, x = read(args.prompt_wav)
    b, y = read(args.continuation_wav)
    if a[:3] != b[:3] or a.comptype != b.comptype:
        raise ValueError('WAV formats differ')
    Path(args.output_wav).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(args.output_wav, 'wb') as stream:
        stream.setparams(a)
        stream.writeframes(x+y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser('concat-csv')
    p.add_argument('input_csvs', nargs='*')
    p.add_argument('--input_list')
    p.add_argument('--input_json', help='JSON array of paths; supports arbitrary filenames')
    p.add_argument('--output_csv', required=True)
    p.add_argument('--manifest_json')
    p.set_defaults(func=cmd_concat_csv)
    p = sub.add_parser('prepare')
    p.add_argument('input_csv')
    p.add_argument('--output_root', default='mel_mc_int')
    p.add_argument('--train_ratio', type=float, default=.9)
    p.add_argument('--states_per_column', type=int)
    p.add_argument('--buffer_rows', type=int, default=4096)
    p.add_argument('--block_size', type=int, default=380)
    p.set_defaults(func=cmd_prepare)
    p = sub.add_parser('cut-prompt')
    p.add_argument('mel_csv')
    p.add_argument('--cutoff_s', type=float, required=True)
    p.add_argument('--output_dir', default='mel_mc_int_prompt')
    p.add_argument('--fully_observed', action='store_true')
    p.set_defaults(func=cmd_cut_prompt)
    p = sub.add_parser('wrap-csv')
    p.add_argument('input_csv')
    p.add_argument('--reference_mel_csv', required=True)
    p.add_argument('--output_csv', required=True)
    p.set_defaults(func=cmd_wrap_csv)
    p = sub.add_parser('stitch-wav')
    for name in ['prompt_wav', 'continuation_wav', 'output_wav']:
        p.add_argument('--'+name, required=True)
    p.set_defaults(func=cmd_stitch)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
