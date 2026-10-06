#!/usr/bin/env python3
"""Mel demo orchestration; subprocess arguments are never evaluated by a shell."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import pickle
import random
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import mel_mc_int_tools as tools

ROOT = Path(__file__).resolve().parents[2]
ENCODER = ROOT / 'data/mel_spectrogram/audio_to_token_mel.py'
DECODER = ROOT / 'data/mel_spectrogram/token_mel_to_audio.py'
EXTENSIONS = {'.wav', '.flac', '.mp3', '.m4a', '.aac', '.ogg', '.opus'}
DEFAULTS = {'sample_rate': 48000, 'bands': 384, 'levels': 64, 'hop_ms': 15., 'win_ms': 60.,
            'n_fft': 8192, 'fmin': 10., 'fmax': 20000., 'top_db': 96.}
FLAGS = {'sample_rate': '--samples-per-second', 'bands': '--columns-per-timestep',
         'levels': '--states-per-column', 'hop_ms': '--timestep-ms', 'win_ms': '--win-ms',
         'n_fft': '--n-fft', 'fmin': '--fmin', 'fmax': '--fmax', 'top_db': '--top-db'}


def env(name, default, cast=str):
    value = cast(os.environ.get('MEL_MC_'+name, str(default)))
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'MEL_MC_{name} must be finite')
    return value


def flag(name, default=False):
    value = env(name, '1' if default else '0').lower()
    if value not in {'0', '1', 'true', 'false'}:
        raise ValueError(f'MEL_MC_{name} must be 0/1 or true/false')
    return value in {'1', 'true'}


def run(command):
    command = [str(part) for part in command]
    print('+ ' + shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def transform_config():
    result = {key: env(key.upper(), value, type(value)) for key, value in DEFAULTS.items()}
    if any(value <= 0 for key, value in result.items() if key != 'fmin') or result['fmin'] < 0:
        raise ValueError('Invalid encoder sizes/frequencies')
    result['keep_dc'] = True
    result['reference_mode'] = 'fixed'
    result['reference_power'] = None
    result['encoder_sha256'] = tools.file_sha256(ENCODER)
    return result


def encode(source, directory, prefix, config, device, mode='fixed'):
    command = [sys.executable, ENCODER, source, '--force', '--quiet', '--preset', 'max', '--keep-dc',
               '--reference-mode', mode, '--output-format', 'csv', '--output-dir', directory,
               '--output-prefix', prefix, '--device', device]
    for key, argument in FLAGS.items():
        command.extend([argument, config[key]])
    if mode == 'fixed':
        command.extend(['--reference-power', config['reference_power']])
    run(command)
    output = Path(directory) / f'{prefix}.max.mel.csv'
    tools.validate_csv(output)
    return output


def calibration(train_files, source_hashes, config, work, device):
    explicit = env('REFERENCE_POWER', '')
    if explicit:
        value = float(explicit)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('MEL_MC_REFERENCE_POWER must be positive and finite')
        return value
    identity = {'sources': {str(path): source_hashes[str(path)] for path in train_files}, 'encoder': config}
    cache = work / 'calibration.json'
    if cache.exists():
        old = json.loads(cache.read_text())
        if old.get('identity') == identity:
            value = float(old['reference_power'])
            if math.isfinite(value) and value > 0:
                return value
    references = []
    # Sequential calibration deletes intermediate CSVs after each recording.
    for source in train_files:
        with tempfile.TemporaryDirectory(prefix='calibration-', dir=work) as temporary:
            csv = encode(source, temporary, 'reference', config, device, mode='file_percentile')
            metadata = tools.inspect_mel_csv(csv)[2]
            if not metadata['quantizer']['silent']:
                references.append(float(metadata['quantizer']['reference_power']))
    reference = max(references, default=1.0)
    tools.atomic_json(cache, {'identity': identity, 'reference_power': reference,
                             'method': 'maximum training-recording 99.5th-percentile power'})
    return reference


def cached_encode(source, source_hash, config, work, device, reuse_only):
    identity = {'source': str(source), 'sha256': source_hash, 'encoder': config}
    key = hashlib.sha256(tools.compact_json(identity).encode()).hexdigest()[:24]
    directory = work / 'encoded'
    directory.mkdir(parents=True, exist_ok=True)
    csv = directory / f'{key}.max.mel.csv'
    sidecar = directory / f'{key}.cache.json'
    if csv.exists() and sidecar.exists():
        old = json.loads(sidecar.read_text())
        if old.get('identity') == identity and old.get('csv_sha256') == tools.file_sha256(csv):
            tools.validate_csv(csv)
            return csv
    if reuse_only:
        raise ValueError(f'No valid encoded cache for {source}; unset MEL_MC_SKIP_ENCODE to rebuild')
    encode(source, directory, key, config, device)
    tools.atomic_json(sidecar, {'identity': identity, 'csv_sha256': tools.file_sha256(csv)})
    return csv


def read_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get('schema_version') != 1 or not manifest.get('multicontext_datasets'):
        raise ValueError('Use a manifest produced by the repaired mel adapter; re-prepare legacy datasets')
    config = manifest.get('encoder')
    if not config or config.get('reference_mode') != 'fixed' or config.get('keep_dc') is not True:
        raise ValueError('Manifest needs fixed-reference encoder settings with keep_dc enabled')
    if config.get('encoder_sha256') != tools.file_sha256(ENCODER):
        raise ValueError('Encoder code changed since dataset preparation; re-prepare or use the original encoder')
    metadata = manifest['roundtrip_metadata']
    tools.validate_roundtrip_metadata(metadata)
    if (config['bands'] != metadata['shape']['bands'] or config['levels'] != metadata['quantizer']['levels']
            or config['reference_power'] != metadata['quantizer']['reference_power']):
        raise ValueError('Encoder configuration and roundtrip metadata disagree')
    return manifest


def validate_checkpoint(checkpoint, manifest):
    arguments = checkpoint.get('model_args', {})
    names = checkpoint.get('config', {}).get('multicontext_datasets')
    expected = manifest['multicontext_datasets']
    if not arguments.get('multicontext') or arguments.get('numerical_multicontext', False):
        raise ValueError('A categorical multicontext checkpoint is required')
    if names != expected:
        raise ValueError('Checkpoint and manifest dataset order/identity differ')
    if arguments.get('vocab_sizes') != [manifest['vocab_size']] * len(expected):
        raise ValueError('Checkpoint and manifest vocabularies differ')
    for name, column in zip(expected, manifest['columns']):
        with (ROOT / 'data' / name / 'meta.pkl').open('rb') as stream:
            metadata = pickle.load(stream)
        if metadata.get('source_column') != column or metadata.get('vocab_size') != manifest['vocab_size']:
            raise ValueError('Dataset metadata no longer matches the saved manifest')
    return int(arguments['block_size'])


def crop_prefix(source, output, cutoff):
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError('Cutoff must be positive and finite')
    # Trim decoded samples at the source rate BEFORE any resampling or statistics.
    # The encoder performs resampling later, on this prefix-only file.
    run(['ffmpeg', '-v', 'error', '-y', '-i', source, '-map', '0:a:0',
         '-af', f'atrim=end={cutoff:.17g},asetpts=PTS-STARTPTS', '-c:a', 'pcm_f32le', output])


def infer(out_dir, source, cutoff, frames, manifest_path=None):
    out_dir, source = Path(out_dir).resolve(), Path(source).resolve()
    manifest_path = Path(manifest_path or out_dir / 'mel_manifest.json').resolve()
    manifest = read_manifest(manifest_path)
    if frames < 1:
        raise ValueError('max_new_tokens must be positive')
    device, dtype = env('DEVICE', 'cuda:0'), env('DTYPE', 'bfloat16')
    top_k, temperature, seed = env('TOP_K', 1, int), env('TEMPERATURE', .8, float), env('SEED', 1337, int)
    if not 1 <= top_k <= manifest['vocab_size'] or temperature <= 0:
        raise ValueError('top_k must be in [1,levels] and temperature must be positive')
    import torch
    checkpoint = torch.load(out_dir / 'ckpt.pt', map_location='cpu', weights_only=False)
    block_size = validate_checkpoint(checkpoint, manifest)
    del checkpoint
    parent = out_dir / 'mel_samples'
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='sample-', dir=parent))
    prefix = output / 'original_prefix.wav'
    crop_prefix(source, prefix, cutoff)
    reference = encode(prefix, output, 'prefix', manifest['encoder'], device)
    if tools.metadata_signature(tools.inspect_mel_csv(reference)[2]) != tools.metadata_signature(manifest['roundtrip_metadata']):
        raise ValueError('Prompt transform differs from the trained transform')
    tools.cmd_cut_prompt(SimpleNamespace(mel_csv=reference, cutoff_s=cutoff, output_dir=output/'prompt', fully_observed=True))
    prompt = json.loads((output/'prompt/prompt_manifest.json').read_text())
    run([sys.executable, ROOT/'sample.py', '--init_from', 'resume', '--out_dir', out_dir,
         '--multicontext', '--multicontext_datasets', *manifest['multicontext_datasets'],
         '--multicontext_csv_input', output/'prompt/prompt.csv',
         '--multicontext_csv_output_file', output/'generated.csv', '--multicontext_csv_output_include_prompt',
         '--max_new_tokens', frames, '--num_samples', 1, '--device', device, '--dtype', dtype,
         '--temperature', temperature, '--top_k', top_k, '--seed', seed, '--no-compile', '--no-print_model_info'])
    tools.cmd_wrap_csv(SimpleNamespace(input_csv=output/'generated.csv', reference_mel_csv=reference,
                                       output_csv=output/'generated.mel.csv'))
    for source_csv, target in [(output/'generated.mel.csv', 'generated.wav'),
                               (output/'prompt/prompt.mel.csv', 'codec_prompt.wav')]:
        run([sys.executable, DECODER, source_csv, '--output', output/target, '--force', '--device', device])
    start = prompt['prompt_frames'] * manifest['roundtrip_metadata']['stft']['hop_length']
    run(['ffmpeg', '-v', 'error', '-y', '-i', output/'generated.wav', '-af', f'atrim=start_sample={start}',
         '-c:a', 'pcm_s16le', output/'continuation.wav'])
    settings = {'input_audio': str(source), 'out_dir': str(out_dir), 'manifest': str(manifest_path),
                'cutoff_s': cutoff, 'max_new_tokens': frames, 'device': device, 'dtype': dtype,
                'temperature': temperature, 'top_k': top_k, 'seed': seed, 'prompt': prompt,
                'conditioned_frames': min(block_size, prompt['prompt_frames']),
                'context_seconds': block_size * manifest['roundtrip_metadata']['stft']['hop_length'] / manifest['encoder']['sample_rate']}
    tools.atomic_json(output/'run.json', settings)
    from make_viewer import build_viewer
    build_viewer(output, settings)
    print(f'Viewer: {output / "index.html"}')
    return output


def train(manifest, out_dir, device, dtype):
    layers, width, heads = env('N_LAYER', 12, int), env('N_EMBD', 500, int), env('N_HEAD', 12, int)
    qk, v = env('QK_DIM', 120, int), env('V_DIM', 120, int)
    run([sys.executable, ROOT/'train.py', '--training_mode', 'multicontext', '--multicontext',
         '--dataset', manifest['multicontext_datasets'][0], '--multicontext_datasets', *manifest['multicontext_datasets'],
         '--n_layer', layers, '--n_embd', width, '--n_head', heads, '--n_kv_group', env('N_KV_GROUP', heads, int),
         '--mlp_size', env('MLP_SIZE', 1536, int), '--n_qk_head_dim', qk, '--n_v_head_dim', v,
         '--block_size', manifest['block_size'], '--batch_size', env('BATCH_SIZE', 12, int),
         '--attention_variant', 'infinite', '--use_concat_heads', '--no-use_abs_pos_embeddings',
         '--use_rotary_embeddings', '--optimizer', 'muon', '--weight_decay', 0.0,
         '--use_qk_norm', '--use_qk_norm_scale', '--max_iters', env('MAX_ITERS', 1000, int),
         '--eval_interval', env('EVAL_INTERVAL', 100, int), '--eval_iters', env('EVAL_ITERS', 10, int),
         '--learning_rate', env('LR', .001, float), '--dropout', env('DROPOUT', 0., float),
         '--seed', env('SEED', 1337, int), '--device', device, '--dtype', dtype,
         '--only_save_checkpoint_at_end',
         '--compile' if flag('COMPILE', True) else '--no-compile',
         '--tensorboard_log' if flag('TENSORBOARD') else '--no-tensorboard_log', '--out_dir', out_dir])


def folder(args):
    source_dir = Path(args.music_dir).resolve()
    files = sorted(path.resolve() for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in EXTENSIONS)
    if not files:
        raise ValueError('No supported audio files in the input directory')
    prompt = Path(args.prompt_audio).resolve() if args.prompt_audio else files[0]
    out = Path(env('OUT_DIR', str(ROOT/'out/mel_mc_int_music'))).resolve()
    if flag('SKIP_TRAIN'):
        return infer(out, prompt, args.cutoff, env('MAX_NEW_TOKENS', 200, int))
    if (out/'ckpt.pt').exists():
        raise ValueError('Checkpoint already exists; choose another MEL_MC_OUT_DIR or use MEL_MC_SKIP_TRAIN=1')
    if len(files) < 2:
        raise ValueError('Folder training needs at least two recordings; run.sh provides a guarded single-recording split')
    ratio = env('TRAIN_RATIO', .9, float)
    tools.check_ratio(ratio)
    shuffled = list(files)
    random.Random(env('SEED', 1337, int)).shuffle(shuffled)
    count = max(1, min(len(files)-1, int(len(files)*ratio)))
    train_files, val_files = shuffled[:count], shuffled[count:]
    hashes = {str(path): tools.file_sha256(path) for path in files}
    if len(set(hashes.values())) != len(hashes):
        raise ValueError('Duplicate source-file contents; remove duplicates before splitting')
    work = Path(env('WORK_DIR', str(ROOT/'data/mel_mc_int/music_pipeline_out'))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    device, dtype = env('DEVICE', 'cuda:0'), env('DTYPE', 'bfloat16')
    config = transform_config()
    config['reference_power'] = calibration(train_files, hashes, config, work, device)
    selected = {str(path): cached_encode(path, hashes[str(path)], config, work, device, flag('SKIP_ENCODE')) for path in files}
    splits = {key: [{'csv': str(selected[str(path)])} for path in group]
              for key, group in [('train', train_files), ('val', val_files)]}
    manifest = tools.prepare_segments(splits, env('OUTPUT_ROOT', 'mel_mc_int_music'),
                                       env('BLOCK_SIZE', 380, int), encoder=config)
    tools.atomic_json(work/'selected_sources.json', {'train': [str(p) for p in train_files],
                      'val': [str(p) for p in val_files], 'sha256': hashes, 'encoder': config})
    out.mkdir(parents=True, exist_ok=True)
    tools.atomic_json(out/'mel_manifest.json', manifest)
    if flag('PREPARE_ONLY'):
        print(out/'mel_manifest.json')
        return
    train(manifest, out, device, dtype)
    return infer(out, prompt, args.cutoff, env('MAX_NEW_TOKENS', 200, int))


def single(args):
    config = transform_config()
    config['reference_power'] = float(env('REFERENCE_POWER', 1.0))
    if not math.isfinite(config['reference_power']) or config['reference_power'] <= 0:
        raise ValueError('MEL_MC_REFERENCE_POWER must be positive and finite')
    source = Path(args.input_audio).resolve()
    work = Path(env('WORK_DIR', str(ROOT/'data/mel_mc_int/mel_out'))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    csv = cached_encode(source, tools.file_sha256(source), config, work, env('DEVICE', 'auto'), flag('SKIP_ENCODE'))
    info = tools.validate_csv(csv)
    ratio = env('TRAIN_RATIO', .9, float)
    tools.check_ratio(ratio)
    cut = int(info['rows'] * ratio)
    guard = math.ceil(info['metadata']['stft']['win_length']/info['metadata']['stft']['hop_length'])
    manifest = tools.prepare_segments({'train': [{'csv': str(csv), 'start': 0, 'stop': cut}],
                    'val': [{'csv': str(csv), 'start': cut+guard, 'stop': info['rows']}]},
                    args.output_root, env('BLOCK_SIZE', 380, int), encoder=config)
    print(tools.dataset_root(args.output_root)/'manifest.json')
    print('Temporal holdout with a full-window gap; this is not held-out-recording evaluation.')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='See data/mel_mc_int/README.md for MEL_MC_* settings.')
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser('folder')
    p.add_argument('music_dir')
    p.add_argument('prompt_audio', nargs='?')
    p.add_argument('cutoff', nargs='?', type=float, default=10.)
    p.set_defaults(func=folder)
    p = sub.add_parser('single')
    p.add_argument('input_audio')
    p.add_argument('output_root', nargs='?', default='mel_mc_int')
    p.set_defaults(func=single)
    p = sub.add_parser('infer')
    p.add_argument('out_dir')
    p.add_argument('input_audio')
    p.add_argument('cutoff', type=float)
    p.add_argument('max_new_tokens', nargs='?', type=int, default=200)
    p.add_argument('--manifest')
    p.set_defaults(func=lambda a: infer(a.out_dir, a.input_audio, a.cutoff, a.max_new_tokens, a.manifest))
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
