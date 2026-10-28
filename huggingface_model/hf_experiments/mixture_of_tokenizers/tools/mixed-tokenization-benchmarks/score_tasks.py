"""Score the prepared multilingual tasks on a common byte-budget subset.

This is a specified continuation-scoring protocol, not an official leaderboard
implementation. It preserves the experiment's no-autocast scoring precision.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

LANGUAGES = ('zh', 'ko', 'ja', 'en')
ROUTES = ('byte',) + LANGUAGES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def load_experiment(package_dir):
    """Import the experiment by its actual package directory name."""
    package_dir = Path(package_dir).resolve()
    require(package_dir.name.isidentifier(), 'Package directory name must be a Python identifier')
    for location in (package_dir, package_dir.parent):
        if str(location) not in sys.path:
            sys.path.insert(0, str(location))
    imported = []
    for name in ('evaluate', 'modeling_mixture', 'tokenization_mixture'):
        module = importlib.import_module(f'{package_dir.name}.{name}')
        require(Path(module.__file__).resolve() == package_dir / f'{name}.py', f'Unexpected imported module: {name}')
        imported.append(module)
    return tuple(imported)


def load_checkpoint(package_dir, model_dir, device):
    base, modeling, tokenization = load_experiment(package_dir)
    model_dir = Path(model_dir).resolve()
    tokenizer = tokenization.MixtureTokenizer.from_pretrained(str(model_dir / 'tokenizer'))
    model = modeling.MixtureForCausalLM.from_pretrained(str(model_dir)).to(device).eval()
    require(model.config.lexical_size == tokenizer.lexical_size, 'Model/tokenizer lexical sizes differ')
    expected = getattr(model.config, 'tokenizer_fingerprint', None)
    if expected is not None:
        require(expected == tokenizer.fingerprint, 'Model/tokenizer fingerprints differ')
    return base, model, tokenizer


def read_training_identity(model_dir, expected_seed=None):
    model_dir = Path(model_dir).resolve()
    config = read_json(model_dir / 'config.json')
    run_path, summary_path = model_dir.parent / 'run_config.json', model_dir.parent / 'summary.json'
    run, summary = read_json(run_path), read_json(summary_path)
    seed = run['effective_training']['seed']
    require(type(seed) is int and seed >= 0, 'Invalid effective training seed')
    require(summary['settings']['seed'] == seed, 'Training summary seed differs from effective configuration')
    if 'initializer_seed' in config:
        require(config['initializer_seed'] == seed, 'Checkpoint initializer seed differs from training evidence')
    if expected_seed is not None:
        require(seed == expected_seed, 'Requested seed differs from recorded effective training seed')
    size = config['lexical_size']
    require(type(size) is int and size > 0, 'Invalid checkpoint lexical size')
    fingerprints = [obj.get('tokenizer_fingerprint') for obj in (config, run, summary)]
    fingerprints = [value for value in fingerprints if value is not None]
    require(len(set(fingerprints)) <= 1, 'Training/checkpoint tokenizer fingerprints differ')
    return {'training_seed': seed, 'lexical_size': size, 'model_config': config,
            'run_config_sha256': sha256(run_path), 'summary_sha256': sha256(summary_path)}


def build_provenance(package_dir, model_dir, device, threads, script_path):
    package_dir, model_dir = Path(package_dir).resolve(), Path(model_dir).resolve()
    weights = sorted(set(model_dir.glob('*.safetensors')) | set(model_dir.glob('pytorch_model*.bin')))
    require(bool(weights), f'No HF model weights in {model_dir}')
    tokenizer_files = sorted(path for path in (model_dir / 'tokenizer').rglob('*') if path.is_file())
    require(bool(tokenizer_files), 'Checkpoint tokenizer files missing')
    files = [model_dir / 'config.json'] + weights + sorted(model_dir.glob('*.index.json')) + tokenizer_files
    names = ('evaluate.py', 'modeling_mixture.py', 'configuration_mixture.py', 'tokenization_mixture.py')
    versions = {}
    for name in ('torch', 'transformers', 'tokenizers', 'numpy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        'checkpoint_files_sha256': {path.relative_to(model_dir).as_posix(): sha256(path) for path in files},
        'experiment_sources_sha256': {name: sha256(package_dir / name) for name in names},
        'scorer_sha256': sha256(script_path), 'device': device, 'threads': threads,
        'precision_policy': 'Original FP32 loading and score_ids; no autocast; FP32 log_softmax',
        'packages': versions,
    }


def load_tasks(path):
    records, ids = [], set()
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            require(isinstance(row, dict), f'Task line {line_number} must be an object')
            for key in ('id', 'task', 'language', 'split'):
                require(isinstance(row.get(key), str) and bool(row[key]), f'Task line {line_number}: missing {key}')
            require(row['id'] not in ids, f'Duplicate task ID: {row["id"]}')
            require(row['language'] in LANGUAGES and isinstance(row.get('prompt'), str), f'Invalid task language/prompt: {row["id"]}')
            choices = row.get('choices')
            require(isinstance(choices, list) and len(choices) >= 2
                    and all(isinstance(choice, str) and bool(choice) for choice in choices), f'Invalid choices: {row["id"]}')
            answer = row.get('answer_index')
            require(type(answer) is int and 0 <= answer < len(choices), f'Invalid answer index: {row["id"]}')
            row['prompt'].encode('utf-8')
            for choice in choices:
                choice.encode('utf-8')
            ids.add(row['id']); records.append(row)
    require(bool(records), 'Prepared task file is empty')
    return records


def task_identity(task):
    return {key: task[key] for key in ('id', 'task', 'language', 'split', 'answer_index')} | {
        'num_choices': len(task['choices']), 'chance_accuracy': 1 / len(task['choices']),
        'prompt_utf8_bytes': len(task['prompt'].encode('utf-8')),
        'choices_utf8_bytes': [len(choice.encode('utf-8')) for choice in task['choices']],
        'example_sha256': canonical_hash(task),
    }


def validate_task_manifest(tasks, manifest, digest):
    require(manifest.get('schema_version') == 1 and manifest.get('tasks_sha256') == digest,
            'Manifest schema/task digest differs')
    require(manifest.get('record_count') == len(tasks), 'Manifest record count differs')
    require(manifest.get('split_set') in ('dev', 'final')
            and isinstance(manifest.get('template_version'), str), 'Manifest split/template identity missing')
    groups = {}
    for row in tasks:
        key = (row['task'], row['language'], row['split'])
        groups.setdefault(key, []).append(row)
    declared = manifest.get('tasks')
    require(isinstance(declared, list) and len(declared) == len(groups), 'Manifest task groups differ')
    seen = set()
    for item in declared:
        key = (item['task'], item['language'], item['split'])
        require(key in groups and key not in seen, 'Manifest task group missing or duplicated')
        seen.add(key)
        rows = groups[key]
        require(item['count'] == len(rows)
                and all(len(row['choices']) == item['num_choices'] for row in rows), 'Manifest task counts/choice counts differ')
        require(math.isclose(item['chance_accuracy'], 1 / item['num_choices'], rel_tol=1e-12), 'Manifest chance level differs')


def eligibility(task, context=512):
    info = task_identity(task)
    info['max_sequence_bytes_with_bos'] = 1 + info['prompt_utf8_bytes'] + max(info['choices_utf8_bytes'])
    return info['max_sequence_bytes_with_bos'] <= context, info


def metrics_from_scores(scores, answer):
    require(len(scores) >= 2 and type(answer) is int and 0 <= answer < len(scores), 'Invalid score/answer shape')
    require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value <= 0 for value in scores), 'Invalid/nonfinite/positive log probability')
    best = max(scores)
    ties = [i for i, value in enumerate(scores) if value == best]
    # Avoid cancelling two large negative scores when options are tied/close.
    loss = best - scores[answer] + math.log(math.fsum(math.exp(value - best) for value in scores))
    return {'prediction': ties[0], 'tied_predictions': ties, 'tie_count': len(ties),
            'correct_fraction': 1 / len(ties) if answer in ties else 0.0,
            'option_log_loss': loss,
            'correct_vs_best_incorrect_margin': scores[answer] - max(value for i, value in enumerate(scores) if i != answer)}


def tokenize_task(tokenizer, task):
    prepared = {}
    texts = [task['prompt']] + task['choices']
    byte_lengths = [len(text.encode('utf-8')) for text in texts]
    for label, route in (('native', ROUTES.index(task['language'])), ('byte', 0)):
        encoded = [tokenizer.encode_route(text, route) for text in texts]
        for index, (ids, nbytes) in enumerate(zip(encoded, byte_lengths)):
            require(len(ids) <= nbytes, f'Tokenizer emitted more tokens than UTF-8 bytes: {task["id"]}/{label}/{index}')
            require(index == 0 or len(ids) > 0, f'Empty tokenized answer: {task["id"]}/{label}/{index}')
            if label == 'byte':
                require(ids == list(texts[index].encode('utf-8')), 'Byte route is not the exact UTF-8 sequence')
        prepared[label] = {'route': route, 'prompt': encoded[0], 'choices': encoded[1:]}
    return prepared


def score_one(base, model, tokenizer, task, device='cpu', context=512):
    require(int(model.config.max_position_embeddings) == context, 'Model context differs from eligibility protocol')
    fits, item = eligibility(task, context)
    prepared = tokenize_task(tokenizer, task)
    if not fits:
        item['reason'] = 'byte_context_exceeded'
        return None, item
    item.pop('max_sequence_bytes_with_bos')
    item['routes'] = {}
    for label, data in prepared.items():
        route = data['route']
        scores = []
        answer_start = 1 + len(data['prompt'])
        for answer_ids in data['choices']:
            ids = [tokenizer.bos_id(route)] + data['prompt'] + answer_ids
            require(len(ids) <= context, 'Eligible question unexpectedly exceeds native context')
            losses, expected_position = [], 1
            for position, target, nll in base._score_ids(model, ids, route, context, device):
                require(position == expected_position and 0 < position < len(ids) and target == ids[position], 'Scorer target coverage/order differs')
                require(isinstance(nll, (int, float)) and math.isfinite(nll) and nll >= 0, 'Invalid target NLL')
                if position >= answer_start:
                    losses.append(nll)
                expected_position += 1
            require(expected_position == len(ids) and len(losses) == len(answer_ids), 'Scorer omitted or duplicated answer targets')
            scores.append(-math.fsum(losses))
        normalized = [value / nbytes for value, nbytes in zip(scores, item['choices_utf8_bytes'])]
        item['routes'][label] = {
            'sum_log_probs': scores, 'sum_log_probs_per_utf8_byte': normalized,
            'prompt_tokens': len(data['prompt']), 'choice_tokens': [len(ids) for ids in data['choices']],
            'raw': metrics_from_scores(scores, task['answer_index']),
            'byte_normalized': metrics_from_scores(normalized, task['answer_index']),
        }
    return item, None


def summarize_group(records, skipped):
    count = len(records)
    output = {'count': count, 'skipped_count': len(skipped), 'source_count': count + len(skipped),
              'chance_accuracy': math.fsum(row['chance_accuracy'] for row in records) / count if count else None,
              'routes': {}}
    for route in ('native', 'byte'):
        output['routes'][route] = {}
        for rule in ('raw', 'byte_normalized'):
            rows = [record['routes'][route][rule] for record in records]
            output['routes'][route][rule] = {
                'accuracy': math.fsum(row['correct_fraction'] for row in rows) / count if count else None,
                'option_log_loss': math.fsum(row['option_log_loss'] for row in rows) / count if count else None,
                'correct_vs_best_incorrect_margin': math.fsum(row['correct_vs_best_incorrect_margin'] for row in rows) / count if count else None,
                'exact_tie_count': sum(row['tie_count'] > 1 for row in rows),
            }
    return output


def aggregate(records, skipped):
    all_rows = records + skipped
    tasks = sorted({row['task'] for row in all_rows})
    languages = sorted({row['language'] for row in all_rows})
    select = lambda rows, **criteria: [row for row in rows if all(row[key] == value for key, value in criteria.items())]
    return {'overall': summarize_group(records, skipped),
            'by_task': {task: summarize_group(select(records, task=task), select(skipped, task=task)) for task in tasks},
            'by_language': {lang: summarize_group(select(records, language=lang), select(skipped, language=lang)) for lang in languages},
            'by_task_language': {task: {lang: summarize_group(select(records, task=task, language=lang),
                                                               select(skipped, task=task, language=lang))
                                      for lang in languages if select(all_rows, task=task, language=lang)} for task in tasks}}


def protocol(context):
    return {'name': 'separate_prompt_choice_continuation_v1',
            'choice_encoding': 'prompt and each choice encoded separately; score choice tokens only',
            'primary_score': 'sum_log_probs', 'secondary_score': 'sum_log_probs_per_utf8_byte',
            'eligibility': 'all choices fit BOS + prompt UTF-8 bytes + choice UTF-8 bytes; assert native token bound',
            'eligibility_context': context, 'ties': 'exact_maxima_fractional_credit',
            'eos_policy': 'excluded', 'official_benchmark_score': False,
            'accuracy_interval_policy': 'No unpaired or Wilson intervals; use paired bootstrap report for fractional-credit accuracy'}


def validate_existing(result, tasks, expected):
    for key, value in expected.items():
        require(result.get(key) == value, f'Existing result differs in {key}; choose a new output directory')
    records = result['records']; skipped = result['skipped']
    fits, misses = [], []
    for task in tasks:
        eligible, identity = eligibility(task, result['protocol']['eligibility_context'])
        if eligible:
            identity.pop('max_sequence_bytes_with_bos'); fits.append(identity)
        else:
            identity['reason'] = 'byte_context_exceeded'; misses.append(identity)
    require(skipped == misses and len(records) == len(fits), 'Existing scored/skipped subset differs')
    for record, identity in zip(records, fits):
        require(all(record.get(key) == value for key, value in identity.items()), 'Existing item identity differs')
        require(set(record['routes']) == {'native', 'byte'}, 'Existing route set differs')
        for route, scores in record['routes'].items():
            raw = scores['sum_log_probs']; lengths = record['choices_utf8_bytes']
            require(len(raw) == len(lengths), 'Stored score array length differs')
            normalized = [value / length for value, length in zip(raw, lengths)]
            require(scores['sum_log_probs_per_utf8_byte'] == normalized, 'Stored normalized scores differ')
            for rule, values in (('raw', raw), ('byte_normalized', normalized)):
                require(scores[rule] == metrics_from_scores(values, record['answer_index']), 'Stored metrics differ from option scores')
            require(type(scores['prompt_tokens']) is int and 0 <= scores['prompt_tokens'] <= record['prompt_utf8_bytes'], 'Stored prompt count invalid')
            require(len(scores['choice_tokens']) == len(lengths) and all(type(n) is int and 0 < n <= nb for n, nb in zip(scores['choice_tokens'], lengths)), 'Stored choice counts invalid')
            if route == 'byte':
                require(scores['prompt_tokens'] == record['prompt_utf8_bytes']
                        and scores['choice_tokens'] == lengths, 'Stored byte-route counts differ from UTF-8 bytes')
    require(result['eligible_ids'] == [row['id'] for row in fits]
            and result['skipped_ids'] == [row['id'] for row in misses], 'Stored eligible/skipped IDs differ')
    require(result['subset_sha256'] == canonical_hash([[row['id'], row['example_sha256']] for row in fits]), 'Stored subset digest differs')
    require(result['aggregates'] == aggregate(records, skipped), 'Stored aggregates differ')


def publish_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-dir', default='.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--tasks', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--eligibility-context', type=int, default=512)
    parser.add_argument('--training-seed', type=int)
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()
    require(args.threads > 0 and args.eligibility_context >= 2, 'Invalid threads/context')
    started = time.monotonic()
    package, model_path, task_path, manifest_path, output = [Path(value).resolve() for value in
        (args.package_dir, args.model, args.tasks, args.manifest, args.output)]
    tasks = load_tasks(task_path); manifest = read_json(manifest_path)
    task_digest = sha256(task_path)
    validate_task_manifest(tasks, manifest, task_digest)
    identity = read_training_identity(model_path, args.training_seed)
    require(identity['model_config']['max_position_embeddings'] == args.eligibility_context, 'Model context differs from common eligibility protocol')
    proof = build_provenance(package, model_path, args.device, args.threads, Path(__file__))
    expected = {'schema_version': 1, 'result_type': 'mixed_tokenization_multiple_choice',
                'model': str(model_path), **identity, 'protocol': protocol(args.eligibility_context),
                'tasks_sha256': task_digest, 'manifest_sha256': sha256(manifest_path),
                'manifest': manifest, 'provenance': proof}
    if output.exists():
        require(args.skip_existing, f'Result already exists: {output}; use --skip-existing to verify and reuse')
        validate_existing(read_json(output), tasks, expected)
        print(f'Verified existing result: {output}', flush=True)
        return
    import torch
    torch.set_num_threads(args.threads)
    require(args.device != 'cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
    base, model, tokenizer = load_checkpoint(package, model_path, args.device)
    require(tokenizer.lexical_size == identity['lexical_size'], 'Loaded checkpoint lexical identity differs')
    records, skipped = [], []
    scoring_start = time.monotonic()
    for index, task in enumerate(tasks):
        row, excluded = score_one(base, model, tokenizer, task, args.device, args.eligibility_context)
        if row is not None:
            records.append(row)
        else:
            skipped.append(excluded)
        if (index + 1) % 100 == 0 or index + 1 == len(tasks):
            print(f'{index + 1}/{len(tasks)} questions; scored={len(records)} skipped={len(skipped)}', flush=True)
    scoring_seconds = time.monotonic() - scoring_start
    require(sha256(task_path) == task_digest and sha256(manifest_path) == expected['manifest_sha256'], 'Task inputs changed while scoring')
    require(build_provenance(package, model_path, args.device, args.threads, Path(__file__)) == proof, 'Checkpoint/source provenance changed while scoring')
    result = {**expected, 'records': records, 'skipped': skipped,
              'eligible_ids': [row['id'] for row in records], 'skipped_ids': [row['id'] for row in skipped],
              'subset_sha256': canonical_hash([[row['id'], row['example_sha256']] for row in records]),
              'aggregates': aggregate(records, skipped), 'scoring_seconds': scoring_seconds,
              'total_elapsed_seconds': time.monotonic() - started}
    validate_existing(result, tasks, expected)
    publish_json(output, result)
    print(json.dumps(result['aggregates']['by_task'], indent=2), flush=True)
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
