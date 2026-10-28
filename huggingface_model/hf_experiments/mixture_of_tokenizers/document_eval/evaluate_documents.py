"""Export per-document losses using the installed experiment's scoring functions.

This add-on does not edit evaluate.py or change its context/precision policy.
Outputs are published only after matching the accepted aggregate evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

LANGUAGES = ('zh', 'ko', 'ja', 'en')
POLICIES = {
    'bytes': 'same Unicode chunks <= T-2 UTF-8 bytes for every route; BOS resets each chunk',
    'token': 'T-1 stride; one token overlap; reset positions each window',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_base(project):
    sys.path.insert(0, str(project))
    spec = importlib.util.spec_from_file_location('mixture_document_eval_base', project / 'evaluate.py')
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    return base


def collect_documents(base, data, split, limit):
    documents, hashes = {}, {}
    for language in LANGUAGES:
        documents[language] = list(base.iter_documents(data / language / f'{split}.jsonl', limit))
        if not documents[language]:
            raise ValueError(f'No documents for {language}')
        digest = hashlib.sha256()
        for text in documents[language]:
            encoded = text.encode('utf-8')
            digest.update(len(encoded).to_bytes(8, 'little'))
            digest.update(encoded)
        hashes[language] = digest.hexdigest()
    return documents, hashes


def export_records(base, model, tokenizer, documents, mode, device, progress=True):
    """Return ordinary aggregates plus loss rows; uses original score_text verbatim."""
    records, results = [], {}
    for language in LANGUAGES:
        totals = {route: {key: 0 for key in base.TOTAL_KEYS} for route in (language, 'byte')}
        for index, text in enumerate(documents[language]):
            encoded = text.encode('utf-8')
            row = {'language': language, 'document_index': index,
                   'text_sha256': hashlib.sha256(encoded).hexdigest(),
                   'source_bytes': len(encoded), 'source_characters': len(text), 'routes': {}}
            for route in (language, 'byte'):
                scored = base.score_text(model, tokenizer, text, route, mode, device)
                row['routes'][route] = scored
                for key in base.TOTAL_KEYS:
                    totals[route][key] += scored[key]
            records.append(row)
            if progress and ((index + 1) % 100 == 0 or index + 1 == len(documents[language])):
                print(f'{language}: {index + 1}/{len(documents[language])} documents', flush=True)
        results[language] = {route: base.summarize(values) for route, values in totals.items()}
    macro = {}
    for label in ('native', 'byte'):
        values = [results[lang][lang if label == 'native' else 'byte']['bpb'] for lang in LANGUAGES]
        if any(value is None for value in values):
            raise ValueError('Each language must contain at least one source byte')
        macro[label + '_bpb'] = sum(values) / len(values)
    return records, results, macro


def verify_reference(report, reference, bpb_atol=1e-5):
    """Match source/policy and integer counters exactly, numeric losses within tolerance."""
    for key in ('lexical_size', 'metric', 'context_mode', 'context_tokens', 'context_policy',
                'eos_policy', 'split', 'evaluated_text_sha256'):
        if report[key] != reference[key]:
            raise ValueError(f'Reference mismatch: {key}')
    largest = 0.0
    integers = ('documents', 'source_bytes', 'source_characters', 'text_tokens',
                'fallback_tokens', 'fallback_bytes', 'eos_count', 'windows')
    for language in LANGUAGES:
        for route in (language, 'byte'):
            actual, old = report['languages'][language][route], reference['languages'][language][route]
            for key in integers:
                if actual[key] != old[key]:
                    raise ValueError(f'Reference counter mismatch: {language}/{route}/{key}')
            difference = abs(actual['bpb'] - old['bpb'])
            if not math.isfinite(difference) or difference > bpb_atol:
                raise ValueError(f'BPB differs from reference for {language}/{route}: '
                                 f'{actual["bpb"]} vs {old["bpb"]}; tolerance {bpb_atol}. '
                                 'Check weights, precision, environment and source before proceeding.')
            largest = max(largest, difference)
            if not math.isclose(actual['mean_eos_nll'], old['mean_eos_nll'], rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError(f'EOS loss mismatch for {language}/{route}')
    for key, value in report['macro_average'].items():
        if abs(value - reference['macro_average'][key]) > bpb_atol:
            raise ValueError(f'Reference macro mismatch: {key}')
    return {'status': 'passed', 'bpb_absolute_tolerance': bpb_atol,
            'maximum_language_route_bpb_difference': largest, 'integer_counters': 'exact'}


def provenance(project, model, reference, device):
    weights = sorted(set(model.glob('*.safetensors')) | set(model.glob('pytorch_model*.bin')))
    if not weights:
        raise ValueError(f'No Hugging Face model weight files in {model}')
    files = [model / 'config.json'] + weights
    files += sorted(p for p in (model / 'tokenizer').rglob('*') if p.is_file())
    if not (model / 'tokenizer').is_dir():
        raise ValueError(f'Missing tokenizer directory in {model}')
    source_names = ('evaluate.py', 'modeling_mixture.py', 'configuration_mixture.py', 'tokenization_mixture.py')
    source = {name: sha256(project / name) for name in source_names}
    packages = {}
    for name in ('torch', 'transformers', 'tokenizers', 'numpy'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        'checkpoint_files_sha256': {str(p.relative_to(model)): sha256(p) for p in files},
        'experiment_sources_sha256': source,
        'export_script_sha256': sha256(Path(__file__)),
        'reference_path': str(reference), 'reference_sha256': sha256(reference),
        'device': device, 'precision_policy': 'same as original evaluate.py; no autocast; original score_text',
        'packages': packages,
    }


def publish_json(path, report):
    """Atomic no-clobber publish; a failed evaluation never becomes a final output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', default='data/prepared')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--max-documents', type=int, default=1000)
    parser.add_argument('--context-mode', choices=('bytes', 'token'), required=True)
    parser.add_argument('--training-seed', type=int, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()
    if args.max_documents < 1 or args.threads < 1:
        parser.error('max-documents and threads must be positive')
    project = Path(args.project_root).resolve()
    def resolved(value):
        p = Path(value)
        return p.resolve() if p.is_absolute() else (project / p).resolve()
    model_path, data, reference_path, output = map(resolved, (args.model, args.data, args.reference, args.output))
    reference = read_json(reference_path)
    if resolved(reference['model']) != model_path:
        raise ValueError('Reference model path differs from requested model')
    base = load_base(project)
    documents, text_hashes = collect_documents(base, data, args.split, args.max_documents)
    if text_hashes != reference['evaluated_text_sha256']:
        raise ValueError('Prepared text differs from accepted evaluation; refusing to compare')
    config = read_json(model_path / 'config.json')
    for key, value in [('lexical_size', config['lexical_size']), ('context_mode', args.context_mode),
                       ('context_tokens', int(config['max_position_embeddings'])), ('split', args.split),
                       ('context_policy', POLICIES[args.context_mode])]:
        if reference[key] != value:
            raise ValueError(f'Reference/request mismatch: {key}')
    proof = provenance(project, model_path, reference_path, args.device)
    if output.exists():
        if not args.skip_existing:
            raise FileExistsError(output)
        from paired_bootstrap import validate_report
        saved = read_json(output)
        validate_report(saved, expected_seed=args.training_seed,
                        expected_size=config['lexical_size'], expected_mode=args.context_mode)
        if (saved['provenance'] != proof or saved['evaluated_text_sha256'] != text_hashes
                or resolved(saved['model']) != model_path):
            raise ValueError(f'Existing export provenance changed: {output}; use a new output directory')
        for row in saved['records']:
            text = documents[row['language']][row['document_index']]
            if row['text_sha256'] != hashlib.sha256(text.encode('utf-8')).hexdigest():
                raise ValueError('Existing per-document hash differs from current source')
        verify_reference(saved, reference)
        print(f'Verified existing export; skipped scoring: {output}', flush=True)
        return
    import torch
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    model, tokenizer = base.load_model_and_tokenizer(model_path, device=args.device)
    start = time.monotonic()
    records, languages, macro = export_records(base, model, tokenizer, documents, args.context_mode, args.device)
    report = {
        'schema_version': 1, 'training_seed': args.training_seed,
        'metric': 'canonical_tokenization_bits_per_utf8_byte',
        'context_mode': args.context_mode, 'context_tokens': int(model.config.max_position_embeddings),
        'context_policy': POLICIES[args.context_mode],
        'eos_policy': 'one genuine document EOS, reported separately from BPB',
        'split': args.split, 'evaluated_text_sha256': text_hashes,
        'macro_average': macro, 'languages': languages, 'model': str(Path(args.model)),
        'lexical_size': tokenizer.lexical_size, 'records': records, 'provenance': proof,
        'elapsed_scoring_seconds': time.monotonic() - start,
    }
    report['reference_comparison'] = verify_reference(report, reference)
    from paired_bootstrap import validate_report
    validate_report(report, expected_seed=args.training_seed,
                    expected_size=tokenizer.lexical_size, expected_mode=args.context_mode)
    publish_json(output, report)
    print(json.dumps(macro, indent=2), flush=True)
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
