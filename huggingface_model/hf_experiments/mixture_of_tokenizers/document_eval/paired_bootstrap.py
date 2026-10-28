"""Paired document bootstrap for the fixed two-seed vocabulary comparison.

These intervals condition on the trained checkpoints. They are not confidence
intervals for a population of training seeds, and are not multiplicity adjusted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

LANGUAGES = ('zh', 'ko', 'ja', 'en')
SEEDS = (1337, 1338)
SIZES = (10000, 20000)
MODES = ('bytes', 'token')
ROUTES = ('native', 'byte')
TOTAL_KEYS = ('documents', 'text_nll', 'text_tokens', 'source_bytes',
              'source_characters', 'fallback_tokens', 'fallback_bytes',
              'eos_nll', 'eos_count', 'windows')
FLOAT_KEYS = ('text_nll', 'eos_nll')
POLICIES = {
    'bytes': 'same Unicode chunks <= T-2 UTF-8 bytes for every route; BOS resets each chunk',
    'token': 'T-1 stride; one token overlap; reset positions each window',
}
METRIC = 'canonical_tokenization_bits_per_utf8_byte'
EOS_POLICY = 'one genuine document EOS, reported separately from BPB'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label, integer=False, minimum=0):
    require(not isinstance(value, bool) and isinstance(value, (int, float)), f'{label}: expected number')
    require(math.isfinite(value) and value >= minimum, f'{label}: nonfinite or out of range')
    if integer:
        require(isinstance(value, int), f'{label}: expected integer')
    return value


def same_number(actual, expected, label):
    if expected is None:
        require(actual is None, f'{label}: expected null for zero denominator')
    else:
        number(actual, label)
        require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9),
                f'{label}: {actual!r} differs from recomputed {expected!r}')


def valid_hash(value, label):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
            f'{label}: expected lowercase SHA256')


def check_summary(row, label):
    require(isinstance(row, dict), f'{label}: expected object')
    for key in TOTAL_KEYS:
        require(key in row, f'{label}: missing {key}')
        number(row[key], f'{label}.{key}', integer=key not in FLOAT_KEYS)
    nbytes, ntokens = row['source_bytes'], row['text_tokens']
    require(row['source_characters'] <= nbytes <= 4 * row['source_characters'],
            f'{label}: inconsistent UTF-8 byte/character totals')
    require(row['fallback_tokens'] == row['fallback_bytes'] <= ntokens <= nbytes,
            f'{label}: inconsistent token/fallback counts')
    if nbytes == 0:
        require(row['text_nll'] == 0, f'{label}: empty text has text NLL')
    formulas = {
        'bpb': row['text_nll'] / (nbytes * math.log(2)) if nbytes else None,
        'mean_token_nll': row['text_nll'] / ntokens if ntokens else None,
        'bytes_per_token': nbytes / ntokens if ntokens else None,
        'characters_per_token': row['source_characters'] / ntokens if ntokens else None,
        'fallback_token_fraction': row['fallback_tokens'] / ntokens if ntokens else None,
        'fallback_source_byte_fraction': row['fallback_bytes'] / nbytes if nbytes else None,
        'mean_eos_nll': row['eos_nll'] / row['eos_count'] if row['eos_count'] else None,
    }
    for key, value in formulas.items():
        require(key in row, f'{label}: missing {key}')
        same_number(row[key], value, f'{label}.{key}')


def validate_report(report, expected_seed=None, expected_size=None, expected_mode=None):
    """Validate one export and return records sorted by per-language index."""
    require(isinstance(report, dict), 'report must be an object')
    require(type(report.get('schema_version')) is int and report['schema_version'] == 1, 'unsupported schema_version')
    number(report.get('training_seed'), 'training_seed', integer=True)
    number(report.get('lexical_size'), 'lexical_size', integer=True)
    require(report.get('training_seed') in SEEDS, 'unexpected training_seed')
    require(report.get('lexical_size') in SIZES, 'unexpected lexical_size')
    require(report.get('context_mode') in MODES, 'unexpected context_mode')
    for key, expected in [('training_seed', expected_seed), ('lexical_size', expected_size),
                          ('context_mode', expected_mode)]:
        if expected is not None:
            require(report[key] == expected, f'{key} does not match file condition')
    require(report.get('metric') == METRIC, 'unexpected metric')
    require(report.get('eos_policy') == EOS_POLICY, 'unexpected eos_policy')
    require(report.get('context_policy') == POLICIES[report['context_mode']], 'unexpected context_policy')
    number(report.get('context_tokens'), 'context_tokens', integer=True, minimum=6)
    require(report.get('split') in ('val', 'test'), 'unexpected split')
    require(isinstance(report.get('model'), str) and bool(report['model']), 'missing model identity')
    require(isinstance(report.get('languages'), dict) and set(report['languages']) == set(LANGUAGES),
            'aggregate language set differs')
    hashes = report.get('evaluated_text_sha256')
    require(isinstance(hashes, dict) and set(hashes) == set(LANGUAGES), 'missing evaluated-text hashes')
    for lang, digest in hashes.items():
        valid_hash(digest, f'evaluated_text_sha256.{lang}')
    records = report.get('records')
    require(isinstance(records, list) and len(records) > 0, 'missing document records')
    by_language = {lang: {} for lang in LANGUAGES}
    for row in records:
        require(isinstance(row, dict), 'document record must be an object')
        lang = row.get('language')
        require(lang in LANGUAGES, 'unexpected document language')
        index = number(row.get('document_index'), 'document_index', integer=True)
        require(index not in by_language[lang], f'duplicate document index: {lang}/{index}')
        valid_hash(row.get('text_sha256'), f'{lang}/{index}.text_sha256')
        for key in ('source_bytes', 'source_characters'):
            number(row.get(key), f'{lang}/{index}.{key}', integer=True)
        routes = row.get('routes')
        require(isinstance(routes, dict) and set(routes) == {lang, 'byte'},
                f'{lang}/{index}: route set differs')
        for route, scored in routes.items():
            label = f'{lang}/{index}/{route}'
            check_summary(scored, label)
            require(scored['documents'] == scored['eos_count'] == 1 and scored['windows'] >= 1,
                    f'{label}: document/EOS/window count invalid')
            for key in ('source_bytes', 'source_characters'):
                require(scored[key] == row[key], f'{label}: source count mismatch')
            if route == 'byte':
                require(scored['text_tokens'] == scored['fallback_tokens'] == row['source_bytes'],
                        f'{label}: byte-route token count differs from source bytes')
        by_language[lang][index] = row
    ordered = {}
    for lang, indexed in by_language.items():
        require(len(indexed) > 0 and sorted(indexed) == list(range(len(indexed))),
                f'{lang}: document indices must be contiguous from zero')
        ordered[lang] = [indexed[i] for i in range(len(indexed))]
        require(sum(row['source_bytes'] for row in ordered[lang]) > 0, f'{lang}: all documents are empty')
        aggregate = report['languages'][lang]
        require(isinstance(aggregate, dict) and set(aggregate) == {lang, 'byte'}, f'{lang}: aggregate routes differ')
        for route in (lang, 'byte'):
            check_summary(aggregate[route], f'aggregate/{lang}/{route}')
            for key in TOTAL_KEYS:
                total = sum(row['routes'][route][key] for row in ordered[lang])
                if key in FLOAT_KEYS:
                    same_number(aggregate[route][key], total, f'aggregate/{lang}/{route}/{key}')
                else:
                    require(aggregate[route][key] == total, f'aggregate/{lang}/{route}/{key}: total mismatch')
    macro = report.get('macro_average')
    require(isinstance(macro, dict), 'missing macro_average')
    for route in ROUTES:
        value = sum(report['languages'][lang][lang if route == 'native' else 'byte']['bpb']
                    for lang in LANGUAGES) / len(LANGUAGES)
        same_number(macro.get(route + '_bpb'), value, f'macro_average.{route}_bpb')
    return ordered


def _interval(point, replicates, confidence):
    require(np.isfinite(point) and np.isfinite(replicates).all(), 'nonfinite bootstrap statistic')
    alpha = (1 - confidence) / 2
    bounds = np.quantile(replicates, [alpha, 1 - alpha])
    return {'estimate': float(point), 'ci_low': float(bounds[0]), 'ci_high': float(bounds[1])}


def validate_provenance(report):
    """Require production exports to identify weights, scorer and precision."""
    proof = report.get('provenance')
    require(isinstance(proof, dict) and bool(proof), 'missing export provenance')
    for key in ('checkpoint_files_sha256', 'experiment_sources_sha256'):
        mapping = proof.get(key)
        require(isinstance(mapping, dict) and bool(mapping), f'missing provenance.{key}')
        for name, digest in mapping.items():
            require(isinstance(name, str) and bool(name), f'invalid provenance filename in {key}')
            valid_hash(digest, f'provenance.{key}.{name}')
    files = proof['checkpoint_files_sha256']
    require('config.json' in files and any(name.endswith(('.safetensors', '.bin')) for name in files),
            'checkpoint provenance lacks config or weights')
    require(any(name.startswith('tokenizer/') for name in files), 'checkpoint provenance lacks tokenizer files')
    require(set(proof['experiment_sources_sha256']) == {
        'evaluate.py', 'modeling_mixture.py', 'configuration_mixture.py', 'tokenization_mixture.py'},
        'experiment source provenance is incomplete')
    for key in ('export_script_sha256', 'reference_sha256'):
        valid_hash(proof.get(key), f'provenance.{key}')
    for key in ('reference_path', 'precision_policy'):
        require(isinstance(proof.get(key), str) and bool(proof[key]), f'missing provenance.{key}')
    require(proof.get('device') in ('cpu', 'cuda'), 'unexpected provenance.device')
    packages = proof.get('packages')
    require(isinstance(packages, dict) and set(packages) == {'torch', 'transformers', 'tokenizers', 'numpy'},
            'package provenance incomplete')
    require(all(value is None or isinstance(value, str) for value in packages.values()),
            'invalid package version')
    require(isinstance(report.get('reference_comparison'), dict)
            and report['reference_comparison'].get('status') == 'passed',
            'accepted aggregate reference comparison did not pass')
    return proof


def _comparison(point, draws, confidence):
    # Axis 0 of point / axis 1 of draws is vocabulary size, ordered 10k then 20k.
    require(point[1] > 0 and (draws[:, 1] > 0).all(), '20k BPB is zero; relative difference undefined')
    return {
        'k10000_bpb': _interval(point[0], draws[:, 0], confidence),
        'k20000_bpb': _interval(point[1], draws[:, 1], confidence),
        'delta_bpb_20k_minus_10k': _interval(point[1] - point[0], draws[:, 1] - draws[:, 0], confidence),
        'k10000_reduction_percent': _interval(100 * (point[1] - point[0]) / point[1],
                                              100 * (draws[:, 1] - draws[:, 0]) / draws[:, 1], confidence),
    }


def analyze(input_root, repetitions=10000, seed=20261025, confidence=0.95):
    """Load eight conditions and use shared, language-stratified document draws."""
    number(repetitions, 'repetitions', integer=True, minimum=2)
    number(seed, 'bootstrap seed', integer=True)
    require(isinstance(confidence, (int, float)) and 0 < confidence < 1, 'confidence must be between 0 and 1')
    root = Path(input_root)
    reports, ordered, sources = {}, {}, []
    for mode in MODES:
        for training_seed in SEEDS:
            for size in SIZES:
                condition = (mode, training_seed, size)
                path = root / f'seed{training_seed}' / f'k{size}' / f'{mode}.json'
                raw = path.read_bytes()
                report = json.loads(raw)
                ordered[condition] = validate_report(report, training_seed, size, mode)
                validate_provenance(report)
                reports[condition] = report
                sources.append({'path': str(path.relative_to(root)), 'sha256': hashlib.sha256(raw).hexdigest()})
    reference_key = (MODES[0], SEEDS[0], SIZES[0])
    reference = reports[reference_key]
    for condition, report in reports.items():
        for key in ('metric', 'context_tokens', 'eos_policy', 'split', 'evaluated_text_sha256'):
            require(report[key] == reference[key], f'{condition}: {key} differs across conditions')
        require(report['model'] == reports[('bytes', condition[1], condition[2])]['model'],
                f'{condition}: model identity differs between context modes')
        proof = report['provenance']
        for key in ('experiment_sources_sha256', 'export_script_sha256', 'device', 'precision_policy', 'packages'):
            require(proof[key] == reference['provenance'][key], f'{condition}: provenance.{key} differs')
        paired_proof = reports[('bytes', condition[1], condition[2])]['provenance']
        require(proof['checkpoint_files_sha256'] == paired_proof['checkpoint_files_sha256'],
                f'{condition}: checkpoint hashes differ between context modes')
        tokenizer_hashes = {name: digest for name, digest in proof['checkpoint_files_sha256'].items()
                            if name.startswith('tokenizer/')}
        seed_reference = reports[('bytes', SEEDS[0], condition[2])]['provenance']['checkpoint_files_sha256']
        reference_tokenizer = {name: digest for name, digest in seed_reference.items() if name.startswith('tokenizer/')}
        require(tokenizer_hashes == reference_tokenizer, f'{condition}: tokenizer hashes differ between seeds')
        for lang in LANGUAGES:
            rows = ordered[condition][lang]
            ref_rows = ordered[reference_key][lang]
            require(len(rows) == len(ref_rows), f'{condition}/{lang}: document count differs')
            for row, ref in zip(rows, ref_rows):
                for key in ('document_index', 'text_sha256', 'source_bytes', 'source_characters'):
                    require(row[key] == ref[key], f'{condition}/{lang}/{row["document_index"]}: {key} differs')
    # (replicate, mode, seed, size, route, language): 5.12 MB at 10,000 draws.
    replicates = np.empty((repetitions, 2, 2, 2, 2, 4), dtype=np.float64)
    points = np.empty((2, 2, 2, 2, 4), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for li, lang in enumerate(LANGUAGES):
        ref_rows = ordered[reference_key][lang]
        n = len(ref_rows)
        byte_counts = np.asarray([row['source_bytes'] for row in ref_rows], dtype=np.float64)
        nll = np.empty((2, 2, 2, 2, n), dtype=np.float64)
        for mi, mode in enumerate(MODES):
            for si, training_seed in enumerate(SEEDS):
                for ki, size in enumerate(SIZES):
                    rows = ordered[(mode, training_seed, size)][lang]
                    for ri, route in enumerate((lang, 'byte')):
                        nll[mi, si, ki, ri] = [row['routes'][route]['text_nll'] for row in rows]
        points[..., li] = nll.sum(axis=-1) / (byte_counts.sum() * math.log(2))
        for start in range(0, repetitions, 128):
            end = min(start + 128, repetitions)
            indices = rng.integers(0, n, size=(end - start, n))
            denominators = byte_counts[indices].sum(axis=1) * math.log(2)
            require((denominators > 0).all(),
                    f'{lang}: a bootstrap replicate contains only empty documents; cannot compute BPB')
            # Same indices for all models, modes and routes preserve paired covariance.
            totals = np.take(nll, indices, axis=-1).sum(axis=-1)
            replicates[start:end, ..., li] = np.moveaxis(totals, -1, 0) / denominators[:, None, None, None, None]
    result = {
        'schema_version': 1, 'metric': METRIC,
        'method': 'paired document percentile bootstrap, stratified by language',
        'repetitions': repetitions, 'bootstrap_seed': seed, 'confidence_level': confidence,
        'training_seeds': list(SEEDS), 'lexical_sizes': list(SIZES),
        'split': reference['split'], 'context_tokens': reference['context_tokens'],
        'evaluated_text_sha256': reference['evaluated_text_sha256'],
        'documents_per_language': {lang: len(ordered[reference_key][lang]) for lang in LANGUAGES},
        'sources': sources,
        'aggregation': 'sum document text NLL / sum UTF-8 bytes / ln(2) per language; equal-language macro; mean BPB across two fixed seeds',
        'sign_convention': 'positive delta_bpb_20k_minus_10k or k10000_reduction_percent favors 10k',
        'limitations': [
            'Intervals condition on the trained checkpoints and reflect document resampling, not training-seed population uncertainty.',
            'The mean-of-two-seeds result averages BPB before computing differences and relative reductions; seeds are not resampled.',
            'Intervals are marginal percentile intervals without multiple-comparison adjustment.',
            'Document resampling assumes documents are independent units within each language; correlated source documents can narrow these intervals.',
            'These intervals are not downstream task accuracy or evidence of an isolated tokenizer mechanism.',
        ],
        'modes': {},
    }
    for mi, mode in enumerate(MODES):
        groups = {}
        for label, si in [('1337', 0), ('1338', 1), ('mean_of_two_fixed_seeds', None)]:
            # point: (size, route, lang), draw: (replicate, size, route, lang)
            point = points[mi].mean(axis=0) if si is None else points[mi, si]
            draw = replicates[:, mi].mean(axis=1) if si is None else replicates[:, mi, si]
            groups[label] = {}
            for ri, route in enumerate(ROUTES):
                groups[label][route] = {
                    'macro': _comparison(point[:, ri].mean(axis=-1), draw[:, :, ri].mean(axis=-1), confidence),
                    'languages': {lang: _comparison(point[:, ri, li], draw[:, :, ri, li], confidence)
                                  for li, lang in enumerate(LANGUAGES)},
                }
        result['modes'][mode] = {'context_policy': POLICIES[mode], 'comparisons': groups}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repetitions', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261025)
    parser.add_argument('--confidence', type=float, default=0.95)
    parser.add_argument('--force', action='store_true', help='Explicitly replace an existing comparison output')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.force:
        parser.error(f'Output exists: {output}; use another path or explicit --force')
    result = analyze(args.input_root, args.repetitions, args.seed, args.confidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w' if args.force else 'x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(f'Saved {output.resolve()}')
    for mode in MODES:
        stat = result['modes'][mode]['comparisons']['mean_of_two_fixed_seeds']['native']['macro']['k10000_reduction_percent']
        print(f'{mode}: 10k reduction {stat["estimate"]:.3f}% '
              f'({100 * args.confidence:g}% document CI {stat["ci_low"]:.3f} to {stat["ci_high"]:.3f}%)')


if __name__ == '__main__':
    main()
