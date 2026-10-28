"""Small CPU-only tests for arithmetic, pairing and bootstrap semantics."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from paired_bootstrap import (analyze, validate_report, LANGUAGES, SEEDS, SIZES,
                              MODES, TOTAL_KEYS, POLICIES, METRIC, EOS_POLICY)


def summary(nbytes, bpb, documents=1):
    values = dict(documents=documents, text_nll=nbytes * math.log(2) * bpb,
                  text_tokens=nbytes, source_bytes=nbytes, source_characters=nbytes,
                  fallback_tokens=nbytes, fallback_bytes=nbytes,
                  eos_nll=float(documents), eos_count=documents, windows=documents)
    return summarize(values)


def summarize(values):
    result = dict(values)
    nb, nt = values['source_bytes'], values['text_tokens']
    result.update(bpb=values['text_nll'] / nb / math.log(2) if nb else None,
                  mean_token_nll=values['text_nll'] / nt if nt else None,
                  bytes_per_token=nb / nt if nt else None,
                  characters_per_token=values['source_characters'] / nt if nt else None,
                  fallback_token_fraction=values['fallback_tokens'] / nt if nt else None,
                  fallback_source_byte_fraction=values['fallback_bytes'] / nb if nb else None,
                  mean_eos_nll=values['eos_nll'] / values['eos_count'])
    return result


def report(seed=1337, size=10000, mode='bytes', texts=('a', 'b' * 9), base=(1., 3.)):
    offset = (0.5 if size == 20000 else 0) + (0.2 if seed == 1338 else 0) + (0.1 if mode == 'token' else 0)
    records, languages, hashes = [], {}, {}
    for lang in LANGUAGES:
        digest = hashlib.sha256()
        for index, text in enumerate(texts):
            raw = text.encode()
            digest.update(len(raw).to_bytes(8, 'little')); digest.update(raw)
            records.append(dict(language=lang, document_index=index,
                                text_sha256=hashlib.sha256(raw).hexdigest(),
                                source_bytes=len(raw), source_characters=len(text),
                                routes={lang: summary(len(raw), base[index] + offset),
                                        'byte': summary(len(raw), base[index] + offset + 1)}))
        hashes[lang] = digest.hexdigest()
        languages[lang] = {}
        for route in (lang, 'byte'):
            total = {key: sum(row['routes'][route][key] for row in records if row['language'] == lang)
                     for key in TOTAL_KEYS}
            languages[lang][route] = summarize(total)
    digest = lambda value: hashlib.sha256(str(value).encode()).hexdigest()
    proof = dict(checkpoint_files_sha256={
        'config.json': digest((seed, size, 'config')), 'model.safetensors': digest((seed, size, 'weights')),
        'tokenizer/manifest.json': digest((size, 'tokenizer'))},
        experiment_sources_sha256={name: digest(name) for name in
            ('evaluate.py', 'modeling_mixture.py', 'configuration_mixture.py', 'tokenization_mixture.py')},
        export_script_sha256=digest('export'), reference_path=f'{seed}/{size}/{mode}',
        reference_sha256=digest((seed, size, mode)), device='cpu', precision_policy='test FP32',
        packages={name: 'test' for name in ('torch', 'transformers', 'tokenizers', 'numpy')})
    return dict(schema_version=1, training_seed=seed, lexical_size=size, context_mode=mode,
                metric=METRIC, context_tokens=512, context_policy=POLICIES[mode],
                eos_policy=EOS_POLICY, split='val', model=f'model/{seed}/{size}',
                languages=languages, evaluated_text_sha256=hashes, records=records,
                provenance=proof, reference_comparison={'status': 'passed'},
                macro_average={kind + '_bpb': (sum(languages[lang][lang if kind == 'native' else 'byte']['bpb']
                                                for lang in LANGUAGES) / 4 if any(texts) else None)
                               for kind in ('native', 'byte')})


def write_reports(root):
    for seed in SEEDS:
        for size in SIZES:
            for mode in MODES:
                path = Path(root) / f'seed{seed}' / f'k{size}' / f'{mode}.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report(seed, size, mode)))


class ValidationTests(unittest.TestCase):
    def test_ratio_of_sums_and_order(self):
        obj = report()
        obj['records'].reverse()
        indexed = validate_report(obj)
        self.assertEqual([r['document_index'] for r in indexed['zh']], [0, 1])
        self.assertAlmostEqual(obj['macro_average']['native_bpb'], 2.8)
        self.assertNotAlmostEqual(obj['macro_average']['native_bpb'], 2.0)

    def test_duplicate_and_gap_rejected(self):
        obj = report(); obj['records'].append(copy.deepcopy(obj['records'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_report(obj)
        obj = report(); obj['records'][0]['document_index'] = 2
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            validate_report(obj)

    def test_wrong_aggregate_rejected(self):
        obj = report(); obj['languages']['zh']['zh']['documents'] += 1
        with self.assertRaises(ValueError):
            validate_report(obj)
        obj = report(); obj['macro_average']['native_bpb'] += .1
        with self.assertRaisesRegex(ValueError, 'recomputed'):
            validate_report(obj)

    def test_bad_policy_and_routes_rejected(self):
        obj = report(); obj['context_policy'] = 'rolling'
        with self.assertRaisesRegex(ValueError, 'context_policy'):
            validate_report(obj)
        obj = report(); obj['records'][0]['routes']['other'] = obj['records'][0]['routes'].pop('byte')
        with self.assertRaisesRegex(ValueError, 'route set'):
            validate_report(obj)

    def test_empty_document_permitted_all_empty_rejected(self):
        validate_report(report(texts=('', 'abc')))
        with self.assertRaisesRegex(ValueError, 'all documents are empty'):
            validate_report(report(texts=('', '')))

    def test_nonfinite_and_invalid_hash_rejected(self):
        obj = report(); obj['records'][0]['routes']['zh']['text_nll'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            validate_report(obj)
        obj = report(); obj['records'][0]['text_sha256'] = 'not a hash'
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            validate_report(obj)


class BootstrapTests(unittest.TestCase):
    def test_constant_paired_gap_and_fixed_seed_mean(self):
        with tempfile.TemporaryDirectory() as root:
            write_reports(root)
            result = analyze(root, repetitions=250, seed=42)
            comparison = result['modes']['bytes']['comparisons']['mean_of_two_fixed_seeds']['native']['macro']
            self.assertAlmostEqual(comparison['k10000_bpb']['estimate'], 2.9)
            self.assertAlmostEqual(comparison['k20000_bpb']['estimate'], 3.4)
            for key in ('estimate', 'ci_low', 'ci_high'):
                self.assertAlmostEqual(comparison['delta_bpb_20k_minus_10k'][key], .5)
            self.assertAlmostEqual(comparison['k10000_reduction_percent']['estimate'], 100 * .5 / 3.4)
            other_mode = result['modes']['token']['comparisons']['mean_of_two_fixed_seeds']['native']['macro']
            for key in ('estimate', 'ci_low', 'ci_high'):
                self.assertAlmostEqual(other_mode['k10000_bpb'][key] - comparison['k10000_bpb'][key], .1)
            self.assertEqual(result, analyze(root, repetitions=250, seed=42))

    def test_cross_condition_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            write_reports(root)
            path = Path(root) / 'seed1338/k20000/token.json'
            obj = json.loads(path.read_text()); obj['records'][0]['text_sha256'] = '0' * 64
            path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError, 'text_sha256 differs'):
                analyze(root, 20)

    def test_filename_metadata_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            write_reports(root)
            path = Path(root) / 'seed1338/k20000/token.json'
            obj = json.loads(path.read_text()); obj['training_seed'] = 1337
            path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError, 'training_seed does not match'):
                analyze(root, 20)

    def test_mode_pairing_and_split_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            write_reports(root)
            path = Path(root) / 'seed1338/k20000/token.json'
            obj = json.loads(path.read_text()); obj['split'] = 'test'
            path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError, 'split differs'):
                analyze(root, 20)

    def test_changed_weights_source_and_precision_rejected(self):
        for field in ('checkpoint_files_sha256', 'experiment_sources_sha256', 'precision_policy'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                write_reports(root)
                path = Path(root) / 'seed1338/k20000/token.json'
                obj = json.loads(path.read_text())
                if field == 'checkpoint_files_sha256':
                    obj['provenance'][field]['model.safetensors'] = '0' * 64
                elif field == 'experiment_sources_sha256':
                    obj['provenance'][field]['evaluate.py'] = '0' * 64
                else:
                    obj['provenance'][field] = 'changed'
                path.write_text(json.dumps(obj))
                with self.assertRaisesRegex(ValueError, 'differs|differ'):
                    analyze(root, 20)

    def test_tokenizer_change_between_seeds_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            write_reports(root)
            for mode in MODES:
                path = Path(root) / f'seed1338/k20000/{mode}.json'
                obj = json.loads(path.read_text())
                obj['provenance']['checkpoint_files_sha256']['tokenizer/manifest.json'] = '0' * 64
                path.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError, 'tokenizer hashes differ'):
                analyze(root, 20)

    def test_zero_denominator_draw_has_clear_error(self):
        with tempfile.TemporaryDirectory() as root:
            write_reports(root)
            for seed in SEEDS:
                for size in SIZES:
                    for mode in MODES:
                        path = Path(root) / f'seed{seed}/k{size}/{mode}.json'
                        path.write_text(json.dumps(report(seed, size, mode, texts=('', 'x'))))
            with self.assertRaisesRegex(ValueError, 'bootstrap replicate contains only empty documents'):
                analyze(root, repetitions=100, seed=42)


if __name__ == '__main__':
    unittest.main()
