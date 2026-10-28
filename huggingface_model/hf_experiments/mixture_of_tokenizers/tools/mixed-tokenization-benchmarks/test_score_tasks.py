"""Analytic scorer tests; stdlib only, no model downloads or GPU required."""
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from score_tasks import (aggregate, canonical_hash, eligibility, load_tasks,
                         metrics_from_scores, protocol, publish_json,
                         read_training_identity, score_one, task_identity,
                         validate_existing, validate_task_manifest)


def task(**changes):
    row = {'id': 'sample:val:0', 'task': 'sample', 'language': 'en', 'split': 'val',
           'prompt': 'P', 'choices': ['a', 'bb'], 'answer_index': 0}
    row.update(changes)
    return row


class FakeTokenizer:
    def __init__(self):
        self.encodings = []

    def encode_route(self, text, route):
        self.encodings.append((text, route))
        # A native merged double-b answer; byte encoding remains literal.
        if route and text == 'bb':
            return [300]
        return list(text.encode('utf-8'))

    def bos_id(self, route):
        return 500 + route


class FakeBase:
    def __init__(self):
        self.calls = []

    def _score_ids(self, model, ids, route, context, device):
        self.calls.append((list(ids), route))
        for position, token in enumerate(ids[1:], 1):
            # Distinct prompt NLL should never be counted in continuation score.
            loss = 100.0 if token == ord('P') else (0.9 if token == 300 else (0.6 if token == ord('a') else 0.5))
            yield position, token, loss


MODEL = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))


class ScoreTests(unittest.TestCase):
    def test_answer_boundary_and_no_eos_native_byte_pair(self):
        base, tokenizer = FakeBase(), FakeTokenizer()
        row, skipped = score_one(base, MODEL, tokenizer, task())
        self.assertIsNone(skipped)
        self.assertEqual(row['routes']['native']['sum_log_probs'], [-.6, -.9])
        self.assertEqual(row['routes']['byte']['sum_log_probs'], [-.6, -1.0])
        self.assertEqual(row['routes']['native']['choice_tokens'], [1, 1])
        self.assertEqual(row['routes']['byte']['choice_tokens'], [1, 2])
        self.assertTrue(all(ids[0] == 500 + route for ids, route in base.calls))
        self.assertEqual([ids[-1] for ids, route in base.calls], [97, 300, 97, 98])
        # Prompt and answers must be separate calls; no P+choice encoding.
        self.assertNotIn(('Pa', 4), tokenizer.encodings)
        self.assertAlmostEqual(row['routes']['native']['raw']['option_log_loss'], math.log1p(math.exp(-.3)))

    def test_byte_normalization_changes_decision_and_uses_utf8_not_tokens(self):
        row, _ = score_one(FakeBase(), MODEL, FakeTokenizer(), task())
        native = row['routes']['native']
        self.assertEqual(native['raw']['prediction'], 0)
        self.assertEqual(native['byte_normalized']['prediction'], 1)
        self.assertEqual(native['sum_log_probs_per_utf8_byte'], [-.6, -.45])
        unicode_row, _ = score_one(FakeBase(), MODEL, FakeTokenizer(), task(choices=['é', 'x']))
        self.assertEqual(unicode_row['choices_utf8_bytes'], [2, 1])

    def test_exact_ties_fractional_and_stable_extreme_softmax(self):
        result = metrics_from_scores([-10000., -10000., -10001.], 1)
        self.assertEqual(result['prediction'], 0)
        self.assertEqual(result['tied_predictions'], [0, 1])
        self.assertEqual(result['correct_fraction'], .5)
        self.assertAlmostEqual(result['option_log_loss'], math.log(2 + math.exp(-1)))
        self.assertEqual(result['correct_vs_best_incorrect_margin'], 0)
        self.assertAlmostEqual(metrics_from_scores([-1e300, -1e300], 0)['option_log_loss'], math.log(2))

    def test_whole_question_common_byte_eligibility_and_boundary(self):
        self.assertTrue(eligibility(task(prompt='x' * 510, choices=['a', 'b']))[0])
        too_long = task(prompt='x' * 510, choices=['a', 'bb'])
        self.assertFalse(eligibility(too_long)[0])
        base = FakeBase(); row, skipped = score_one(base, MODEL, FakeTokenizer(), too_long)
        self.assertIsNone(row)
        self.assertEqual(skipped['reason'], 'byte_context_exceeded')
        self.assertEqual(base.calls, [])

    def test_bad_tokenizer_and_scores_fail(self):
        class LongTokenizer(FakeTokenizer):
            def encode_route(self, text, route):
                ids = super().encode_route(text, route)
                return ids + [1] if route else ids
        with self.assertRaisesRegex(ValueError, 'more tokens'):
            score_one(FakeBase(), MODEL, LongTokenizer(), task())
        for scores in ([float('nan'), -1], [float('-inf'), -1], [.1, -1]):
            with self.subTest(scores=scores), self.assertRaisesRegex(ValueError, 'Invalid'):
                metrics_from_scores(scores, 0)
        class MissingTargets(FakeBase):
            def _score_ids(self, *args):
                return iter([])
        with self.assertRaisesRegex(ValueError, 'omitted'):
            score_one(MissingTargets(), MODEL, FakeTokenizer(), task())

    def test_aggregate_fractional_accuracy_and_empty_groups(self):
        row, _ = score_one(FakeBase(), MODEL, FakeTokenizer(), task())
        tied = copy.deepcopy(row)
        for route in tied['routes'].values():
            route['raw'] = metrics_from_scores([-1, -1], 0)
        summary = aggregate([row, tied], [])
        self.assertEqual(summary['overall']['routes']['native']['raw']['accuracy'], .75)
        self.assertEqual(summary['overall']['routes']['native']['raw']['exact_tie_count'], 1)
        _, skipped = score_one(FakeBase(), MODEL, FakeTokenizer(), task(prompt='x' * 512))
        summary = aggregate([], [skipped])
        self.assertIsNone(summary['overall']['routes']['native']['raw']['accuracy'])
        self.assertEqual(summary['overall']['skipped_count'], 1)

    def test_existing_validation_and_atomic_no_overwrite(self):
        tasks = [task()]
        record, _ = score_one(FakeBase(), MODEL, FakeTokenizer(), tasks[0])
        expected = {'protocol': protocol(512)}
        result = {**expected, 'records': [record], 'skipped': [],
                  'eligible_ids': [record['id']], 'skipped_ids': [],
                  'subset_sha256': canonical_hash([[record['id'], record['example_sha256']]]),
                  'aggregates': aggregate([record], [])}
        validate_existing(result, tasks, expected)
        wrong = copy.deepcopy(result); wrong['records'][0]['routes']['native']['raw']['correct_fraction'] = 0
        with self.assertRaisesRegex(ValueError, 'Stored metrics'):
            validate_existing(wrong, tasks, expected)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'result.json'
            publish_json(p, result)
            with self.assertRaises(FileExistsError):
                publish_json(p, {'other': True})
            self.assertEqual(json.loads(p.read_text()), result)
            self.assertEqual(list(Path(directory).glob('*.partial')), [])

    def test_training_seed_comes_from_effective_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint = root / 'checkpoint-best'; checkpoint.mkdir()
            (checkpoint / 'config.json').write_text(json.dumps({'lexical_size': 10000, 'initializer_seed': 1338}))
            (root / 'run_config.json').write_text(json.dumps({'config': {'training': {'seed': 1337}}, 'effective_training': {'seed': 1338}}))
            (root / 'summary.json').write_text(json.dumps({'settings': {'seed': 1338}}))
            identity = read_training_identity(checkpoint, 1338)
            self.assertEqual(identity['training_seed'], 1338)
            with self.assertRaisesRegex(ValueError, 'Requested seed'):
                read_training_identity(checkpoint, 1337)

    def test_prepared_manifest_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tasks.jsonl'
            path.write_text(json.dumps(task()) + '\n')
            rows = load_tasks(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = {'schema_version': 1, 'tasks_sha256': digest, 'record_count': 1, 'split_set': 'dev',
                        'template_version': 'v1', 'tasks': [{'task': 'sample', 'language': 'en', 'split': 'val',
                        'count': 1, 'num_choices': 2, 'chance_accuracy': .5}]}
            validate_task_manifest(rows, manifest, digest)
            manifest['record_count'] = 2
            with self.assertRaisesRegex(ValueError, 'record count'):
                validate_task_manifest(rows, manifest, digest)
            path.write_text((json.dumps(task()) + '\n') * 2)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_tasks(path)


if __name__ == '__main__':
    unittest.main()
