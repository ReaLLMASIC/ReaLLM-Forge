"""Analytic CPU checks against the original evaluator; no HF downloads needed."""
import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from evaluate_documents import (collect_documents, export_records, load_base,
                                publish_json, verify_reference)


class ByteTokenizer:
    lexical_size = 10000

    def encode_route(self, text, route):
        return list(text.encode('utf-8'))

    def bos_id(self, route):
        return 256 + 3 * int(route)

    def eos_id(self, route):
        return 257 + 3 * int(route)


class UniformModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.placeholder = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_position_embeddings=8)

    def forward(self, input_ids, tokenizer_ids, **kwargs):
        logits = torch.full((*input_ids.shape, 271), -float('inf'))
        logits[..., :256] = 0
        logits[..., 257 + 3 * int(tokenizer_ids)] = 0
        return SimpleNamespace(logits=logits)


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ.get('MIXTURE_EVALUATOR_SOURCE', 'evaluate.py')).resolve()
        if not source.is_file():
            raise unittest.SkipTest('Run from the experiment directory containing evaluate.py')
        cls.base = load_base(source.parent)
        torch.set_num_threads(1)

    def test_export_preserves_every_original_aggregate_in_both_modes(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            texts = ['中 A🙂\t\r\n0123456789', '', 'repeat', 'repeat', 'x' * 23]
            for language in ('zh', 'ko', 'ja', 'en'):
                (data / language).mkdir()
                (data / language / 'val.jsonl').write_text(
                    '\n'.join(json.dumps({'text': t}) for t in texts), encoding='utf-8')
            docs, hashes = collect_documents(self.base, data, 'val', 1000)
            for mode in ('bytes', 'token'):
                with self.subTest(mode=mode):
                    model, tokenizer = UniformModel(), ByteTokenizer()
                    reference = self.base.evaluate_corpus(model, tokenizer, data, max_documents=1000,
                                                          context_mode=mode, device='cpu')
                    reference.update(lexical_size=10000)
                    records, languages, macro = export_records(self.base, model, tokenizer, docs,
                                                                mode, 'cpu', progress=False)
                    self.assertEqual(languages, reference['languages'])
                    self.assertEqual(macro, reference['macro_average'])
                    self.assertEqual(hashes, reference['evaluated_text_sha256'])
                    self.assertEqual(len(records), 20)
                    report = dict(reference, records=records, languages=languages, macro_average=macro)
                    self.assertEqual(verify_reference(report, reference)['status'], 'passed')
                    for row in records:
                        for scored in row['routes'].values():
                            self.assertEqual(scored['eos_count'], 1)
                            self.assertEqual(scored['text_tokens'], row['source_bytes'])
                            self.assertAlmostEqual(scored['text_nll'], row['source_bytes'] * math.log(257), places=4)
                    # Same text appears twice: occurrences retain distinct indices.
                    self.assertEqual(records[2]['text_sha256'], records[3]['text_sha256'])
                    self.assertNotEqual(records[2]['document_index'], records[3]['document_index'])
                    wrong = copy.deepcopy(report)
                    wrong['languages']['zh']['zh']['bpb'] += 0.01
                    with self.assertRaisesRegex(ValueError, 'BPB differs'):
                        verify_reference(wrong, reference)
                    wrong = copy.deepcopy(report)
                    wrong['languages']['zh']['zh']['text_tokens'] += 1
                    with self.assertRaisesRegex(ValueError, 'counter mismatch'):
                        verify_reference(wrong, reference)

    def test_atomic_publisher_refuses_to_replace_results(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'nested/report.json'
            publish_json(p, {'value': 1})
            with self.assertRaises(FileExistsError):
                publish_json(p, {'value': 2})
            self.assertEqual(json.loads(p.read_text()), {'value': 1})
            self.assertEqual(list(p.parent.glob('*.partial')), [])


if __name__ == '__main__':
    unittest.main()
