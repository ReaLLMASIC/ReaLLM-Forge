import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import run_benchmarks as runner


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.output = self.project / 'output'
        self.args = argparse.Namespace(device='cpu', threads=1, split_set='final', max_documents=1)
        self.jobs = runner.make_jobs(self.project, self.output, self.args)
        for name in ('evaluate.py', 'modeling_mixture.py', 'configuration_mixture.py', 'tokenization_mixture.py'):
            (self.project / name).write_text('# fixture\n')
        for lang in runner.LANGUAGES:
            for split in ('train', 'val', 'test'):
                path = self.project / 'data/prepared' / lang / (split + '.jsonl')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({'text': lang + split}) + '\n')
        for job in self.jobs:
            model = job['model']
            (model / 'tokenizer').mkdir(parents=True)
            fp = 'tokenizer-' + str(job['size'])
            effective = {'seed': job['seed'], 'target_tokens': 100_000_000}
            (job['run'] / 'run_config.json').write_text(json.dumps({'effective_training': effective, 'tokenizer_fingerprint': fp}))
            (job['run'] / 'summary.json').write_text(json.dumps({'settings': effective, 'tokenizer_fingerprint': fp,
                'budget_reached': True, 'interrupted': False, 'predicted_tokens': 100_000_001}))
            (model / 'config.json').write_text(json.dumps({'lexical_size': job['size'], 'max_position_embeddings': 512,
                'tokenizer_fingerprint': fp}))
            (model / 'model.safetensors').write_bytes(b'fixture-not-a-real-model')

    def test_four_models_eight_sequential_evaluations(self):
        self.assertEqual(runner.preflight(self.project, self.jobs, self.args), [])
        self.assertEqual([(j['seed'], j['size']) for j in self.jobs], [(1337,10000),(1337,20000),(1338,10000),(1338,20000)])
        commands = [c for j in self.jobs for c in j['commands']]
        self.assertEqual(len(commands), 8)
        self.assertTrue(all('--skip-existing' in c for c in commands))
        self.assertTrue(all('train.py' not in ' '.join(c) for c in commands))

    def test_seed_mismatch_rejected_before_scoring(self):
        path = self.jobs[0]['run'] / 'run_config.json'
        value = json.loads(path.read_text()); value['effective_training']['seed'] = 5
        path.write_text(json.dumps(value))
        self.assertTrue(any('training seed' in e for e in runner.preflight(self.project, self.jobs, self.args)))

    def test_document_shortage_is_not_silently_truncated(self):
        self.args.max_documents = 2
        self.assertEqual(sum('has 1 documents' in e for e in runner.preflight(self.project, self.jobs, self.args)), 4)

    def test_dry_run_has_no_output_or_subprocess(self):
        with mock.patch.object(runner.subprocess, 'run') as launch, redirect_stdout(io.StringIO()) as output:
            code = runner.main(['--project-root', str(self.project), '--output', 'output', '--max-documents', '1', '--dry-run'])
        self.assertEqual(code, 0)
        launch.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertIn('prepare_tasks.py', output.getvalue())
        self.assertIn('report_tasks.py', output.getvalue())

    def test_lock_does_not_remove_someone_elses_lock(self):
        with runner.output_lock(self.output):
            lock = self.output / '.benchmark-run.lock'
            contents = lock.read_bytes()
            with self.assertRaises(RuntimeError):
                with runner.output_lock(self.output):
                    pass
            self.assertEqual(lock.read_bytes(), contents)
        self.assertFalse(lock.exists())

    def populate_outputs(self):
        for job in self.jobs:
            for key in ('lm_output', 'task_output'):
                job[key].parent.mkdir(parents=True, exist_ok=True)
                job[key].write_text('{}\n')
        (self.output / 'tasks').mkdir(parents=True)
        (self.output / 'tasks/manifest.json').write_text('{}\n')
        (self.output / 'report').mkdir()
        (self.output / 'report/report.md').write_text('Synthetic fixture only\n')

    def test_archive_closed_checked_and_preserves_previous_run(self):
        self.populate_outputs()
        archive = runner.collect_archive(self.project, self.output, self.jobs, [], self.args)
        original = archive.read_bytes()
        second = runner.collect_archive(self.project, self.output, self.jobs, [], self.args)
        self.assertNotEqual(archive, second)
        self.assertEqual(archive.read_bytes(), original)
        with zipfile.ZipFile(archive) as z:
            self.assertIsNone(z.testzip())
            manifest = json.loads(z.read('manifest.json'))
            for entry in manifest['files']:
                self.assertEqual(hashlib.sha256(z.read(entry['path'])).hexdigest(), entry['sha256'])
            self.assertFalse(any(name.endswith('.safetensors') for name in z.namelist()))

    def test_preexisting_staging_file_is_preserved(self):
        self.populate_outputs()
        with mock.patch.object(runner, 'datetime') as clock, mock.patch.object(runner.uuid, 'uuid4') as identifier:
            clock.now.return_value.strftime.return_value = 'fixture'
            clock.now.return_value.isoformat.return_value = 'fixture'
            identifier.return_value.hex = 'a' * 32
            path = self.project / 'language-benchmark-results-fixture-aaaaaaaa.zip.partial'
            path.write_bytes(b'older interrupted run')
            with self.assertRaises(FileExistsError):
                runner.collect_archive(self.project, self.output, self.jobs, [], self.args)
            self.assertEqual(path.read_bytes(), b'older interrupted run')


if __name__ == '__main__':
    unittest.main()
