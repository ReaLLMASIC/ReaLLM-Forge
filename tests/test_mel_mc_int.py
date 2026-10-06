"""CPU regressions for the mel adapter. Run: python -m unittest discover -s tests -p test_mel_mc_int.py"""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'data/mel_mc_int'), str(ROOT/'data/mel_spectrogram'), str(ROOT)]
import mel_mc_int_tools as mel
import pipeline
import audio_to_token_mel as enc
import token_mel_to_audio as dec
from make_viewer import COMMAND_JS, build_viewer
from train_variations.sequence_windows import SequenceWindows, load_sequence_windows


class MelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.root_patch = patch.object(mel, 'REPO_ROOT', self.directory)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def fixture(self, name='input.csv', rows=30, bands=2, hop=20., levels=64, reference=1., silent=False):
        states = np.zeros((rows, bands), dtype=np.int64) if silent else np.arange(rows*bands).reshape(rows,bands) % levels
        preset = enc.Preset('test','test',16000,1024,60.,hop,bands,10.,7000.,96.,levels)
        metadata = enc.build_roundtrip_metadata(preset, rows*preset.hop_length, rows, False, reference,
                   'fixed', silent, 1., dec.canonical_state_crc(states, levels))
        path = self.directory/name
        mel.write_mel_state_csv(path, [f'mel_{i:03d}_q' for i in range(bands)], states, metadata)
        return path, states, metadata

    def concat(self, paths, name='combined.csv'):
        output = self.directory/name
        mel.cmd_concat_csv(NS(input_csvs=paths, input_list=None, input_json=None,
                              output_csv=output, manifest_json=None))
        return output

    def prepare(self, block=4):
        a, _, _ = self.fixture('a.csv')
        b, _, _ = self.fixture('b.csv')
        return mel.prepare_segments({'train':[{'csv':str(a)}], 'val':[{'csv':str(b)}]}, 'dataset', block)

    def test_metadata_after_or_before_header_and_nondefault_hop(self):
        path, _, _ = self.fixture()
        lines = path.read_text().splitlines(True)
        other = self.directory/'metadata_first.csv'
        other.write_text(lines[1]+lines[0]+''.join(lines[2:]))
        for source in (path, other):
            self.assertIsNotNone(mel.read_mel_csv(source)[3])
            mel.cmd_cut_prompt(NS(mel_csv=source, cutoff_s=.1, output_dir=self.directory/'prompt', fully_observed=False))
            manifest=json.loads((self.directory/'prompt/prompt_manifest.json').read_text())
            self.assertEqual(manifest['prompt_frames'], 5)

    def test_corrupt_crc_and_declared_rows_are_rejected(self):
        path, _, _ = self.fixture()
        original=path.read_text()
        path.write_text(original.replace('0,1\n', '2,3\n', 1))
        with self.assertRaisesRegex(ValueError,'CRC'):
            self.concat([path])
        path.write_text(original + '0,1\n')
        with self.assertRaisesRegex(ValueError,'frame count'):
            mel.validate_csv(path)

    def test_reference_or_normalizer_mismatch_is_rejected(self):
        a, _, _=self.fixture('a.csv')
        b, states, metadata=self.fixture('b.csv',reference=100.)
        with self.assertRaisesRegex(ValueError,'Incompatible'):
            self.concat([a,b])
        metadata['quantizer']['reference_power']=1.
        metadata['stft']['power_normalizer']=2.
        mel.write_mel_state_csv(b,['mel_000_q','mel_001_q'],states,metadata)
        with self.assertRaisesRegex(ValueError,'Incompatible'):
            self.concat([a,b])

    def test_silent_input_does_not_mute_concat_or_generation(self):
        a, _, _=self.fixture('silent.csv',silent=True)
        b, _, _=self.fixture('signal.csv')
        result=dec.read_csv_container(self.concat([a,b]))
        self.assertFalse(result.metadata['quantizer']['silent'])
        generated=self.directory/'generated.csv'
        generated.write_text('mel_000_q,mel_001_q\n63,63\n63,63\n')
        output=self.directory/'wrapped.csv'
        mel.cmd_wrap_csv(NS(input_csv=generated,reference_mel_csv=a,output_csv=output))
        result=dec.read_csv_container(output)
        self.assertGreater(float(dec.dequantize_mel_power(result.states,result.metadata,'zero').max()),0)

    def test_columns_and_quantizer_override_are_validated(self):
        path, _, _=self.fixture()
        path.write_text(path.read_text().replace('mel_000_q,mel_001_q','mel_001_q,mel_000_q'))
        with self.assertRaisesRegex(ValueError,'canonical'):
            mel.validate_csv(path)
        path, _, _=self.fixture()
        with self.assertRaisesRegex(ValueError,'disagrees'):
            mel.cmd_prepare(NS(input_csv=path,train_ratio=.8,states_per_column=32,
                               output_root='bad',block_size=1,buffer_rows=4))

    def test_ratios_and_short_splits_fail_before_publish(self):
        for ratio in [0,1,-.1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):
                mel.check_ratio(ratio)
        with self.assertRaisesRegex(ValueError,'at least 381'):
            self.prepare(block=380)
        self.assertFalse((self.directory/'data/dataset/manifest.json').exists())

    def test_temporal_guard_and_small_context(self):
        path, _, metadata=self.fixture(rows=80)
        mel.cmd_prepare(NS(input_csv=path,train_ratio=.5,states_per_column=64,
                           output_root='temporal',block_size=4,buffer_rows=3))
        manifest=json.loads((self.directory/'data/temporal/manifest.json').read_text())
        self.assertEqual(manifest['train_rows'],40)
        self.assertEqual(manifest['val_rows'],37)
        self.assertEqual(manifest['splits']['val'][0]['start'],43)

    def test_uint16_uint32_boundaries_read_like_training(self):
        for levels in [65535,65536]:
            paths=[]
            for name in ['train','val']:
                path, _, metadata=self.fixture(f'{name}{levels}.csv',rows=2,levels=levels)
                states=np.array([[0,1],[levels-1,2]],dtype=np.int64)
                metadata['integrity']['state_crc32']=f'{dec.canonical_state_crc(states,levels):08x}'
                mel.write_mel_state_csv(path,['mel_000_q','mel_001_q'],states,metadata)
                paths.append(path)
            manifest=mel.prepare_segments({'train':[{'csv':str(paths[0])}], 'val':[{'csv':str(paths[1])}]}, f'd{levels}',1)
            dtype=np.uint32 if levels>np.iinfo(np.uint16).max else np.uint16
            values=np.fromfile(self.directory/'data'/manifest['multicontext_datasets'][0]/'train.bin',dtype=dtype)
            self.assertEqual(values.tolist(),[0,levels-1])

    def test_all_window_ranks_stay_within_one_recording(self):
        windows=SequenceWindows([[0,9],[9,16],[16,27]],27,4)
        starts=windows.starts(np.arange(windows.total))
        self.assertEqual(starts.tolist(),list(range(5))+list(range(9,12))+list(range(16,23)))
        for start in starts:
            self.assertTrue(any(lo<=start and start+4<hi for lo,hi in windows.ranges))

    def test_manifest_reuse_and_channel_alignment(self):
        manifest=self.prepare()
        self.assertEqual(manifest,self.prepare())
        directories=[self.directory/'data'/name for name in manifest['multicontext_datasets']]
        windows=load_sequence_windows(directories,{'train':[30,30],'val':[30,30]},4)
        self.assertEqual(windows['train'].total,26)
        with self.assertRaisesRegex(ValueError,'different lengths'):
            load_sequence_windows(directories,{'train':[30,29],'val':[30,30]},4)
        with (directories[0]/'train.bin').open('ab') as stream:
            stream.write(b'x')
        with self.assertRaisesRegex(ValueError,'immutable'):
            self.prepare()

    def test_checkpoint_order_and_vocab_are_checked(self):
        manifest=self.prepare()
        checkpoint={'model_args':{'multicontext':True,'vocab_sizes':[64,64],'block_size':4},
                    'config':{'multicontext_datasets':manifest['multicontext_datasets']}}
        with patch.object(pipeline,'ROOT',self.directory):
            self.assertEqual(pipeline.validate_checkpoint(checkpoint,manifest),4)
            checkpoint['config']['multicontext_datasets']=list(reversed(manifest['multicontext_datasets']))
            with self.assertRaisesRegex(ValueError,'order'):
                pipeline.validate_checkpoint(checkpoint,manifest)
            checkpoint['config']['multicontext_datasets']=manifest['multicontext_datasets']
            checkpoint['model_args']['vocab_sizes']=[32,64]
            with self.assertRaisesRegex(ValueError,'vocabularies'):
                pipeline.validate_checkpoint(checkpoint,manifest)

    def test_boundaries_are_opt_in_and_required_for_every_channel(self):
        manifest=self.prepare()
        directories=[self.directory/'data'/name for name in manifest['multicontext_datasets']]
        for i, directory in enumerate(directories):
            path=directory/'meta.pkl'
            metadata=pickle.loads(path.read_bytes())
            metadata.pop('sequence_ranges_file')
            path.write_bytes(pickle.dumps(metadata))
            if i == 0:
                with self.assertRaisesRegex(ValueError,'All aligned channels'):
                    load_sequence_windows(directories,{'train':[30,30],'val':[30,30]},4)
        # Legacy datasets retain their existing sampler, including existing validation behavior.
        self.assertIsNone(load_sequence_windows(directories,{'train':[30,29],'val':[30,30]},100))

    @unittest.skipUnless(shutil.which('node'),'Node is needed to execute viewer JavaScript')
    def test_viewer_shell_arguments_roundtrip_without_expansion(self):
        for value in ["a'b.wav",'double"quote.wav','space name.wav','track$(printf injected).wav',
                      'track`printf injected`.wav','a\\b.wav','two\nlines.wav']:
            code=COMMAND_JS+'\nprocess.stdout.write(shellQuote('+json.dumps(value)+'));'
            quoted=subprocess.check_output(['node','-e',code],text=True)
            actual=subprocess.check_output(['bash','-c',"printf '%s' "+quoted],text=True)
            self.assertEqual(actual,value)
        settings={'input_audio':'</script><script>bad()</script>','out_dir':'out','manifest':'m',
                  'cutoff_s':1,'max_new_tokens':2,'device':'cpu','dtype':'float32','temperature':.8,'top_k':1,'seed':0}
        page=build_viewer(self.directory,settings).read_text()
        self.assertNotIn('<script>bad()',page)

    def test_poetry_schema_and_empty_failure(self):
        converter=ROOT/'data/public-domain-poetry/json_poetry_to_espeak_text.py'
        source=self.directory/'poems.json'
        source.write_text(json.dumps([{'Author':'Author','Title':'Title','text':'One little poem.'}]))
        command=[sys.executable,str(converter),str(source),'-o',str(self.directory/'poems.txt'),'--max-output-size','0']
        valid=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(valid.returncode,0,valid.stderr)
        self.assertIn('One little poem.',(self.directory/'poems.txt').read_text())
        source.write_text('[{"Title":"Title","text":"No author"}]')
        self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)

    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg required')
    def test_prefix_invariance_with_different_future_audio(self):
        sample_rate=12000
        t=np.arange(sample_rate)/sample_rate
        original=.05*np.sin(2*np.pi*440*t)
        changed=original.copy()
        changed[sample_rate//2:]=.8*np.sin(2*np.pi*880*t[sample_rate//2:])+.1
        config={'sample_rate':12000,'bands':8,'levels':64,'hop_ms':5.,'win_ms':20.,'n_fft':512,
                'fmin':10.,'fmax':5000.,'top_db':96.,'reference_power':1.}
        matrices=[]
        for i, signal in enumerate([original,changed]):
            source=self.directory/f'{i}.wav'
            with wave.open(str(source),'wb') as stream:
                stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(sample_rate)
                stream.writeframes((signal*32767).astype('<i2').tobytes())
            prefix=self.directory/f'prefix{i}.wav'
            pipeline.crop_prefix(source,prefix,.5)
            csv=pipeline.encode(prefix,self.directory,f'encoded{i}',config,'cpu')
            out=self.directory/f'prompt{i}'
            mel.cmd_cut_prompt(NS(mel_csv=csv,cutoff_s=.5,output_dir=out,fully_observed=True))
            matrices.append(dec.read_csv_container(out/'prompt.mel.csv').states)
        np.testing.assert_array_equal(*matrices)
        self.assertEqual(len(matrices[0]),99)

    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg required')
    def test_folder_prepare_uses_selected_files_and_validates_cache(self):
        sources=self.directory/'audio'
        sources.mkdir()
        for filename,hz in [('track.wav',440),('track.flac',660),('third.wav',880)]:
            t=np.arange(12000)/12000
            signal=.1*np.sin(2*np.pi*hz*t)
            # FFmpeg detects the container, independently of the filename extension.
            with wave.open(str(sources/filename),'wb') as stream:
                stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(12000)
                stream.writeframes((signal*32767).astype('<i2').tobytes())
        work=self.directory/'work'
        (work/'encoded').mkdir(parents=True)
        (work/'encoded/unused.max.mel.csv').write_text('bad stale data')
        overrides={'MEL_MC_DEVICE':'cpu','MEL_MC_DTYPE':'float32','MEL_MC_WORK_DIR':str(work),
                   'MEL_MC_OUT_DIR':str(self.directory/'out'),'MEL_MC_PREPARE_ONLY':'1',
                   'MEL_MC_SAMPLE_RATE':'12000','MEL_MC_BANDS':'4','MEL_MC_LEVELS':'64',
                   'MEL_MC_HOP_MS':'5','MEL_MC_WIN_MS':'20','MEL_MC_N_FFT':'512',
                   'MEL_MC_FMIN':'10','MEL_MC_FMAX':'5000','MEL_MC_TOP_DB':'96',
                   'MEL_MC_BLOCK_SIZE':'4','MEL_MC_SKIP_ENCODE':'0','MEL_MC_SKIP_TRAIN':'0'}
        args=NS(music_dir=sources,prompt_audio=None,cutoff=.5)
        with patch.dict(os.environ,overrides),patch.object(pipeline,'ROOT',self.directory):
            pipeline.folder(args)
            manifest=json.loads((self.directory/'out/mel_manifest.json').read_text())
            self.assertEqual(sum(len(v) for v in manifest['splits'].values()),3)
            selected=json.loads((work/'selected_sources.json').read_text())
            self.assertTrue(set(selected['train']).isdisjoint(selected['val']))
            with patch.dict(os.environ,{'MEL_MC_SKIP_ENCODE':'1'}):
                pipeline.folder(args)
            (sources/'third.wav').unlink()
            pipeline.folder(args)
            selected=json.loads((work/'selected_sources.json').read_text())
            self.assertEqual(len(selected['sha256']),2)
            calibration=json.loads((work/'calibration.json').read_text())
            self.assertEqual(set(calibration['identity']['sources']),set(selected['train']))


if __name__ == '__main__':
    unittest.main()
