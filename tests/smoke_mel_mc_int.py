#!/usr/bin/env python3
"""Tiny real train/checkpoint/sample/decode check; requires the repository CPU dependencies."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='_mel_smoke_', dir=ROOT/'data') as directory:
        work = Path(directory)
        sources = work/'audio'
        sources.mkdir()
        t = np.arange(12000)/12000
        for name, frequency in [('a.wav',440),('b.wav',660)]:
            with wave.open(str(sources/name),'wb') as stream:
                stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(12000)
                stream.writeframes((np.sin(2*np.pi*frequency*t)*3000).astype('<i2').tobytes())
        environment = {key:value for key,value in os.environ.items() if not key.startswith('MEL_MC_')}
        environment.update({'MEL_MC_DEVICE':args.device,'MEL_MC_DTYPE':'float32','MEL_MC_COMPILE':'0',
            'MEL_MC_OUT_DIR':str(work/'out'),'MEL_MC_WORK_DIR':str(work/'work'),
            'MEL_MC_OUTPUT_ROOT':work.name+'/prepared','MEL_MC_SAMPLE_RATE':'12000','MEL_MC_BANDS':'4',
            'MEL_MC_LEVELS':'64','MEL_MC_HOP_MS':'5','MEL_MC_WIN_MS':'20','MEL_MC_N_FFT':'512',
            'MEL_MC_FMIN':'10','MEL_MC_FMAX':'5000','MEL_MC_TOP_DB':'96','MEL_MC_N_LAYER':'1',
            'MEL_MC_N_EMBD':'16','MEL_MC_N_HEAD':'2','MEL_MC_N_KV_GROUP':'2','MEL_MC_MLP_SIZE':'32',
            'MEL_MC_QK_DIM':'8','MEL_MC_V_DIM':'8','MEL_MC_BATCH_SIZE':'1','MEL_MC_BLOCK_SIZE':'4',
            'MEL_MC_MAX_ITERS':'2','MEL_MC_EVAL_INTERVAL':'1','MEL_MC_EVAL_ITERS':'2',
            'MEL_MC_MAX_NEW_TOKENS':'3','MEL_MC_SEED':'0','MEL_MC_TENSORBOARD':'0',
            'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        # Run outside the repo to exercise wrapper path handling.
        subprocess.run(['bash',str(ROOT/'demos/mel_mc_int_music_pipeline.sh'),str(sources),str(sources/'a.wav'),'.25'],
                       cwd=work,env=environment,check=True)
        checkpoint=work/'out/ckpt.pt'
        if not checkpoint.is_file():
            raise AssertionError('No checkpoint saved')
        manifest=json.loads((work/'out/mel_manifest.json').read_text())
        import torch
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if payload['iter_num'] < 2 or not np.isfinite(float(payload['best_val_loss'])):
            raise AssertionError('Training did not complete with a finite validation loss')
        if not all(torch.isfinite(value).all() for value in payload['model'].values() if torch.is_tensor(value)):
            raise AssertionError('Checkpoint contains nonfinite parameters')
        samples=list((work/'out/mel_samples').iterdir())
        if len(samples)!=1:
            raise AssertionError('Expected one complete inference run')
        settings=json.loads((samples[0]/'run.json').read_text())
        for name in ['generated.wav','codec_prompt.wav','continuation.wav']:
            with wave.open(str(samples[0]/name),'rb') as stream:
                if stream.getnframes()<1:
                    raise AssertionError(f'Empty audio: {name}')
        # Sampling already reloads the saved checkpoint; this additionally exercises SKIP_TRAIN.
        environment['MEL_MC_SKIP_TRAIN']='1'
        subprocess.run(['bash',str(ROOT/'demos/mel_mc_int_music_pipeline.sh'),str(sources),str(sources/'b.wav'),'.25'],
                       cwd=work,env=environment,check=True)
        print(json.dumps({'result':'passed','train_rows':manifest['train_rows'],'val_rows':manifest['val_rows'],
                          'prompt_frames':settings['prompt']['prompt_frames'],'new_frames':3,
                          'checkpoint_reload_and_reuse':True},indent=2))


if __name__=='__main__':
    main()
