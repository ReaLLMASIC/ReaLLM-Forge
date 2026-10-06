"""Verify actual GPU profiles match the requested study before a long run."""
from dataclasses import asdict
from pathlib import Path
import json
import sys
import torch
from slerp_context.summary.config import Study,digest
from slerp_context.summary.data import load_split
from slerp_context.storage import versions

config,warmup,directory,*methods=sys.argv[1:]
if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
for method,path in [('native',warmup)]+[(m,config) for m in methods]:
    study=Study.load(path);study.model['method']=method;study.validate()
    p=Path(directory)/(method+'.json')
    result=json.loads(p.read_text())
    _,meta=load_split(study,'train')
    assert result['study_sha256']==digest(asdict(study)),f'{method}: study changed; profile again in a fresh directory'
    assert result['device_name']==torch.cuda.get_device_name(),f'{method}: profile belongs to another GPU model'
    assert result['versions']==versions(),f'{method}: runtime changed; profile again'
    assert result['model_revision']==meta['model_revision'],f'{method}: profile and prepared tokenizer revisions differ'
    assert result['status']=='ok' and result['headroom_ok'] is True,f'{method}: GPU profile did not pass'
    assert result['source_tokens']>=study.train_source_limit and result['target_tokens']>=study.train_target_limit
    assert result['steps']>=2 and result['full_weight_training'] is True
print('Matching full-weight GPU profiles passed for all selected methods.')
