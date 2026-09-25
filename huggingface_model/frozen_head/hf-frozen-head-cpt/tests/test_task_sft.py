import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2ForCausalLM

from common import read_json, run_dir, write_json
from experiment import latest_checkpoint
from model_utils import configure_mode
from task_data import collate_examples, encode_example, final_number, prepare_task, score_answer
from task_metrics import response_loss_sum
from task_report import first_accuracy_crossing, report_task
from task_sft import baseline, baseline_dir, train_task
from test_correctness import fixture_config, small_model


def test_numeric_scoring_separates_format_and_math():
    assert final_number('Work 3 + 9. #### 1,200.00') == 1200
    assert final_number('The answer is 42.') is None
    score = score_answer('The answer is 42.', 'work #### 42')
    assert score['relaxed_correct'] and not score['strict_correct'] and not score['format_valid']
    assert score_answer('#### -0.50', '#### -0.5')['strict_correct']
    assert not score_answer('I do not know.', '#### 2')['relaxed_correct']


@pytest.mark.parametrize('mode', ['full','freeze_head','freeze_embed','freeze_both'])
def test_response_only_loss_masks_prompt_and_padding(mode):
    model = small_model()
    configure_mode(model, mode)
    direct = copy.deepcopy(model)
    rows = [{'input_ids':[2,3,4,5,1], 'labels':[-100,-100,-100,5,1]},
            {'input_ids':[6,7,1], 'labels':[-100,7,1]}]
    batch = collate_examples(rows, 1)
    total, count = response_loss_sum(model, batch, 2)
    logits = direct(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).logits
    reference = F.cross_entropy(logits[:,:-1].reshape(-1,32), batch['labels'][:,1:].reshape(-1), ignore_index=-100, reduction='sum')
    assert count == 4
    torch.testing.assert_close(total, reference)
    total.backward()
    reference.backward()
    for (n,p),(n2,q) in zip(model.named_parameters(),direct.named_parameters()):
        assert n == n2
        if p.requires_grad:
            torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=2e-4)
        else:
            assert p.grad is None


def task_fixture(tmp_path, name):
    c=fixture_config(tmp_path,name)
    c['data'].pop('b')
    c.update(seq_len=16,b_steps=3,eval_every=1,save_every=1)
    mapping={}
    for split,values in [('train',range(0,12)),('validation',range(12,14)),('test',range(14,16))]:
        path=tmp_path/f'math_{split}.jsonl'
        path.write_text(''.join(json.dumps({'question':f't{i}', 'answer':'t0 #### 3'})+'\n' for i in values))
        mapping[split]=str(path)
    c['sft']={'local_jsonl':mapping,'max_new_tokens':2,'prompt':'t0 {question} t1 ', 'accuracy_targets':[0.5]}
    return c


def test_prompt_mask_and_length_filter(tmp_path):
    c=task_fixture(tmp_path,'mask')
    tok=AutoTokenizer.from_pretrained(c['model'])
    row=encode_example(tok,'t1','t2 #### 3',32,'t0 {question} t1 ')
    assert row['input_ids'][:len(row['prompt_ids'])] == row['prompt_ids']
    assert all(x == -100 for x in row['labels'][:len(row['prompt_ids'])])
    assert row['labels'][-1] == tok.eos_token_id
    assert encode_example(tok,'t1','t2 #### 3',2,'t0 {question} t1 ') is None


def test_task_end_to_end_resume_and_report(tmp_path):
    c=task_fixture(tmp_path,'task_continuous')
    prepare_task(c)
    baseline(c)
    assert train_task(c,12,'full',0.003) == 0
    assert train_task(c,12,'freeze_head',0.003) == 0
    root=run_dir(c,12,'freeze_head',0.003)
    assert read_json(root/'frozen_initial.json') == read_json(root/'frozen_final.json')
    report_task(c,plots=False)
    assert (Path(c['output'])/'task_analysis'/'final_test.csv').exists()
    resumed=task_fixture(tmp_path,'task_resumed')
    prepare_task(resumed)
    baseline(resumed)
    assert train_task(resumed,12,'full',0.003,stop_after=1) == 75
    assert train_task(resumed,12,'full',0.003) == 0
    a=Qwen2ForCausalLM.from_pretrained(latest_checkpoint(run_dir(c,12,'full',0.003)))
    b=Qwen2ForCausalLM.from_pretrained(latest_checkpoint(run_dir(resumed,12,'full',0.003)))
    assert all(torch.equal(t,b.state_dict()[k]) for k,t in a.state_dict().items())
    assert read_json(run_dir(c,12,'full',0.003)/'test.json') == read_json(run_dir(resumed,12,'full',0.003)/'test.json')


def test_accuracy_crossing_does_not_interpolate():
    rows=[{'step':0,'math':{'relaxed_accuracy':0.1}}, {'step':10,'math':{'relaxed_accuracy':0.3}}]
    assert first_accuracy_crossing(rows,0.2)['step'] == 10
    assert first_accuracy_crossing(rows,0.4) is None


def test_benchmark_summary_detects_matching_records(tmp_path):
    from benchmarks import sample_fingerprints, summarize_benchmarks
    c={'output':str(tmp_path)}
    root=tmp_path/'benchmarks'/'limit_2'
    root.mkdir(parents=True)
    samples={'boolq':[{'doc_id':0,'doc':{'q':'x'},'acc':1},{'doc_id':1,'doc':{'q':'y'},'acc':0}]}
    a={'name':'baseline','protocol':{'tasks':['boolq']},'sample_fingerprints':sample_fingerprints(samples),
       'harness':{'results':{'boolq':{'acc,none':0.5}},'samples':samples}}
    b=copy.deepcopy(a)
    b['name']='seed_1_full'
    b['harness']['results']['boolq']['acc,none']=0.0
    b['harness']['samples']['boolq'][0]['acc']=0
    write_json(root/'baseline.json',a)
    write_json(root/'seed_1_full.json',b)
    summarize_benchmarks(c,2)
    text=(root/'retention.csv').read_text()
    assert '50.0' in text
    b['sample_fingerprints']={'boolq':'different'}
    write_json(root/'seed_1_full.json',b)
    with pytest.raises(RuntimeError,match='mismatched'):
        summarize_benchmarks(c,2)
