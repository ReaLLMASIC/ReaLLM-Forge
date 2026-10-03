import pytest
import torch
from context_sweep.recipe import ROOT, read, build
from depth_sweep.planning import depth_spec
from depth_sweep.worker import measure


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for real capacity worker')
@pytest.mark.parametrize('variant',['softmax_sdpa','relu2_bias_scale_one','linear16'])
def test_real_probe_includes_optimizer_and_validation(variant):
    spec=depth_spec(read(ROOT/'configs/context_ablation_50m.json'),2)
    spec['model'].update(vocab_size=64,hidden_size=128,intermediate_size=256,
                         num_attention_heads=1,num_key_value_heads=1,bos_token_id=0,eos_token_id=1,pad_token_id=1)
    spec.update(variants=[variant],output_norms=['capped'],microbatch_token_target=512,
                tokens_per_step=1024,max_micro_batch_size=1)
    plan=build(spec)
    cell=plan['cells'][0]
    result=measure(dict(config=plan['configs'][cell['config_key']],cell=cell,
                        device='cuda:0',reserve_gib=0))
    assert result['status']=='fit'
    assert result['optimizer_steps']==2 and result['synthetic_probe']
    assert result['peak_reserved_bytes']>=result['peak_allocated_bytes']>0
