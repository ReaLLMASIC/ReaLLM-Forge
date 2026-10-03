"""GPU checks for all four attention rules, Muon and output norms on/off."""
import pytest
import torch
from context_sweep.recipe import ROOT,read,build
from experiment.common import make_model,token_loss,amp,backend_labels
from experiment.optimizers import optimizer_for

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA GPU required")


@pytest.mark.parametrize("variant",["softmax_sdpa","relu2","linear16","kda"])
@pytest.mark.parametrize("output_norm",["none","capped"])
def test_three_cuda_training_steps_and_optimizer_restore(variant,output_norm):
    name="muon"
    cfg=build(read(ROOT/"configs/sweep_50m.json"))["configs"]["512_"+output_norm]
    cfg["model"]["num_hidden_layers"]=2
    m,_=make_model(cfg,variant,0,"cuda:0")
    opt=optimizer_for(m,name,cfg["optimizer_recipes"][name],"cuda:0")
    x=torch.randint(0,100,(1,32),device="cuda:0");y=torch.roll(x,-1,-1)
    for _ in range(3):
        opt.zero_grad()
        with amp("cuda:0","bfloat16"):loss=token_loss(m,x,y,32)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0,error_if_nonfinite=True)
        opt.step()
    assert backend_labels(m)==(["sdpa_flash"] if variant=="softmax_sdpa" else ["fla_triton_kda_chunk"] if variant=="kda" else ["triton_fused"])
    assert all(torch.isfinite(p).all() for p in m.parameters())
    restored=optimizer_for(m,name,cfg["optimizer_recipes"][name],"cuda:0")
    restored.load_state_dict(opt.state_dict())
    assert set(restored.state_dict()["states"])==set(opt.state_dict()["states"])
