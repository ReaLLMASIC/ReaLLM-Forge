"""Real optimizer updates, scheduler propagation, state restore and HF Trainer."""
import copy
import json
from pathlib import Path
import torch
from transformers import TrainingArguments
from hf_model.configuration_comparison import ComparisonConfig
from hf_model.modeling_comparison import ComparisonForCausalLM
from hf_search.hf_trainer import MuonAdamW, MuonTrainer

ROOT = Path(__file__).resolve().parents[1]


def recipe():
    r = json.loads((ROOT / "configs/context_ablation_50m.json").read_text())["optimizer_recipes"]["muon"]
    r["weight_decay"] = 0.0
    return r


def model():
    return ComparisonForCausalLM(ComparisonConfig(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        variant="relu2", require_fused=False, relu2max_accelerator="torch",
        max_position_embeddings=16, qk_norm_scale_init=2.0,
        pad_token_id=0, bos_token_id=1, eos_token_id=2))


def update(m, opt):
    x = torch.tensor([[1, 3, 7, 2, 5, 6]])
    opt.zero_grad()
    m(x, labels=x).loss.backward()
    opt.step()


def test_scheduler_routing_and_exact_optimizer_restore():
    torch.manual_seed(7)
    a = model()
    opt = MuonAdamW(a, recipe())
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 0.5 ** step)
    update(a, opt)
    schedule.step()
    assert [g["lr"] for g in opt.param_groups] == [g["lr"] for g in opt.bundle.param_groups]
    assert {g["group_name"]: g["weight_decay"] for g in opt.param_groups} == {
        "muon_decoder": 0.0, "adamw_decay": 0.1, "adamw_no_decay": 0.0}
    b = model()
    b.load_state_dict(a.state_dict())
    other = MuonAdamW(b, recipe())
    other.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert all(x is y for x, y in zip(other.param_groups, other.bundle.param_groups))
    update(a, opt)
    update(b, other)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_hf_trainer_model_init_runs_two_steps(tmp_path):
    torch.manual_seed(8)
    x = torch.tensor([1, 3, 7, 2, 5, 6])
    dataset = [dict(input_ids=x, labels=x) for _ in range(4)]
    trainer = MuonTrainer(model_init=model, muon_recipe=recipe(), train_dataset=dataset,
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True,
            per_device_train_batch_size=1, max_steps=2, report_to=[],
            save_strategy="no", disable_tqdm=True, dataloader_pin_memory=False))
    result = trainer.train()
    assert result.global_step == 2
    assert len(trainer.optimizer.state_dict()["states"]) == 2
