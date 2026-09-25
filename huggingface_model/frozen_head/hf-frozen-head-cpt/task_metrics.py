"""Response-token objective and fixed-protocol greedy GSM8K evaluation."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from common import append_json
from model_utils import autocast, ce_sum
from task_data import collate_examples, score_answer


def response_loss_sum(model, batch, chunk_size):
    out = model.base_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                           use_cache=False, return_dict=True).last_hidden_state
    labels = batch["labels"][:, 1:].reshape(-1)
    states = out[:, :-1].reshape(-1, out.shape[-1])
    keep = labels != -100
    labels, states = labels[keep], states[keep]
    if not len(labels):
        raise ValueError("No supervised response tokens in this microbatch")
    head = model.get_output_embeddings()
    total = states.new_zeros((), dtype=torch.float32)
    for i in range(0, len(labels), chunk_size):
        args = (states[i:i + chunk_size], head.weight, head.bias, labels[i:i + chunk_size])
        total = total + (checkpoint(ce_sum, *args, use_reentrant=False) if torch.is_grad_enabled() else ce_sum(*args))
    return total, len(labels)


@torch.no_grad()
def evaluate_math(model, tokenizer, examples, c, device, predictions_path=None):
    model.eval()
    count, strict, relaxed, formatted, capped, loss_sum, tokens = 0, 0, 0, 0, 0, 0.0, 0
    if predictions_path:
        Path(predictions_path).write_text("")
    max_new = c["sft"].get("max_new_tokens", 384)
    context = getattr(model.config, "max_position_embeddings", getattr(model.config, "n_positions", 10**9))
    for row in examples:
        batch = collate_examples([row], tokenizer.eos_token_id, device)
        with autocast(device):
            total, n = response_loss_sum(model, batch, c["head_chunk"])
        loss_sum += total.item()
        tokens += n
        prompt = torch.tensor([row["prompt_ids"]], device=device)
        budget = min(max_new, context - prompt.shape[1])
        if budget <= 0:
            raise ValueError("Generation prompt leaves no context room")
        with autocast(device):
            sequence = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                   do_sample=False, max_new_tokens=budget, use_cache=True,
                   pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        new = sequence[0, prompt.shape[1]:]
        prediction = tokenizer.decode(new, skip_special_tokens=True)
        score = score_answer(prediction, row["answer"])
        hit_cap = len(new) == budget and int(new[-1]) != tokenizer.eos_token_id
        strict += score["strict_correct"]
        relaxed += score["relaxed_correct"]
        formatted += score["format_valid"]
        capped += hit_cap
        count += 1
        if predictions_path:
            append_json(predictions_path, {"question_hash": row["question_hash"], "prediction": prediction,
                        "hit_generation_cap": hit_cap, **score})
    return {"examples": count, "strict_accuracy": strict / count, "relaxed_accuracy": relaxed / count,
            "format_rate": formatted / count, "generation_cap_rate": capped / count,
            "response_loss": loss_sum / tokens, "response_tokens": tokens,
            "protocol": "Fixed zero-shot plain-text prompt, greedy decoding; not interchangeable with harness few-shot GSM8K."}
