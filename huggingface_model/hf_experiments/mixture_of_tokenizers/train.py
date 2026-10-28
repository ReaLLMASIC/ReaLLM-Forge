#!/usr/bin/env python3
"""Five equally weighted routes per optimizer update, using a HF causal model.

The small explicit loop makes route weighting and resumable sampling inspectable.
Checkpoints are ordinary HF model folders plus a local trusted trainer-state file.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import math
from pathlib import Path
import random
import signal
import shutil
import time

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

from configuration_mixture import MixtureConfig
from modeling_mixture import MixtureForCausalLM
from tokenization_mixture import MixtureTokenizer, ROUTES
from data_pipeline import LanguageWindowDataset


def atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate(step: int, tokens: int, settings: dict) -> float:
    base = float(settings['learning_rate'])
    minimum = float(settings.get('min_learning_rate', base / 10))
    warmup = int(settings.get('warmup_steps', 100))
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    fraction = min(1.0, tokens / max(1, int(settings['target_tokens'])))
    return minimum + (base - minimum) * 0.5 * (1 + math.cos(math.pi * fraction))


def autocast_context(device: torch.device, precision: str):
    if precision == 'bf16':
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return nullcontext()


def estimate_eta_seconds(tokens, target, step, max_steps, initial_tokens, initial_step, elapsed):
    """Estimate the earliest active budget limit using this process's rates."""
    elapsed = max(elapsed, 1e-9)
    token_rate = max(tokens - initial_tokens, 1) / elapsed
    remaining = max(0, target - tokens) / token_rate
    if max_steps is not None:
        step_rate = max(step - initial_step, 1) / elapsed
        remaining = min(remaining, max(0, max_steps - step) / step_rate)
    return remaining


def validate(model, sampler, settings, device, initial_state):
    """Fixed validation windows; route NLLs are not cross-tokenizer perplexities."""
    sampler.load_state_dict(initial_state)
    model.eval()
    losses = {}
    with torch.no_grad():
        for route in ROUTES:
            nll, count = 0.0, 0
            for _ in range(int(settings.get('eval_batches', 4))):
                batch = sampler.get_batch(route, int(settings['batch_size_per_route']), device=str(device))
                batch.pop('batch_stats', None)
                targets = int((batch['labels'][:, 1:] != -100).sum())
                with autocast_context(device, settings['precision']):
                    loss = model(**batch, use_cache=False, return_logits=False).loss
                nll += float(loss) * targets
                count += targets
            losses[route] = nll / max(1, count)
    model.train()
    return {'route_nll': losses, 'mean_route_nll': sum(losses.values()) / len(losses)}


def save_checkpoint(path, model, tokenizer, optimizer, sampler, state):
    temp = path.with_name(path.name + '.tmp')
    previous = path.with_name(path.name + '.previous')
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    model.save_pretrained(temp, safe_serialization=True)
    tokenizer.save_pretrained(temp / 'tokenizer')
    training = dict(state)
    training.update({
        'optimizer': optimizer.state_dict(), 'sampler': sampler.state_dict(),
        'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    })
    torch.save(training, temp / 'trainer_state.pt')
    atomic_json(temp / 'training_summary.json', state)
    if previous.exists():
        shutil.rmtree(previous)
    if path.exists():
        path.rename(previous)
    temp.rename(path)
    if previous.exists():
        shutil.rmtree(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/4090.yaml')
    parser.add_argument('--tokenizers', required=True)
    parser.add_argument('--data')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--max-steps', type=int, help='Total optimizer-step ceiling, including resumed steps')
    parser.add_argument('--target-tokens', type=int, help='Total non-padding predicted-token budget')
    parser.add_argument('--resume', nargs='?', const='auto', help='Checkpoint folder, or checkpoint-last in output')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    settings = dict(config['training'])
    for key in ('device', 'max_steps', 'target_tokens', 'seed'):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    settings.setdefault('precision', 'bf16' if settings.get('device') == 'cuda' else 'fp32')
    settings.setdefault('batch_size_per_route', 4)
    settings.setdefault('target_tokens', 20_000_000)
    settings.setdefault('learning_rate', 3e-4)
    settings.setdefault('seed', 1337)
    device = torch.device(settings.get('device', 'cuda'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable. Use --device cpu with configs/smoke.yaml for verification.')
    if settings['precision'] not in ('fp32', 'bf16'):
        raise SystemExit('precision must be fp32 or bf16; fp16 scaling is deliberately not implicit.')
    if device.type == 'cpu':
        settings['precision'] = 'fp32'
    if device.type == 'cuda' and settings['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise SystemExit('This GPU does not support BF16; set training.precision: fp32.')
    if int(settings['batch_size_per_route']) <= 0 or int(settings['target_tokens']) <= 0:
        raise SystemExit('Batch size and target-token budget must be positive.')
    torch.set_num_threads(int(settings.get('threads', 4)))
    seed_all(int(settings['seed']))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    resume = output / 'checkpoint-last' if args.resume == 'auto' else Path(args.resume) if args.resume else None
    if not resume and (output / 'metrics.jsonl').exists():
        raise SystemExit('Output already contains a run. Use --resume or choose a new --output.')
    tokenizers_path = Path(args.tokenizers)
    tokenizer = MixtureTokenizer.from_pretrained(tokenizers_path)
    fingerprint = tokenizer.fingerprint
    data_dir = args.data or config['data']['output']
    model_settings = dict(config['model'])
    model_config = MixtureConfig(lexical_size=tokenizer.lexical_size, tokenizer_fingerprint=fingerprint,
                                initializer_seed=int(settings['seed']), **model_settings)
    model = MixtureForCausalLM.from_pretrained(resume) if resume else MixtureForCausalLM(model_config)
    for key in set(model_settings) | {'lexical_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
                                    'intermediate_size', 'max_position_embeddings', 'initializer_seed',
                                    'tokenizer_fingerprint'}:
        if getattr(model.config, key) != getattr(model_config, key):
            raise SystemExit(f'Resume model differs from requested configuration: {key}')
    model.to(device)
    decay, no_decay = [], []
    for _, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([
        {'params': decay, 'weight_decay': float(settings.get('weight_decay', .1))},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=float(settings['learning_rate']), betas=(.9, .95), eps=1e-8)
    sampler = LanguageWindowDataset(data_dir, tokenizer, model.config.max_position_embeddings,
                                    split='train', seed=int(settings['seed']))
    val_sampler = LanguageWindowDataset(data_dir, tokenizer, model.config.max_position_embeddings,
                                        split='val', seed=int(settings['seed']) + 1)
    val_state = val_sampler.state_dict()
    state = {'step': 0, 'predicted_tokens': 0, 'best_validation_nll': None,
             'tokenizer_fingerprint': fingerprint, 'settings': settings, 'validation_identity': val_state,
             'route_predicted_tokens': {r: 0 for r in ROUTES},
             'route_source_bytes': {r: 0 for r in ROUTES}, 'elapsed_seconds': 0.0}
    console = Console()
    if resume:
        # This is our own local checkpoint. Never load an untrusted trainer_state.pt.
        restored = torch.load(resume / 'trainer_state.pt', map_location='cpu', weights_only=False)
        if restored['tokenizer_fingerprint'] != fingerprint:
            raise SystemExit('Tokenizer fingerprint differs from checkpoint. Use the original tokenizer bundle.')
        for key in ('learning_rate', 'min_learning_rate', 'warmup_steps', 'weight_decay',
                    'batch_size_per_route', 'precision', 'seed', 'eval_batches'):
            if restored['settings'].get(key) != settings.get(key):
                raise SystemExit(f'Resume setting {key} differs; use the original setting or a fresh output directory.')
        if restored.get('validation_identity') != val_state:
            raise SystemExit('Validation source or sampler identity differs from checkpoint; best scores are not comparable.')
        optimizer.load_state_dict(restored.pop('optimizer'))
        sampler.load_state_dict(restored.pop('sampler'))
        random.setstate(restored.pop('python_rng'))
        np.random.set_state(restored.pop('numpy_rng'))
        torch.set_rng_state(restored.pop('torch_rng'))
        cuda_rng = restored.pop('cuda_rng')
        if device.type == 'cuda' and cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        old_budget = restored['settings']['target_tokens']
        state.update(restored)
        state['settings'] = settings
        if old_budget != settings['target_tokens']:
            console.print(f'Budget changed {old_budget:,} → {settings["target_tokens"]:,}; cosine horizon is resized.')
    versions = {name: importlib.metadata.version(name) for name in
                ['torch', 'transformers', 'tokenizers', 'datasets', 'numpy']}
    atomic_json(output / 'run_config.json', {
        'config': config, 'effective_training': settings, 'tokenizers': str(tokenizers_path.resolve()),
        'tokenizer_fingerprint': fingerprint, 'data': str(Path(data_dir).resolve()),
        'parameters': sum(p.numel() for p in model.parameters()), 'versions': versions,
        'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'budget_note': 'Non-padding text targets plus genuine EOS; BOS excluded. Stop after complete five-route update.',
    })
    target = int(settings['target_tokens'])
    max_steps = settings.get('max_steps')
    max_steps = int(max_steps) if max_steps is not None else None
    interval = max(1, int(settings.get('eval_interval', 100)))
    save_interval = max(1, int(settings.get('save_interval', 100)))
    log_interval = max(1, int(settings.get('log_interval', 1)))
    model.train()
    start = time.monotonic()
    elapsed_before = state['elapsed_seconds']
    initial_tokens = state['predicted_tokens']
    initial_step = state['step']
    interrupted = False
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    last_validation_step = -1
    progress = Progress(TextColumn('{task.description}'), BarColumn(),
                        TextColumn('{task.completed:,.0f}/{task.total:,.0f} tokens'),
                        TimeElapsedColumn(), TextColumn('ETA {task.fields[eta_seconds]:,.0f}s'),
                        console=console, disable=args.no_progress or not console.is_terminal)
    with progress:
        task = progress.add_task('Five-route training', total=target, completed=state['predicted_tokens'], eta_seconds=0)
        try:
            while not stop_requested and state['predicted_tokens'] < target and (max_steps is None or state['step'] < max_steps):
                tic = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                lr = learning_rate(state['step'], state['predicted_tokens'], settings)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                route_losses, route_counts, route_stats = {}, {}, {}
                for route in ROUTES:
                    batch = sampler.get_batch(route, int(settings['batch_size_per_route']), device=str(device))
                    route_stats[route] = batch.pop('batch_stats', {})
                    count = int((batch['labels'][:, 1:] != -100).sum())
                    if count == 0:
                        raise RuntimeError(f'No predicted targets in route {route}')
                    with autocast_context(device, settings['precision']):
                        loss = model(**batch, use_cache=False, return_logits=False).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Non-finite loss for route {route} at step {state["step"]}')
                    (loss / len(ROUTES)).backward()
                    route_losses[route] = float(loss.detach())
                    route_counts[route] = count
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.get('grad_clip', 1.0)),
                                                          error_if_nonfinite=True)
                optimizer.step()
                step_tokens = sum(route_counts.values())
                state['step'] += 1
                state['predicted_tokens'] += step_tokens
                for route in ROUTES:
                    state['route_predicted_tokens'][route] += route_counts[route]
                    state['route_source_bytes'][route] += int(route_stats[route].get('source_bytes', 0))
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                duration = time.monotonic() - tic
                elapsed = time.monotonic() - start
                state['elapsed_seconds'] = elapsed_before + elapsed
                record = {'event': 'train', 'step': state['step'], 'predicted_tokens': state['predicted_tokens'],
                          'loss': sum(route_losses.values()) / len(ROUTES), 'route_loss': route_losses,
                          'route_tokens': route_counts, 'route_batch_stats': route_stats, 'learning_rate': lr,
                          'grad_norm': float(grad_norm), 'tokens_per_second': step_tokens / max(duration, 1e-9),
                          'eta_seconds': estimate_eta_seconds(state['predicted_tokens'], target, state['step'],
                                                              max_steps, initial_tokens, initial_step, elapsed),
                          'elapsed_seconds': state['elapsed_seconds'],
                          'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else None,
                          'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30 if device.type == 'cuda' else None}
                with (output / 'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
                progress.update(task, completed=state['predicted_tokens'],
                                description=f'Step {state["step"]} | loss {record["loss"]:.4f}',
                                eta_seconds=record['eta_seconds'])
                if state['step'] % log_interval == 0 and (args.no_progress or not console.is_terminal):
                    console.print(f'step={state["step"]} tokens={state["predicted_tokens"]:,} '
                                  f'loss={record["loss"]:.5f} tok/s={record["tokens_per_second"]:.0f} '
                                  f'lr={lr:.3g} eta={record["eta_seconds"]:.0f}s')
                if state['step'] % interval == 0:
                    validation = validate(model, val_sampler, settings, device, val_state)
                    last_validation_step = state['step']
                    state['last_validation'] = validation
                    with (output / 'metrics.jsonl').open('a') as handle:
                        handle.write(json.dumps({'event': 'validation', 'step': state['step'], **validation}) + '\n')
                    if state['best_validation_nll'] is None or validation['mean_route_nll'] < state['best_validation_nll']:
                        state['best_validation_nll'] = validation['mean_route_nll']
                        save_checkpoint(output / 'checkpoint-best', model, tokenizer, optimizer, sampler, state)
                    console.print(f'validation={validation["mean_route_nll"]:.5f} best={state["best_validation_nll"]:.5f}')
                if state['step'] % save_interval == 0:
                    save_checkpoint(output / 'checkpoint-last', model, tokenizer, optimizer, sampler, state)
        except KeyboardInterrupt:
            interrupted = True
            console.print('Interrupted; saving the last completed optimizer step.')
    interrupted = interrupted or stop_requested
    signal.signal(signal.SIGINT, previous_sigint)
    signal.signal(signal.SIGTERM, previous_sigterm)
    # No partially accumulated gradients are saved; only complete updates are resumable.
    optimizer.zero_grad(set_to_none=True)
    if last_validation_step != state['step'] and not interrupted:
        validation = validate(model, val_sampler, settings, device, val_state)
        state['last_validation'] = validation
        with (output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps({'event': 'validation', 'step': state['step'], **validation}) + '\n')
        if state['best_validation_nll'] is None or validation['mean_route_nll'] < state['best_validation_nll']:
            state['best_validation_nll'] = validation['mean_route_nll']
            save_checkpoint(output / 'checkpoint-best', model, tokenizer, optimizer, sampler, state)
    state['elapsed_seconds'] = elapsed_before + time.monotonic() - start
    save_checkpoint(output / 'checkpoint-last', model, tokenizer, optimizer, sampler, state)
    summary = dict(state)
    summary.update({'interrupted': interrupted, 'parameters': sum(p.numel() for p in model.parameters()),
                    'budget_reached': state['predicted_tokens'] >= target, 'versions': versions,
                    'gpu_tested': device.type == 'cuda'})
    atomic_json(output / 'summary.json', summary)
    console.print(f'Saved {output / "checkpoint-last"}; {state["predicted_tokens"]:,} predicted tokens.')


if __name__ == '__main__':
    main()
