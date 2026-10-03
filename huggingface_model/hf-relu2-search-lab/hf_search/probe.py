"""One disposable GPU process for one depth/condition; no data download required."""
import argparse
import time
import traceback
from context_sweep.recipe import read, write


def measure(request):
    import torch
    from hf_search.common import (configure_device, make_model, amp, token_loss,
                                   parameter_counts, backend_labels, environment)
    from experiment.optimizers import optimizer_for
    from hf_search.train import learning_rate
    from hf_model.output_norm import start_output_diagnostics, stop_output_diagnostics
    cfg, cell, device = request['config'], request['cell'], request['device']
    configure_device(device, cfg['dtype'])
    if not device.startswith('cuda'):
        raise ValueError('Capacity measurement requires CUDA, never CPU emulation')
    torch.set_num_threads(cfg.get('cpu_threads', 4))
    free, total = torch.cuda.mem_get_info()
    reserve = request['reserve_gib'] * 2**30
    if free <= reserve:
        raise RuntimeError('Insufficient free GPU memory before probing; stop other GPU workloads')
    torch.cuda.reset_peak_memory_stats()
    model, _ = make_model(cfg, cell['variant'], cell['seed'], device)
    model.model.gradient_checkpointing = cfg['train']['gradient_checkpointing']
    model.train()
    opt = optimizer_for(model, cell['optimizer'], cfg['optimizer_recipes'][cell['optimizer']], device)
    t = cfg['train']
    b, n = t['micro_batch_size'], t['sequence_length']
    # Same allocation shapes as token streams, but never counted as training/data.
    x = torch.randint(cfg['model']['vocab_size'], (b, n), device=device)
    y = torch.randint(cfg['model']['vocab_size'], (b, n), device=device)

    def validation():
        model.eval()
        start_output_diagnostics(model)
        try:
            with torch.no_grad(), amp(device, cfg['dtype']):
                value = token_loss(model, x, y, t['loss_chunk_tokens'])
                if not torch.isfinite(value):
                    raise FloatingPointError('Nonfinite probe validation loss')
                common = t['common_validation_context']
                if common != n:
                    value = token_loss(model, x.reshape(-1, common), y.reshape(-1, common), t['loss_chunk_tokens'])
                    if not torch.isfinite(value):
                        raise FloatingPointError('Nonfinite probe common-context validation loss')
        finally:
            stop_output_diagnostics(model)
            model.train()

    validation()  # Training also starts with validation.
    begin = time.perf_counter()
    # Two complete optimizer steps: includes lazy Muon/AdamW state, retained
    # gradients on later microbatches, clipping and the real chunked LM-head loss.
    for step in range(2):
        for group in opt.param_groups:
            group['lr'] = learning_rate(step, cell['steps'], t, group['initial_lr'])
        opt.zero_grad(set_to_none=True)
        for _ in range(t['gradient_accumulation']):
            with amp(device, cfg['dtype']):
                loss = token_loss(model, x, y, t['loss_chunk_tokens'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite probe training loss; not a VRAM-capacity failure')
            (loss / t['gradient_accumulation']).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t['grad_clip'], error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize()
    validation()  # Optimizer states + residual gradients coexist with validation.
    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    # The initial free figure already excludes the CUDA context/non-PyTorch use.
    # Retain an additional physical-memory reserve (not an allocator fraction).
    fits = reserved <= free - reserve
    return dict(status='fit' if fits else 'budget_exceeded', peak_allocated_bytes=allocated,
                peak_reserved_bytes=reserved, initial_free_bytes=free, total_bytes=total,
                reserve_bytes=reserve, parameter_counts=parameter_counts(model),
                backends=backend_labels(model), seconds=time.perf_counter()-begin,
                environment=environment(), synthetic_probe=True, optimizer_steps=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', required=True)
    p.add_argument('--result', required=True)
    a = p.parse_args()
    try:
        result = measure(read(a.request))
    except Exception as error:
        import torch
        result = dict(status='oom' if isinstance(error, torch.cuda.OutOfMemoryError) else 'error',
                      error=str(error), traceback=traceback.format_exc())
    write(a.result, result)
    if result['status'] == 'error':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
