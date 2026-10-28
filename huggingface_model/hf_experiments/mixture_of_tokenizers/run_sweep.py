"""Launch vocabulary conditions sequentially, preserving a single-GPU budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/4090.yaml')
    parser.add_argument('--sizes', type=int, nargs='+', help='Override configured private vocabulary sizes')
    parser.add_argument('--target-tokens', type=int, help='Total predicted-token budget per run, including resumed tokens')
    parser.add_argument('--max-steps', type=int, help='Total optimizer-step limit per run, including resumed steps')
    parser.add_argument('--device', help='Override the configured device')
    parser.add_argument('--output-root', '--output', dest='output', help='Override parent run directory')
    parser.add_argument('--no-resume', action='store_true', help='Do not pass an existing checkpoint to train.py')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without starting or modifying runs')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the best checkpoint after each run')
    parser.add_argument('--eval-documents', type=int, default=100)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    sizes = args.sizes or config['tokenizers']['sizes']
    if not sizes or len(set(sizes)) != len(sizes) or any(k < 10 for k in sizes):
        parser.error('sizes must be distinct integers of at least 10')
    root = Path(__file__).resolve().parent
    # Config-relative values intentionally use the launch working directory,
    # exactly as train.py and prepare_data.py do. README commands cd here first.
    tokenizer_root = Path(config['tokenizers']['output']).resolve()
    output_root = Path(args.output or config['training']['output']).resolve()
    data_root = Path(config['data']['output']).resolve()
    summaries = []
    for size in sizes:
        tokenizers = tokenizer_root / f'k{size}'
        output = output_root / f'k{size}'
        command = [sys.executable, str(root / 'train.py'), '--config', str(config_path),
                   '--tokenizers', str(tokenizers), '--output', str(output)]
        for flag, value in (('--target-tokens', args.target_tokens),
                            ('--max-steps', args.max_steps), ('--device', args.device)):
            if value is not None:
                command.extend([flag, str(value)])
        checkpoint = output / 'checkpoint-last'
        if checkpoint.is_dir() and not args.no_resume:
            command.extend(['--resume', str(checkpoint)])
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            if not tokenizers.is_dir():
                raise SystemExit(f'Missing {tokenizers}; run train_tokenizers.py first.')
            subprocess.run(command, check=True)
            summary_file = output / 'summary.json'
            summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
            summaries.append({'lexical_size': size, 'output': str(output), 'training': summary})
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / 'sweep-summary.json').write_text(
                json.dumps({'sequential': True, 'config': str(config_path), 'runs': summaries}, indent=2) + '\n',
                encoding='utf-8')
            # train.py saves a resumable checkpoint after a handled SIGTERM or
            # SIGINT, then exits normally. Do not mistake that for completion.
            if summary.get('interrupted'):
                print(f'Training interrupted at K={size}; sweep stopped. Restart this command to resume.',
                      file=sys.stderr)
                raise SystemExit(130)

        if args.evaluate:
            eval_command = [sys.executable, str(root / 'evaluate.py'),
                            '--model', str(output / 'checkpoint-best'),
                            '--data', str(data_root), '--split', 'val',
                            '--context-mode', 'bytes', '--max-documents', str(args.eval_documents),
                            '--device', args.device or config['training']['device'],
                            '--output', str(output / 'evaluation-val-bytes.json')]
            print(shlex.join(eval_command), flush=True)
            if not args.dry_run:
                subprocess.run(eval_command, check=True)

if __name__ == '__main__':
    main()
