import argparse
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    class _TqdmFallback:  # minimal shim when tqdm is unavailable
        def __init__(self, iterable=None, total=None, desc=None, position=None, leave=True):
            self.iterable = iterable

        def update(self, n=1):
            return None

        def close(self):
            return None

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        @staticmethod
        def set_lock(_lock):
            return None

    tqdm = _TqdmFallback


def _run_prepare_command(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "prepare.py subprocess failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def _run_prepare_for_files(files, desc, max_parallel, command_builder):
    if not files:
        return

    commands = [command_builder(file) for file in files]

    if max_parallel == 1:
        bar = tqdm(total=len(commands), desc=f"{desc} [worker 1]", position=0, leave=True)
        try:
            for command in commands:
                _run_prepare_command(command)
                bar.update(1)
        finally:
            bar.close()
        return

    worker_count = min(max_parallel, len(commands))
    # Reserve one visual spacer line above worker bars as a placeholder region
    # when subprocess tokenizers print progress updates.
    print()
    worker_buckets = [[] for _ in range(worker_count)]
    for idx, command in enumerate(commands):
        worker_buckets[idx % worker_count].append(command)

    tqdm.set_lock(threading.RLock())
    progress_lock = threading.Lock()
    errors = []
    errors_lock = threading.Lock()

    worker_bars = [
        tqdm(total=len(worker_buckets[i]), desc=f"{desc} [worker {i + 1}]", position=i, leave=True)
        for i in range(worker_count)
    ]
    total_bar = tqdm(total=len(commands), desc=f"{desc} [total]", position=worker_count, leave=True)

    def _worker_run(worker_index, worker_commands):
        bar = worker_bars[worker_index]
        try:
            for command in worker_commands:
                _run_prepare_command(command)
                with progress_lock:
                    bar.update(1)
                    total_bar.update(1)
        except Exception as exc:  # pragma: no cover - thread error path
            with errors_lock:
                errors.append(exc)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_worker_run, worker_index, worker_commands)
            for worker_index, worker_commands in enumerate(worker_buckets)
        ]
        for future in futures:
            future.result()

    for bar in worker_bars:
        bar.close()
    total_bar.close()

    if errors:
        raise errors[0]


def _build_common_command(prepare_script, file):
    return [
        'python3',
        prepare_script,
        '--train_input',
        file,
        '--train_output',
        file.replace('.txt', '.bin'),
        '-p',
        '1.0',
    ]


def batch_prepare(
    input_dir,
    train_output,
    val_output,
    prepare_script,
    tokenizer,
    spm_model_file,
    spm_vocab_file,
    char_bpe_vocab_path,
    json_tokens_file,
    max_parallel,
    train_ratio=0.9,
):
    files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.txt')])
    files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    print(files)
    num_train = int(len(files) * train_ratio)

    if max_parallel < 1:
        raise ValueError('--max_parallel must be >= 1')

    if tokenizer == 'tiktoken':
        train_builder = lambda file: _build_common_command(prepare_script, file)
        val_builder = train_builder
    elif tokenizer == 'sentencepiece':
        train_builder = lambda file: _build_common_command(prepare_script, file) + [
            '--method',
            tokenizer,
            '--spm_model_file',
            spm_model_file,
            '--spm_vocab_file',
            spm_vocab_file,
        ]
        val_builder = train_builder
    elif tokenizer == 'char':
        train_builder = lambda file: _build_common_command(prepare_script, file) + ['--method', tokenizer, '--reuse_chars']
        val_builder = train_builder
    elif tokenizer == 'char_bpe':
        if not char_bpe_vocab_path:
            raise ValueError(
                '--char_bpe_vocab_path is required when --tokenizer char_bpe. '
                'Provide the path to a previously generated char_bpe meta.pkl file.'
            )
        train_builder = lambda file: _build_common_command(prepare_script, file) + [
            '--method',
            tokenizer,
            '--char_bpe_vocab_path',
            char_bpe_vocab_path,
        ]
        val_builder = train_builder
    elif tokenizer == 'json_byte_fallback':
        if not json_tokens_file:
            raise ValueError(
                '--json_tokens_file is required when --tokenizer json_byte_fallback. '
                'Provide a path to a JSON array of tokens.'
            )
        train_builder = lambda file: _build_common_command(prepare_script, file) + [
            '--method',
            tokenizer,
            '--json_tokens_file',
            json_tokens_file,
        ]
        val_builder = train_builder
    else:
        print(f'tokenizer {tokenizer} not currently supported')
        return

    _run_prepare_for_files(
        files[:num_train],
        desc='Processing training files',
        max_parallel=max_parallel,
        command_builder=train_builder,
    )
    _run_prepare_for_files(
        files[num_train:],
        desc='Processing validation files',
        max_parallel=max_parallel,
        command_builder=val_builder,
    )

    combine_bins(files[:num_train], train_output)
    combine_bins(files[num_train:], val_output)
    print(f'Created {train_output} and {val_output}')


def combine_bins(files, output_file):
    with open(output_file, 'wb') as fout:
        for file in files:
            bin_file = file.replace('.txt', '.bin')
            if os.path.exists(bin_file):
                with open(bin_file, 'rb') as fin:
                    while chunk := fin.read(1024):
                        fout.write(chunk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare training and validation binary files from divided text files.')
    parser.add_argument('--input_dir', type=str, default='partitioned_file', help='Directory containing the divided files.')
    parser.add_argument('--train_output', type=str, default='train.bin', help='Output binary file for training data.')
    parser.add_argument('--val_output', type=str, default='val.bin', help='Output binary file for validation data.')
    parser.add_argument('--prepare_script', type=str, default='prepare.py', help='Path to the prepare.py script.')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='Ratio of training data to total data.')
    parser.add_argument('--tokenizer', type=str, required=True, help='Tokenizer method.')
    parser.add_argument('--spm_model', type=str, default='trained_spm_model.model', help='SPM model file.')
    parser.add_argument('--spm_vocab', type=str, default='trained_spm_model.vocab', help='SPM vocab file.')
    parser.add_argument(
        '--char_bpe_vocab_path',
        type=str,
        default=None,
        help='Path to an existing char_bpe meta.pkl used to reuse a prebuilt vocab.',
    )
    parser.add_argument(
        '--json_tokens_file',
        type=str,
        default=None,
        help='Path to a JSON token list used by json_byte_fallback tokenization.',
    )
    parser.add_argument(
        '--max_parallel',
        type=int,
        default=1,
        help='Maximum number of shard tokenization subprocesses to run concurrently.',
    )

    args = parser.parse_args()
    batch_prepare(
        args.input_dir,
        args.train_output,
        args.val_output,
        args.prepare_script,
        args.tokenizer,
        args.spm_model,
        args.spm_vocab,
        args.char_bpe_vocab_path,
        args.json_tokens_file,
        args.max_parallel,
        train_ratio=args.train_ratio,
    )
