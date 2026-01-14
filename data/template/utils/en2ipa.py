# data/template/utils/en2ipa.py

import subprocess
from konlpy.tag import Okt
import argparse
import re
import json
from typing import List, Tuple
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading

counter = 0
counter_lock = threading.Lock()

def transcribe_english(sentence, wrapper=False):
    """Transcribe an English sentence into its phonemes using espeak."""
    try:
        result = subprocess.run(
            ["espeak-ng", "-q", "-v", "en", "--ipa", sentence],
            capture_output=True,
            text=True
        )
        transcription = result.stdout.strip().replace("ㆍ", " ")
        if "(en)" in transcription:
            return f"[[[[[{sentence}]]]]]" if wrapper else sentence
        return transcription
    except Exception as e:
        return f"Error in transcribing English: {str(e)}"

def handle_mixed_language(word, wrapper=False):
    """Handle a word with potential English, Language, or number content."""
    global counter
    if word.isdigit():
        return word
    elif any('a' <= char.lower() <= 'z' for char in word):
        return transcribe_english(word, wrapper=wrapper)
    else:
        if wrapper:
            return "[[[[[" + word + "]]]]]"
        else:
            # thread-safe increment
            with counter_lock:
                counter += 1
            return word

_WORD_RE = re.compile(r'\w+|[^\w\s]', re.UNICODE)

def transcribe_tokens_to_string(tokens: List[str], wrapper: bool) -> str:
    result = []
    for tok in tokens:
        if re.match(r'\w+', tok):
            result.append(handle_mixed_language(tok, wrapper=wrapper))
        else:
            result.append(tok)
    return " ".join(result)

def _worker_sentence(sentence: str, wrapper: bool) -> str:
    """Worker function: tokenize and transcribe one sentence/line."""
    tokens = _WORD_RE.findall(sentence)
    return transcribe_tokens_to_string(tokens, wrapper=wrapper)

def _progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    )

def transcribe_multilingual(sentences, input_json_key=None, output_json_key='ipa', wrapper=False,
                            multithread: bool = False, workers: int = 0):
    """Transcribe multilingual sentences (JSON list mode)."""
    try:
        data = json.loads(sentences) if isinstance(sentences, str) else sentences
        if not isinstance(data, list):
            raise ValueError("JSON data should be a list of objects.")

        n = len(data)
        if n == 0:
            return json.dumps(data, ensure_ascii=False, indent=4)

        if not multithread or workers <= 1:
            # Single-threaded path (original behavior)
            with _progress() as progress:
                task = progress.add_task("Processing JSON items", total=n)
                for item in data:
                    if input_json_key in item:
                        sentence = item[input_json_key]
                        item[output_json_key] = _worker_sentence(sentence, wrapper)
                    progress.update(task, advance=1)
        else:
            # Multithreaded path with ordered assembly
            results: List[Tuple[int, str]] = [None] * n  # type: ignore
            # prepare jobs
            jobs = []
            for idx, item in enumerate(data):
                sentence = item.get(input_json_key, "")
                jobs.append((idx, sentence))

            with _progress() as progress:
                task = progress.add_task(f"Processing JSON items (mt x{workers})", total=n)
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    future_to_idx = {
                        ex.submit(_worker_sentence, sentence, wrapper): idx
                        for idx, sentence in jobs
                    }
                    for fut in as_completed(future_to_idx):
                        idx = future_to_idx[fut]
                        try:
                            res = fut.result()
                        except Exception as e:
                            res = f"Error: {e}"
                        results[idx] = (idx, res)
                        progress.update(task, advance=1)

            # write back in original order
            for idx, item in enumerate(data):
                if input_json_key in item:
                    item[output_json_key] = results[idx][1]

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error: {e}")
        return None

    return json.dumps(data, ensure_ascii=False, indent=4)

def transcribe_text_lines(lines: List[str], wrapper: bool, multithread: bool = False, workers: int = 0) -> List[str]:
    """Transcribe a plain-text file line-by-line."""
    n = len(lines)
    if n == 0:
        return []

    if not multithread or workers <= 1:
        # Single-threaded
        out_lines: List[str] = []
        with _progress() as progress:
            task = progress.add_task("Processing text lines", total=n)
            for line in lines:
                raw = line.rstrip("\n")
                out_lines.append(_worker_sentence(raw, wrapper))
                progress.update(task, advance=1)
        return out_lines
    else:
        # Multithreaded with ordered assembly
        out_lines: List[str] = [None] * n  # type: ignore
        with _progress() as progress:
            task = progress.add_task(f"Processing text lines (mt x{workers})", total=n)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                future_to_idx = {
                    ex.submit(_worker_sentence, lines[i].rstrip("\n"), wrapper): i
                    for i in range(n)
                }
                for fut in as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    try:
                        out_lines[idx] = fut.result()
                    except Exception as e:
                        out_lines[idx] = f"Error: {e}"
                    progress.update(task, advance=1)
        return out_lines

def main():
    parser = argparse.ArgumentParser(
        description='Transcribe multilingual content into IPA phonemes. Supports JSON list mode and plain-text line mode.'
    )
    parser.add_argument('input_file', type=str, help='Path to the input file (JSON list or plain text).')

    # Mode selection
    parser.add_argument('--mode', choices=['json', 'text'], default='json',
                        help='Processing mode. "json" expects a JSON list; "text" treats file as plain text.')

    # JSON mode params
    parser.add_argument('--input_json_key', type=str, help='JSON key to read sentences from (required for --mode json).')
    parser.add_argument('--output_json_key', type=str, default='ipa', help='JSON key to store IPA (default: "ipa").')

    # Text mode params
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file path for text mode. Defaults to overwriting input.')

    # Common options
    parser.add_argument("--wrapper", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Wrap unparseable non-English text with [[[[[...]]]]] for later recovery.")

    # Multithreading options
    parser.add_argument("--multithread", default=False, action=argparse.BooleanOptionalAction,
                        help="Enable multithreading for faster processing while preserving output order.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                        help="Number of worker threads when --multithread is enabled (default: CPU count).")

    args = parser.parse_args()

    # clamp workers
    if args.workers is None or args.workers < 1:
        args.workers = 1

    try:
        if args.mode == 'json':
            if not args.input_json_key:
                raise ValueError("--input_json_key is required when --mode json")
            with open(args.input_file, 'r', encoding='utf-8') as f:
                input_content = f.read()
            updated_json_data = transcribe_multilingual(
                input_content,
                args.input_json_key,
                args.output_json_key,
                wrapper=args.wrapper,
                multithread=args.multithread,
                workers=args.workers
            )
            if updated_json_data:
                with open(args.input_file, 'w', encoding='utf-8') as f:
                    f.write(updated_json_data)
                print(f"✅ Successfully updated JSON data in '{args.input_file}'")
        else:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            out_lines = transcribe_text_lines(
                lines,
                wrapper=args.wrapper,
                multithread=args.multithread,
                workers=args.workers
            )
            target_path = args.output_file if args.output_file else args.input_file
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
            print(f"✅ Successfully wrote transcribed text to '{target_path}'")

        print(f"📊 Stats: {counter} unparseable words")
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
    except ValueError as ve:
        print(f"Error: {ve}")

if __name__ == '__main__':
    main()

