# tests.py

import unittest
import os
import sys
import pickle
import json
from tokenizer_options import (
    NumericRangeTokenizer,
    SentencePieceTokenizer,
    TiktokenTokenizer,
    CustomTokenizer,
    CharTokenizer,
    CustomCharTokenizerWithByteFallback,
    Qwen2Tokenizer,
    JsonByteTokenizerWithByteFallback,
)
from argparse import Namespace
from rich.console import Console
from rich.theme import Theme
from rich.table import Table

console = Console(theme=Theme({
    "pass": "bold green",
    "fail": "bold red",
    "test_name": "bold yellow",
    "separator": "grey50",
    "input": "bold cyan",
    "output": "bold magenta",
    "info": "bold blue",
}))

class RichTestResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.test_results = []

    def addSuccess(self, test):
        self.test_results.append((test, 'PASS'))
        console.print("[bold green]Test Passed.[/bold green]")
        super().addSuccess(test)

    def addFailure(self, test, err):
        self.test_results.append((test, 'FAIL'))
        console.print("[bold red]Test Failed.[/bold red]")
        super().addFailure(test, err)

    def addError(self, test, err):
        self.test_results.append((test, 'FAIL'))
        console.print("[bold red]Test Error.[/bold red]")
        super().addError(test, err)

    def startTest(self, test):
        console.print('-' * 80, style='separator')
        console.print(f"Running test: [bold]{test._testMethodName}[/bold]", style='test_name')
        super().startTest(test)

    def stopTest(self, test):
        super().stopTest(test)


def run_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestTokenizers)
    result = RichTestResult()
    suite.run(result)
    # Print final table
    console.print('=' * 80, style='separator')
    console.print("[bold]Test Results:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Test")
    table.add_column("Result", justify="center")
    for test, status in result.test_results:
        test_name = test._testMethodName
        style = "pass" if status == 'PASS' else "fail"
        table.add_row(test_name, f"[{style}]{status}[/{style}]")
    console.print(table)

    # Exit with error code if any test failed
    if not result.wasSuccessful():
        sys.exit(1)  # Exit with status code 1 if tests failed
    else:
        sys.exit(0)  # Exit with status code 0 if all tests passed


class TestTokenizers(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.sample_text = "Hello\nworld\nThis is a test."
        self.numeric_data = "123\n456\n789"
        self.tokens_file = "tokens.txt"

        # Create a tokens file for custom tokenizers
        with open(self.tokens_file, 'w') as f:
            f.write("Hello\nworld\nThis is a test.\n")

    def tearDown(self):
        # Clean up tokens file
        if os.path.exists(self.tokens_file):
            os.remove(self.tokens_file)
        # Remove temporary files created by SentencePiece
        for fname in ["spm_input.txt", "trained_spm_model"]:
            for ext in ["", ".model", ".vocab"]:
                full_name = f"{fname}{ext}"
                if os.path.exists(full_name):
                    os.remove(full_name)
        if os.path.exists("meta.pkl"):
            os.remove("meta.pkl")
        if os.path.exists("remaining.txt"):
            os.remove("remaining.txt")

    # --------------------------------------------------------------------------
    # Helper Method to Print Token Count Histogram
    # --------------------------------------------------------------------------
    def _print_token_count_histogram(self, token_counts, itos):
        """
        Prints a histogram of all tokens in `token_counts`, sorted by descending frequency.
        Columns: Token ID, Actual Token, Count, Bar
        """

        if not token_counts:
            console.print("[info]No token counts to display.[/info]")
            return

        console.print("[info]Token Count Histogram (All Tokens):[/info]")
        table = Table("Token ID", "Token", "Count", "Bar", title="Histogram")

        # Sort all tokens in descending order by count
        sorted_counts = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        max_count = max(token_counts.values())

        for token_id, count in sorted_counts:
            token_str = itos.get(token_id, f"<UNK:{token_id}>")
            bar_len = 20  # max width in characters
            filled = int((count / max_count) * bar_len)
            bar_str = "█" * filled
            table.add_row(str(token_id), repr(token_str), str(count), bar_str)

        console.print(table)
        console.print()  # extra newline

    # --------------------------------------------------------------------------
    # Tokenizer Tests
    # --------------------------------------------------------------------------
    def test_numeric_range_tokenizer(self):
        args = Namespace(min_token=100, max_token=1000)
        tokenizer = NumericRangeTokenizer(args)
        ids = tokenizer.tokenize(self.numeric_data)
        detokenized = tokenizer.detokenize(ids)

        console.print("[input]Input:[/input]")
        console.print(self.numeric_data.strip(), style="input")
        console.print("[output]Detokenized Output:[/output]")
        console.print(detokenized, style="output")

        self.assertEqual(self.numeric_data.strip(), detokenized)

    def test_sentencepiece_tokenizer(self):
        args = Namespace(
            vocab_size=30,
            spm_model_file=None,
            spm_vocab_file=None,
            skip_tokenization=False
        )
        # Simulate training data
        with open("spm_input.txt", "w") as f:
            f.write(self.sample_text)
        tokenizer = SentencePieceTokenizer(args, input_files="spm_input.txt")
        ids = tokenizer.tokenize(self.sample_text)
        detokenized = tokenizer.detokenize(ids)

        console.print("[input]Input:[/input]")
        console.print(self.sample_text, style="input")
        console.print("[output]Detokenized Output:[/output]")
        console.print(detokenized, style="output")

        self.assertEqual(self.sample_text, detokenized)

    def test_tiktoken_tokenizer(self):
        args = Namespace(tiktoken_encoding='gpt2')
        tokenizer = TiktokenTokenizer(args)
        ids = tokenizer.tokenize(self.sample_text)
        detokenized = tokenizer.detokenize(ids)

        console.print("[input]Input:[/input]")
        console.print(self.sample_text, style="input")
        console.print("[output]Detokenized Output:[/output]")
        console.print(detokenized, style="output")

        self.assertEqual(self.sample_text, detokenized)

    def test_custom_tokenizer(self):
        args = Namespace(tokens_file=self.tokens_file)
        tokenizer = CustomTokenizer(args)
        ids = tokenizer.tokenize(self.sample_text)
        detokenized = tokenizer.detokenize(ids)

        console.print("[input]Input:[/input]")
        console.print(self.sample_text, style="input")
        console.print("[output]Detokenized Output:[/output]")
        console.print(detokenized, style="output")

        tokens_to_check = ["Hello", "world", "This", "is", "a", "test"]
        for token in tokens_to_check:
            self.assertIn(token, detokenized)

    def test_char_tokenizer(self):
        args = Namespace(reuse_chars=False)
        tokenizer = CharTokenizer(args, self.sample_text, None)
        ids = tokenizer.tokenize(self.sample_text)
        detokenized = tokenizer.detokenize(ids)

        console.print("[input]Input:[/input]")
        console.print(self.sample_text, style="input")
        console.print("[output]Detokenized Output:[/output]")
        console.print(detokenized, style="output")

        self.assertEqual(self.sample_text, detokenized)

    def test_custom_char_tokenizer_with_byte_fallback(self):
        args = Namespace(custom_chars_file="custom_chars.txt")
        # Create a custom characters file for testing
        with open(args.custom_chars_file, 'w', encoding='utf-8') as f:
            f.write('a\nb\nc\n\\n')

        tokenizer = CustomCharTokenizerWithByteFallback(args)
        test_string = "abc😊😊dd\nefg"

        ids = tokenizer.tokenize(test_string)
        detokenized = tokenizer.detokenize(ids)

        console.print("[input]Input:[/input]")
        console.print(test_string, style="input")
        console.print("[output]Detokenized Output:[/output]")
        console.print(detokenized, style="output")

        console.print("[info]Characters that used byte fallback:[/info]")
        bft = []
        for char in detokenized:
            # If it's not in the custom tokens, we consider it fallback
            if char not in tokenizer.custom_tokens:
                bft.append(repr(char))

        console.print(", ".join(bft), style="info")

        self.assertEqual(test_string, detokenized)
        console.print("CustomCharTokenizerWithByteFallback test passed.")

        # Clean up
        if os.path.exists(args.custom_chars_file):
            os.remove(args.custom_chars_file)

    def test_qwen2_tokenizer(self):
        args = Namespace(qwen2_model="qwen2_0p5b")
        tokenizer = Qwen2Tokenizer(args)
        
        # Tokenize
        ids = tokenizer.tokenize(self.sample_text)
        detokenized = tokenizer.detokenize(ids)

        console.print("[input]Input:[/input]")
        console.print(self.sample_text, style="input")
        console.print("[output]Detokenized Output:[/output]")
        console.print(detokenized, style="output")

        # Assert that detokenized text matches original
        self.assertEqual(self.sample_text, detokenized)

    def test_json_byte_tokenizer_with_byte_fallback(self):
        # Create a temporary JSON file with test tokens
        json_tokens_file = "test_tokens.json"
        test_tokens = ["Hello", "world", "This", "is", "a", "test"]
        with open(json_tokens_file, 'w', encoding='utf-8') as f:
            json.dump(test_tokens, f)

        args = Namespace(json_tokens_file=json_tokens_file, track_token_counts=True)
        test_string = "Hello world😊😊 This is a test"

        tokenizer = JsonByteTokenizerWithByteFallback(args)
        ids = tokenizer.tokenize(test_string)
        detokenized = tokenizer.detokenize(ids)

        console.print("[input]Input:[/input]")
        console.print(test_string, style="input")
        console.print("[output]Detokenized Output:[/output]")
        console.print(detokenized, style="output")

        # Get token counts from meta.pkl
        with open("meta.pkl", "rb") as f:
            meta = pickle.load(f)
        token_counts = meta.get("token_counts", {})
        itos = meta.get("itos", {})

        # Print histogram
        self._print_token_count_histogram(token_counts, itos)

        self.assertEqual(test_string, detokenized)
        self.assertEqual(meta["tokenizer"], "json_byte_fallback")
        self.assertEqual(meta["custom_token_count"], len(test_tokens))

        # Clean up
        if os.path.exists(json_tokens_file):
            os.remove(json_tokens_file)

        console.print("JsonByteTokenizerWithByteFallback test passed.")

    # --------------------------------------------------------------------------
    # Tests for Token Counts (with histogram printing)
    # --------------------------------------------------------------------------
    def test_numeric_range_tokenizer_counts(self):
        args = Namespace(min_token=100, max_token=1000, track_token_counts=True)
        tokenizer = NumericRangeTokenizer(args)
        ids = tokenizer.tokenize(self.numeric_data)

        with open("meta.pkl", "rb") as f:
            meta = pickle.load(f)
        token_counts = meta.get("token_counts", {})

        # Retrieve the itos mapping so we can display actual tokens in the histogram
        itos = meta.get("itos", {})

        # Print histogram
        self._print_token_count_histogram(token_counts, itos)

        self.assertEqual(
            sum(token_counts.values()), 
            len(ids),
            "Total token counts should match number of tokens."
        )
        for token_id in ids:
            self.assertIn(token_id, token_counts, 
                          "Each token id should appear in token_counts.")

    def test_sentencepiece_tokenizer_counts(self):
        with open("spm_input.txt", "w") as f:
            f.write(self.sample_text)

        args = Namespace(
            vocab_size=30,
            spm_model_file=None,
            spm_vocab_file=None,
            skip_tokenization=False,
            track_token_counts=True
        )
        tokenizer = SentencePieceTokenizer(args, input_files="spm_input.txt")
        ids = tokenizer.tokenize(self.sample_text)

        with open("meta.pkl", "rb") as f:
            meta = pickle.load(f)
        token_counts = meta.get("token_counts", {})

        itos = meta.get("itos", {})

        # Print histogram
        self._print_token_count_histogram(token_counts, itos)

        self.assertEqual(
            sum(token_counts.values()), 
            len(ids),
            "Total token counts should match number of tokens for SentencePiece."
        )
        for token_id in ids:
            self.assertIn(token_id, token_counts)

    def test_tiktoken_tokenizer_counts(self):
        args = Namespace(tiktoken_encoding='gpt2', track_token_counts=True)
        tokenizer = TiktokenTokenizer(args)
        ids = tokenizer.tokenize(self.sample_text)

        with open("meta.pkl", "rb") as f:
            meta = pickle.load(f)
        token_counts = meta.get("token_counts", {})
        itos = meta.get("itos", {})

        # Print histogram
        self._print_token_count_histogram(token_counts, itos)

        self.assertEqual(
            sum(token_counts.values()), 
            len(ids),
            "Total token counts should match for Tiktoken."
        )
        for token_id in ids:
            self.assertIn(token_id, token_counts)

    def test_custom_tokenizer_counts(self):
        args = Namespace(tokens_file=self.tokens_file, track_token_counts=True)
        tokenizer = CustomTokenizer(args)
        ids = tokenizer.tokenize(self.sample_text)

        with open("meta.pkl", "rb") as f:
            meta = pickle.load(f)
        token_counts = meta.get("token_counts", {})
        itos = meta.get("itos", {})

        # Print histogram
        self._print_token_count_histogram(token_counts, itos)

        self.assertEqual(
            sum(token_counts.values()), 
            len(ids),
            "Total token counts should match for CustomTokenizer."
        )
        for token_id in ids:
            self.assertIn(token_id, token_counts)

    def test_char_tokenizer_counts(self):
        args = Namespace(reuse_chars=False, track_token_counts=True)
        tokenizer = CharTokenizer(args, self.sample_text, None)
        ids = tokenizer.tokenize(self.sample_text)

        with open("meta.pkl", "rb") as f:
            meta = pickle.load(f)
        token_counts = meta.get("token_counts", {})
        itos = meta.get("itos", {})

        # Print histogram
        self._print_token_count_histogram(token_counts, itos)

        self.assertEqual(
            sum(token_counts.values()), 
            len(ids),
            "Total token counts should match for CharTokenizer."
        )
        for token_id in ids:
            self.assertIn(token_id, token_counts)

    def test_custom_char_tokenizer_with_byte_fallback_counts(self):
        args = Namespace(custom_chars_file="custom_chars.txt", track_token_counts=True)
        test_string = "abc😊😊dd\nefg"
        with open(args.custom_chars_file, 'w', encoding='utf-8') as f:
            f.write('a\nb\nc\n\\n')

        tokenizer = CustomCharTokenizerWithByteFallback(args)
        ids = tokenizer.tokenize(test_string)

        with open("meta.pkl", "rb") as f:
            meta = pickle.load(f)
        token_counts = meta.get("token_counts", {})
        itos = meta.get("itos", {})

        # Print histogram
        self._print_token_count_histogram(token_counts, itos)

        self.assertEqual(
            sum(token_counts.values()), 
            len(ids),
            "Total token counts should match for CustomCharTokenizerWithByteFallback."
        )
        for token_id in ids:
            self.assertIn(token_id, token_counts)

        # Clean up
        if os.path.exists(args.custom_chars_file):
            os.remove(args.custom_chars_file)


if __name__ == '__main__':
    run_tests()

