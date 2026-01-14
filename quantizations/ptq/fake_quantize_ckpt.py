import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import textwrap
from collections import Counter
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import torch


_RICH_SPEC = importlib.util.find_spec("rich")
if _RICH_SPEC:
    from rich import box
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table

    _RICH_CONSOLE = Console(highlight=False)
else:
    _RICH_CONSOLE = None


_TEXTUAL_AVAILABLE = False
_TEXTUAL_SPEC = importlib.util.find_spec("textual")
if _TEXTUAL_SPEC:
    try:
        from textual import events, on, work
        from textual.app import App, ComposeResult
        from textual.containers import Container
        from textual.screen import Screen
        from textual.widgets import Button, DataTable, Footer, Header, Input, Label
    except Exception:  # pragma: no cover - import guard for optional dependency
        _TEXTUAL_SPEC = None
        App = ComposeResult = Container = Screen = Button = DataTable = Footer = Header = Input = Label = events = on = work = None  # type: ignore[assignment]
    else:
        _TEXTUAL_AVAILABLE = True

_YAML_SPEC = importlib.util.find_spec("yaml")
if _YAML_SPEC:
    import yaml  # type: ignore[import]
else:
    yaml = None  # type: ignore[assignment]

if TYPE_CHECKING:
    # Textual exposes RowKey for cursor information; importing for typing only.
    try:
        from textual.widgets._data_table import RowKey as TextualRowKey
    except Exception:  # pragma: no cover - typing import fallback
        TextualRowKey = str  # type: ignore[assignment]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply uniform fake quantization to all weights in a checkpoint. "
            "Supports optional per-tensor overrides and an interactive TUI."
        )
    )
    parser.add_argument(
        "ckpt_dir",
        type=str,
        help="Directory containing ckpt.pt and meta.pkl from a previous training run",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to write the quantized checkpoint (defaults to <ckpt_dir>_ptq)",
    )
    parser.add_argument(
        "--num_bits",
        type=int,
        default=8,
        help="Number of bits for uniform quantization",
    )
    parser.add_argument(
        "--per-tensor-bits",
        type=str,
        default=None,
        metavar="SPEC",
        help=(
            "Optional per-tensor bit-width overrides. Provide a path to a JSON file "
            "or an inline mapping such as 'tensor=4,other=8'."
        ),
    )
    parser.add_argument(
        "--tui-default-quantization",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a YAML file with per-tensor bit-width defaults to pre-populate "
            "the interactive UI."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Launch an interactive text UI to choose bit-widths for each tensor before "
            "quantization."
        ),
    )
    parser.add_argument(
        "--min-bits",
        type=int,
        default=1,
        help=(
            "Minimum allowed bit-width when selecting per-tensor values interactively. "
            "Use 0 to allow keeping tensors in floating point."
        ),
    )
    parser.add_argument(
        "--max-bits",
        type=int,
        default=16,
        help=(
            "Maximum allowed bit-width when selecting per-tensor values interactively."
        ),
    )
    parser.add_argument(
        "--tui-page-size",
        type=int,
        default=20,
        help="Number of tensors to display per page in the interactive TUI.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="symmetric",
        choices=("symmetric", "asymmetric"),
        help=(
            "Quantization scheme to use: symmetric signed (two's complement) or "
            "asymmetric unsigned"
        ),
    )
    args = parser.parse_args()
    if args.num_bits < 0:
        parser.error("--num_bits must be non-negative")
    if args.min_bits < 0:
        parser.error("--min-bits must be non-negative")
    if args.max_bits is not None and args.max_bits <= 0:
        parser.error("--max-bits must be positive")
    if args.max_bits is not None and args.min_bits > args.max_bits:
        parser.error("--min-bits cannot exceed --max-bits")
    if args.tui_page_size <= 0:
        parser.error("--tui-page-size must be positive")
    return args


@dataclass
class TensorConfigEntry:
    name: str
    shape: Tuple[int, ...]
    numel: int
    dtype: str
    default_bits: int
    bits: int
    prior_bits: Optional[int] = None


LAST_QUANTIZATION_BASENAME = "last_fake_ptq_quantization.yaml"


def _print_info(message: str) -> None:
    if _RICH_CONSOLE:
        _RICH_CONSOLE.print(f"[cyan]{message}[/cyan]")
    else:
        print(message)


def _print_warning(message: str) -> None:
    if _RICH_CONSOLE:
        _RICH_CONSOLE.print(f"[yellow]Warning:[/yellow] {message}")
    else:
        print(f"Warning: {message}")


def _format_bits_label(bits: int) -> str:
    if bits <= 0:
        return "fp32"
    return f"{bits}-bit"


def _parse_bits_value(value) -> int:
    if isinstance(value, bool):
        raise ValueError("Bit-width must be an integer, not a boolean value")
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"Bit-width must be an integer, got {value}")
            bits = int(round(value))
        else:
            bits = int(value)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("Bit-width value cannot be empty")
        lowered = text.lower()
        if lowered in {"fp32", "float", "skip", "none"}:
            return 0
        bits = int(text, 0)
    if bits < 0:
        raise ValueError("Bit-width must be non-negative")
    return bits


def _parse_simple_mapping(text: str) -> Optional[Dict[str, int]]:
    mapping: Dict[str, int] = {}
    found_entry = False
    for raw_line in text.replace(",", "\n").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        found_entry = True
        if "=" in line:
            key, bits_str = line.split("=", 1)
        elif ":" in line:
            key, bits_str = line.split(":", 1)
        else:
            return None
        key = key.strip()
        bits_str = bits_str.strip()
        if not key:
            raise ValueError("Missing tensor name in per-tensor bit specification")
        if not bits_str:
            raise ValueError(f"Missing bit-width for tensor '{key}'")
        bits = _parse_bits_value(bits_str)
        mapping[key] = bits
    if not found_entry:
        return {}
    return mapping


def _load_yaml_mapping(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    if yaml is not None:
        data = yaml.safe_load(text) or {}
        if not isinstance(data, MutableMapping):
            raise ValueError("YAML file must contain a mapping of tensor names to bit-widths")
        return {str(key): _parse_bits_value(value) for key, value in data.items()}

    mapping: Dict[str, int] = {}
    for index, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(
                f"Line {index} is not a valid 'tensor: bits' entry: {raw_line.strip()}"
            )
        name, bits_text = line.split(":", 1)
        name = name.strip()
        bits_text = bits_text.strip()
        if not name:
            raise ValueError(f"Line {index} is missing a tensor name")
        if not bits_text:
            raise ValueError(f"Line {index} is missing a bit-width for '{name}'")
        mapping[name] = _parse_bits_value(bits_text)
    return mapping


def save_quantization_yaml(path: str, mapping: Dict[str, Optional[int]]) -> None:
    normalized: Dict[str, int] = {}
    for name, bits in mapping.items():
        if bits is None:
            normalized[name] = 0
        else:
            normalized[name] = _parse_bits_value(bits)

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if yaml is not None:
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(sorted(normalized.items())), handle, sort_keys=False)
        return

    lines = ["# Fake PTQ per-tensor bit-widths"]
    for name, bits in sorted(normalized.items()):
        value = "fp32" if bits <= 0 else str(int(bits))
        lines.append(f"{name}: {value}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_per_tensor_bits(spec: Optional[str]) -> Dict[str, int]:
    if spec is None:
        return {}

    spec = spec.strip()
    if not spec:
        return {}

    data = None
    if os.path.exists(spec):
        with open(spec, "r", encoding="utf-8") as handle:
            text = handle.read()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            mapping = _parse_simple_mapping(text)
            if mapping is None:
                raise ValueError(
                    "Unable to parse per-tensor bit specification file. "
                    "Use JSON or key=value pairs."
                )
            return mapping
    else:
        try:
            data = json.loads(spec)
        except json.JSONDecodeError:
            mapping = _parse_simple_mapping(spec)
            if mapping is None:
                raise ValueError(
                    "Unable to parse per-tensor bit specification. "
                    "Use JSON or comma-separated key=value pairs."
                )
            return mapping

    if not isinstance(data, dict):
        raise ValueError("Per-tensor bit specification must be a mapping")

    mapping: Dict[str, int] = {}
    for key, value in data.items():
        mapping[str(key)] = _parse_bits_value(value)
    return mapping


def _filter_tensor_bit_mapping(
    mapping: Dict[str, int], state_dict, *, label: str
) -> Dict[str, int]:
    valid: Dict[str, int] = {}
    for name, bits in mapping.items():
        if name not in state_dict:
            _print_warning(f"Ignoring {label} for unknown tensor '{name}'")
            continue
        value = state_dict[name]
        if not torch.is_tensor(value):
            _print_warning(f"Ignoring {label} for non-tensor entry '{name}'")
            continue
        if not torch.is_floating_point(value):
            _print_warning(
                f"Ignoring {label} for non-floating tensor '{name}' (dtype: {value.dtype})"
            )
            continue
        valid[name] = bits
    return valid


def iter_state_items(state_dict) -> Iterable[Tuple[str, torch.Tensor]]:
    if isinstance(state_dict, torch.nn.Module):
        iterable = state_dict.state_dict().items()
    elif isinstance(state_dict, dict):
        iterable = state_dict.items()
    else:
        iterable = getattr(state_dict, "state_dict", lambda: {})().items()

    for key, value in iterable:
        if torch.is_tensor(value):
            yield key, value


def build_tensor_config_entries(
    state_dict,
    default_bits: int,
    overrides: Dict[str, int],
    default_overrides: Optional[Dict[str, int]] = None,
    prior_bits: Optional[Dict[str, int]] = None,
) -> List[TensorConfigEntry]:
    entries: List[TensorConfigEntry] = []
    default_overrides = default_overrides or {}
    prior_bits = prior_bits or {}
    for name, tensor in iter_state_items(state_dict):
        if not torch.is_floating_point(tensor):
            continue
        baseline_bits = default_overrides.get(name, default_bits)
        initial_bits = overrides.get(name, baseline_bits)
        dtype_str = str(tensor.dtype)
        if dtype_str.startswith("torch."):
            dtype_str = dtype_str.split(".", 1)[1]
        entries.append(
            TensorConfigEntry(
                name=name,
                shape=tuple(tensor.shape),
                numel=tensor.numel(),
                dtype=dtype_str,
                default_bits=baseline_bits,
                bits=initial_bits,
                prior_bits=prior_bits.get(name),
            )
        )
    return entries


def _format_shape(shape: Tuple[int, ...]) -> str:
    if not shape:
        return "scalar"
    return "×".join(str(dim) for dim in shape)


def _resolve_entry(
    entries: List[TensorConfigEntry], target: str
) -> Tuple[Optional[TensorConfigEntry], Optional[str]]:
    identifier = target.strip()
    if not identifier:
        return None, "Tensor identifier cannot be empty"

    if identifier.isdigit():
        index = int(identifier)
        if index < 1 or index > len(entries):
            return None, f"Index {index} is out of range (1-{len(entries)})"
        return entries[index - 1], None

    for entry in entries:
        if entry.name == identifier:
            return entry, None

    matches = [entry for entry in entries if identifier in entry.name]
    if not matches:
        return None, f"No tensor matching '{target}'"
    if len(matches) > 1:
        preview = ", ".join(entry.name for entry in matches[:5])
        if len(matches) > 5:
            preview += ", ..."
        return None, f"Ambiguous tensor name '{target}' (matches: {preview})"
    return matches[0], None


def _interactive_instruction_text(min_bits: int, max_bits: Optional[int]) -> str:
    if max_bits is None:
        if min_bits <= 0:
            allowed_line = (
                "Allowed bit-widths: 0 (keep float32) or any positive integer"
            )
        else:
            allowed_line = (
                f"Allowed bit-widths: {min_bits}+ (use 0 to keep float32)"
            )
    else:
        if min_bits <= 0:
            allowed_line = (
                f"Allowed bit-widths: 0 (keep float32) or 1-{max_bits}"
            )
        else:
            allowed_line = (
                f"Allowed bit-widths: {min_bits}-{max_bits} (use 0 to keep float32)"
            )

    return textwrap.dedent(
        f"""
        Commands:
          set <index|name> <bits>  Set bit-width for a tensor (use 0 to keep float32)
          all <bits>               Apply a bit-width to every tensor
          reset [<index|name>]     Reset all or a single tensor to the default value
          next / prev              Move between pages of tensors
          page <number>            Jump to a specific page (1-based)
          done / apply             Finish configuration and continue
          quit / cancel            Abort the operation
        {allowed_line}
        """
    ).strip()


def _render_tensor_table(
    entries: List[TensorConfigEntry],
    page: int,
    page_size: int,
    total_pages: int,
    instructions: str,
    status_message: str,
    status_style: str,
) -> None:
    total = len(entries)
    start = page * page_size
    end = min(start + page_size, total)

    if _RICH_CONSOLE:
        _RICH_CONSOLE.clear()
        instructions_panel = Panel(
            instructions,
            title="Per-tensor bit-width selection",
            border_style="bright_blue",
            padding=(0, 1),
            highlight=False,
        )
        _RICH_CONSOLE.print(instructions_panel)

        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.SIMPLE_HEAVY,
            expand=True,
        )
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
        table.add_column("Tensor", style="bright_white")
        table.add_column("Shape", style="green")
        table.add_column("DType", style="cyan")
        table.add_column("Elements", justify="right", style="yellow")
        table.add_column("Current", style="bright_white")
        table.add_column("Default", style="dim")
        table.add_column("Prior", style="cyan")

        for idx, entry in enumerate(entries[start:end], start=start + 1):
            shape_str = _format_shape(entry.shape)
            numel_str = f"{entry.numel:,}"
            bits_label = _format_bits_label(entry.bits)
            default_label = _format_bits_label(entry.default_bits)
            if entry.bits <= 0:
                bits_display = f"[dim]{bits_label}[/dim]"
            elif entry.bits != entry.default_bits:
                bits_display = f"[bold yellow]{bits_label}[/bold yellow]"
            else:
                bits_display = bits_label
            if entry.prior_bits is None:
                prior_display = "[dim]—[/dim]"
            else:
                prior_label = _format_bits_label(entry.prior_bits)
                if (
                    entry.prior_bits <= 0
                    or entry.prior_bits == entry.bits
                    or entry.prior_bits == entry.default_bits
                ):
                    prior_display = f"[dim]{prior_label}[/dim]"
                else:
                    prior_display = f"[cyan]{prior_label}[/cyan]"
            table.add_row(
                str(idx),
                entry.name,
                shape_str,
                entry.dtype,
                numel_str,
                bits_display,
                default_label,
                prior_display,
            )

        table.caption = (
            f"Showing tensors {start + 1}-{end} of {total} "
            f"(page {page + 1}/{total_pages})"
        )
        table.caption_style = "dim"
        _RICH_CONSOLE.print(table)

        if status_message:
            _RICH_CONSOLE.print(f"[{status_style}]{status_message}[/{status_style}]")
        return

    print("\033c", end="")
    print("Per-tensor bit-width selection")
    print("=" * 72)
    print(instructions)
    print("")
    header = (
        f"{'#':>4}  {'Tensor':<48}  {'Shape':<24}  {'DType':<12}  "
        f"{'Elements':>12}  {'Bits':>8}  {'Default':>8}  {'Prior':>8}"
    )
    print(header)
    print("-" * len(header))
    for idx, entry in enumerate(entries[start:end], start=start + 1):
        name_display = entry.name if len(entry.name) <= 48 else entry.name[:45] + "..."
        shape_str = _format_shape(entry.shape)
        if len(shape_str) > 24:
            shape_str = shape_str[:21] + "..."
        bits_label = _format_bits_label(entry.bits)
        default_label = _format_bits_label(entry.default_bits)
        if entry.prior_bits is None:
            prior_plain = "-"
        else:
            prior_plain = _format_bits_label(entry.prior_bits)
        print(
            f"{idx:>4}  {name_display:<48}  {shape_str:<24}  {entry.dtype:<12}  "
            f"{entry.numel:>12,d}  {bits_label:>8}  {default_label:>8}  {prior_plain:>8}"
        )
    print("-" * len(header))
    print(
        f"Showing tensors {start + 1}-{end} of {total} (page {page + 1}/{total_pages})"
    )
    if status_message:
        print(status_message)


def _legacy_interactive_select_tensor_bits(
    entries: List[TensorConfigEntry],
    min_bits: int,
    max_bits: Optional[int],
    page_size: int,
) -> Dict[str, int]:
    if not entries:
        _print_warning(
            "No floating-point tensors were found for interactive configuration."
        )
        return {}

    page_size = max(page_size, 1)
    total_pages = max(1, math.ceil(len(entries) / page_size))
    page = 0
    instructions = _interactive_instruction_text(min_bits, max_bits)
    status_message = ""
    status_style = "cyan"

    def set_status(message: str, style: str = "cyan") -> None:
        nonlocal status_message, status_style
        status_message = message
        status_style = style

    _render_tensor_table(
        entries, page, page_size, total_pages, instructions, status_message, status_style
    )

    while True:
        try:
            raw = input("tui> ").strip()
        except EOFError:
            set_status("EOF received; applying current configuration.", "yellow")
            break
        except KeyboardInterrupt:
            raise SystemExit("Interactive configuration canceled by user.") from None

        if not raw:
            set_status("", "cyan")
            _render_tensor_table(
                entries,
                page,
                page_size,
                total_pages,
                instructions,
                status_message,
                status_style,
            )
            continue

        parts = raw.split()
        command = parts[0].lower()

        if command in {"done", "apply"}:
            set_status("Applying selected configuration.", "green")
            break

        if command in {"quit", "exit", "cancel"}:
            raise SystemExit("Interactive configuration canceled by user.")

        if command in {"next", "n"}:
            if page + 1 < total_pages:
                page += 1
                set_status(f"Moved to page {page + 1}/{total_pages}.")
            else:
                set_status("Already on the last page.", "yellow")
        elif command in {"prev", "p"}:
            if page > 0:
                page -= 1
                set_status(f"Moved to page {page + 1}/{total_pages}.")
            else:
                set_status("Already on the first page.", "yellow")
        elif command == "page":
            if len(parts) < 2:
                set_status("Usage: page <number>", "red")
            else:
                try:
                    new_page = int(parts[1]) - 1
                except ValueError:
                    set_status("Page number must be an integer.", "red")
                else:
                    if 0 <= new_page < total_pages:
                        page = new_page
                        set_status(f"Moved to page {page + 1}/{total_pages}.")
                    else:
                        set_status("Page number out of range.", "red")
        elif command == "set":
            if len(parts) < 3:
                set_status("Usage: set <index|name> <bits>", "red")
            else:
                target = parts[1]
                bits_str = parts[2]
                try:
                    bits = _parse_bits_value(bits_str)
                except ValueError as exc:
                    set_status(str(exc), "red")
                else:
                    if bits > 0 and bits < min_bits:
                        set_status(
                            f"Bit-width must be at least {min_bits} for quantized tensors.",
                            "red",
                        )
                    elif max_bits is not None and bits > 0 and bits > max_bits:
                        set_status(
                            f"Bit-width must be at most {max_bits} for quantized tensors.",
                            "red",
                        )
                    else:
                        entry, error = _resolve_entry(entries, target)
                        if entry is None:
                            set_status(error or "Unknown tensor", "red")
                        else:
                            entry.bits = bits
                            set_status(
                                f"Set {entry.name} to {_format_bits_label(bits)}.",
                                "green",
                            )
        elif command == "all":
            if len(parts) < 2:
                set_status("Usage: all <bits>", "red")
            else:
                try:
                    bits = _parse_bits_value(parts[1])
                except ValueError as exc:
                    set_status(str(exc), "red")
                else:
                    if bits > 0 and bits < min_bits:
                        set_status(
                            f"Bit-width must be at least {min_bits} for quantized tensors.",
                            "red",
                        )
                    elif max_bits is not None and bits > 0 and bits > max_bits:
                        set_status(
                            f"Bit-width must be at most {max_bits} for quantized tensors.",
                            "red",
                        )
                    else:
                        for entry in entries:
                            entry.bits = bits
                        set_status(
                            f"Applied {_format_bits_label(bits)} to {len(entries)} tensor(s).",
                            "green",
                        )
        elif command == "reset":
            if len(parts) == 1:
                for entry in entries:
                    entry.bits = entry.default_bits
                set_status("Reset all tensors to their default bit-widths.", "green")
            else:
                target = parts[1]
                entry, error = _resolve_entry(entries, target)
                if entry is None:
                    set_status(error or "Unknown tensor", "red")
                else:
                    entry.bits = entry.default_bits
                    set_status(
                        f"Reset {entry.name} to {_format_bits_label(entry.default_bits)}.",
                        "green",
                    )
        elif command in {"help", "?"}:
            set_status("Help is shown above.")
        else:
            set_status(f"Unknown command: {command}", "red")

        _render_tensor_table(
            entries,
            page,
            page_size,
            total_pages,
            instructions,
            status_message,
            status_style,
        )

    if _RICH_CONSOLE:
        _RICH_CONSOLE.print("[green]Interactive configuration complete.[/green]")
    else:
        print("Interactive configuration complete.")

    return {entry.name: entry.bits for entry in entries}


if _TEXTUAL_AVAILABLE:

    def _textual_help_text(min_bits: int, max_bits: Optional[int]) -> str:
        if max_bits is None:
            if min_bits <= 0:
                allowed_line = (
                    "Allowed bit-widths: 0 (keep float32) or any positive integer."
                )
            else:
                allowed_line = (
                    f"Allowed bit-widths: {min_bits}+ (use 0 to keep float32)."
                )
        else:
            if min_bits <= 0:
                allowed_line = (
                    f"Allowed bit-widths: 0 (keep float32) or 1-{max_bits}."
                )
            else:
                allowed_line = (
                    f"Allowed bit-widths: {min_bits}-{max_bits} (use 0 to keep float32)."
                )

        return textwrap.dedent(
            f"""[b]Hotkeys[/b]:
  • Arrow keys / Tab move between cells
  • Enter or b starts editing the selected tensor; type digits then press Enter
  • k increases and j decreases the selected tensor one bit at a time
  • f toggles float32 for the current tensor
  • a selects every tensor for a bulk update
  • g matches tensors by regular expression for a bulk update
  • r resets the highlighted tensor; R resets all tensors
  • p shows this command reference in the notification tray
  • Esc cancels the current entry or clears bulk selection highlights
  • Ctrl+Z / Ctrl+Y undo or redo recent changes
  • Ctrl+S applies the configuration, Ctrl+C cancels the session
  • Ctrl+Shift+S saves the current bit-widths to a YAML file
{allowed_line}
"""
        ).strip()


    class PromptScreen(Screen[str | None]):
        def __init__(
            self,
            prompt: str,
            placeholder: str = "",
            title: str = "Input",
            initial: Optional[str] = None,
        ) -> None:
            super().__init__()
            self.prompt_text = prompt
            self.placeholder = placeholder
            self.title = title
            self.initial_value = initial or ""

        def compose(self) -> ComposeResult:
            yield Label(self.prompt_text, id="prompt")
            yield Input(
                placeholder=self.placeholder,
                id="value",
                value=self.initial_value,
            )
            with Container(id="buttons"):
                yield Button("OK", id="ok", variant="success")
                yield Button("Cancel", id="cancel", variant="error")

        def on_mount(self) -> None:
            input_widget = self.query_one("#value", Input)
            input_widget.focus()

        @on(Input.Submitted, "#value")
        def _submitted(self, event: Input.Submitted) -> None:
            self.dismiss(event.value.strip() or None)

        @on(Button.Pressed, "#ok")
        def _ok(self) -> None:
            value = self.query_one("#value", Input).value.strip()
            self.dismiss(value or None)

        @on(Button.Pressed, "#cancel")
        def _cancel(self) -> None:
            self.dismiss(None)


    class TensorBitwidthApp(App[Dict[str, int]]):
        CSS = """
        Screen { align: center middle; }
        #body { height: 1fr; width: 1fr; }
        DataTable#table {
            height: 1fr;
            width: 1fr;
            overflow-x: auto;
            overflow-y: auto;
        }
        PromptScreen Container#buttons {
            layout: horizontal;
            padding-top: 1;
        }
        PromptScreen Container#buttons Button#cancel {
            margin-left: 1;
        }
        """

        BINDINGS = [
            ("ctrl+s", "confirm", "Apply"),
            ("ctrl+c", "cancel", "Cancel"),
            ("ctrl+z", "undo", "Undo"),
            ("ctrl+y", "redo", "Redo"),
            ("ctrl+shift+s", "save_yaml", "Save YAML"),
            ("enter", "noop", "Edit tensor"),
            ("b", "noop", "Edit tensor"),
            ("k", "noop", "Bit +1"),
            ("j", "noop", "Bit -1"),
            ("f", "noop", "Toggle fp32"),
            ("a", "noop", "Set all"),
            ("g", "noop", "Regex bulk"),
            ("r", "noop", "Reset tensor"),
            ("R", "noop", "Reset all"),
            ("p", "noop", "Commands"),
        ]

        def __init__(
            self,
            entries: List[TensorConfigEntry],
            min_bits: int,
            max_bits: Optional[int],
            page_size: int,
            save_directory: Optional[str] = None,
            last_save_path: Optional[str] = None,
        ) -> None:
            super().__init__()
            self.entries = entries
            self.min_bits = min_bits
            self.max_bits = max_bits
            self.page_size = page_size
            self.cancelled = False
            self._result: Optional[Dict[str, int]] = None
            self._entries_by_name = {entry.name: entry for entry in entries}
            self._highlight_matches: set[str] = set()
            self._digit_mode: Optional[str] = None
            self._digit_buffer: list[str] = []
            self._pending_targets: list[str] = []
            self._pending_label: Optional[str] = None
            self._undo_stack: list[tuple[list[tuple[str, int, int]], str]] = []
            self._redo_stack: list[tuple[list[tuple[str, int, int]], str]] = []
            self._history_limit = 10
            self.table: Optional[DataTable] = None
            self.help_text = _textual_help_text(min_bits, max_bits)
            self._save_directory = save_directory
            self._default_save_filename = "fake_ptq_bits.yaml"
            self._last_save_path = (
                os.path.abspath(last_save_path)
                if last_save_path
                else None
            )

        def compose(self) -> ComposeResult:
            yield Header(show_clock=False)
            with Container(id="body"):
                yield DataTable(id="table", zebra_stripes=True)
            yield Footer()

        def on_mount(self) -> None:
            self.table = self.query_one(DataTable)
            self.title = "Fake PTQ Bit-width Planner"
            self.sub_title = (
                "Arrows move • Enter edits • j/k adjust • p commands • "
                "Ctrl+Shift+S saves YAML"
            )
            if self.table is not None:
                self.table.cursor_type = "cell"
                try:
                    self.table.styles.height = max(self.page_size, 1) + 4
                except Exception:
                    pass
            self.refresh_table(preserve_cursor=False)
            self._msg("Use the arrow keys to highlight a tensor. Press p for commands.")

        def refresh_table(self, preserve_cursor: bool = True) -> None:
            if not self.table:
                return
            table = self.table
            cursor = table.cursor_coordinate if preserve_cursor else None
            table.clear(columns=True)
            table.add_column("#", key="index", width=6)
            table.add_column("Tensor", key="tensor")
            table.add_column("Shape", key="shape")
            table.add_column("DType", key="dtype")
            table.add_column("Elements", key="elements")
            table.add_column("Current", key="current")
            table.add_column("Default", key="default")
            table.add_column("Prior", key="prior")

            for idx, entry in enumerate(self.entries, start=1):
                tensor_label = entry.name
                if entry.name in self._highlight_matches:
                    tensor_label = f"[reverse]{tensor_label}[/reverse]"
                table.add_row(
                    str(idx),
                    tensor_label,
                    _format_shape(entry.shape),
                    entry.dtype,
                    f"{entry.numel:,}",
                    self._format_current(entry),
                    _format_bits_label(entry.default_bits),
                    self._format_prior(entry),
                    key=entry.name,
                )

            table.cursor_type = "cell"
            if self.entries:
                if cursor is not None:
                    row = max(0, min(cursor.row, len(self.entries) - 1))
                    col = max(0, min(cursor.column, len(table.columns) - 1))
                    try:
                        table.cursor_coordinate = (row, col)
                    except Exception:
                        table.cursor_coordinate = (row, 5)
                else:
                    try:
                        table.cursor_coordinate = (0, 5)
                    except Exception:
                        pass
            try:
                table.focus()
            except Exception:
                pass
            self._update_summary()

        def _format_current(self, entry: TensorConfigEntry) -> str:
            label = _format_bits_label(entry.bits)
            if entry.bits <= 0:
                return f"[dim]{label}[/dim]"
            if entry.bits != entry.default_bits:
                return f"[bold yellow]{label}[/bold yellow]"
            return label

        def _format_prior(self, entry: TensorConfigEntry) -> str:
            prior = entry.prior_bits
            if prior is None:
                return "[dim]—[/dim]"
            label = _format_bits_label(prior)
            if prior <= 0 or prior == entry.bits or prior == entry.default_bits:
                return f"[dim]{label}[/dim]"
            return f"[cyan]{label}[/cyan]"

        def _update_summary(self) -> None:
            if not self.entries:
                self.sub_title = "No tensors available for configuration."
                return
            counts = Counter(entry.bits for entry in self.entries)
            parts: List[str] = []
            for bits in sorted(counts.keys(), key=lambda value: (value <= 0, value)):
                parts.append(f"{counts[bits]}×{_format_bits_label(bits)}")
            summary = "Summary: " + ", ".join(parts)
            changed = sum(entry.bits != entry.default_bits for entry in self.entries)
            if changed:
                summary += f" • {changed} tensor(s) modified"
            summary += " • Ctrl+S to apply, Ctrl+C to cancel, p for commands"
            self.sub_title = summary

        def _current_entry(self) -> Optional[TensorConfigEntry]:
            if not self.table:
                return None
            coord = self.table.cursor_coordinate
            if coord is None:
                return None
            row_idx = coord.row
            if row_idx < 0 or row_idx >= len(self.entries):
                return None
            return self.entries[row_idx]

        def _min_quant_bits(self) -> int:
            return self.min_bits if self.min_bits > 0 else 1

        def _clamp_bits(self, bits: int) -> int:
            if bits <= 0:
                return 0
            if self.min_bits > 0 and bits < self.min_bits:
                bits = self.min_bits
            if self.max_bits is not None and bits > self.max_bits:
                bits = self.max_bits
            return bits

        def _push_history(
            self, changes: List[tuple[str, int, int]], description: str
        ) -> None:
            if not changes:
                return
            self._undo_stack.append((changes, description))
            if len(self._undo_stack) > self._history_limit:
                self._undo_stack.pop(0)
            self._redo_stack.clear()

        def _apply_changes(
            self,
            changes: List[tuple[str, int, int]],
            *,
            description: str = "",
            record: bool = True,
            notify: bool = True,
        ) -> None:
            if not changes:
                return
            for name, _old, new in changes:
                entry = self._entries_by_name.get(name)
                if entry is None:
                    continue
                entry.bits = new
            if record:
                self._push_history(changes, description)
            self.refresh_table()
            if notify and description:
                self._msg(description)

        def _nudge_entry(self, entry: TensorConfigEntry, delta: int) -> None:
            old = entry.bits
            if delta > 0:
                if old <= 0:
                    new = self._clamp_bits(self._min_quant_bits())
                else:
                    new = self._clamp_bits(old + delta)
            else:
                if old <= 0:
                    new = 0
                else:
                    new = old + delta
                    if new <= 0:
                        new = 0
                    else:
                        new = self._clamp_bits(new)
            if new == old:
                limit = "minimum" if delta < 0 else "maximum"
                self._msg(
                    f"{entry.name} already at the {limit}: {_format_bits_label(old)}."
                )
                self.bell()
                return
            description = (
                f"{entry.name}: {_format_bits_label(old)} → {_format_bits_label(new)}"
            )
            self._apply_changes([(entry.name, old, new)], description=description)

        def _toggle_float(self, entry: TensorConfigEntry) -> None:
            old = entry.bits
            if old <= 0:
                new = self._clamp_bits(self._min_quant_bits())
            else:
                new = 0
            if new == old:
                self._msg(f"{entry.name} already at {_format_bits_label(old)}.")
                return
            description = (
                f"{entry.name}: {_format_bits_label(old)} → {_format_bits_label(new)}"
            )
            self._apply_changes([(entry.name, old, new)], description=description)

        def _reset_entry(self, entry: TensorConfigEntry) -> None:
            old = entry.bits
            new = entry.default_bits
            if new == old:
                self._msg(
                    f"{entry.name} already at default {_format_bits_label(new)}."
                )
                return
            description = f"Reset {entry.name} to {_format_bits_label(new)}"
            self._apply_changes([(entry.name, old, new)], description=description)

        def _reset_all(self) -> None:
            changes: List[tuple[str, int, int]] = []
            for entry in self.entries:
                if entry.bits != entry.default_bits:
                    changes.append((entry.name, entry.bits, entry.default_bits))
            if not changes:
                self._msg("All tensors are already at their default bit-widths.")
                return
            description = (
                f"Reset {len(changes)} tensor(s) to their default bit-widths."
            )
            self._apply_changes(changes, description=description)

        def _start_single_digit(self, entry: TensorConfigEntry) -> None:
            was_bulk = self._digit_mode == "bulk"
            if self._digit_mode:
                self._cancel_digit_mode(clear_highlight=was_bulk)
            self._digit_mode = "single"
            self._digit_buffer = []
            self._pending_targets = [entry.name]
            self._pending_label = entry.name
            self._msg(
                "Type a new bit-width for "
                f"{entry.name} (use f for fp32), then press Enter (Esc to cancel)."
            )

        def _start_bulk_digits(self, names: List[str], label: str) -> None:
            was_bulk = self._digit_mode == "bulk"
            if self._digit_mode:
                self._cancel_digit_mode(clear_highlight=was_bulk)
            unique_names = list(dict.fromkeys(names))
            if not unique_names:
                self._msg("No tensors matched that selection.")
                return
            self._digit_mode = "bulk"
            self._digit_buffer = []
            self._pending_targets = unique_names
            self._pending_label = label
            self._msg(
                f"{len(unique_names)} tensor(s) selected ({label}). "
                "Type the new bit-width (use f for fp32) and press Enter (Esc to cancel)."
            )

        def _start_all_digits(self) -> None:
            if not self.entries:
                self._msg("No tensors available to update.")
                return
            self._highlight_matches = {entry.name for entry in self.entries}
            self.refresh_table()
            self._start_bulk_digits(
                [entry.name for entry in self.entries],
                "all tensors",
            )

        def _cancel_digit_mode(self, *, clear_highlight: bool = False) -> None:
            if clear_highlight and self._highlight_matches:
                self._highlight_matches.clear()
                self.refresh_table()
            self._digit_mode = None
            self._digit_buffer = []
            self._pending_targets = []
            self._pending_label = None

        def _clear_highlight(self) -> None:
            if self._highlight_matches:
                self._highlight_matches.clear()
                self.refresh_table()
                self._msg("Cleared highlighted selection.")

        def _apply_digit_buffer(self) -> None:
            if not self._pending_targets:
                self._cancel_digit_mode()
                return
            bits_text = "".join(self._digit_buffer)
            if not bits_text:
                was_bulk = self._digit_mode == "bulk"
                self._cancel_digit_mode(clear_highlight=was_bulk)
                self._msg("Bit entry cancelled.")
                return
            lowered = bits_text.lower()
            if lowered == "f":
                bits = 0
            else:
                try:
                    bits = int(bits_text)
                except ValueError:
                    was_bulk = self._digit_mode == "bulk"
                    self._cancel_digit_mode(clear_highlight=was_bulk)
                    self._msg("Bit-width must be an integer or 'f' for fp32.")
                    self.bell()
                    return
            new_bits = self._clamp_bits(bits)
            total = len(self._pending_targets)
            changes: List[tuple[str, int, int]] = []
            for name in self._pending_targets:
                entry = self._entries_by_name.get(name)
                if entry is None:
                    continue
                if entry.bits != new_bits:
                    changes.append((name, entry.bits, new_bits))
            was_bulk = self._digit_mode == "bulk"
            if was_bulk:
                self._highlight_matches.clear()
            if changes:
                label = _format_bits_label(new_bits)
                if total == 1:
                    entry_name = self._pending_targets[0]
                    old_label = _format_bits_label(changes[0][1])
                    description = f"Set {entry_name} to {label} (was {old_label})."
                else:
                    description = (
                        f"Updated {len(changes)}/{total} tensor(s) to {label}"
                    )
                    if self._pending_label:
                        description += f" ({self._pending_label})"
                self._apply_changes(changes, description=description)
            else:
                self.refresh_table()
                if total == 1:
                    entry_name = self._pending_targets[0]
                    self._msg(
                        f"{entry_name} already at {_format_bits_label(new_bits)}."
                    )
                else:
                    message = f"No changes needed for {total} tensor(s)."
                    if self._pending_label:
                        message += f" ({self._pending_label})"
                    self._msg(message)
            self._cancel_digit_mode()

        @work(exclusive=True)
        async def _regex_prompt_worker(self) -> None:
            await self._run_regex_prompt()

        async def _run_regex_prompt(self) -> None:
            if not self.entries:
                self._msg("No tensors available for selection.")
                return
            if self._digit_mode:
                self._cancel_digit_mode(clear_highlight=self._digit_mode == "bulk")
            result = await self.push_screen_wait(
                PromptScreen(
                    "Regular expression to match tensor names:",
                    placeholder="e.g. attn.*",
                )
            )
            if result is None:
                self._msg("Regex entry cancelled.")
                return
            try:
                pattern = re.compile(result)
            except re.error as exc:
                self._msg(f"Invalid regex: {exc}")
                self.bell()
                return
            matches = [
                entry.name for entry in self.entries if pattern.search(entry.name)
            ]
            if not matches:
                self._highlight_matches.clear()
                self.refresh_table()
                self._msg(f"No tensors matched /{result}/.")
                return
            self._highlight_matches = set(matches)
            self.refresh_table()
            self._start_bulk_digits(matches, f"regex /{result}/")

        def action_confirm(self) -> None:
            self._result = {entry.name: entry.bits for entry in self.entries}
            self.exit(result=self._result)

        def action_cancel(self) -> None:
            self.cancelled = True
            self.exit(result=None)

        def action_undo(self) -> None:
            if not self._undo_stack:
                self.bell()
                self._msg("Nothing to undo.")
                return
            if self._digit_mode:
                self._cancel_digit_mode(clear_highlight=self._digit_mode == "bulk")
            changes, description = self._undo_stack.pop()
            self._redo_stack.append((changes, description))
            reverse = [(name, new, old) for name, old, new in changes]
            self._apply_changes(reverse, record=False, notify=False)
            if description:
                self._msg(f"Undid {description}")
            else:
                self._msg("Undo performed.")

        def action_redo(self) -> None:
            if not self._redo_stack:
                self.bell()
                self._msg("Nothing to redo.")
                return
            if self._digit_mode:
                self._cancel_digit_mode(clear_highlight=self._digit_mode == "bulk")
            changes, description = self._redo_stack.pop()
            self._apply_changes(changes, record=False, notify=False)
            self._undo_stack.append((changes, description))
            if len(self._undo_stack) > self._history_limit:
                self._undo_stack.pop(0)
            if description:
                self._msg(f"Redid {description}")
            else:
                self._msg("Redo performed.")

        def action_noop(self) -> None:
            """Placeholder for footer bindings that are handled elsewhere."""

        def action_show_commands(self) -> None:
            self._msg(self.help_text, timeout=8.0)

        def action_show_help(self) -> None:
            self.action_show_commands()

        def _msg(self, text: str, timeout: float = 3.0) -> None:
            try:
                self.notify(text, timeout=timeout)
            except AttributeError:
                self.bell()

        def _default_yaml_path(self) -> str:
            if self._last_save_path:
                return self._last_save_path
            base = self._save_directory or os.getcwd()
            return os.path.join(base, self._default_save_filename)

        def _resolve_save_path(self, raw_path: str) -> str:
            expanded = os.path.expanduser(raw_path)
            if not os.path.isabs(expanded):
                base = self._save_directory or os.getcwd()
                expanded = os.path.join(base, expanded)
            return os.path.abspath(expanded)

        @work(exclusive=True)
        async def _save_yaml_worker(self) -> None:
            await self._run_save_yaml_prompt()

        async def _run_save_yaml_prompt(self) -> None:
            if not self.entries:
                self._msg("No tensors available to save.")
                return
            default_path = self._default_yaml_path()
            result = await self.push_screen_wait(
                PromptScreen(
                    "Path to save per-tensor bit-widths as YAML:",
                    placeholder=default_path,
                    title="Save quantization YAML",
                    initial=default_path,
                )
            )
            if result is None:
                self._msg("YAML save cancelled.")
                return
            path_text = result.strip() or default_path
            resolved = self._resolve_save_path(path_text)
            try:
                save_quantization_yaml(
                    resolved,
                    {entry.name: entry.bits for entry in self.entries},
                )
            except Exception as exc:
                self._msg(f"Failed to save YAML: {exc}")
                self.bell()
                return
            self._last_save_path = resolved
            self._msg(
                f"Saved bit-widths to {os.path.abspath(resolved)}",
                timeout=5.0,
            )

        def action_save_yaml(self) -> None:
            self._save_yaml_worker()

        async def on_key(self, event: events.Key) -> None:
            if not self.table or not self.entries:
                return
            key = event.key
            if self._digit_mode:
                if key == "escape":
                    was_bulk = self._digit_mode == "bulk"
                    self._cancel_digit_mode(clear_highlight=was_bulk)
                    self._msg("Bit entry cancelled.")
                    return
                if key == "enter":
                    self._apply_digit_buffer()
                    return
                if key == "backspace":
                    if self._digit_buffer:
                        self._digit_buffer.pop()
                        if self._digit_buffer:
                            preview = "".join(self._digit_buffer)
                            if preview.lower() == "f":
                                self._msg("Pending bit-width: fp32")
                            else:
                                self._msg(f"Pending bit-width: {preview}")
                        else:
                            self._msg("Cleared pending digits.")
                    else:
                        self.bell()
                    return
                if key.lower() == "f":
                    self._digit_buffer = ["f"]
                    self._msg("Pending bit-width: fp32")
                    return
                if key.isdigit():
                    if self._digit_buffer == ["f"]:
                        self._digit_buffer.clear()
                    self._digit_buffer.append(key)
                    preview = "".join(self._digit_buffer)
                    self._msg(f"Pending bit-width: {preview}")
                    return
                self.bell()
                return

            lower = key.lower()

            if lower == "ctrl+s":
                self.action_confirm()
                return
            if lower == "ctrl+c":
                self.action_cancel()
                return
            if lower == "ctrl+z":
                self.action_undo()
                return
            if lower == "ctrl+y":
                self.action_redo()
                return
            if lower == "ctrl+shift+s":
                self.action_save_yaml()
                return
            if key == "?":
                self.action_show_help()
                return
            if lower == "p":
                self.action_show_commands()
                return
            if lower == "escape":
                if self._highlight_matches:
                    self._clear_highlight()
                return

            entry = self._current_entry()
            if lower == "k":
                if entry:
                    self._nudge_entry(entry, 1)
                return
            if lower == "j":
                if entry:
                    self._nudge_entry(entry, -1)
                return
            if lower == "enter":
                if entry:
                    self._start_single_digit(entry)
                return
            if lower == "g":
                self._regex_prompt_worker()
                return

            if lower == "b":
                if entry:
                    self._start_single_digit(entry)
                else:
                    self._msg("Select a tensor to edit.")
                return
            if lower == "f":
                if entry:
                    self._toggle_float(entry)
                return
            if key == "R":
                self._reset_all()
                return
            if lower == "r":
                if entry:
                    self._reset_entry(entry)
                return
            if lower == "a":
                self._start_all_digits()
                return

else:  # pragma: no cover - textual is optional at runtime
    TensorBitwidthApp = None  # type: ignore[assignment]


def interactive_select_tensor_bits(
    entries: List[TensorConfigEntry],
    min_bits: int,
    max_bits: Optional[int],
    page_size: int,
    *,
    save_directory: Optional[str] = None,
    last_save_path: Optional[str] = None,
) -> Dict[str, int]:
    if not entries:
        _print_warning(
            "No floating-point tensors were found for interactive configuration."
        )
        return {}

    if not _TEXTUAL_AVAILABLE:
        _print_warning(
            "The textual package is not available; falling back to the legacy prompt-based UI. "
            "Install textual to use the richer interface."
        )
        return _legacy_interactive_select_tensor_bits(entries, min_bits, max_bits, page_size)

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        _print_warning(
            "The interactive UI requires a TTY. Falling back to the legacy prompt-based workflow."
        )
        return _legacy_interactive_select_tensor_bits(entries, min_bits, max_bits, page_size)

    app = TensorBitwidthApp(
        entries,
        min_bits,
        max_bits,
        page_size,
        save_directory=save_directory,
        last_save_path=last_save_path,
    )
    try:
        result = app.run()
    except KeyboardInterrupt as exc:  # pragma: no cover - user interruption
        raise SystemExit("Interactive configuration canceled by user.") from exc

    if app.cancelled or result is None:
        raise SystemExit("Interactive configuration canceled by user.")

    _print_info("Interactive configuration complete.")
    return result


def _fake_quant_symmetric(tensor: torch.Tensor, num_bits: int) -> torch.Tensor:
    # Signed two's-complement style quantization covering
    #   qmin = -2^{B-1} and qmax = 2^{B-1} - 1
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -1 << (num_bits - 1)
    if qmax <= 0:
        return tensor

    if tensor.numel() == 0:
        return tensor

    max_abs = tensor.abs().max()
    if max_abs.numel() == 0:
        return tensor
    max_abs_val = max_abs.item()
    if max_abs_val == 0.0 or not math.isfinite(max_abs_val):
        return tensor

    scale = max_abs_val / qmax
    if scale == 0.0 or not math.isfinite(scale):
        return tensor

    q = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    return (q * scale).to(tensor.dtype)


def _fake_quant_asymmetric(tensor: torch.Tensor, num_bits: int) -> torch.Tensor:
    # Unsigned quantization with range [0, 2^{B}-1]
    qmin = 0
    qmax = (1 << num_bits) - 1
    if qmax <= qmin:
        return tensor

    if tensor.numel() == 0:
        return tensor

    # min/max provide scalar tensors; handle degenerate ranges gracefully
    min_val = tensor.min()
    max_val = tensor.max()
    if min_val.numel() == 0 or max_val.numel() == 0:
        return tensor

    min_float = min_val.item()
    max_float = max_val.item()
    if not (math.isfinite(min_float) and math.isfinite(max_float)):
        return tensor
    if max_float <= min_float:
        return tensor

    scale = (max_float - min_float) / float(qmax - qmin)
    if scale == 0.0 or not math.isfinite(scale):
        return tensor

    zero_point = qmin - round(min_float / scale)
    zero_point = max(qmin, min(qmax, int(zero_point)))

    q = torch.round(tensor / scale + zero_point)
    q = torch.clamp(q, qmin, qmax)
    return ((q - zero_point) * scale).to(tensor.dtype)


def fake_quant_tensor(
    tensor: torch.Tensor, num_bits: int, scheme: str
) -> torch.Tensor:
    """Uniform quantize then dequantize a tensor."""

    if not torch.is_floating_point(tensor):
        return tensor

    if num_bits <= 0:
        return tensor

    if scheme == "symmetric":
        return _fake_quant_symmetric(tensor, num_bits)
    if scheme == "asymmetric":
        return _fake_quant_asymmetric(tensor, num_bits)

    raise ValueError(f"Unsupported quantization scheme: {scheme}")


def iter_state_tensors(state_dict) -> Iterable[torch.Tensor]:
    for _, tensor in iter_state_items(state_dict):
        yield tensor


def estimate_checkpoint_sizes(
    state_dict,
    default_num_bits: int,
    tensor_bitwidths: Optional[Dict[str, int]] = None,
) -> Tuple[float, float]:
    """Estimate raw and quantized storage requirements for tensors in a state dict."""

    tensor_bitwidths = tensor_bitwidths or {}

    original_bytes = 0.0
    quantized_bytes = 0.0

    for name, tensor in iter_state_items(state_dict):
        numel = tensor.numel()
        elem_bytes = tensor.element_size()
        original = numel * elem_bytes
        original_bytes += original

        if torch.is_floating_point(tensor):
            bits = tensor_bitwidths.get(name, default_num_bits)
            if bits is None or bits <= 0:
                quantized_bytes += original
            else:
                quantized_bytes += numel * int(bits) / 8.0
        else:
            quantized_bytes += original

    return original_bytes, quantized_bytes


def _size_breakdown(num_bytes: float) -> Tuple[str, str, str, str]:
    kb = num_bytes / 1024.0
    mb = kb / 1024.0
    gb = mb / 1024.0
    return (
        f"{num_bytes:,.0f} bytes",
        f"{kb:,.2f} KB",
        f"{mb:,.2f} MB",
        f"{gb:,.4f} GB",
    )


def format_size(num_bytes: float) -> str:
    base, kb, mb, gb = _size_breakdown(num_bytes)
    return f"{base} ({kb} / {mb} / {gb})"


def _format_size_rich(
    num_bytes: float,
    value_style: str = "white",
    detail_style: str = "dim",
) -> str:
    base, kb, mb, gb = _size_breakdown(num_bytes)
    return (
        f"[{value_style}]{base}[/{value_style}]\n"
        f"[{detail_style}]{kb} | {mb} | {gb}[/{detail_style}]"
    )


def print_quantization_summary(
    scheme: str,
    num_bits: int,
    original_bytes: float,
    quantized_bytes: float,
    tensor_bitwidths: Optional[Dict[str, int]] = None,
) -> None:
    bit_counts: Counter[int] = Counter()
    skipped_count = 0
    if tensor_bitwidths:
        for bits in tensor_bitwidths.values():
            if bits is None or bits <= 0:
                skipped_count += 1
            else:
                bit_counts[int(bits)] += 1

    if _RICH_CONSOLE:
        scheme_label = f"{scheme} ({num_bits}-bit)"
        table = Table(
            title="Quantization Summary",
            title_style="bold magenta",
            show_header=False,
            box=box.SIMPLE_HEAVY,
            expand=True,
        )
        table.add_column("Metric", style="bold cyan", no_wrap=True)
        table.add_column("Value", style="bright_white")
        table.add_row("Scheme", f"[bold green]{scheme_label}[/bold green]")
        table.add_row(
            "Original Size",
            _format_size_rich(original_bytes, value_style="bright_white"),
        )
        table.add_row(
            "Quantized Size",
            _format_size_rich(quantized_bytes, value_style="cyan"),
        )

        if original_bytes > 0:
            if quantized_bytes > 0:
                ratio = original_bytes / quantized_bytes
                pct_of_original = (quantized_bytes / original_bytes) * 100.0
                reduction_pct = 100.0 - pct_of_original
                bytes_saved = max(original_bytes - quantized_bytes, 0.0)
                table.add_row("Compression", f"[bold green]{ratio:.2f}x[/bold green]")
                table.add_row(
                    "Size Reduction",
                    f"[green]{reduction_pct:.2f}%[/green]",
                )
                table.add_row(
                    "Remaining Size",
                    f"[yellow]{pct_of_original:.2f}%[/yellow] of original",
                )
                table.add_row(
                    "Bytes Saved",
                    _format_size_rich(bytes_saved, value_style="bright_green"),
                )
            else:
                table.add_row(
                    "Compression",
                    "[bold green]∞[/bold green] ([green]100.00% size reduction[/green])",
                )
                table.add_row(
                    "Remaining Size",
                    "[yellow]0.00%[/yellow] of original",
                )
                bytes_saved = max(original_bytes - quantized_bytes, 0.0)
                table.add_row(
                    "Bytes Saved",
                    _format_size_rich(bytes_saved, value_style="bright_green"),
                )
        else:
            table.add_row("Compression", "[yellow]n/a[/yellow]")

        renderables = [table]
        if bit_counts or skipped_count:
            bit_table = Table(
                title="Bit-width Usage",
                title_style="bold magenta",
                header_style="bold cyan",
                box=box.SIMPLE_HEAVY,
                expand=True,
            )
            bit_table.add_column("Bit-width", justify="center", style="bright_white")
            bit_table.add_column("Tensors", justify="right", style="cyan")
            for bits, count in sorted(bit_counts.items()):
                bit_table.add_row(f"{bits}-bit", str(count))
            if skipped_count:
                bit_table.add_row("fp32", str(skipped_count))
            renderables.append(bit_table)

        panel_content = renderables[0] if len(renderables) == 1 else Group(*renderables)

        panel = Panel.fit(
            panel_content,
            title="[bold bright_white on blue] Fake PTQ [/bold bright_white on blue]",
            border_style="bright_blue",
            padding=(1, 2),
        )
        _RICH_CONSOLE.print(panel)
        return

    # Plain-text fallback when rich is unavailable.
    print("Quantization summary:")
    print(f"  Scheme: {scheme}, bits: {num_bits}")
    print("  Estimated checkpoint size before quantization:")
    print(f"    {format_size(original_bytes)}")
    print("  Estimated checkpoint size after quantization:")
    print(f"    {format_size(quantized_bytes)}")
    if original_bytes > 0:
        if quantized_bytes > 0:
            ratio = original_bytes / quantized_bytes
            pct_of_original = (quantized_bytes / original_bytes) * 100.0
            reduction_pct = 100.0 - pct_of_original
            bytes_saved = max(original_bytes - quantized_bytes, 0.0)
            print(
                "  Estimated compression factor:",
                f" {ratio:.2f}x ({reduction_pct:.2f}% size reduction,",
                f"{pct_of_original:.2f}% of original size)",
            )
            print(f"  Bytes saved: {format_size(bytes_saved)}")
        else:
            print("  Estimated compression factor: ∞ (100.00% size reduction)")
            print("  Remaining size: 0.00% of original")
            bytes_saved = max(original_bytes - quantized_bytes, 0.0)
            print(f"  Bytes saved: {format_size(bytes_saved)}")
    else:
        print("  Estimated compression factor: n/a")

    if bit_counts or skipped_count:
        print("  Per-tensor bit-widths:")
        for bits, count in sorted(bit_counts.items()):
            print(f"    {bits}-bit: {count} tensor(s)")
        if skipped_count:
            print(f"    fp32 (skipped): {skipped_count} tensor(s)")

def main():
    args = parse_args()
    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_obj = checkpoint["model"]
    else:
        state_obj = checkpoint

    if isinstance(state_obj, MutableMapping):
        state_dict = state_obj
    else:
        to_state_dict = getattr(state_obj, "state_dict", None)
        if callable(to_state_dict):
            state_dict = to_state_dict()
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                checkpoint["model"] = state_dict
            else:
                checkpoint = state_dict
        else:
            raise TypeError(
                "Unsupported checkpoint format: expected a mapping for the model state"
            )

    last_quant_path = os.path.join(args.ckpt_dir, LAST_QUANTIZATION_BASENAME)

    try:
        overrides = parse_per_tensor_bits(args.per_tensor_bits)
    except ValueError as exc:
        raise SystemExit(f"Failed to parse --per-tensor-bits: {exc}") from None
    except OSError as exc:
        raise SystemExit(f"Unable to read --per-tensor-bits: {exc}") from None

    valid_overrides = _filter_tensor_bit_mapping(
        overrides, state_dict, label="override"
    )

    if valid_overrides:
        _print_info(
            f"Loaded per-tensor overrides for {len(valid_overrides)} tensor(s)."
        )

    default_yaml_mapping: Dict[str, int] = {}
    if args.tui_default_quantization:
        default_spec = os.path.expanduser(args.tui_default_quantization)
        try:
            raw_defaults = _load_yaml_mapping(default_spec)
        except (OSError, ValueError) as exc:
            raise SystemExit(
                f"Failed to read --tui-default-quantization: {exc}"
            ) from None
        default_yaml_mapping = _filter_tensor_bit_mapping(
            raw_defaults, state_dict, label="default mapping"
        )
        if default_yaml_mapping:
            _print_info(
                f"Loaded default bit-widths for {len(default_yaml_mapping)} tensor(s) "
                f"from {os.path.abspath(default_spec)}."
            )

    prior_mapping: Dict[str, int] = {}
    if os.path.exists(last_quant_path):
        try:
            prior_raw = _load_yaml_mapping(last_quant_path)
        except (OSError, ValueError) as exc:
            _print_warning(
                f"Unable to load previous quantization from "
                f"{os.path.abspath(last_quant_path)}: {exc}"
            )
        else:
            prior_mapping = _filter_tensor_bit_mapping(
                prior_raw, state_dict, label="previous quantization"
            )
            if prior_mapping:
                _print_info(
                    f"Loaded prior quantization for {len(prior_mapping)} tensor(s) "
                    f"from {os.path.abspath(last_quant_path)}."
                )

    default_baseline = default_yaml_mapping if default_yaml_mapping else valid_overrides

    entries = build_tensor_config_entries(
        state_dict,
        args.num_bits,
        valid_overrides,
        default_overrides=default_baseline,
        prior_bits=prior_mapping,
    )

    if os.path.exists(last_quant_path):
        initial_save_hint = os.path.abspath(last_quant_path)
    elif args.tui_default_quantization:
        initial_save_hint = os.path.abspath(
            os.path.expanduser(args.tui_default_quantization)
        )
    else:
        initial_save_hint = None

    if entries:
        _print_info(
            f"Detected {len(entries)} floating-point tensor(s) available for quantization."
        )

    positive_bits = [entry.bits for entry in entries if entry.bits > 0]
    if args.num_bits > 0:
        positive_bits.append(args.num_bits)

    min_positive = min(positive_bits) if positive_bits else None
    max_positive = max(positive_bits) if positive_bits else None

    effective_min_bits = args.min_bits if args.min_bits > 0 else (min_positive or 1)
    if min_positive is not None and min_positive < effective_min_bits:
        if args.min_bits > 0:
            _print_info(
                f"Lowering minimum interactive bit-width to {min_positive} to accommodate overrides."
            )
        effective_min_bits = min_positive

    effective_max_bits = args.max_bits
    if max_positive is not None:
        if effective_max_bits is None:
            effective_max_bits = max_positive
        elif max_positive > effective_max_bits:
            _print_info(
                f"Raising maximum interactive bit-width to {max_positive} to accommodate overrides."
            )
            effective_max_bits = max_positive

    if effective_max_bits is not None and effective_max_bits < effective_min_bits:
        effective_max_bits = effective_min_bits

    if args.interactive:
        tensor_bitwidths = interactive_select_tensor_bits(
            entries,
            effective_min_bits,
            effective_max_bits,
            args.tui_page_size,
            save_directory=args.ckpt_dir,
            last_save_path=initial_save_hint,
        )
        if not tensor_bitwidths and entries:
            tensor_bitwidths = {entry.name: entry.bits for entry in entries}
    else:
        tensor_bitwidths = {entry.name: entry.bits for entry in entries}

    original_bytes, quantized_bytes = estimate_checkpoint_sizes(
        state_dict, args.num_bits, tensor_bitwidths
    )

    applied_tensor_bits: Dict[str, int] = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if not torch.is_floating_point(value):
            continue
        bits = tensor_bitwidths.get(key, args.num_bits)
        applied_tensor_bits[key] = bits
        if bits is None or bits <= 0:
            continue
        state_dict[key] = fake_quant_tensor(value, int(bits), args.quantization)

    if applied_tensor_bits:
        quantized_count = sum(
            1 for bits in applied_tensor_bits.values() if bits and bits > 0
        )
        skipped_count = sum(
            1 for bits in applied_tensor_bits.values() if bits is None or bits <= 0
        )
        _print_info(
            f"Configured per-tensor bit-widths for {len(applied_tensor_bits)} tensor(s): "
            f"{quantized_count} quantized, {skipped_count} kept as fp32."
        )

    out_dir = args.out_dir or f"{args.ckpt_dir}_ptq"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    meta_in = os.path.join(args.ckpt_dir, "meta.pkl")
    meta_out = os.path.join(out_dir, "meta.pkl")
    if os.path.exists(meta_in):
        shutil.copy(meta_in, meta_out)

    try:
        save_quantization_yaml(last_quant_path, applied_tensor_bits)
    except Exception as exc:
        _print_warning(
            f"Unable to record last quantization in {os.path.abspath(last_quant_path)}: {exc}"
        )
    else:
        _print_info(
            f"Recorded last quantization in {os.path.abspath(last_quant_path)}."
        )

    print_quantization_summary(
        args.quantization,
        args.num_bits,
        original_bytes,
        quantized_bytes,
        applied_tensor_bits if applied_tensor_bits else None,
    )

    if _RICH_CONSOLE:
        _RICH_CONSOLE.print(
            f"[cyan]Saved quantized checkpoint to[/cyan] "
            f"[bold]{os.path.abspath(out_dir)}[/bold]"
        )
    else:
        print(f"Saved quantized checkpoint to {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    main()
