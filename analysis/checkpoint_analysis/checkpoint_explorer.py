import argparse
import sys
from pathlib import Path

import torch

# Ensure the repository root is on the import path regardless of CWD
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import GPT, GPTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Model Parameter Explorer")
    parser.add_argument("ckpt_path", help="Path to the checkpoint file")
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on')
    return parser.parse_args()

def load_model(ckpt_path, device):
    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint.get('model_args', None)
    if model_args is None:
        sys.exit("Model arguments not found in checkpoint.")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def get_parameter_tree(state_dict):
    tree = {}
    for full_key in state_dict.keys():
        parts = full_key.split('.')
        current_level = tree
        for part in parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
        # Last part is the parameter tensor
        current_level[parts[-1]] = state_dict[full_key]
    return tree

IMAGES_DIR = SCRIPT_DIR / "images"


def display_heatmap(tensor, full_key):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    tensor = tensor.detach().cpu().numpy()
    if tensor.ndim != 2:
        print("Heatmap can only be displayed for 2D tensors.")
        input("Press Enter to continue...")
        return

    # Create the images directory if it doesn't exist
    IMAGES_DIR.mkdir(exist_ok=True)

    # Clean the full_key to create a valid filename
    filename = full_key.replace('.', '_').replace('/', '_')
    image_path = IMAGES_DIR / f"{filename}_heatmap.png"

    # Generate the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(tensor, cmap='viridis')
    plt.title(f"Heatmap of {full_key}")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

    print(f"\nHeatmap saved to {image_path}")

    # Attempt to display the image in the terminal
    try:
        from PIL import Image
        import shutil

        columns = shutil.get_terminal_size().columns
        img = Image.open(image_path)
        # Resize image to fit terminal width
        aspect_ratio = img.height / img.width
        new_width = columns // 2  # Adjust as necessary
        new_height = int(aspect_ratio * new_width)
        img = img.resize((new_width, new_height))

        # Convert image to ASCII
        img = img.convert('L')  # Convert to grayscale
        pixels = np.array(img)
        chars = np.asarray(list(' .:-=+*#%@'))
        normalized = (pixels - pixels.min()) / (pixels.max() - pixels.min())
        indices = (normalized * (len(chars) - 1)).astype(int)
        ascii_image = "\n".join("".join(chars[pixel] for pixel in row) for row in indices)
        print(ascii_image)
    except Exception as e:
        print("Unable to display image in terminal.")
        print(f"Error: {e}")
    input("Press Enter to continue...")

def display_histogram(tensor, full_key):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from rich.console import Console
    from rich.text import Text
    from rich.style import Style

    tensor = tensor.detach().cpu().numpy().flatten()

    # Prompt for number of bins
    while True:
        bins_input = input("Enter the number of bins for the histogram (default 20): ")
        if bins_input == '':
            num_bins = 20
            break
        elif bins_input.isdigit() and int(bins_input) > 0:
            num_bins = int(bins_input)
            break
        else:
            print("Invalid input. Please enter a positive integer.")

    # Create the images directory if it doesn't exist
    IMAGES_DIR.mkdir(exist_ok=True)

    # Clean the full_key to create a valid filename
    filename = full_key.replace('.', '_').replace('/', '_')
    image_path = IMAGES_DIR / f"{filename}_histogram.png"

    # Generate the histogram using seaborn
    plt.figure(figsize=(10, 8))
    sns.histplot(tensor, bins=num_bins, kde=False)
    plt.title(f"Histogram of {full_key}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

    print(f"\nHistogram saved to {image_path}")

    # Display histogram in terminal using rich
    hist, bin_edges = np.histogram(tensor, bins=num_bins)
    max_height = 10
    max_count = hist.max()
    console = Console()

    # Prepare styles
    bar_style = Style(color="cyan")
    bin_label_style = Style(color="magenta")

    print("\nHistogram:")
    for i in range(len(hist)):
        bar_length = int((hist[i] / max_count) * max_height)
        bar = "█" * bar_length
        # Format bin edges with alignment
        bin_label = f"{bin_edges[i]: .4f} -{bin_edges[i+1]: .4f}"
        # Align bin labels
        bin_label = bin_label.ljust(25)
        # Display the bar with color
        text = Text(bin_label, style=bin_label_style)
        text.append(bar, style=bar_style)
        console.print(text)
    input("Press Enter to continue...")

def display_stats(tensor, full_key):
    import numpy as np
    from rich.console import Console
    from rich.table import Table
    from rich import box

    tensor = tensor.detach().cpu().numpy()
    flat_tensor = tensor.flatten()
    min_val = np.min(flat_tensor)
    max_val = np.max(flat_tensor)
    q1 = np.percentile(flat_tensor, 25)
    median = np.median(flat_tensor)
    q3 = np.percentile(flat_tensor, 75)
    mean = np.mean(flat_tensor)
    zeros = np.sum(flat_tensor == 0)
    total = flat_tensor.size
    percent_zeros = zeros / total * 100

    # Updated table with borders and right-aligned value column
    table = Table(title=f"Summary Statistics for {full_key}", box=box.SQUARE)
    table.add_column("Statistic", style="bold magenta")
    table.add_column("Value", style="bold cyan", justify="right")

    table.add_row("Shape", str(tensor.shape))
    table.add_row("Min", f"{min_val:.6f}")
    table.add_row("Max", f"{max_val:.6f}")
    table.add_row("Q1 (25%)", f"{q1:.6f}")
    table.add_row("Median", f"{median:.6f}")
    table.add_row("Q3 (75%)", f"{q3:.6f}")
    table.add_row("Mean", f"{mean:.6f}")
    table.add_row("Zeros (%)", f"{percent_zeros:.2f}%")

    console = Console()
    console.print(table)
    input("Press Enter to continue...")

def explore_tree(tree, path=[]):
    while True:
        current_level = tree
        for part in path:
            current_level = current_level[part]

        if isinstance(current_level, dict):
            keys = list(current_level.keys())
            print("\nCurrent Path: " + ('.'.join(path) if path else "root"))
            print("Submodules/Parameters:")
            for idx, key in enumerate(keys):
                print(f"{idx}: {key}")
            print("b: Go back, q: Quit")
            choice = input("Enter the number of the submodule/parameter to explore (or 'b' to go back, 'q' to quit): ")
            if choice == 'b':
                if path:
                    path.pop()
                else:
                    print("Already at root.")
            elif choice == 'q':
                break
            elif choice.isdigit() and int(choice) < len(keys):
                path.append(keys[int(choice)])
            else:
                print("Invalid choice.")
        else:
            # It's a parameter tensor
            full_key = '.'.join(path)
            while True:
                print(f"\nReached parameter: {full_key}")
                print("Options:")
                print("1: View value")
                print("2: Display 2D heatmap")
                print("3: Display histogram")
                print("4: Display summary statistics")
                print("b: Go back")
                choice = input("Enter your choice: ")
                if choice == '1':
                    # View value
                    tensor = current_level
                    tensor_str = str(tensor.detach().cpu().numpy())
                    if len(tensor_str) > 1000:
                        tensor_str = tensor_str[:1000] + '...'
                    print(f"\nValue of {full_key}:")
                    print(tensor_str)
                    input("Press Enter to continue...")
                elif choice == '2':
                    # Display heatmap
                    display_heatmap(current_level, full_key)
                elif choice == '3':
                    # Display histogram
                    display_histogram(current_level, full_key)
                elif choice == '4':
                    # Display stats
                    display_stats(current_level, full_key)
                elif choice == 'b':
                    path.pop()
                    break
                else:
                    print("Invalid choice.")
            if not path:
                # At root level, break the loop
                break

def main():
    args = parse_args()

    model = load_model(args.ckpt_path, args.device)
    state_dict = model.state_dict()

    parameter_tree = get_parameter_tree(state_dict)

    print("Model Parameter Explorer")
    print("Navigate through the parameters using numbers. Press 'b' to go back, 'q' to quit.")

    explore_tree(parameter_tree)

if __name__ == '__main__':
    main()

