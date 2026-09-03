# export_nanogpt.py

from export_weights.loader import load_model
from export_weights.exporters import export_to_executorch
import torch

if __name__ == "__main__":
    # Load the model.
    model = load_model("gpt2")

    # Create example inputs. This is used in the export process to provide
    # hints on the expected shape of the model input.
    example_inputs = (torch.randint(0, 100, (1, model.config.block_size), dtype=torch.long), )
    export_to_executorch(model, example_inputs, "nanogpt")
