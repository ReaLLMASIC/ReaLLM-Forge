# export_nanogpt_xnnpack.py

from export_weights.loader import load_model
from export_weights.exporters import export_to_executorch
import torch

try:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
except ImportError:
    XnnpackPartitioner = None
    get_xnnpack_edge_compile_config = None

if __name__ == "__main__":
    # Load the nanoGPT model.
    model = load_model("gpt2")

    # Create example inputs. This is used in the export process to provide
    # hints on the expected shape of the model input.
    example_inputs = (
            torch.randint(0, 100, (1, model.config.block_size - 1), dtype=torch.long),
        )
    if XnnpackPartitioner and get_xnnpack_edge_compile_config:
        export_to_executorch(model, example_inputs, "nanogpt_xnnpack", backend="xnnpack")
    else:
        print("XNNPACK backend not available.")
