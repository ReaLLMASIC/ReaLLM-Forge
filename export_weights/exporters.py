import torch
import ml_dtypes
from collections import OrderedDict
from executorch.exir import EdgeCompileConfig, to_edge
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.export import export, export_for_training
from export_weights.savers import save_to_pkl, save_to_pte

def export_to_pkl(state_dict, file_name):
    to_save = OrderedDict()
    for k, v in state_dict.items():
        if any(key in k for key in ["mlp_act", "attn_act"]) or k.endswith(("quantized_bias","bias_norm","zero_point","quantized_weight","weight_norm")):
            if v.dtype == torch.bfloat16:
                to_save[k] = v.cpu().float().numpy().astype(ml_dtypes.bfloat16)
            else:
                to_save[k] = v.cpu().numpy()
    save_to_pkl(to_save, file_name)

def export_to_executorch(model, example_inputs, file_name, backend=None):
    # Set up dynamic shape configuration. This allows the sizes of the input tensors
    # to differ from the sizes of the tensors in `example_inputs` during runtime, as
    # long as they adhere to the rules specified in the dynamic shape configuration.
    # Here we set the range of 0th model input's 1st dimension as
    # [0, model.config.block_size].
    # See https://pytorch.org/executorch/main/concepts.html#dynamic-shapes
    # for details about creating dynamic shapes.
    if backend == "xnnpack":
        dynamic_shape = ({1: torch.export.Dim("token_dim", max=model.config.block_size - 1)},)
    else:
        dynamic_shape = ({1: torch.export.Dim("token_dim", max=model.config.block_size)},)

    # Trace the model, converting it to a portable intermediate representation.
    # The torch.no_grad() call tells PyTorch to exclude training-specific logic.
    with sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
        m = export_for_training(model, example_inputs, dynamic_shapes=dynamic_shape).module()
        traced_model = export(m, example_inputs, dynamic_shapes=dynamic_shape)
    if backend == "xnnpack":
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
            # Convert the model into a runnable ExecuTorch program.
            # To be further lowered to Xnnpack backend, `traced_model` needs xnnpack-specific edge compile config
            edge_config = get_xnnpack_edge_compile_config()
            edge_manager = to_edge(traced_model, compile_config=edge_config)
            edge_manager = edge_manager.to_backend(XnnpackPartitioner())
        except ImportError:
            raise RuntimeError("XNNPACK backend not available.")
    else:
        # Convert the model into a runnable ExecuTorch program.
        edge_config = EdgeCompileConfig(_check_ir_validity=False)
        edge_manager = to_edge(traced_model, compile_config=edge_config)
    et_program = edge_manager.to_executorch()

    # Save the ExecuTorch program to a file.
    save_to_pte(et_program, file_name)
