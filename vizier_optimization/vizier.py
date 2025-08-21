import argparse
from datetime import datetime
import json
import os
import subprocess

from rich import print
from rich.console import Console
from rich.table import Table

from absl import app
from absl import flags
from absl import logging

import torch
from vizier.service import clients, pyvizier as vz

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run vizier optimization based on json configuration file."
    )

    parser.add_argument(
        "--c",
        type=str,
        required=True,
        help="Path to the configuration JSON file."
    )

    parser.add_argument(
        "--add_names",
        action="store_true",
        help="Include names of values of the configuration parameters in addition to values (may cause too long a file name).",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="Directory to place the set of output checkpoints.",
    )

    parser.add_argument(
        "--vizier_iterations", type=int, default=20, help="Number of Vizier iterations."
    )

    parser.add_argument(
        "--vizier_algorithm",
        choices=[
            "GP_UCB_PE",
            "GAUSSIAN_PROCESS_BANDIT",
            "RANDOM_SEARCH",
            "QUASI_RANDOM_SEARCH",
            "GRID_SEARCH",
            "SHUFFLED_GRID_SEARCH",
            "EAGLE_STRATEGY",
            "CMA_ES",
            "EMUKIT_GP_EI",
            "NSGA2",
            "BOCS",
            "HARMONICA",
        ],
        default="GAUSSIAN_PROCESS_BANDIT",
        help="Choose the Vizier algorithm to use.",
    )

    parser.add_argument(
    "--parameters",
    type=str,
    nargs="+",
    help="List of parameters to optimize. Example: --parameters n_layer n_head block_size",
)
    parser.add_argument(
        "--ranges",
        type=str,
        nargs="+",
        help="List of ranges for each parameter in the same order. Example: --ranges '[6,24]' '[4,16]' '[512,2048]'",
    )

    parser.add_argument(
        "--scaling_types",
        type=str,
        nargs="+",
        help="List of scaling_type for each parameter. Example: --scaling_types linear linar log",
    )

    parser.add_argument(
        "--data_types",
        type=str,
        nargs="+",
        help="List of data types for each parameter. Example: --data_types int int folat",
    )
    return parser.parse_args()


def get_best_val_loss(out_dir):
    best_val_loss_file = out_dir + "/best_val_loss_and_iter.txt"
    if os.path.exists(best_val_loss_file):
        with open(best_val_loss_file, "r") as file:
            try:
                best_val_loss = float(file.readline().strip().split(",")[0])
                return best_val_loss
            except ValueError:
                print("val_loss file not found, trying checkpoint...")

    # if contained file doesn't exist, try ckpt.pt file
    checkpoint_file = out_dir + "/ckpt.pt"
    checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
    best_val_loss = checkpoint["best_val_loss"]
    return best_val_loss

def get_num_parameter(out_dir):
    num_parameter_file = "num_parameter.txt"
    if os.path.exists(num_parameter_file):
        print("opened")
        with open(num_parameter_file, "r") as file:
            try:
                num_parameter = float(file.readline().strip().split(",")[0])
                print(num_parameter)
                return num_parameter
            except ValueError:
                print("num_parameter file not found, trying checkpoint...")



def get_num_bits(config):
    """
    Calculate the total number of bits used by quantized attention activations.

    Parameters:
    - config: dict containing all quantization-related configurations.

    Returns:
    - total_bits: int, total number of bits used by quantized activations.
    """
    H = float(config.get("n_head", 6))
    D = float(config.get("n_embd", 6))
    B = float(config.get("batch_size", 6))
    L = float(config.get("block_size", 6))

    d_k = D // H

    default_bits = float(config.get("quantize_attn_act_bits", 16))

    activation_names = [
        "attn_act_input",
        "attn_act_qk_mult_q_input",
        "attn_act_qk_mult_k_input",
        "attn_act_softmax_input",
        "attn_act_pv_mult_p_input",
        "attn_act_pv_mult_v_input",
        "attn_act_pv_mult_output",
        "attn_act_output"
    ]

    tensor_sizes = {
        "attn_act_input": B * L * D,
        "attn_act_qk_mult_q_input": B * H * L * d_k,
        "attn_act_qk_mult_k_input": B * H * L * d_k,
        "attn_act_softmax_input": B * H * L * L,
        "attn_act_pv_mult_p_input": B * H * L * L,
        "attn_act_pv_mult_v_input": B * H * L * d_k,
        "attn_act_pv_mult_output": B * H * L * d_k,
        "attn_act_output": B * L * D
    }

    total_bits = 0

    for name in activation_names:
        quantize_flag_key = f"quantize_{name}"
        quantize_bits_key = f"{quantize_flag_key}_bits"

        quantize_flag = config.get(quantize_flag_key, False)

        quantize_bits = float(config.get(quantize_bits_key, default_bits))

        size = tensor_sizes.get(name, 0)

        act_bits = quantize_bits * size
        total_bits += act_bits

    print(f"Total bits for quantized attention activations: {total_bits}")
    return total_bits

def get_layer_bits(config):
    """
    Calculate the total number of bits used by quantized linear layer weights, embeddings, and LayerNorm parameters.

    Parameters:
    - config: dict containing all quantization-related configurations.

    Returns:
    - total_bits: float, total number of bits used by quantized weights and embeddings.
    """
    H = float(config.get("n_head", 6))
    D = float(config.get("n_embd", 6))
    D_ff = float(config.get("n_mlp", 4 * D))
    V = float(config.get("vocab_size", 50257))
    n_positions = float(config.get("n_positions", 1024))
    n_layer = int(config.get("n_layer", 12))

    default_bits = float(config.get("quantize_linear_weight_bits", 16))
    default_full_precision_bits = float(config.get("default_full_precision_bits", 16))

    total_bits = 0

    # Include word embedding table (wte)
    quantize_wte_bits = float(config.get("quantize_wte_bits", default_bits))
    wte_shape = (V, D)  # Shape of the word embedding table
    num_wte_params = wte_shape[0] * wte_shape[1]
    wte_bits = num_wte_params * quantize_wte_bits
    total_bits += wte_bits

    # Include positional embedding table (wpe)
    quantize_wpe_bits = float(config.get("quantize_wpe_bits", default_bits))
    wpe_shape = (n_positions, D)  # Shape of the positional embedding table
    num_wpe_params = wpe_shape[0] * wpe_shape[1]
    wpe_bits = num_wpe_params * quantize_wpe_bits
    total_bits += wpe_bits

    # Include LayerNorm parameters
    num_layernorms = (2 * n_layer) + 1
    num_layernorm_params = num_layernorms * D
    layernorm_bits = num_layernorm_params * default_full_precision_bits
    total_bits += layernorm_bits

    layer_names = [
        "linear_attn_q",
        "linear_attn_k",
        "linear_attn_v",
        "linear_attn_proj",
        "linear_mlp_up",
        "linear_mlp_down"
    ]

    weight_shapes = {
        "linear_attn_q": (D, D),
        "linear_attn_k": (D, D),
        "linear_attn_v": (D, D),
        "linear_attn_proj": (D, D),
        "linear_mlp_up": (D, D_ff),
        "linear_mlp_down": (D_ff, D)
    }

    for name in layer_names:
        quantize_method_key = f"quantize_{name}_method"
        quantize_bits_key = f"quantize_{name}_bits"

        quantize_method = config.get(quantize_method_key, None)

        if quantize_method is None:
            quantize_bits = default_full_precision_bits
        else:
            quantize_bits = float(config.get(quantize_bits_key, default_bits))

        shape = weight_shapes.get(name, (0, 0))
        num_params_per_layer = shape[0] * shape[1]

        num_layers = n_layer  # Number of transformer layers
        total_params = num_params_per_layer * num_layers

        layer_bits = total_params * quantize_bits

        total_bits += layer_bits

    print(f"Total bits for quantized linear layer weights, embeddings, and LayerNorm parameters: {total_bits}")
    return total_bits

def format_config_name(config, config_basename, add_names):
    if add_names:
        config_items = [f"{k}_{v}" for k, v in config.items()]
    else:
        config_items = [f"{v}" for _, v in config.items()]

    return f"{config_basename}-{'-'.join(config_items)}"


def run_command(config, config_basename, output_dir, add_names):
    formatted_name = format_config_name(config, config_basename, add_names)
    base_command = ["python3", "train.py"]
    config["tensorboard_run_name"] = formatted_name
    timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["out_dir"] = os.path.join(output_dir, f"{timestamp_prefix}_{formatted_name}")
    base_command.extend(["--timestamp", timestamp_prefix])

    # Print the entered arguments before each run
    console = Console()
    table = Table(
        title="Entered Arguments", show_header=True, header_style="bold magenta"
    )
    table.add_column("Argument", style="cyan")
    table.add_column("Value", style="green")

    for key, value in config.items():
        table.add_row(key, str(value))

    console.print(table)

    # Create train.py command with argparse flags
    for key, value in config.items():
        if isinstance(value, bool):
            print(key, value, "bool")
            base_command.extend([f"--{'' if value else 'no-'}{key}"])
        elif value == "True":
            base_command.extend([f"--{key}"])
        elif value == "False":
            base_command.extend([f"--no-{key}"])
        elif isinstance(value, list):
            print(key, value, "list")
            for val in value:
                base_command.extend([f"--{key}", str(val)])
        else:
            print(key, value, "else")
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            base_command.extend([f"--{key}", str(value)])

    print(f"Running command: {' '.join(base_command)}")
    subprocess.run(base_command)
    return config


def run_experiment_with_vizier(
    config, config_basename, output_dir, add_names, vizier_algorithm, vizier_iterations, parameters, ranges, scaling_types, data_types
):
    #search_space = vz.SearchSpace()
    study_config = vz.StudyConfig()  # Search space, metrics, and algorithm.
    root = study_config.search_space.root
    for k, v in config.items():
        if isinstance(v, list):
            param_type = type(v[0]).__name__.upper()
            if param_type == "INT":
                root.add_int_param(
                    name=k, min_value=min(map(int, v)), max_value=max(map(int, v))
                )
            elif param_type == "FLOAT":
                root.add_float_param(
                    name=k, min_value=min(map(float, v)), max_value=max(map(float, v))
                )
            elif param_type == "STR":
                if k == "n_head_nonexist":
                    print(k, v)
                    print("str")
                    min = float(v[0]) - 4.0
                    max = float(v[0]) + 4.0
                    root.add_int_param(
                        name=k, min_value=int(min), max_value=int(max),scale_type=vz.ScaleType.LOG
                    )
                if k == "block_size_nonexist":
                    print(k, v)
                    print("str")
                    min = float(v[0]) - 4.0
                    max = float(v[0]) + 4.0
                    root.add_int_param(
                        name=k, min_value=int(min), max_value=int(max),scale_type=vz.ScaleType.LOG
                    )
                else:
                    root.add_categorical_param(name=k, feasible_values=v)
            elif param_type == "BOOL":
                root.add_categorical_param(
                    name=k, feasible_values=[str(val) for val in v]
                )
        elif isinstance(v, dict) and "range" in v:
            range_def = v["range"]
            start, end, step = range_def["start"], range_def["end"], range_def["step"]
            param_type = type(start).__name__.upper()
            if param_type == "INT":
                root.add_int_param(
                    name=k,
                    min_value=start,
                    max_value=end,
                    scale_type=vz.ScaleType.LINEAR,
                )
            elif param_type == "FLOAT":
                root.add_float_param(
                    name=k,
                    min_value=start,
                    max_value=end,
                    scale_type=vz.ScaleType.LINEAR,
                )
        else:
            param_type = type(v).__name__.upper()
            if param_type == "INT":
                root.add_int_param(name=k, min_value=v, max_value=v)
            elif param_type == "FLOAT":
                root.add_float_param(name=k, min_value=v, max_value=v)
            elif param_type == "STR":
                root.add_categorical_param(name=k, feasible_values=[v])
            elif param_type == "BOOL":
                root.add_categorical_param(
                    name=k, feasible_values=[bool(v)]
                )

    study_config.algorithm = vizier_algorithm
    study_config.metric_information.append(
      vz.MetricInformation(
          name='num_parameters',
          goal=vz.ObjectiveMetricGoal.MINIMIZE,
          min_value=0.0,
          max_value=1.0,
      )
  )
    study_config.metric_information.append(
        vz.MetricInformation(
            name='loss', goal=vz.ObjectiveMetricGoal.MINIMIZE
        )
    )
    study_client = clients.Study.from_study_config(
        study_config, owner="owner1", study_id="example_study_id"
    )

    for i in range(vizier_iterations):
        print("Vizier Iteration", i)
        suggestions = study_client.suggest(count=1)
        for suggestion in suggestions:
            params = suggestion.parameters
            print("Suggested parameters:")
            print(params)
            config = run_command(params, config_basename, output_dir, add_names)
            loss = get_best_val_loss(config["out_dir"])
            num_parameters = get_num_parameter(config["out_dir"])
            num_bits = get_num_bits(params)
            #test = Hardware_Efficiency_Run(params)
            measurement = evaluate_trial(loss, num_parameters)
            suggestion.complete(measurement)
            #suggestion.complete(vz.Measurement(metrics={"loss": loss}))

    optimal_trials = study_client.optimal_trials()
    for trial in optimal_trials:
        best_trial = trial.materialize()
        print(
            f"Best trial: {best_trial.parameters}, Loss: {best_trial.final_measurement.metrics['loss']}, num_parameters: {best_trial.final_measurement.metrics['num_parameters']}"
        )

def evaluate_trial(loss, PPA):
  m = vz.Measurement()
  m.metrics = {'loss': loss}
  print(m.metrics)
  print(PPA)
  m.metrics['num_parameters'] = PPA
  return m

def Hardware_Efficiency_Run(config):
    #Simplified hardware efficiency computation only for testing
    shared_attn_size = 1
    shared_attn_sym =  1
    shared_mlp_size = 1
    shared_mlp_sym = 1
    num_neads = 1
    num_layers = 1
    for k, v in config.items():
        if k == "n_head":
            num_heads = float(v)
            print(num_heads)
        elif k == "n_layer":
            num_layers = float(v)
            print(num_layers)
        elif k == "shared_mlp_size":
            shared_mlp_size = float(v)
            print(shared_mlp_size)
        elif k == "shared_attn_size":
            shared_attn_size = float(v)
            print(shared_attn_size)
        elif k == "shared_mlp_sym":
            shared_mlp_sym = 1/2
            print(shared_mlp_sym)
        elif k == "shared_attn_sym":
            shared_attn_sym = 1/2
            print(shared_attn_sym)
    hardware_efficiency = num_heads * num_layers * shared_mlp_sym * shared_attn_sym /(shared_mlp_size * shared_attn_size)
    print()
    return hardware_efficiency

def main():
    args = parse_args()
    config_basename = os.path.splitext(os.path.basename(args.c))[0]

    with open(args.c, "r") as file:
        original_configurations = json.load(file)
    for config in original_configurations:
        print(config)
        run_experiment_with_vizier(
            config,
            config_basename,
            args.output_dir,
            args.add_names,
            args.vizier_algorithm,
            args.vizier_iterations,
            args.parameters,
            args.ranges,
            args.scaling_types,
            args.data_types,
        )


if __name__ == "__main__":
    main()
