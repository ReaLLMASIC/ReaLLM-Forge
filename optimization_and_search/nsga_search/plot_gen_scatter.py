import numpy as np
import pandas as pd
from visualization.evolution_history import LogParser
from nsga2 import Population
import json, os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def get_pareto_front(obj1, obj2):
    """Identify the Pareto front from a set of points."""
    points = np.array(list(zip(obj1, obj2)))
    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i, point in enumerate(points):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(points[is_pareto] < point, axis=1)  # Keep any point with a lower value
            is_pareto[i] = True  # And keep self
    return points[is_pareto]

def plot_gen_scatter(
    file_name_base: str,
    save_path: str,
    start_gen: int,
    end_gen: int,
    x_axis: str = 'ttft',
    y_axis: str = 'validation_loss',
    color_axis: str = 'generation',
    cmap: str = 'Blues',           # blue (light to dark) sequential colormap
    fix_color_scale: bool = True,  # normalize color scale using global min/max
    point_size: float = 10,       # size of scatter markers (points^2)
    cmap_min: float = 0.2,        # lower bound of colormap (0..1); raise to avoid too-light colors
    cmap_max: float = 1.0         # upper bound of colormap (0..1)
):
    pop_data = []
    generations = list(range(start_gen, end_gen + 1))
    file_name_base2 = "ckpts/infi_attn_exp_2/10090627_ckpt_gen"
    for gen in generations:
        json_file_name = f"{file_name_base}{gen}.json"
        if not os.path.exists(json_file_name):
            json_file_name = f"{file_name_base2}{gen}.json"
            # rename this file if exists
            if os.path.exists(json_file_name):
                # rename to the first file name
                os.rename(json_file_name, f"{file_name_base}{gen}.json")

    for gen in generations:
        json_file_name = f"{file_name_base}{gen}.json"
        if not os.path.exists(json_file_name):
            print(f"❌ Checkpoint file not found: {json_file_name}")
            exit(f"❌ Checkpoint file not found: {json_file_name}\nPlease ensure all generation checkpoint files are present.")
                
        population = Population.load_checkpoint(json_file_name, from_pkl=False)

        val_loss_vals = [eva.objs[0] for eva in population.evaluations ]
        energy_vals = [eva.objs[1] for eva in population.evaluations ]
        ttft_vals = [eva.objs[2] for eva in population.evaluations ]

        perplexity = [np.exp(va) for va in val_loss_vals]

        # Ensure all lists have the same length
        min_len = min(len(val_loss_vals), len(energy_vals), len(ttft_vals))
        
        for i in range(min_len):
            pop_data.append({
                'generation': gen,
                'validation_loss': val_loss_vals[i],
                'energy_per_token': energy_vals[i],
                'ttft': ttft_vals[i],
                'perplexity': perplexity[i],
                'individual_id': i
            })

    df_pop = pd.DataFrame(pop_data)

    # based on the x_axis and y_axis, slect the design on the first pareto front and highlight them
    if x_axis not in df_pop.columns or y_axis not in df_pop.columns or color_axis not in df_pop.columns:
        exit(f"❌ Invalid axis names. Available columns: {df_pop.columns.tolist()}")

    # Select the designs on the first Pareto front
    pareto_front = get_pareto_front(df_pop[x_axis], df_pop[y_axis])
    pareto_df = pd.DataFrame(pareto_front, columns=[x_axis, y_axis])

    print("Detected Pareto front designs:")
    print(pareto_df)

    # Create scatter plot
    fig, ax = plt.subplots(dpi=400)
    # Normalize color scale across all points so the colorbar is consistent
    vmin = df_pop[color_axis].min() if fix_color_scale else None
    vmax = df_pop[color_axis].max() if fix_color_scale else None
    # Build a truncated colormap to avoid overly light tones
    base_cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    cmap_vals = base_cmap(np.linspace(max(0.0, cmap_min), min(1.0, cmap_max), 256))
    truncated_cmap = mcolors.LinearSegmentedColormap.from_list(
        f"{getattr(base_cmap, 'name', 'cmap')}_trunc",
        cmap_vals
    )

    # plot all designs
    scatter = ax.scatter(
        df_pop[x_axis],
        df_pop[y_axis],
        c=df_pop[color_axis],
        cmap=truncated_cmap,
        vmin=vmin,
        vmax=vmax,
        s=point_size,
        label='NSGA-II searched architectures'
    )

    # Highlight Pareto front designs with red hollow star markers
    # pareto_scatter = ax.scatter(
    #     pareto_df[x_axis],
    #     pareto_df[y_axis],
    #     marker='*',
    #     facecolors='none',
    #     edgecolors='red',
    #     linewidths=1.2,
    #     s=120,
    #     zorder=3,
    #     label='Pareto front designs'
    # )

    # add special reference point
    ax.scatter(
        123.55,
        3.004,
        c='purple',
        marker='*',
        s=25,
        label='GPT-2 Small'
    )

    # show legend at top-right
    ax.legend(loc='upper right')

    # set axis ranges
    ax.set_xlim(right=130)
    ax.set_ylim(top=3.6)

    # ax.set_xlabel(x_axis)
    ax.set_xlabel("Size (M)")
    ax.set_ylabel("Validation Loss")

    # use blue (light to dark) for color axis (colorbar from full population scatter)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_axis)
    plt.title("Optimizing Accuracy and Size")
    plt.savefig(save_path)
    plt.close()


def main():
    """Main function to create the interactive plots"""
    import os
    
    # take arguments from command line
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Interactive Generational Scatter Plots")
    parser.add_argument("--ckpt_base", type=str, default="ckpts/infi_attn_exp_2/1009_0627_ckpt_gen", help="Path to the evolution log file")
    parser.add_argument("--start_gen", type=int, default=1, help="Starting generation index (default: 1)")
    parser.add_argument("--end_gen", type=int, default=80, help="Ending generation index (inclusive, default: 30)")
    parser.add_argument("--output", type=str, default="plots/gen_scatter.png", help="Output png file path")
    args = parser.parse_args()
    
    file_name_base = args.ckpt_base
    start_gen = args.start_gen
    end_gen = args.end_gen
    output_path = args.output

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_gen_scatter(file_name_base, output_path, start_gen, end_gen)

    # Print completion message
    print(f"✅ Generational scatter plot saved to {output_path}")


if __name__ == "__main__":
    main()