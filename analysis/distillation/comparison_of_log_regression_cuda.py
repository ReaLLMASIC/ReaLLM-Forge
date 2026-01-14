import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse
import os
from tqdm import tqdm
from itertools import cycle
import csv

# Define function for logarithmic fit
def log_func(x, a, b):
    return a * np.log1p(x) + b

# Define function for exponential decay fit
def exp_decay_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Define MSE Inverse (MSEI) metric
def mse_inverse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return 1 / (1 + mse)  # Higher MSEI means better fit

def save_regression_plot(x_vals, y_vals, fit_results, dim, metric_name, regression_type):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_vals, y_vals, color="red", s=1, alpha=0.3, label=f"{metric_name}")
    if "log" in fit_results:
        plt.plot(x_vals, fit_results["log"][0], color="black", linestyle="--", linewidth=2,
                 label=f"Log Fit (MSEI={fit_results['log'][1]:.4f})")
    if "exp" in fit_results:
        plt.plot(x_vals, fit_results["exp"][0], color="blue", linestyle="-.", linewidth=2,
                 label=f"Exp Fit (MSEI={fit_results['exp'][1]:.4f})")
    plt.xlabel("Number of Vectors Added")
    plt.ylabel(metric_name)
    plt.title(f"Regression Fit for Min {metric_name} ({dim}-Dim)")
    plt.legend()
    plt.savefig(f"angle_distribution_{dim}d_regression_{metric_name.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()

def save_comparison_plot(regression_trends, metric_name, regression_type):
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    plt.figure(figsize=(12, 8))
    for label, (x_vals, min_vals, trend, msei) in regression_trends.items():
        color = next(color_cycle)
        plt.scatter(
            x_vals,
            min_vals,
            color=color,
            s=1,
            alpha=0.3,
            label=label.replace("Log", f"Min {metric_name}").replace("Exp", f"Min {metric_name}")
        )
        linestyle = "--" if "Log" in label else "-."
        plt.plot(
            x_vals,
            trend,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=f"{label.replace('Min Log', 'Log Fit').replace('Min Exp', 'Exp Fit')} (MSEI={msei:.4f})"
        )

    plt.xlabel("Number of Vectors Added")
    plt.ylabel(metric_name)
    plt.title(f"Comparison of Regression Models Across Dimensions - {metric_name}")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc='upper right', framealpha=0.7)
    plt.savefig(f"angle_distribution_{regression_type}_comparison_{metric_name.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()

def process_embedding_dims(min_pow, max_pow, regression_type, num_vectors, mean, stddev, use_cuda):
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    embedding_dims = [2 ** i for i in range(min_pow, max_pow + 1)]
    regression_trends_angle = {}
    regression_trends_abs_dot = {}
    regression_trends_norm_abs_dot = {}
    all_regression_params = []

    for dim in tqdm(embedding_dims, desc="Processing Dimensions", unit="dim"):
        data_filename = f"angle_distribution_{dim}d.npy"

        if os.path.exists(data_filename):
            print(f"\nLoading existing data file: {data_filename}")
            data_dict = np.load(data_filename, allow_pickle=True).item()
        else:
            print(f"\nGenerating new data file: {data_filename}")

            vectors = torch.normal(mean, stddev, size=(1, dim), device=device)
            min_angles = []
            max_abs_dots = []
            max_abs_norm_dots = []

            for _ in tqdm(range(1, num_vectors + 1), desc=f"{dim}D Vectors", unit="vec", leave=False):
                new_vector = torch.normal(mean, stddev, size=(dim,), device=device)
                
                # Compute absolute dot products
                dots = torch.matmul(vectors, new_vector)
                max_abs_dots.append(torch.max(torch.abs(dots)).item())
                
                # Compute absolute normalized dot products (cosine similarity)
                norm_vectors = vectors / torch.norm(vectors, dim=1, keepdim=True)
                norm_new_vector = new_vector / torch.norm(new_vector)
                norm_dots = torch.matmul(norm_vectors, norm_new_vector)
                max_abs_norm_dots.append(torch.max(torch.abs(norm_dots)).item())
                
                # Compute angles from normalized dot products
                angles = torch.rad2deg(torch.acos(torch.clamp(norm_dots, -1.0, 1.0)))
                min_angles.append(torch.min(angles).item())

                vectors = torch.cat((vectors, new_vector.unsqueeze(0)), dim=0)

            data_dict = {
                "x_vals": np.arange(1, num_vectors + 1),
                "min_angles": np.array(min_angles),
                "max_abs_dots": np.array(max_abs_dots),
                "max_abs_norm_dots": np.array(max_abs_norm_dots)
            }
            np.save(data_filename, data_dict)

        x_vals = data_dict["x_vals"]
        metrics = {
            "Angle": data_dict["min_angles"],
            "Absolute Dot Product": data_dict.get("max_abs_dots", []),  # Use get() for backward compatibility
            "Absolute Normalized Dot Product": data_dict.get("max_abs_norm_dots", [])  # Use get() for backward compatibility
        }

        for metric_name, vals in metrics.items():
            if len(vals) == 0:  # Skip if data not available (for backward compatibility)
                continue
                
            fit_results = {}
            if regression_type in ["log", "both"]:
                popt_log, _ = curve_fit(log_func, x_vals, vals)
                trend_log = log_func(x_vals, *popt_log)
                msei_log = mse_inverse(np.array(vals), trend_log)
                fit_results["log"] = (trend_log, msei_log)

                all_regression_params.append({
                    "dimension": dim,
                    "metric": metric_name,
                    "regression_type": "log",
                    "params": popt_log,
                    "MSEI": msei_log
                })
                print(f"Dimension {dim}, {metric_name} Log Fit: a={popt_log[0]:.6f}, b={popt_log[1]:.6f}, MSEI={msei_log:.6f}")

                trends_dict = {
                    "Angle": regression_trends_angle,
                    "Absolute Dot Product": regression_trends_abs_dot,
                    "Absolute Normalized Dot Product": regression_trends_norm_abs_dot
                }[metric_name]
                trends_dict[f"{dim}D Log"] = (x_vals, vals, trend_log, msei_log)

            if regression_type in ["exp", "both"]:
                popt_exp, _ = curve_fit(exp_decay_func, x_vals, vals, p0=[5, 0.001, 80])
                trend_exp = exp_decay_func(x_vals, *popt_exp)
                msei_exp = mse_inverse(np.array(vals), trend_exp)
                fit_results["exp"] = (trend_exp, msei_exp)

                all_regression_params.append({
                    "dimension": dim,
                    "metric": metric_name,
                    "regression_type": "exp",
                    "params": popt_exp,
                    "MSEI": msei_exp
                })
                print(f"Dimension {dim}, {metric_name} Exp Fit: a={popt_exp[0]:.6f}, b={popt_exp[1]:.6f}, c={popt_exp[2]:.6f}, MSEI={msei_exp:.6f}")

                trends_dict = {
                    "Angle": regression_trends_angle,
                    "Absolute Dot Product": regression_trends_abs_dot,
                    "Absolute Normalized Dot Product": regression_trends_norm_abs_dot
                }[metric_name]
                trends_dict[f"{dim}D Exp"] = (x_vals, vals, trend_exp, msei_exp)

            # Save individual plots
            plt.figure(figsize=(10, 6))
            plt.scatter(x_vals, vals, color="red", s=1, alpha=0.3, label=f"{metric_name}")
            plt.xlabel("Number of Vectors Added")
            plt.ylabel(metric_name)
            plt.title(f"{metric_name} Distribution ({dim}-Dim)")
            plt.legend()
            plt.savefig(f"angle_distribution_{dim}d_{metric_name.lower().replace(' ', '_')}.png", dpi=300)
            plt.close()

            # Save regression plots
            save_regression_plot(x_vals, vals, fit_results, dim, metric_name, regression_type)

        print(f"Saved plots for {dim}D")

    # Generate final comparison charts for each metric
    save_comparison_plot(regression_trends_angle, "Angle", regression_type)
    save_comparison_plot(regression_trends_abs_dot, "Absolute Dot Product", regression_type)
    save_comparison_plot(regression_trends_norm_abs_dot, "Absolute Normalized Dot Product", regression_type)

    # Write all regression parameters to a CSV file
    with open("regression_constants.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Dimension", "Metric", "Regression Type", "a", "b", "c", "MSEI"])
        for row in all_regression_params:
            dimension = row["dimension"]
            metric = row["metric"]
            regtype = row["regression_type"]
            p = row["params"]
            msei = row["MSEI"]
            if len(p) == 2:
                a, b = p
                c = ""
            else:
                a, b, c = p
            writer.writerow([dimension, metric, regtype, a, b, c, msei])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process embedding dimensions for CUDA-accelerated regression analysis.")
    parser.add_argument("--min_pow", type=int, default=2, help="Minimum power of 2 for embedding dims (default: 2)")
    parser.add_argument("--max_pow", type=int, default=10, help="Maximum power of 2 for embedding dims (default: 10)")
    parser.add_argument("--regression", choices=["log", "exp", "both"], default="both",
                        help="Choose regression type: log, exp, or both (default: both)")
    parser.add_argument("--num_vectors", type=int, default=5000, help="Number of vectors (default: 5000)")
    parser.add_argument("--mean", type=float, default=0.0, help="Mean for Gaussian initialization (default: 0.0)")
    parser.add_argument("--stddev", type=float, default=0.02, help="Stddev for Gaussian initialization (default: 0.02)")
    parser.add_argument("--use_cuda", action="store_true", help="Enable CUDA acceleration (if available)")

    args = parser.parse_args()

    process_embedding_dims(args.min_pow, args.max_pow, args.regression, args.num_vectors, args.mean, args.stddev, args.use_cuda)

