# scripts/plot_bit_alloc.py
import os
import csv
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def read_csv(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="e.g. out/repro_adaptive_bitbalanced_w1e-5")
    ap.add_argument("--out_dir", type=str, default=None, help="where to save figures; default: <run_dir>/bit_alloc/figs")
    args = ap.parse_args()

    if os.path.exists(os.path.join(args.run_dir, "bit_layers.csv")):
        bit_dir = args.run_dir
    else:
        bit_dir = os.path.join(args.run_dir, "bit_alloc")
        
    layers_csv = os.path.join(bit_dir, "bit_layers.csv")
    types_csv  = os.path.join(bit_dir, "bit_types.csv")
    events_csv = os.path.join(bit_dir, "bit_events.csv")

    out_dir = args.out_dir or os.path.join(bit_dir, "figs")
    os.makedirs(out_dir, exist_ok=True)

    layers = read_csv(layers_csv)
    types_ = read_csv(types_csv)

    # -------- plot: per-layer avg_b_int vs iter --------
    by_layer = defaultdict(list)
    for r in layers:
        it = int(float(r["iter"]))
        layer = int(float(r["layer"]))
        b = float(r["avg_b_int_weighted"])
        by_layer[layer].append((it, b))

    plt.figure()
    for layer in sorted(by_layer.keys()):
        xs = [x for x, _ in sorted(by_layer[layer], key=lambda t: t[0])]
        ys = [y for _, y in sorted(by_layer[layer], key=lambda t: t[0])]
        plt.plot(xs, ys, label=f"layer {layer}")
    plt.xlabel("iter")
    plt.ylabel("avg_b_int (weighted)")
    plt.title("Per-layer average integer bitwidth")
    plt.legend()
    fig_path = os.path.join(out_dir, "layer_avg_b_int.png")
    plt.savefig(fig_path, dpi=160)
    plt.close()

    # -------- plot: attn vs mlp avg_b_int vs iter --------
    by_type = defaultdict(list)
    for r in types_:
        it = int(float(r["iter"]))
        fam = r["family"]
        b = float(r["avg_b_int_weighted"])
        by_type[fam].append((it, b))

    plt.figure()
    for fam in sorted(by_type.keys()):
        xs = [x for x, _ in sorted(by_type[fam], key=lambda t: t[0])]
        ys = [y for _, y in sorted(by_type[fam], key=lambda t: t[0])]
        plt.plot(xs, ys, label=fam)
    plt.xlabel("iter")
    plt.ylabel("avg_b_int (weighted)")
    plt.title("Attention vs MLP average integer bitwidth")
    plt.legend()
    fig_path = os.path.join(out_dir, "type_avg_b_int.png")
    plt.savefig(fig_path, dpi=160)
    plt.close()

    # -------- report: who drops bits first --------
    if os.path.exists(events_csv):
        events = read_csv(events_csv)
        first_drop = {}
        for e in events:
            name = e["module_name"]
            it = int(float(e["iter"]))
            if name not in first_drop:
                first_drop[name] = it

        ranked = sorted(first_drop.items(), key=lambda kv: kv[1])
        txt_path = os.path.join(out_dir, "first_drop_ranking.txt")
        with open(txt_path, "w") as f:
            for name, it in ranked:
                f.write(f"{it}\t{name}\n")
        print("Saved:", txt_path)

    print("Saved figures to:", out_dir)


if __name__ == "__main__":
    main()
