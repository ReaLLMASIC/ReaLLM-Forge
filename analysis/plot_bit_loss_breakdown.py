import sys, os
import pandas as pd
import matplotlib.pyplot as plt

run_dir = sys.argv[1]
w = float(sys.argv[2])

layers = pd.read_csv(os.path.join(run_dir, "bit_layers.csv"))
mods   = pd.read_csv(os.path.join(run_dir, "bit_modules.csv"))

# ---- total params per iter ----
total_param = layers.groupby("iter")["param_count"].sum().rename("total_param")
layers = layers.merge(total_param, on="iter", how="left")

# ---- average bit loss (global) ----
avg_bit = layers.groupby("iter").apply(
    lambda g: g["total_bits_int"].sum() / g["param_count"].sum()
).rename("avg_b_int").reset_index()

avg_bit["avg_bit_loss"] = w * avg_bit["avg_b_int"]

plt.figure()
plt.plot(avg_bit["iter"], avg_bit["avg_bit_loss"])
plt.xlabel("iter"); plt.ylabel("avg_bit_loss")
plt.title("Average bit loss (normalized)")
plt.tight_layout()
plt.savefig("avg_bit_loss.png", dpi=200)
plt.close()

# ---- per-layer contribution ----
layers["layer_bit_loss"] = w * (layers["total_bits_int"] / layers["total_param"])

plt.figure()
for lid, g in layers.groupby("layer"):
    g = g.sort_values("iter")
    plt.plot(g["iter"], g["layer_bit_loss"], label=f"layer {int(lid)}")
plt.legend()
plt.xlabel("iter"); plt.ylabel("layer_bit_loss")
plt.title("Per-layer bit loss contribution (normalized)")
plt.tight_layout()
plt.savefig("layer_bit_loss.png", dpi=200)
plt.close()

# ---- per-module contribution (top-k by average) ----
mods["module_total_bits_int"] = mods["b_int"] * mods["param_count"]
total_param_m = mods.groupby("iter")["param_count"].sum().rename("total_param")
mods = mods.merge(total_param_m, on="iter", how="left")
mods["module_bit_loss"] = w * (mods["module_total_bits_int"] / mods["total_param"])

topk = (
    mods.groupby("module_name")["module_bit_loss"].mean()
    .sort_values(ascending=False)
    .head(10)
    .index
)

plt.figure(figsize=(10,6))
for name in topk:
    g = mods[mods["module_name"]==name].sort_values("iter")
    plt.plot(g["iter"], g["module_bit_loss"], label=name)
plt.legend(fontsize=7)
plt.xlabel("iter"); plt.ylabel("module_bit_loss")
plt.title("Top-10 module bit loss contributions (normalized)")
plt.tight_layout()
plt.savefig("top10_module_bit_loss.png", dpi=200)
plt.close()

print("Saved: avg_bit_loss.png, layer_bit_loss.png, top10_module_bit_loss.png")
