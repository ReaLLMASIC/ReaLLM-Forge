# graph_sine_bins.py
import numpy as np
import matplotlib.pyplot as plt


def load_and_plot(filename, label):
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype=np.uint16)
    plt.plot(data, label=label)


def main():
    plt.figure(figsize=(12, 4))
    load_and_plot("train.bin", "train.bin")
    load_and_plot("val.bin", "val.bin")
    plt.title("Sine Wave Token Values from train.bin and val.bin")
    plt.xlabel("Index")
    plt.ylabel("Token Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
