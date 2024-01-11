import os
from glob import glob
import matplotlib.pyplot as plt # type: ignore
import numpy as np


def plot_distribution(dest_root, dataset_type="train", save_path="reports/figures"):
    subfolders = [
        name
        for name in os.listdir(os.path.join(dest_root, dataset_type))
        if os.path.isdir(os.path.join(dest_root, dataset_type, name))
    ]
    subfolders.sort()
    counts = [len(glob(os.path.join(dest_root, dataset_type, subfolder, "*"))) for subfolder in subfolders]

    plt.figure(figsize=(20, 10))  # Increase figure size for better visibility
    ax = plt.subplot()

    indices = np.arange(len(subfolders))
    bar_width = 0.35  # Adjust bar width
    opacity = 0.8

    ax.bar(indices, counts, bar_width, alpha=opacity, color="b", label=dataset_type.capitalize())

    ax.set_xlabel("Subfolder indices")
    ax.set_ylabel("Number of images")
    ax.set_title(f"Distribution of images in {dataset_type}")
    ax.set_xticks(indices[::25])  # Display every 10th index
    ax.set_xticklabels(indices[::25], fontsize=16)  # Adjust font size

    plt.tight_layout()
    file_name = f"data_distribution_{dataset_type}.png"
    plt.savefig(os.path.join(save_path, file_name), bbox_inches="tight")  # Save with enough space


if __name__ == "__main__":
    root = "data/raw/"
    plot_distribution(root, dataset_type="train", save_path="reports/figures")
    plot_distribution(root, dataset_type="validation", save_path="reports/figures")
    plot_distribution(root, dataset_type="test", save_path="reports/figures")
