import os
from typing import Tuple

import cv2
import numpy as np
import yaml


def load_image(path: str) -> np.ndarray:
    """Load an image from the specified path."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # Convert to RGB and normalize
    return image


def process_images_in_folder(folder_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Process all .jpg images in the specified folder and its subfolders."""
    sum_rgb = np.zeros(3)
    sum_sq_rgb = np.zeros(3)
    pixel_count = 0

    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg"):
                path = os.path.join(subdir, file)
                image = load_image(path)
                if image is None:
                    print(f"Failed to load image: {path}")
                    continue

                pixels = image.reshape(-1, 3)
                sum_rgb += np.sum(pixels, axis=0)
                sum_sq_rgb += np.sum(np.square(pixels), axis=0)
                pixel_count += pixels.shape[0]

    if pixel_count == 0:
        raise ValueError("No valid images processed. Check the image folder and file paths.")

    mean_rgb = sum_rgb / pixel_count
    std_rgb = np.sqrt((sum_sq_rgb / pixel_count) - np.square(mean_rgb))
    return mean_rgb, std_rgb


def save_data_config(mean, std, file_path):
    """Save mean and std to a YAML config file with three significant decimals."""
    # Round mean and std to three decimal places
    formatted_mean = [round(m, 3) for m in mean]
    formatted_std = [round(s, 3) for s in std]

    config = {"dataset_mean": formatted_mean, "dataset_std": formatted_std}
    with open(file_path, "w") as file:
        yaml.dump(config, file)


if __name__ == "__main__":
    root = "data/raw/"
    dest_config = "data/train_data_config.yaml"

    # Process training images and calculate mean and std
    train_folder = os.path.join(root, "train")
    try:
        mean_rgb, std_rgb = process_images_in_folder(train_folder)
        print(f"Training Data - Mean RGB: {mean_rgb}, Standard Deviation RGB: {std_rgb}")
        # Save mean and std to a config file
        save_data_config(mean_rgb, std_rgb, dest_config)
    except ValueError as e:
        print(e)
