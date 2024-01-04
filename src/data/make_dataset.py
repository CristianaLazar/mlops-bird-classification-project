import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

def load_image(path: str) -> np.ndarray:
    """Load an image from the specified path."""
    image = cv2.imread(path)[..., ::-1]  # Load the image in RGB mode
    return image

def load_all_images(folder_path: str) -> Dict[str, List[np.ndarray]]:
    """Load all .jpg images from the specified folder and its subfolders."""
    images = {}
    for subdir, _, files in os.walk(folder_path):
        subdir_images = []
        for file in files:
            if file.endswith('.jpg'):
                path = os.path.join(subdir, file)
                image = load_image(path)
                subdir_images.append(image)
        if subdir_images:
            relative_subdir = os.path.relpath(subdir, folder_path)
            images[relative_subdir] = subdir_images
    return images

def calculate_mean_std(images: List[np.ndarray]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Calculate the mean and standard deviation for each channel across all images."""
    # Accumulators for sum, sum of squares, and number of pixels
    sum_rgb = np.zeros(3)
    sum_sq_rgb = np.zeros(3)
    pixel_count = 0

    for img in images:
        sum_rgb += np.sum(img, axis=(0, 1))  # Sum over height and width for each channel
        sum_sq_rgb += np.sum(np.square(img), axis=(0, 1))  # Sum of squares
        pixel_count += img.shape[0] * img.shape[1]  # Total number of pixels (height * width)

    mean_rgb = sum_rgb / pixel_count
    std_rgb = np.sqrt(sum_sq_rgb / pixel_count - np.square(mean_rgb))

    return tuple(mean_rgb), tuple(std_rgb)

def normalize_images(images: List[np.ndarray], mean_rgb: Tuple[float, float, float], std_rgb: Tuple[float, float, float]) -> List[np.ndarray]:
    """Normalize each image in the list channel-wise."""
    normalized_images = []
    for image in images:
        # Normalize each channel separately
        normalized_image = np.zeros_like(image, dtype=np.float32)
        for c in range(3):  # Assuming image has three channels (RGB)
            normalized_image[:, :, c] = (image[:, :, c] - mean_rgb[c]) / std_rgb[c]
        normalized_images.append(normalized_image)
    return normalized_images

def save_data_config(mean, std, file_path):
    """Save mean and std to a YAML config file."""
    config = {'mean': mean, 'std': std}
    with open(file_path, 'w') as file:
        yaml.dump(config, file)

def save_images(images: Dict[str, List[np.ndarray]], base_folder: str):
    """Save images as numpy arrays, preserving the subfolder structure."""
    for subdir, imgs in images.items():
        subdir_path = os.path.join(base_folder, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
        for i, image in enumerate(imgs):
            np.save(os.path.join(subdir_path, f'image_{i}.npy'), image)


if __name__ == '__main__':
    root = 'data/raw/'
    dest_root = 'data/processed/'

    # Load and process training images
    train_folder = os.path.join(root, 'train')
    train_images = load_all_images(train_folder)

    # Calculate mean and std for training images
    all_train_images = [img for sublist in train_images.values() for img in sublist]
    mean_rgb, std_rgb = calculate_mean_std(all_train_images)
    print(f"Training Data - Mean RGB: {mean_rgb}, Standard Deviation RGB: {std_rgb}")

    # Save mean and std to a config file for training data
    save_data_config(mean_rgb, std_rgb, 'data/train_data_config.yaml')

    # Normalize train images and save them in the corresponding subfolder structure
    normalized_train_images = {subdir: normalize_images(imgs, mean_rgb, std_rgb) 
                               for subdir, imgs in train_images.items()}
    save_images(normalized_train_images, os.path.join(dest_root, 'train'))

    # Process and normalize validation and test images using the training mean and std
    for sub_folder in ['validation', 'test']:
        folder_path = os.path.join(root, sub_folder)
        images = load_all_images(folder_path)
        normalized_images = {subdir: normalize_images(imgs, mean_rgb, std_rgb) 
                             for subdir, imgs in images.items()}
        save_images(normalized_images, os.path.join(dest_root, sub_folder))