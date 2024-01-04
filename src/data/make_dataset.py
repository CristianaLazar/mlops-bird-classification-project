import os
import cv2
import numpy as np
import yaml

def load_config(file_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def load_image(path: str) -> np.ndarray:
    """Load an image from the specified path."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Convert to RGB and normalize
    return image

def normalize_image(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize an image using the provided mean and standard deviation."""
    normalized_image = (image - mean) / std
    return normalized_image

def process_and_save_images(src_folder: str, dest_folder: str, mean: np.ndarray, std: np.ndarray):
    """Process and save images from source to destination folder."""
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for subdir, _, files in os.walk(src_folder):
        for file in files:
            if file.endswith('.jpg'):
                src_path = os.path.join(subdir, file)
                image = load_image(src_path)
                normalized_image = normalize_image(image, mean, std)

                # Construct destination path
                rel_path = os.path.relpath(subdir, src_folder)
                dest_subdir = os.path.join(dest_folder, rel_path)
                if not os.path.exists(dest_subdir):
                    os.makedirs(dest_subdir)
                
                dest_path = os.path.join(dest_subdir, os.path.splitext(file)[0] + '.npy')
                np.save(dest_path, normalized_image)

if __name__ == '__main__':
    config_path = 'data/train_data_config.yaml'
    root = 'data/raw/'
    dest_root = 'data/processed/'

    # Load configuration
    config = load_config(config_path)
    mean_rgb = np.array(config['mean'])
    std_rgb = np.array(config['std'])

    # Normalize and save images for each dataset
    for dataset in ['train', 'validation', 'test']:
        src_folder = os.path.join(root, dataset)
        dest_folder = os.path.join(dest_root, dataset)
        process_and_save_images(src_folder, dest_folder, mean_rgb, std_rgb)


