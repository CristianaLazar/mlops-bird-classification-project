import os

from PIL import Image
import numpy as np


train_images_folder = 'data/raw/corruptmnist/train/images'
test_images_folder = 'data/raw/corruptmnist/test/images'

def load_image(path):
    """Load all .pt files from the specified folder."""
    imgage = Image.load(path)
    for file in os.listdir(folder_path):
        if file.endswith('.pt'):
            path = os.path.join(folder_path, file)
            tensors.append(torch.load(path))
    return image

def calculate_mean_std(tensors):
    """Calculate the mean and standard deviation across all tensors."""
    all_data = torch.cat(tensors)
    mean = torch.mean(all_data)
    std = torch.std(all_data)
    return mean.item(), std.item()

def normalize_tensors(tensors, mean, std):
    """Normalize each tensor in the list."""
    normalized = [(tensor - mean) / std for tensor in tensors]
    return normalized

def load_normalize_and_save_tensors(folder_path, mean, std, save_folder=None):
    """Load, normalize, and save tensors from a folder."""
    tensors = load_all_tensors(folder_path)
    normalized_tensors = normalize_tensors(tensors, mean, std)

    if save_folder is None:
        save_folder = folder_path

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, tensor in enumerate(normalized_tensors):
        save_path = os.path.join(save_folder, f'normalized_{i}.pt')
        torch.save(tensor, save_path)

if __name__ == '__main__':
    train_tensors = load_all_tensors(train_images_folder)

    mean, std = calculate_mean_std(train_tensors)
    print(f"Mean: {mean}, Standard Deviation: {std}")

    # Optionally, specify a different folder to save normalized tensors
    normalized_train_folder = 'data/processed/corruptmnist/train/images'
    normalized_test_folder = 'data/processed/corruptmnist/test/images'

    load_normalize_and_save_tensors(train_images_folder, mean, std, normalized_train_folder)
    load_normalize_and_save_tensors(test_images_folder, mean, std, normalized_test_folder)