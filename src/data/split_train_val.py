import os
import shutil
import random
from glob import glob
import matplotlib.pyplot as plt

root = 'data/raw/'
sub_folders = ['train', 'validation']
sub_sub_folders = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']


def split_dataset(train_folder, val_folder, split_ratio=0.8, seed=42):
    # Set the seed for random number generator
    random.seed(seed)

    subfolders = [f.path for f in os.scandir(train_folder) if f.is_dir()]

    for subfolder in subfolders:
        # Get all JPG files in the subfolder
        images = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith('.jpg')]
        random.shuffle(images)

        # Calculate the split index for 80%-20% distribution
        split_index = int(len(images) * split_ratio)

        # Split images into train (remaining) and validation sets
        val_images = images[split_index:]

        # Create corresponding subfolder in validation folder
        subfolder_name = os.path.basename(subfolder)
        val_subfolder = os.path.join(val_folder, subfolder_name)
        os.makedirs(val_subfolder, exist_ok=True)

        # Move images to the validation subfolder
        for image in val_images:
            shutil.move(image, os.path.join(val_subfolder, os.path.basename(image)))

def plot_distribution(dest_root, subfolders):
    train_counts = []
    validation_counts = []
    test_counts = []

    for subfolder in subfolders:
        train_path = os.path.join(dest_root, 'train', subfolder)
        validation_path = os.path.join(dest_root, 'validation', subfolder)
        test_path = os.path.join(dest_root, 'test', subfolder)
        train_counts.append(len(glob(os.path.join(train_path, '*'))))
        validation_counts.append(len(glob(os.path.join(validation_path, '*'))))
        test_counts.append(len(glob(os.path.join(test_path, '*'))))

    # Plotting the distribution
    fig, ax = plt.subplots()
    index = range(len(subfolders))
    bar_width = 0.25
    opacity = 0.8

    rects1 = ax.bar(index, train_counts, bar_width, alpha=opacity, color='b', label='Train')
    rects2 = ax.bar([i + bar_width for i in index], validation_counts, bar_width, alpha=opacity, color='r', label='Validation')
    rects3 = ax.bar([i + bar_width * 2 for i in index], test_counts, bar_width, alpha=opacity, color='g', label='Test')

    ax.set_xlabel('Subfolders')
    ax.set_ylabel('Number of images')
    ax.set_title('Distribution of images in train, validation, and test sets')
    ax.set_xticks([i + bar_width for i in index])
    ax.set_xticklabels(subfolders)
    ax.legend()

    plt.tight_layout()
    plt.savefig('reports/figures/data_set_distributions.png')


if __name__ == '__main__':
    split_dataset(root+sub_folders[0], root+sub_folders[1])
    plot_distribution(root, sub_sub_folders)