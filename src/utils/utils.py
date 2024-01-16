import os
from typing import Dict, Tuple

import yaml


# Function to load config from YAML
def load_yaml_config(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


def create_class_mappings(train_dir: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create class-to-index and index-to-class mappings from a directory structure.

    Parameters:
    train_dir (str): Path to the training directory.

    Returns:
    Tuple[Dict[str, int], Dict[int, str]]: A tuple containing two dictionaries,
                                           (class_to_idx, idx_to_class).
    """
    # List all subdirectories (classes)
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    classes.sort()  # Sort to ensure consistent ordering

    # Create mappings
    class_to_idx: Dict[str, int] = {classes[i]: i for i in range(len(classes))}
    idx_to_class: Dict[int, str] = {i: classes[i] for i in range(len(classes))}

    return class_to_idx, idx_to_class


"""
# Save mapping
train_dir = "data/raw/test/"
_, idx_to_class = create_class_mappings(train_dir)

with open("src/utils/idx_to_class.json", 'w') as fp:
    json.dump(idx_to_class, fp)
"""