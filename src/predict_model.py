import os
import hydra
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.model import ImageClassifier
from src.utils.utils import create_class_mappings


@hydra.main(config_path="configs", config_name="default_config.yaml", version_base="1.1")
def predict(config: DictConfig) -> None:
    config = config.predict

    train_dir = hydra.utils.to_absolute_path(config.data)
    class_to_idx, idx_to_class = create_class_mappings(train_dir)

    model = ImageClassifier.load_from_checkpoint(
        checkpoint_path=config.model_checkpoint,
        map_location="cpu",
    )
    model.eval()

    # Define the transforms
    base_transforms = transforms.Compose(
        [
            transforms.Resize((config.resize_dim, config.resize_dim)),
            transforms.ToTensor(),
            # Assuming normalization mean and std are part of your config
            transforms.Normalize(mean=config.normalization.mean, std=config.normalization.std),
        ]
    )

    # Create the dataset and DataLoader
    dataset = datasets.ImageFolder(root=config.data, transform=base_transforms)
    inference_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = {}
    with torch.no_grad():
        for images, indices in inference_loader:
            logits = model(images)
            probabilities = F.softmax(logits, dim=1)
            top_prob, top_idx = probabilities.topk(1, dim=1)

            # Get the image file name
            image_path = dataset.samples[indices.item()][0]
            image_file_name = os.path.basename(image_path)

            # Store the results in a dictionary
            results[image_file_name] = {"certainty": top_prob.item(), "class_name": idx_to_class[top_idx.item()]}

    return results


if __name__ == "__main__":
    predict()
