import click
import torch
from torch.utils.data import datasets, DataLoader
from torchvision import transforms

from src.models.model import ImageClassifier
from src.utils.utils import create_class_mappings, load_yaml_config


# Example usage
train_dir = 'data/raw/train'

class_to_idx, idx_to_class = create_class_mappings(train_dir)

@click.group()
def cli():
    """Command line interface."""
    pass

data_config = load_yaml_config("data/data_config.yaml")
normalization_mean = data_config['mean']
normalization_std = data_config['std']

@click.command()
@click.argument("model_checkpoint")
@click.option("--data", default='data/raw/test/', help="Folder containing the data. Data can be .jpg, .pngs, .np, .pt")
@click.option("--resize-dim", required=True, type=int, help="Resize dimension for the images")
def predict(model_checkpoint: str, data: str, resize_dim: int):
    model = ImageClassifier.load_from_checkpoint(
        checkpoint_path=model_checkpoint,
        map_location='cpu',
    )
    model.eval()

    # Define the transforms
    base_transforms = transforms.Compose([
        transforms.Resize((resize_dim, resize_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization_mean, std=normalization_std),
    ])

    # Create the dataset and DataLoader
    dataset = datasets.ImageFolder(root=data, transform=base_transforms)
    inference_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    _, idx_to_class = create_class_mappings('data/raw/train')

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
            results[image_file_name] = {
                "certainty": top_prob.item(),
                "class_name": idx_to_class[top_idx.item()]
            }

    return results

cli.add_command(predict)

if __name__ == "__main__":
    cli()



