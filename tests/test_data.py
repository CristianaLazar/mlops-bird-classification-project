import pytest
import os
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import ToPILImage
from src.data.data import ImageFolderClassificationModule

# Sample data for testing
def create_sample_images(directory, num_images):
    os.makedirs(directory, exist_ok=True)
    to_img = ToPILImage()
    for i in range(num_images):
        img = torch.rand(3, 32, 32)  # Random image data
        img = to_img(img)
        img.save(os.path.join(directory, f"{i}.png"))

@pytest.fixture
def sample_data():
    data_dir = os.getcwd() + "/tests/data/raw/"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "test")
    create_sample_images(train_dir, 50)
    create_sample_images(val_dir, 20)
    yield train_dir, val_dir

def test_data_loading(sample_data):
    train_dir, val_dir = sample_data

    resize_dims = (32, 32)
    normalization_mean = (0.5, 0.5, 0.5)
    normalization_std = (0.5, 0.5, 0.5)
    augmentation_strategy = 'light'

    data_module = ImageFolderClassificationModule(train_dir, val_dir, resize_dims, normalization_mean, normalization_std, augmentation_strategy)

    # Test if setup method creates datasets
    data_module.setup()
    assert isinstance(data_module.train_dataset, torch.utils.data.Dataset)
    assert isinstance(data_module.val_dataset, torch.utils.data.Dataset)

    # Test dataloaders
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(val_dataloader, DataLoader)

    # Test batch sizes
    assert train_dataloader.batch_size == data_module.batch_size
    assert val_dataloader.batch_size == data_module.batch_size

    # Test number of workers
    assert train_dataloader.num_workers == data_module.num_workers
    assert val_dataloader.num_workers == data_module.num_workers

    # Test persistent_workers
    assert train_dataloader.persistent_workers == True
    assert val_dataloader.persistent_workers == True

    # Test iterating through dataloaders
    for batch in train_dataloader:
        x, y = batch
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

        assert x.shape[0] <= data_module.batch_size
        assert y.shape[0] <= data_module.batch_size
        break  # Stop after iterating through one batch

    for batch in val_dataloader:
        x, y = batch
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

        assert x.shape[0] <= data_module.batch_size
        assert y.shape[0] <= data_module.batch_size
        break  # Stop after iterating through one batch