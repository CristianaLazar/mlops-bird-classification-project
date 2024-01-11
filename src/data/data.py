from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import datasets, transforms


class ImageFolderClassificationModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir,
        val_dir,
        resize_dims,
        normalization_mean,
        normalization_std,
        augmentation_strategy,
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.resize_dims = resize_dims
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.augmentation_strategy = augmentation_strategy
        self.train_transform, self.val_transform = self._get_transform()

    def _get_transform(self):
        # Base transforms with dynamic normalization
        base_transforms = [
            transforms.Resize(self.resize_dims),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalization_mean, std=self.normalization_std),
        ]

        # Additional transforms based on augmentation level
        if self.augmentation_strategy == "light":
            augment_transforms = [
                transforms.RandomHorizontalFlip(),
            ]
        elif self.augmentation_strategy == "moderate":
            augment_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
            ]
        else:  # heavy
            augment_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.RandomResizedCrop(self.resize_dims, scale=(0.8, 1.2)),
                # We can add more transforms if necessary
            ]

        return transforms.Compose(base_transforms + augment_transforms), transforms.Compose(base_transforms)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = datasets.ImageFolder(root=self.train_dir, transform=self.train_transform)
        self.val_dataset = datasets.ImageFolder(root=self.val_dir, transform=self.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
