import click
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.models.model import ImageClassifier
from src.data.data import ImageFolderClassificationModule
from src.utils.utils import load_yaml_config


@click.group()
def cli():
    """Command line interface."""
    pass

@click.command()
@click.option("--model_name", default='efficientnet_es.ra_in1k', help="Name of the model.")
@click.option("--num_classes", default=525, help="Number of classes.")
@click.option("--drop_rate", default=0.5, help="Dropout rate.")
@click.option("--pretrained", is_flag=True, help="Use pretrained model.")
@click.option("--lr_encoder", default=3e-6, help="Learning rate for the encoder.")
@click.option("--lr_head", default=3e-4, help="Learning rate for the head.")
@click.option("--optimizer", default="AdamW", help="Optimizer to use.")
@click.option("--criterion", default="cross_entropy", help="Criterion for training.")
@click.option("--batch_size", default=32, help="Batch size for training.")
@click.option("--num_workers", default=4, help="Number of workers for data loading.")
@click.option("--num_epochs", default=100, help="Number of epochs for training.")
@click.option("--log_every_n_steps", default=10, help="Logging frequency.")
@click.option("--wandb_run_name", default="training_run", help="Wandb run name.")
@click.option("--wandb_project_name", default="mlops-bird-classification", help="Wandb project name.")
@click.option("--file_name", default="model", help="Model file name for checkpoints.")
@click.option("--resize_dims", nargs=2, type=int, default=(224, 224), help="Dimensions to resize images (width height).")
@click.option("--train_dir", default="data/raw/train", type=str, help="Directory path for training data.")
@click.option("--val_dir", default="data/raw/validation", type=str, help="Directory path for validation data.")
@click.option("--augmentation_strategy", default="moderate", type=click.Choice(['light', 'moderate', 'heavy']), help="Augmentation strategy to use.")
def train(model_name, num_classes, drop_rate, pretrained, lr_encoder, lr_head, optimizer, 
          criterion, batch_size, num_workers, num_epochs, log_every_n_steps, 
          wandb_run_name, wandb_project_name, file_name, resize_dims, train_dir, val_dir, augmentation_strategy):
    """Train a model."""
    # Load normalization parameters from the YAML file
    data_config = load_yaml_config("data/data_config.yaml")
    normalization_mean = data_config['mean']
    normalization_std = data_config['std']

    data_module = ImageFolderClassificationModule(
        train_dir=train_dir, 
        val_dir=val_dir,
        resize_dims=resize_dims, 
        normalization_mean=normalization_mean, 
        normalization_std=normalization_std,
        augmentation_strategy=augmentation_strategy,
        batch_size = batch_size,
        num_workers=num_workers,

    )

    data_module.setup()

    # Create the model
    model = ImageClassifier(
        model_name=model_name,
        num_classes=num_classes,
        drop_rate=drop_rate,
        pretrained=pretrained,
        lr_encoder=lr_encoder,
        lr_head=lr_head,
        optimizer=optimizer,
        criterion=criterion,
    )

    # Create a PyTorch Lightning trainer
    trainer = Trainer(
        max_epochs=num_epochs,
        logger=WandbLogger(name=wandb_run_name, project=wandb_project_name),
        log_every_n_steps=log_every_n_steps,
        callbacks=[ModelCheckpoint(
            dirpath="checkpoints/",
            filename=file_name + '-{epoch:02d}-{val_accuracy:.2f}',
            save_top_k=1,
            monitor="val_accuracy",
            mode="max"
        )],
        accelerator="auto"
    )

    # Start training
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

if __name__ == "__main__":
    cli.add_command(train)
    cli()
