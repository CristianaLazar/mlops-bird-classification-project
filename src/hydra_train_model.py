from typing import Any, Dict

import hydra
from hydra.utils import to_absolute_path
import yaml
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data.data import ImageFolderClassificationModule
from models.model import ImageClassifier



def load_yaml_config(filepath: str) -> Dict[str, Any]:
    """
    Function to load config from YAML file.
    
    :param filepath: Path to the YAML file.
    :return: Dictionary with the loaded configuration.
    """
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)


@hydra.main(config_path="configs", config_name="default_config.yaml", version_base="1.1")
def train(config: DictConfig) -> None:
    """
    Train a model using the provided configuration.

    :param config: Configuration object provided by Hydra.
    """
    # Load model and hyperparameter configs
    print(f"Configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment
    seed_everything(hparams["seed"], workers=True)

    # Load normalization parameters
    data_config = load_yaml_config(to_absolute_path("data/train_data_config.yaml"))
    normalization_mean = data_config['mean']
    normalization_std = data_config['std']

    # Set up data module
    data_module = ImageFolderClassificationModule(
        train_dir = to_absolute_path(hparams["train_dir"]), 
        val_dir = to_absolute_path(hparams["val_dir"]),
        resize_dims = tuple(hparams["resize_dims"]), 
        normalization_mean = normalization_mean, 
        normalization_std = normalization_std,
        augmentation_strategy = hparams["augmentation_strategy"],
        batch_size = hparams["batch_size"],
        num_workers = hparams["num_workers"],
    )
    data_module.setup()

    # Create the model
    model = ImageClassifier(
        model_name=hparams["model_name"],
        num_classes=hparams["num_classes"],
        drop_rate=hparams["drop_rate"],
        pretrained=hparams["pretrained"],
        lr_encoder=hparams["lr_encoder"],
        lr_head=hparams["lr_head"],
        optimizer=hparams["optimizer"],
        criterion=hparams["criterion"],
    )

    # Create a PyTorch Lightning trainer
    trainer = Trainer(
        max_epochs=hparams["num_epochs"],
        logger=WandbLogger(name=hparams["wandb_run_name"], project=hparams["wandb_project_name"]),
        log_every_n_steps=hparams["log_every_n_steps"],
        callbacks=[ModelCheckpoint(
            dirpath="checkpoints/",
            filename=hparams["file_name"] + '-{epoch:02d}-{val_accuracy:.2f}',
            save_top_k=1,
            monitor="val_accuracy",
            mode="max"
        )],
        accelerator="auto"
    )

    # Start training
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

if __name__ == "__main__":
    train()

    # To run file with specified config: python src/hydra_train_model.py experiment=exp1