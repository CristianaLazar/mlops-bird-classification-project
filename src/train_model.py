import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.data.data import ImageFolderClassificationModule
from src.models.model import ImageClassifier

@hydra.main(config_path="configs", config_name="default_config.yaml", version_base="1.1")
def train(config: DictConfig) -> None:
    """
    Train a model using the provided configuration.

    :param config: Configuration object provided by Hydra.

    """

    hparams = config.experiment

    # Load model and hyperparameter configs
    print(f"Configuration: \n {OmegaConf.to_yaml(config)}")
    seed_everything(hparams.reproducibility.seed, workers=True)

    # Set up data module
    data_module = ImageFolderClassificationModule(
        train_dir=to_absolute_path(hparams.data.train_dir),
        val_dir=to_absolute_path(hparams.data.val_dir),
        resize_dims=tuple(hparams.data.resize_dims),
        normalization_mean=hparams.data.normalization.mean,
        normalization_std=hparams.data.normalization.std,
        augmentation_strategy=hparams.data.augmentation_strategy,
        batch_size=hparams.training.batch_size,
        num_workers=hparams.training.num_workers
    )
    data_module.setup()

    # Create the model
    model = ImageClassifier(
        model_name=hparams.model.model_name,
        num_classes=hparams.model.num_classes,
        drop_rate=hparams.model.drop_rate,
        pretrained=hparams.model.pretrained,
        lr_encoder=hparams.optim.lr_encoder,
        lr_head=hparams.optim.lr_head,
        optimizer=hparams.optim.optimizer,
        criterion=hparams.optim.criterion,
    )

    # Create a PyTorch Lightning trainer
    trainer = Trainer(
        max_epochs=hparams.training.num_epochs,
        logger=WandbLogger(name=hparams.logging.wandb_run_name, project=hparams.logging.wandb_project_name),
        log_every_n_steps=hparams.logging.log_every_n_steps,
        callbacks=[ModelCheckpoint(
            dirpath="checkpoints/",
            filename=hparams.checkpoint.file_name + '-{epoch:02d}-{val_accuracy:.2f}',
            save_top_k=1,
            monitor="val_accuracy",
            mode="max"
        )],
        accelerator="auto", profiler="simple"
    )

    # Start training
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

if __name__ == "__main__":
    train()

