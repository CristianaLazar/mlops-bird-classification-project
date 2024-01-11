import os
import pytest
from omegaconf import OmegaConf, DictConfig
from unittest.mock import patch, MagicMock
from hydra.utils import to_absolute_path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.data.data import ImageFolderClassificationModule
from src.models.model import ImageClassifier
from src.train_model import train  # Updated import


@pytest.fixture
def config(tmp_path):
    config_file = tmp_path / "default_config.yaml"
    # Define the configuration for exp1
    exp1_config = """
        hydra:
            run:
                dir: .

        reproducibility:
            seed: 47

        model:
            model_name: 'efficientnet_b0'
            num_classes: 525
            drop_rate: 0.5
            pretrained: true

        optim:
            lr_encoder: 3e-6
            lr_head: 3e-4
            optimizer: 'AdamW'
            criterion: 'cross_entropy'

        training:
            batch_size: 32
            num_epochs: 100
            num_workers: 4

        logging:
            log_every_n_steps: 10
            wandb_run_name: 'training_run'
            wandb_project_name: 'mlops-bird-classification'

        checkpoint:
            file_name: 'model'

        data:
            train_dir: 'data/raw/train'
            val_dir: 'data/raw/validation'
            resize_dims: [224, 224]
            augmentation_strategy: 'moderate'
            normalization:
                mean:
                    - 0.473
                    - 0.468
                    - 0.395
                std:
                    - 0.240
                    - 0.234
                    - 0.255
    """

    # Create a DictConfig for exp1
    exp1_dict = OmegaConf.create(exp1_config)
    exp1 = DictConfig(exp1_dict)

    # Create the main config with exp1
    config_content = DictConfig({"experiment": exp1})
    return config_file, config_content


@patch('src.data.data.ImageFolderClassificationModule.setup')
@patch('src.models.model.ImageClassifier')
@patch('pytorch_lightning.Trainer')
@patch('src.train_model.Trainer.fit')  # Updated import
@patch('hydra.utils.to_absolute_path')
@patch('hydra.main')
def test_train(
    mock_hydra_main,
    mock_to_absolute_path,
    mock_trainer_fit,
    mock_trainer,
    mock_image_classifier,
    mock_data_module_setup,
    config
):
    config_path, config_content = config
    # Mock Hydra, Trainer, to_absolute_path, and fit method
    mock_hydra_main.return_value = MagicMock()
    mock_trainer.return_value = MagicMock(spec=Trainer)
    mock_trainer_fit.return_value = None
    mock_to_absolute_path.side_effect = lambda x: x  # Mock to_absolute_path to return the same path

    # Mock ImageClassifier configure_optimizers method
    mock_image_classifier.return_value.configure_optimizers.return_value = MagicMock()

    # Set the original working directory
    original_cwd = os.getcwd()
    os.chdir(os.path.dirname(config_path))

    try:
        # Call the train function with the given config dict
        train(config_content)
    finally:
        # Reset the working directory to the original value
        os.chdir(original_cwd)

    # Assert that the required components were created with the correct configurations
    mock_hydra_main.assert_called_once_with(config_path="configs", config_name="default_config.yaml", version_base="1.1")
    mock_to_absolute_path.assert_called_once_with('data/raw/train')
    mock_data_module_setup.assert_called_once_with()
    mock_image_classifier.assert_called_once_with(
        model_name='efficientnet_b0',
        num_classes=525,
        drop_rate=0.5,
        pretrained=True,
        lr_encoder=3e-6,
        lr_head=3e-4,
        optimizer='AdamW',
        criterion='cross_entropy'
    )

    # Assert optimizer configuration
    mock_image_classifier.return_value.configure_optimizers.assert_called_once()
    optimizer_args = mock_image_classifier.return_value.configure_optimizers.call_args[1]
    assert optimizer_args['params'][0]['lr'] == 3e-6  # Check lr_encoder
    assert optimizer_args['params'][1]['lr'] == 3e-4  # Check lr_head

    mock_trainer.assert_called_once_with(
        max_epochs=100,
        logger=WandbLogger(name='training_run', project='mlops-bird-classification'),
        log_every_n_steps=10,
        callbacks=[ModelCheckpoint(
            dirpath="checkpoints/",
            filename='model-{epoch:02d}-{val_accuracy:.2f}',
            save_top_k=1,
            monitor="val_accuracy",
            mode="max"
        )],
        accelerator="auto"
    )
    mock_trainer_fit.assert_called_once_with(
        mock_image_classifier.return_value,
        MagicMock().train_dataloader.return_value,
        MagicMock().val_dataloader.return_value
    )
