import torch
import pytest
import pytorch_lightning as pl
from src.models.model import ImageClassifier


# Define a fixture to create an instance of the ImageClassifier for testing
@pytest.fixture
def image_classifier():
    model = ImageClassifier(
        model_name="efficientnet_es.ra_in1k",
        num_classes=2,  # dummy classes
        drop_rate=0.5,
        pretrained=False,
        lr_encoder=0.001,
        lr_head=0.01,
        optimizer="Adam",
        criterion="cross_entropy",
    )
    return model


# Test the forward pass of the model
def test_forward_pass(image_classifier):
    batch_size = 8
    input_channels = 3
    image_height = 224
    image_width = 224

    # Create a sample input tensor (batch_size, channels, height, width)
    x = torch.randn((batch_size, input_channels, image_height, image_width))

    # Ensure forward pass works without errors
    output = image_classifier.forward(x)

    # Check output shape and type
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, image_classifier.model.num_classes)  # Check the output shape


# Test the training step of the model
def test_training_step(image_classifier):
    batch_size = 8
    input_channels = 3
    image_height = 224
    image_width = 224
    num_classes = 2

    # Create a sample input tensor (batch_size, channels, height, width)
    x = torch.randn((batch_size, input_channels, image_height, image_width))

    # Create a sample target tensor (batch_size,)
    y = torch.randint(0, num_classes, (batch_size,))

    # Pack the input and target tensors into a tuple (simulating a batch)
    batch = (x, y)

    # Ensure training step works without errors
    loss = image_classifier.training_step(batch)

    # Check loss type and value
    assert isinstance(loss, torch.Tensor)
    assert loss >= 0  # Loss should be a non-negative value


# Test the validation step of the model
def test_validation_step(image_classifier):
    batch_size = 8
    input_channels = 3
    image_height = 224
    image_width = 224
    num_classes = 2

    # Create a sample input tensor (batch_size, channels, height, width)
    x = torch.randn((batch_size, input_channels, image_height, image_width))

    # Create a sample target tensor (batch_size,)
    y = torch.randint(0, num_classes, (batch_size,))

    # Pack the input and target tensors into a tuple (simulating a batch)
    batch = (x, y)

    # Ensure validation step works without errors
    loss = image_classifier.validation_step(batch)

    # Check loss type and value
    assert isinstance(loss, torch.Tensor)
    assert loss >= 0  # Loss should be a non-negative value
