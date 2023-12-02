# model_config.py
import torch
import torch.nn as nn
import torchvision.models as models
from config import Config

class ModelConfig:
    """A class to encapsulate the configuration parameters for the model."""

    def __init__(self):
        """Initialize the ModelConfig instance.

        Initializes model parameters, text processing parameters, and training parameters.
        Also, initializes ShuffleNetV2 to determine input channels.
        """

        # Model parameters
        self.latent_dim = Config.latent_dim  # Dimension of the latent space for noise input to the generator
        self.text_dim = Config.text_dim  # Dimension of the text embedding input to the generator
        self.shufflenet_feature_dim = Config.shufflenet_feature_dim  # Dimension of ShuffleNetV2 features
        self.output_dim = Config.output_dim  # Dimension of the generator output

        # Text processing parameters
        self.vocab_size = Config.vocab_size  # Size of the vocabulary for text embedding
        self.embedding_dim = Config.embedding_dim  # Dimension of the text embedding

        # Training parameters
        self.learning_rate = Config.learning_rate  # Learning rate for the optimizer
        self.beta1 = Config.beta1  # Beta1 parameter for the Adam optimizer
        self.beta2 = Config.beta2  # Beta2 parameter for the Adam optimizer
        self.batch_size = Config.batch_size  # Batch size for training data

        # Initialize ShuffleNetV2 model to determine input channels
        self.shufflenet_model = models.shufflenet_v2_x1_0(pretrained=True)
        self.shufflenet_input_channels = self.shufflenet_model.conv1[0].in_channels  # Input channels for ShuffleNetV2

# Usage example:
# from model_config import ModelConfig
# config = ModelConfig()
# print(f"ShuffleNetV2 Input Channels: {config.shufflenet_input_channels}")
