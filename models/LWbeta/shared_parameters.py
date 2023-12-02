import torch
import torch.nn as nn
import torchvision.models as models  # Import models from torchvision
from config import Config

class SharedParameters(nn.Module):
    def __init__(self):
        super(SharedParameters, self).__init__()

        # Text processing branch
        self.text_branch = nn.Sequential(
            nn.Embedding(Config.vocab_size, Config.embedding_dim),
            nn.Linear(Config.embedding_dim, Config.text_dim),
            nn.ReLU()
        )

        # ShuffleNetV2-based image processing branch
        # Assuming shufflenet_model is ShuffleNetV2 with proper input channels
        self.shufflenet_model = models.shufflenet_v2_x1_0(pretrained=True)
        self.shufflenet_branch = nn.Sequential(
            self.shufflenet_model,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.latent_dim = Config.latent_dim
        self.text_dim = Config.text_dim
        self.shufflenet_feature_dim = Config.shufflenet_feature_dim


class Generator(nn.Module, SharedParameters):
    def __init__(self):
        super(Generator, self).__init__()
        SharedParameters.__init__(self)

        # Combined branch
        self.combined_branch = nn.Sequential(
            nn.Linear(self.latent_dim + self.text_dim + self.shufflenet_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, Config.output_dim),
            nn.Tanh()
        )

    def forward(self, z, text, image):
        text_embedding = self.text_branch(text)
        shufflenet_features = self.shufflenet_branch(image)
        
        # Concatenate text, image, and noise vectors
        combined_input = torch.cat([z, text_embedding, shufflenet_features], dim=1)
        return self.combined_branch(combined_input)

class Discriminator(nn.Module, SharedParameters):
    def __init__(self):
        super(Discriminator, self).__init__()
        SharedParameters.__init__(self)

        # Combined branch
        self.combined_branch = nn.Sequential(
            nn.Linear(self.text_dim + self.shufflenet_feature_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, text, image):
        text_embedding = self.text_branch(text)
        shufflenet_features = self.shufflenet_branch(image)
        
        # Concatenate text and image features
        combined_input = torch.cat([text_embedding, shufflenet_features], dim=1)
        return self.combined_branch(combined_input)
