import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_0

class Generator(nn.Module):
    def __init__(self, latent_dim, feature_dim, output_dim, hidden_dim=1024, num_layers=3, upscale_factor=2):
        super(Generator, self).__init__()

        # Feature extractor dimensions
        self.feature_dim = feature_dim

        # Load the ShuffleNetV2 model
        self.shuffle_net = shufflenet_v2_x1_0(pretrained=True)

        # Generator architecture
        layers = []
        layers.append(nn.Linear(latent_dim + feature_dim, hidden_dim))
        layers.append(nn.GELU())

        # Upscaling hidden layers
        for hidden_layer in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim * upscale_factor))
            layers.append(nn.GELU())
            hidden_dim *= upscale_factor

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.generator = nn.Sequential(*layers)

    def shuffle_features(self, x):
        return self.shuffle_net.features(x)

    def forward(self, features, z):
        # Extract ShuffleNet features
        shuffle_features = self.shuffle_features(features)

        # Reshape ShuffleNet features
        shuffle_features = shuffle_features.view(shuffle_features.size(0), -1)

        # Concatenate ShuffleNet features and noise vector
        combined = torch.cat([shuffle_features, z], dim=1)

        # Pass through the generator
        return self.generator(combined)
