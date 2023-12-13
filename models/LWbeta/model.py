import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import shufflenet_v2_x1_0  # Add this import statement

class Generator(nn.Module):
    def __init__(self, latent_dim, feature_dim, output_dim, hidden_dim=1024, num_layers=3, upscale_factor=2):
        super(Generator, self).__init__()

        # Feature extractor dimensions
        self.feature_dim = feature_dim

        # Load the ShuffleNetV2 model
        self.shuffle_net = shufflenet_v2_x1_0(weights='imagenet1k')

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

    def shuffle_features(self, input_data):
        # Pass input through ShuffleNetV2 model
        shuffle_net_output = self.shuffle_net(input_data)
    
        # Print the shape of the output tensor
        print("ShuffleNet Output Shape:", shuffle_net_output.shape)
    
        # Check if the output tensor has at least three dimensions
        while len(shuffle_net_output.shape) < 3:
            # Add an extra dimension
            shuffle_net_output = shuffle_net_output.unsqueeze(2)
    
        # Use Global Average Pooling (GAP) to reduce spatial dimensions
        shuffle_features = F.adaptive_avg_pool2d(shuffle_net_output, (1, 1))
    
        return shuffle_features.view(shuffle_features.size(0), -1)


    def forward(self, features, z):
        # Extract ShuffleNet features
        shuffle_features = self.shuffle_features(features)

        # Print shapes for debugging
        print("ShuffleNet Features Shape:", shuffle_features.shape)
        print("Z Shape:", z.shape)

        # Concatenate ShuffleNet features and noise vector
        combined = torch.cat([shuffle_features, z], dim=1)

        # Print combined shape
        print("Combined Shape:", combined.shape)

        # Pass through the generator
        return self.generator(combined)
