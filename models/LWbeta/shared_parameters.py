import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import shufflenet_v2_x1_0

class SharedModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.GELU()):
        super(SharedModule, self).__init__()

        layers = [nn.Linear(input_dim, hidden_dim), activation]

        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation])

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.shared_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.shared_layers(x)

class Generator(nn.Module):
    def __init__(self, latent_dim, feature_dim, output_dim, hidden_dim=1024, num_layers=3, upscale_factor=2):
        super(Generator, self).__init__()

        # Feature extractor dimensions
        self.feature_dim = feature_dim

        # Load the ShuffleNetV2 model
        self.shuffle_net = shufflenet_v2_x1_0(pretrained=True)

        # Generator architecture
        self.shared_layers = SharedModule(latent_dim + feature_dim, hidden_dim * upscale_factor, output_dim, num_layers)

    def shuffle_features(self, input_data):
        # Pass input through ShuffleNetV2 model
        shuffle_net_output = self.shuffle_net(input_data)

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

        # Concatenate ShuffleNet features and noise vector
        combined = torch.cat([shuffle_features, z], dim=1)

        #debugging statement
        print("Generator Combined Size:", combined.size()) #find out the size of the generator output to match the discriminator input

        # Pass through the shared layers
        return self.shared_layers(combined)

class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=1024, num_layers=3, downscale_factor=2):
        super(Discriminator, self).__init__()

        # Discriminator architecture
        self.shared_layers = SharedModule(input_dim, hidden_dim, output_dim, num_layers, activation=nn.LeakyReLU(0.2))

    def forward(self, images):
        # Flatten the images
        flattened_images = images.view(images.size(0), -1)

        print("Discriminator Input Size:", flattened_images.size())  # Debugging statement

        # Determine input_dim dynamically based on the size of flattened_images
        input_dim = flattened_images.size(1)

        # Pass through the shared layers
        return self.shared_layers(flattened_images)


if __name__ == "__main__":
    main()
