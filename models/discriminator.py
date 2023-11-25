import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=1024, num_layers=3, downscale_factor=2):
        super(Discriminator, self).__init__()

        # Discriminator architecture
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LeakyReLU(0.2))

        # Downscaling hidden layers
        for hidden_layer in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, int(hidden_dim / downscale_factor)))
            layers.append(nn.LeakyReLU(0.2))
            hidden_dim = int(hidden_dim / downscale_factor)

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())  # Sigmoid activation for binary classification

        self.discriminator = nn.Sequential(*layers)

    def forward(self, images):
        # Flatten the images
        flattened_images = images.view(images.size(0), -1)

        # Pass through the discriminator
        return self.discriminator(flattened_images)
