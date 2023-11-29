import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from generator import Generator
from discriminator import Discriminator
from shared_parameters import SharedParameters  # Assuming you saved the shared parameters in a file named shared_parameters.py
from torchvision.models import shufflenet_v2_x1_0  # Import the ShuffleNetV2 model

# Define constants
latent_dim = 100
text_dim = 16
shufflenet_feature_dim = 1024
output_dim = 3
vocab_size = 10000  # Adjust based on the size of your vocabulary
embedding_dim = 128  # Adjust based on the desired size of text embeddings
batch_size = 64
epochs = 50
learning_rate = 0.0002

# Instantiate the shared parameters
shared_parameters = SharedParameters(latent_dim=latent_dim, text_dim=text_dim, shufflenet_feature_dim=shufflenet_feature_dim)

# Instantiate the models with shared parameters
generator = Generator(output_dim=output_dim, **shared_parameters.__dict__)
discriminator = Discriminator(**shared_parameters.__dict__)

# Define loss and optimizers
criterion = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Load ShuffleNetV2 model
shufflenet_model = shufflenet_v2_x1_0(pretrained=True)
shufflenet_model = nn.Sequential(*list(shufflenet_model.children())[:-1])

# Training loop
for epoch in range(epochs):
    for batch_idx, data in enumerate(custom_loader):  # Replace custom_loader with your actual DataLoader
        real_images = data['image']
        text_descriptions = data['description']

        # Forward pass
        discriminator_outputs = discriminator(text_descriptions, real_images)
        generator_loss = criterion(discriminator_outputs, real_labels)

        # Backward pass and optimization for generator
        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

        # Train discriminator
        discriminator_optimizer.zero_grad()
        real_labels = torch.ones(real_images.size(0), 1)
        fake_labels = torch.zeros(real_images.size(0), 1)

        real_outputs = discriminator(text_descriptions, real_images.view(-1, shufflenet_feature_dim, 1, 1))
        real_loss = criterion(real_outputs, real_labels)
        real_loss.backward()

        z = torch.randn(real_images.size(0), latent_dim)
        fake_images = generator(z, text_descriptions, real_images)
        fake_outputs = discriminator(text_descriptions, fake_images.detach())
        fake_loss = criterion(fake_outputs, fake_labels)
        fake_loss.backward()

        discriminator_optimizer.step()

        # Print training progress
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Batch {batch_idx}/{len(custom_loader)}, "
                  f"Generator Loss: {generator_loss.item():.4f}, "
                  f"Discriminator Loss: {real_loss.item() + fake_loss.item():.4f}")

# Save models
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
