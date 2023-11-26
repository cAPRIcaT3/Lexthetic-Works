import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from generator import Generator
from discriminator import Discriminator

# Define constants
latent_dim = 100
image_dim = 28 * 28  # Assuming MNIST-like images
batch_size = 64
epochs = 50
learning_rate = 0.0002

# Define transformations for the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Instantiate the models
generator = Generator(latent_dim=latent_dim, output_dim=image_dim)
discriminator = Discriminator(input_dim=image_dim)

# Define loss and optimizers
criterion = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for batch_idx, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        
        # Train discriminator
        discriminator_optimizer.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Discriminator on real data
        real_outputs = discriminator(real_images.view(-1, image_dim))
        real_loss = criterion(real_outputs, real_labels)
        real_loss.backward()

        # Discriminator on fake data
        z = torch.randn(batch_size, latent_dim)
        fake_images = generator(z)
        fake_outputs = discriminator(fake_images.detach())
        fake_loss = criterion(fake_outputs, fake_labels)
        fake_loss.backward()

        discriminator_optimizer.step()

        # Train generator
        generator_optimizer.zero_grad()
        z = torch.randn(batch_size, latent_dim)
        generated_images = generator(z)
        discriminator_outputs = discriminator(generated_images)
        generator_loss = criterion(discriminator_outputs, real_labels)
        generator_loss.backward()
        generator_optimizer.step()

        # Print training progress
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                  f"Generator Loss: {generator_loss.item():.4f}, "
                  f"Discriminator Loss: {real_loss.item() + fake_loss.item():.4f}")

# Save models
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
