import torch
import torch.nn as nn
import torch.optim as optim
from model import Generator
from discriminator import Discriminator

# Initialize generator and discriminator
generator = Generator(latent_dim=100, feature_dim=1024, output_dim=3)
discriminator = Discriminator(input_dim=3 * 7 * 7)

# Define loss function and optimizers
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training parameters
num_epochs = 100
batch_size = 32

# Training loop
for epoch in range(num_epochs):
    for _ in range(num_batches):  # You need to define num_batches based on your dataset
        # 1. Train Discriminator
        discriminator.zero_grad()
        real_images = # Load real images from your dataset
        real_labels = torch.ones(batch_size, 1)
        real_outputs = discriminator(real_images)
        real_loss = criterion(real_outputs, real_labels)

        noise_vector = torch.randn(batch_size, 100)
        generated_images = generator(shuffle_features, noise_vector)
        fake_labels = torch.zeros(batch_size, 1)
        fake_outputs = discriminator(generated_images.detach())  # Detach to avoid generator gradients
        fake_loss = criterion(fake_outputs, fake_labels)

        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # 2. Train Generator
        generator.zero_grad()
        generated_images = generator(shuffle_features, noise_vector)
        discriminator_outputs = discriminator(generated_images)
        generator_loss = criterion(discriminator_outputs, real_labels)  # Fool the discriminator

        generator_loss.backward()
        generator_optimizer.step()

    # Print training progress (optional)
    print(f'Epoch [{epoch+1}/{num_epochs}], Generator Loss: {generator_loss.item()}, Discriminator Loss: {discriminator_loss.item()}')

# Save models
torch.save(generator.state_dict(), 'generator_model.pth')
torch.save(discriminator.state_dict(), 'discriminator_model.pth')
