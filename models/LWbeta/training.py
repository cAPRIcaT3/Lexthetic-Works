# Import necessary libraries
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from CustomDataset import CustomDataset
from model import Generator, Discriminator
from config import Config
from google.colab import files

# Upload the dataset CSV file
uploaded = files.upload()

# Initialize the GAN model, discriminator, and generator
generator = Generator()
discriminator = Discriminator()

# Define the loss function (Binary Cross Entropy for GANs)
criterion = nn.BCELoss()

# Set up optimizers for the generator and discriminator
optimizer_G = Adam(generator.parameters(), lr=Config.learning_rate, betas=(Config.beta1, Config.beta2))
optimizer_D = Adam(discriminator.parameters(), lr=Config.learning_rate, betas=(Config.beta1, Config.beta2))

# Set up transformations for the dataset
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

# Create an instance of the CustomDataset class
dataset = CustomDataset(csv_file='data_sample.csv', transform=transform)

# Create a DataLoader for the dataset
dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)

# Training loop
num_epochs = 50  # Adjust the number of epochs as needed

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        # Get real samples
        real_text, real_image = data['description'], data['image']

        # Generate random noise
        z = torch.randn(real_text.size(0), Config.latent_dim)

        # Generate fake samples
        fake_output_with_text = generator(z, real_text, real_image)

        # Train the discriminator
        discriminator.zero_grad()

        # Real samples
        real_labels = torch.ones(real_text.size(0), 1)
        real_output = discriminator(real_text, real_image, real_labels)
        real_loss = criterion(real_output, real_labels)

        # Fake samples
        fake_labels = torch.zeros(real_text.size(0), 1)
        fake_output = discriminator(real_text, real_image, fake_output_with_text.detach())
        fake_loss = criterion(fake_output, fake_labels)

        # Total discriminator loss
        total_discriminator_loss = real_loss + fake_loss
        total_discriminator_loss.backward()
        optimizer_D.step()

        # Train the generator
        generator.zero_grad()

        # Adversarial loss
        adversarial_loss = criterion(fake_output, real_labels)

        # Reconstruction loss (optional, depending on your objectives)
        # recon_loss = your_custom_loss_function(fake_output_with_text, real_text)

        # Total generator loss
        total_generator_loss = adversarial_loss  # + recon_loss if using reconstruction loss
        total_generator_loss.backward()
        optimizer_G.step()

        # Print training information
        if i % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], "
                f"Discriminator Loss: {total_discriminator_loss.item()}, "
                f"Generator Loss: {total_generator_loss.item()}"
            )
