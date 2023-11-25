import torch
import os
import torch.nn as nn
import torch.optim as optim
from model import Generator
from discriminator import Discriminator
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import pandas as pd

# Training parameters
num_epochs = 100
batch_size = 32

# Path to your CSV file
csv_file_path = os.path.abspath("/content/Lexthetic-Works/data/data_sample.csv") #its weird bc I'm training on google colab and couldn't get the import straight

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_path)

# Calculate the number of batches based on the dataset size and batch size
num_samples = len(df)
num_batches = num_samples // batch_size

# Function to load and preprocess images from URLs
def load_images_from_data(data):
    images = []
    for url in data['image_url']:
        response = requests.get(url)

        # Check if the response status is OK (200)
        if response.status_code == 200:
            try:
                # Attempt to open the image
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img = transforms(img)
                images.append(img)
            except Exception as e:
                print(f"Error processing image from URL {url}: {e}")
        else:
            print(f"Failed to fetch image from URL {url}. Status code: {response.status_code}")

    return torch.stack(images)

generator = Generator(latent_dim=100, feature_dim=1024, output_dim=3)
discriminator = Discriminator(input_dim=3 * 64 * 64)

# Define loss function and optimizers
criterion = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
for epoch in range(num_epochs):
    for _ in range(num_batches):  # You need to define num_batches based on your dataset
        # 1. Train Discriminator
        discriminator.zero_grad()
        real_data = load_images_from_data(df.sample(batch_size, replace=True))  # Load random real images from your dataset
        real_labels = torch.ones(batch_size, 1)
        real_outputs = discriminator(real_data)
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
        generator_loss = criterion(discriminator_outputs, real_labels)

        generator_loss.backward()
        generator_optimizer.step()

    # Print training progress (optional)
    print(f'Epoch [{epoch+1}/{num_epochs}], Generator Loss: {generator_loss.item()}, Discriminator Loss: {discriminator_loss.item()}')

# Save models
torch.save(generator.state_dict(), 'models/LWbeta/generator_model.pth')
torch.save(discriminator.state_dict(), 'models/LWbeta/discriminator_model.pth')
