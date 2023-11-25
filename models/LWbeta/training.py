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
csv_file_path = os.path.abspath("/content/Lexthetic-Works/data/data_sample.csv")

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_path)

# Calculate the number of batches based on the dataset size and batch size
num_samples = len(df)
num_batches = num_samples // batch_size

# Function to load and preprocess images from URLs
def load_images_from_data(data):
    transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to (224, 224) for ShuffleNetV2
            transforms.ToTensor()  # Convert the image to a PyTorch tensor
        ])
    images = []
    for url in data['image_url']:
        response = requests.get(url)

        # Check if the response status is OK (200)
        if response.status_code == 200:
            try:
                # Attempt to open the image
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img = transform(img)
                images.append(img)
            except Exception as e:
                print(f"Error processing image from URL {url}: {e}")
        else:
            print(f"Failed to fetch image from URL {url}. Status code: {response.status_code}")

    return torch.stack(images)

# Instantiate the Generator and Discriminator models
generator = Generator(latent_dim=100, feature_dim=1024, output_dim=3)
discriminator = Discriminator(input_dim=3 * 64 * 64)

# Define loss function and optimizers
criterion = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Calculate the adjusted batch size to ensure it's a multiple of batch size
adjusted_batch_size = batch_size - (num_samples % batch_size)

for epoch in range(num_epochs):
    # If there's a difference in batch sizes, add records
    if adjusted_batch_size != batch_size:
        additional_samples = batch_size - adjusted_batch_size
        additional_data = df.sample(additional_samples, replace=True)
        df = pd.concat([df, additional_data], ignore_index=True)

    for _ in range(num_batches):
        # 1. Train Discriminator
        discriminator.zero_grad()

        # Load random real images from your dataset with the adjusted batch size
        sampled_data = df.sample(batch_size, replace=True)  # Use original batch_size here
        real_data = load_images_from_data(sampled_data)

        real_labels = torch.ones(batch_size, 1)  # Use original batch_size here

        real_outputs = discriminator(real_data)
        real_loss = criterion(real_outputs, real_labels)

        if batch_size > 0:  # Ensure batch size is greater than zero
            noise_vector = torch.randn(batch_size, 100)
            shuffle_features_real = generator.shuffle_features(real_data)
            generated_images = generator(shuffle_features_real, noise_vector)

            fake_labels = torch.zeros(batch_size, 1)
            fake_outputs = discriminator(generated_images.detach())
            fake_loss = criterion(fake_outputs, fake_labels)

            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # 2. Train Generator
            generator.zero_grad()

            shuffle_features_fake = generator.shuffle_features(generated_images)
            discriminator_outputs = discriminator(shuffle_features_fake)
            generator_loss = criterion(discriminator_outputs, real_labels)

            generator_loss.backward()
            generator_optimizer.step()

            # Print training progress (optional)
            print(f'Epoch [{epoch+1}/{num_epochs}], Generator Loss: {generator_loss.item()}, Discriminator Loss: {discriminator_loss.item()}')

    # Update adjusted batch size based on the actual size of the last batch
    adjusted_batch_size = len(sampled_data)


# Save models
torch.save(generator.state_dict(), 'models/LWbeta/generator_model.pth')
torch.save(discriminator.state_dict(), 'models/LWbeta/discriminator_model.pth')
