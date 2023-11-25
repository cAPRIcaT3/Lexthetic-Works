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
            transforms.Resize((64, 64)),  # Resize the image to (64, 64)
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

generator = Generator(latent_dim=100, feature_dim=1024, output_dim=3)
discriminator = Discriminator(input_dim=3 * 64 * 64)

# Define loss function and optimizers
criterion = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(num_epochs):
    for _ in range(num_batches):
        # 1. Train Discriminator
        discriminator.zero_grad()
        
        # Load random real images from your dataset
        real_data = load_images_from_data(df.sample(batch_size, replace=True))
        
        # Adjust the size of real_labels to match the current batch size
        current_batch_size = real_data.size(0)
        real_labels = torch.ones(current_batch_size, 1)
        
        real_outputs = discriminator(real_data)
        real_loss = criterion(real_outputs, real_labels)

        noise_vector = torch.randn(current_batch_size, 100)
        shuffle_features_real = generator.shuffle_features(real_data)
        generated_images = generator(shuffle_features_real, noise_vector)
        
        fake_labels = torch.zeros(current_batch_size, 1)
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

# Save models
torch.save(generator.state_dict(), 'models/LWbeta/generator_model.pth')
torch.save(discriminator.state_dict(), 'models/LWbeta/discriminator_model.pth')
