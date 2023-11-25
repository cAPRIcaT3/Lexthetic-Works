import torch
from discriminator import Discriminator
from torchvision import transforms

# Function to preprocess images
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Adjust dimensions based on your model
    transforms.ToTensor(),
])

def test_discriminator():
    # Example usage to test the Discriminator
    discriminator_model = Discriminator(input_dim=3 * 64 * 64)  # Adjust input_dim based on your image size
    example_image = torch.randn(1, 3, 64, 64)  # Example image, adjust dimensions based on your model

    discriminator_output = discriminator_model(example_image)

    print("Discriminator Output:", discriminator_output)

if __name__ == "__main__":
    test_discriminator()
