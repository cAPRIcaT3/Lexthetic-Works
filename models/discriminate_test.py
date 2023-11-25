import torch
from discriminate import Discriminator

def test_discriminator():
    # Example usage to test the Discriminator
    discriminator_model = Discriminator(input_dim=3 * 7 * 7)  # Adjust input_dim based on your image size
    example_image = torch.randn(1, 3, 7, 7)  # Example image, adjust dimensions based on your model

    discriminator_output = discriminator_model(example_image)

    print("Discriminator Output:", discriminator_output)

if __name__ == "__main__":
    test_discriminator()
