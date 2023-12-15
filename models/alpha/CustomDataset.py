import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Sample data for a given index
        sample = self.data.iloc[idx]

        # Load pre-processed image
        image_path = os.path.join(self.image_folder, f"{sample['label']}_{idx}.png")
        image = Image.open(image_path)

        # Apply transformations to the image if specified
        if self.transform:
            image = self.transform(image)

        # Return a dictionary containing relevant information
        data_dict = {
            'label': sample['label'],
            'description': sample['description'],
            'color_palette': sample['color_palette'],
            'image': image,
            'keywords': sample['keywords'],
            'content': sample['content']  # Assuming 'content' is a field in your dataset
        }

        return data_dict

# Example of using the CustomDataset class
# Assuming you have a folder containing pre-processed images named 'preprocessed_images'
# and corresponding CSV file named 'data_sample.csv'
#dataset = CustomDataset(csv_file='data_sample.csv', image_folder='preprocessed_images', transform=transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()]))
