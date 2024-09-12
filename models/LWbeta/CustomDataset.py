import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms
import requests
from io import BytesIO

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Sample data for a given index
        sample = self.data.iloc[idx]

        # Load image from URL
        response = requests.get(sample['image_url'])
        image = Image.open(BytesIO(response.content))
        
        # Apply transformations to the image if specified
        if self.transform:
            image = self.transform(image)

        # Return a dictionary containing relevant information
        data_dict = {
            'label': sample['label'],
            'description': sample['description'],
            'color_palette': sample['color_palette'],
            'image': image,
            'keywords': sample['keywords']
        }

        return data_dict

# Example of using the CustomDataset class
# Assuming you have uploaded your CSV file using the Google Colab upload widget
#from google.colab import files
#uploaded = files.upload()
#dataset = CustomDataset(csv_file='data_sample.csv', transform=transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()]))
