import os
import pandas as pd
import requests
from urllib.parse import urlparse

# Load the dataset
df = pd.read_csv('data_sample.csv')

# Create a folder to store the images
output_folder = 'image_folder'
os.makedirs(output_folder, exist_ok=True)

# Iterate through the rows of the dataset
for index, row in df.iterrows():
    # Extract relevant information
    label = row['label']
    image_url = row['image_url']

    # Download the image
    response = requests.get(image_url)
    if response.status_code == 200:
        # Extract the filename from the URL
        filename = os.path.join(output_folder, f'{label}_{os.path.basename(urlparse(image_url).path)}')

        # Save the image
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Image for '{label}' downloaded successfully.")
    else:
        print(f"Failed to download image for '{label}'. Status code: {response.status_code}")
