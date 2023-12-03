import os
import cv2
import numpy as np

def save_stylized_text(image_path, output_folder):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lower_color = np.array([0, 0, 100], dtype="uint8")
    upper_color = np.array([100, 100, 255], dtype="uint8")
    color_mask = cv2.inRange(image_rgb, lower_color, upper_color)

    combined_mask = cv2.bitwise_or(edges, color_mask)
    result = cv2.bitwise_and(image_rgb, image_rgb, mask=combined_mask)

    # Save the stylized text image as a PNG
    output_filename = os.path.join(output_folder, f'stylized_text_{os.path.basename(image_path)}')
    cv2.imwrite(output_filename, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

# Example usage:
image_folder = 'image_folder'
output_folder = 'output_stylized_text_images'
os.makedirs(output_folder, exist_ok=True)

# Process each image in the image folder
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_folder, filename)
        save_stylized_text(image_path, output_folder)
