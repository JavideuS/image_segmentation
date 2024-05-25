import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os

class ImageTransform:
    def __init__(self):
        # Define the transformations with a wide rotation range
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=(0, 359)),  # Rotation range from 0 to 359 degrees
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Random resized crop
        ])

    def __call__(self, image):
        # Convert image to PIL Image
        image = transforms.ToPILImage()(image)

        # Apply the transformation
        transformed_image = self.transform(image)

        # Convert back to tensor
        transformed_image = transforms.ToTensor()(transformed_image)

        return transformed_image

def process_folder(input_folder, output_folder, transform):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    files_processed = 0  # Counter to track processed files
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            files_processed += 1  # Increment the counter for each valid file processed

            try:
                # Load the image
                img_path = os.path.join(input_folder, filename)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transforms.ToTensor()(img)

                # Apply the custom transformation to the image
                transformed_img_tensor = transform(img_tensor)

                # Convert tensor back to PIL image
                transformed_img = transforms.ToPILImage()(transformed_img_tensor)

                # Save the transformed image
                transformed_img.save(os.path.join(output_folder, f'transformed_{filename}'))

                print(f"Processed and saved: {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    print(f"Total files processed: {files_processed}")  # Print total files processed

# Example usage
input_folder = 'DATASET_5R/IMAGES/TRAIN'
output_folder = 'xd'

# Create an instance of the custom transformation
image_transform = ImageTransform()

# Process all images in the folder
process_folder(input_folder, output_folder, image_transform)
