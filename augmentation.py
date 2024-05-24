import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class SegmentationTransform:
    def __init__(self):
        # Define the transformations you want to apply
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=20),  # Random rotation within 20 degrees
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Random resized crop
        ])

    def __call__(self, image, masks):
        # Convert image and masks to PIL Images
        image = transforms.ToPILImage()(image)
        masks = [transforms.ToPILImage()(mask) for mask in masks]

        # Ensure both image and masks receive the same random transformations
        seed = np.random.randint(2147483647)  # Get a random seed
        torch.manual_seed(seed)  # Set the seed for torch random transformations
        image = self.transform(image)  # Apply the transformation to the image

        torch.manual_seed(seed)  # Reset the seed for torch random transformations
        transformed_masks = [self.transform(mask) for mask in masks]  # Apply the same transformation to each mask

        # Convert back to tensor
        transformed_image = transforms.ToTensor()(image)
        transformed_masks = [transforms.ToTensor()(mask) for mask in transformed_masks]

        return transformed_image, transformed_masks

# Example usage
# Load an example image and its corresponding masks as tensors
img = Image.open('DATASET_5R/LABELS/MASK/TRAIN/mask_5r_setup_6/5r_setup_6.png').convert('RGB')
mask1 = Image.open('DATASET_5R/LABELS/MASK/TRAIN/mask_5r_setup_6/mask_5r_setup_6_0.png').convert('L')  # Assuming masks are single-channel
mask2 = Image.open('DATASET_5R/LABELS/MASK/TRAIN/mask_5r_setup_6/mask_5r_setup_6_1.png').convert('L')  # Assuming masks are single-channel


img_tensor = transforms.ToTensor()(img)
mask1_tensor = transforms.ToTensor()(mask1)
mask2_tensor = transforms.ToTensor()(mask2)

masks_tensors = [mask1_tensor, mask2_tensor]

# Create an instance of the custom transformation
segmentation_transform = SegmentationTransform()

# Apply the custom transformation to the image and masks
transformed_img_tensor, transformed_masks_tensors = segmentation_transform(img_tensor, masks_tensors)

# Plot the original and transformed images and masks
fig, axs = plt.subplots(3, 2, figsize=(10, 15))
axs[0, 0].imshow(img)
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

axs[1, 0].imshow(mask1, cmap='gray')
axs[1, 0].set_title('Original Mask 1')
axs[1, 0].axis('off')

axs[2, 0].imshow(mask2, cmap='gray')
axs[2, 0].set_title('Original Mask 2')
axs[2, 0].axis('off')

axs[0, 1].imshow(transformed_img_tensor.permute(1, 2, 0))
axs[0, 1].set_title('Transformed Image')
axs[0, 1].axis('off')

axs[1, 1].imshow(transformed_masks_tensors[0].permute(1, 2, 0), cmap='gray')
axs[1, 1].set_title('Transformed Mask 1')
axs[1, 1].axis('off')

axs[2, 1].imshow(transformed_masks_tensors[1].permute(1, 2, 0), cmap='gray')
axs[2, 1].set_title('Transformed Mask 2')
axs[2, 1].axis('off')

plt.show()
