import os
import glob
import torch
import random
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import time

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class ImageTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=(0, 359)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(size=(1242, 2208), scale=(0.8, 1.0)),
        ])

    def __call__(self, image):
        image = transforms.ToPILImage()(image)
        transformed_image = self.transform(image)
        transformed_image = transforms.ToTensor()(transformed_image)
        return transformed_image

def process_images_and_masks(image_folder, mask_folder, output_image_folder, output_mask_folder, startR, startS, verbose):
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_mask_folder):
        os.makedirs(output_mask_folder)

    image_files = sorted(glob.glob(os.path.join(image_folder, '*.jpg')) + glob.glob(os.path.join(image_folder, '*.png')))
    files_processed = 0

    transform = ImageTransform()
    r = startR
    s = startS

    total_time = 0

    for image_path in tqdm(image_files, desc="Processing Images and Masks"):
        start_time = time.time()

        filename = os.path.basename(image_path)
        image = Image.open(image_path).convert('RGBA')

        if filename.startswith('5r'):
            adjusted_filename = "mask_" + filename
            output_name = f"real_{r}.png"
            r += 1
        elif filename.startswith('real'):
            adjusted_filename = filename
            output_name = f"real_{r}.png"
            r += 1
        else:
            adjusted_filename = filename.replace("_defaultImage", "")
            output_name = f"simulated_{s}.png"
            s += 1

        image_tensor = transforms.ToTensor()(image)

        seed = random.randint(0, 2 ** 32 - 1)
        set_seed(seed)

        transformed_image = transform(image_tensor)

        output_image_path = os.path.join(output_image_folder, output_name)
        save_image(transformed_image, output_image_path)

        mask_subfolder = os.path.join(mask_folder, adjusted_filename.split('.')[0])

        if verbose:
            print(f"Processing masks for {filename}: original mask_subfolder = {mask_subfolder}")

        if os.path.exists(mask_subfolder):
            output_mask_subfolder = os.path.join(output_mask_folder, output_name.split('.')[0])
            if not os.path.exists(output_mask_subfolder):
                os.makedirs(output_mask_subfolder)
            if verbose:
                print(f"Output mask subfolder created: {output_mask_subfolder}")

            mask_files = sorted(glob.glob(os.path.join(mask_subfolder, '*.jpg')) + glob.glob(os.path.join(mask_subfolder, '*.png')))
            if verbose:
                print(f"Mask files found: {mask_files}")

            for mask_path in mask_files:
                mask_filename = os.path.basename(mask_path)
                mask = Image.open(mask_path).convert('L')
                mask_tensor = transforms.ToTensor()(mask)

                set_seed(seed)

                transformed_mask = transform(mask_tensor)

                output_mask_path = os.path.join(output_mask_subfolder, mask_filename)
                save_image(transformed_mask, output_mask_path)
                if verbose:
                    print(f"Saved transformed mask to: {output_mask_path}")
        else:
            if verbose:
                print(f"Mask subfolder does not exist: {mask_subfolder}")

        files_processed += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        if verbose:
            print(f"Time taken for {filename}: {elapsed_time:.2f} seconds")

    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time / files_processed:.2f} seconds")

    return r, s

def save_image(tensor, path):
    image = transforms.ToPILImage()(tensor.cpu())
    image.save(path)

def main():
    image_folder = 'DATASET_5R/IMAGES/TRAIN'
    mask_folder = 'TRAIN_LABELS_CLEANED'
    output_image_folder = 'output_images'
    output_mask_folder = 'output_masks'

    start1 = 0
    start2 = 0

    start1, start2 = process_images_and_masks(image_folder, mask_folder, output_image_folder, output_mask_folder, start1, start2, False)

if __name__ == '__main__':
    main()

