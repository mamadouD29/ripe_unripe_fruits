import os
import cv2
import numpy as np
import albumentations as A
from sklearn.model_selection import train_test_split
import shutil

# Augmentation Transformations | albumentations(p = probability)
augmentations = A.Compose([
    A.GaussNoise(var_limit=(10, 50), p=0.5),  # Adds slight random noise
    A.Rotate(limit=30, p=0.7),  # Rotates between -30° and 30°
    A.HorizontalFlip(p=0.5),  # Flips image horizontally
    A.RandomBrightnessContrast(p=0.6)  # Adjusts brightness & contrast
])

# File path directory 
base_dir = r"C:\Users\timne\OneDrive\Desktop\CS898 Image Analysis and Computer Vision\CS898 Project\Datasets" # Directory 
output_dir = r"C:\Users\timne\OneDrive\Desktop\CS898 Image Analysis and Computer Vision\CS898 Project\TTS" # Directory
os.makedirs(output_dir, exist_ok=True)

fruits = ["Banana", "Mango", "Rambutan", "Strawberry", "Tomato"]
categories = ["Ripe", "Unripe"]

# Function to augment an image
def augment_image(image, count):
    augmented_images = [] # Augmented images list 
    while len(augmented_images) < count:
        augmented = augmentations(image=image)['image'] # Apply augmentation
        augmented_images.append(augmented) # append to augmented images list 
    return augmented_images

# Loop through each fruit and category in file path 
for fruit in fruits:
    for category in categories:
        input_folder = os.path.join(base_dir, fruit, category)
        output_folder = os.path.join(output_dir, fruit, category)
        os.makedirs(output_folder, exist_ok=True) # Create folder if it does not exist

        images = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
        
        if len(images) != 20:
            print(f"Error: {input_folder} only has {len(images)}")
            continue
        
        augmented_images = []
        
        # Load original images
        for img_name in images:
            img_path = os.path.join(input_folder, img_name)
            image = cv2.imread(img_path)
            
            # Error-handling if image cannot load
            if image is None:
                print(f"Error loading image: {img_path}")
                continue  # Skip corrupt/missing images

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert for correct color rendering
            
            
            augmented_images.append(image) # Add originial image to list 
            
            # Ensure image dont exceed 120 
            augmented_images.extend(augment_image(image, count=5 if len(augmented_images) + 5 <= 120 else 120 - len(augmented_images)))

        # Save augmented images (ensure 120 images total)
        for i, img in enumerate(augmented_images[:120]):
            output_path = os.path.join(output_folder, f"{category}_{i+1}.jpg") # Output path directory
            cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Convert back to BGR

        # Message report for fruit/category
        print(f"Augmented {fruit}/{category} to {len(augmented_images[:120])} images.")

# File path/directory for Train/Test datasets
train_dir = os.path.join(output_dir, "train_data")
test_dir = os.path.join(output_dir, "test_data")

for fruit in fruits: # loop through fruits
    for category in categories: # loop through categories
        src_folder = os.path.join(output_dir, fruit, category) # source folder of img
        images = [f for f in os.listdir(src_folder) if f.endswith(('.jpg', '.png'))]

        train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42) # split imgs (80% - training) (20% - test)

        # Path for dataset categories
        train_category_path = os.path.join(train_dir, fruit, category)
        test_category_path = os.path.join(test_dir, fruit, category)

        # Create file paths if they don't exist
        os.makedirs(train_category_path, exist_ok=True)
        os.makedirs(test_category_path, exist_ok=True)

        # Checking if files exists in source folder before copying into TRAINING
        for img in train_imgs:
            if os.path.exists(os.path.join(src_folder, img)):
                shutil.copy(os.path.join(src_folder, img), os.path.join(train_category_path, img))
            else:
                print(f"Warning: File {img} not found in {src_folder}")

        # Checking if files exists in source folder before copying into TEST 
        for img in test_imgs:
            if os.path.exists(os.path.join(src_folder, img)):
                shutil.copy(os.path.join(src_folder, img), os.path.join(test_category_path, img))
            else:
                print(f"Warning: File {img} not found in {src_folder}")

        print(f"Split {fruit}/{category}: {len(train_imgs)} train, {len(test_imgs)} test.")

print("Dataset augmentation and train-test split performed.")
