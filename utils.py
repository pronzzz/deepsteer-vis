import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import csv
from math import ceil
from random import shuffle

def preprocess_image(img):
    """
    Preprocess the image for the model.
    1. Crop: Remove top 70 pixels (sky) and bottom 25 pixels (hood).
    2. Resize: Resize to 200x66 (NVIDIA standard).
    3. Color Space: Convert RGB to YUV.
    """
    # Crop top 70 and bottom 25
    new_img = img[70:-25, :, :]
    
    # Resize to 200x66
    new_img = cv2.resize(new_img, (200, 66), interpolation=cv2.INTER_AREA)
    
    # Convert to YUV
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2YUV)
    
    return new_img

def random_augment(image, measurement):
    """
    Randomly augment the image and steering angle.
    1. Random Flip: Flip image and invert steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        measurement = -measurement
    return image, measurement

def get_image(source_path):
    """
    Load image from path.
    """
    # Adjust path if needed (Udacity sim data often has absolute paths or relative to a specific folder)
    # Check if file exists, if not, try prepending a common data directory if needed.
    # For this implementation, we assume the csv contains relative or correct absolute paths.
    if not os.path.exists(source_path):
        # Fallback for common issue where path is just filename
        filename = source_path.split('/')[-1]
        # Trying to find where the data might be. Assuming 'data/IMG/'
        if os.path.exists(os.path.join('data/IMG', filename)):
            return cv2.imread(os.path.join('data/IMG', filename))
            
    # Read in BGR (OpenCV default)
    image = cv2.imread(source_path)
    # Convert to RGB (Sim drives in RGB, we want consistent pipeline)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def generator(samples, batch_size=32):
    """
    Generator to yield batches of images and steering angles.
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            
            correction = 0.2 # Steering correction for side cameras

            for batch_sample in batch_samples:
                # We have 3 images: center, left, right
                # Center
                center_path = batch_sample[0]
                center_angle = float(batch_sample[3])
                
                # Left
                left_path = batch_sample[1]
                left_angle = center_angle + correction
                
                # Right
                right_path = batch_sample[2]
                right_angle = center_angle - correction
                
                # Choose one of the 3 randomly to avoid bias, or add all 3?
                # Stacking all 3 increases dataset size 3x.
                # To keep batch size consistent, let's just add all 3 if we want to maximize data, 
                # or randomly sample.
                # Typically for behavior cloning, adding all is better.
                # However, to respect the batch_size exactly, we can just pick one random camera per sample
                # OR we count each "line" in CSV as 3 samples.
                # Let's add all 3, but note that the actual batch output size will be 3 * batch_size of CSV lines.
                # The generator signature usually expects batch_size to be the NUMBER OF IMAGES yielded.
                # So we should process `batch_size // 3` lines per yield roughly.
                
                # SIMPLIFICATION: Using all 3 images per CSV line.
                
                # Load images
                # Note: paths in csv might need adjustment (handled in get_image partial logic or manually here)
                # Assuming batch_sample entries are paths
                
                img_center = get_image(center_path.strip())
                img_left = get_image(left_path.strip())
                img_right = get_image(right_path.strip())
                
                if img_center is not None:
                     images.append(preprocess_image(img_center))
                     angles.append(center_angle)
                if img_left is not None:
                     images.append(preprocess_image(img_left))
                     angles.append(left_angle)
                if img_right is not None:
                     images.append(preprocess_image(img_right))
                     angles.append(right_angle)

            # Convert to numpy arrays
            X_train = np.array(images)
            y_train = np.array(angles)
            
            # Additional Augmentation on the Batch
            augmented_images, augmented_angles = [], []
            for image, angle in zip(X_train, y_train):
                aug_image, aug_angle = random_augment(image, angle)
                augmented_images.append(aug_image)
                augmented_angles.append(aug_angle)
            
            X_train_shuffled, y_train_shuffled = sklearn.utils.shuffle(np.array(augmented_images), np.array(augmented_angles))
            yield (X_train_shuffled, y_train_shuffled)

