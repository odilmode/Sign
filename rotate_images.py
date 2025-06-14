import cv2
import numpy as np
import os

def rotate_image(image, angle):
    """Rotate an image by a given angle"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255))
    
    return rotated

def augment_dataset():
    """Augment the dataset with rotated images"""
    # Get all gesture directories
    gesture_dirs = [d for d in os.listdir("data/train") 
                   if os.path.isdir(os.path.join("data/train", d))]
    
    for gesture in gesture_dirs:
        print(f"Augmenting images for gesture '{gesture}'...")
        gesture_dir = os.path.join("data/train", gesture)
        
        # Get all images in the directory
        images = [f for f in os.listdir(gesture_dir) 
                 if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in images:
            img_path = os.path.join(gesture_dir, img_name)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Generate rotated versions
            angles = [-15, -10, -5, 5, 10, 15]  # Rotation angles
            
            for angle in angles:
                # Rotate image
                rotated = rotate_image(img, angle)
                
                # Save rotated image
                base_name = os.path.splitext(img_name)[0]
                new_name = f"{base_name}_rot{angle}.jpg"
                new_path = os.path.join(gesture_dir, new_name)
                
                cv2.imwrite(new_path, rotated)
        
        print(f"Completed augmentation for gesture '{gesture}'")

if __name__ == "__main__":
    augment_dataset() 