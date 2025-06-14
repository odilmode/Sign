import cv2
import numpy as np
import os

def display_gestures():
    """Display all gestures in the dataset"""
    # Get all gesture directories
    gesture_dirs = sorted([d for d in os.listdir("data/train") 
                         if os.path.isdir(os.path.join("data/train", d))])
    
    if not gesture_dirs:
        print("No gesture directories found!")
        return
    
    # Create window
    cv2.namedWindow("Gestures", cv2.WINDOW_NORMAL)
    
    # Display each gesture
    for gesture in gesture_dirs:
        gesture_dir = os.path.join("data/train", gesture)
        
        # Get first image from the directory
        images = [f for f in os.listdir(gesture_dir) 
                 if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            continue
        
        # Read and display first image
        img_path = os.path.join(gesture_dir, images[0])
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        # Resize image
        img = cv2.resize(img, (200, 200))
        
        # Add gesture label
        cv2.putText(img, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        
        # Show image
        cv2.imshow("Gestures", img)
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    
    # Release resources
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_gestures() 