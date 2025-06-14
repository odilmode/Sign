import cv2
import numpy as np
import os
from set_hand_histogram import create_hand_histogram

def create_gesture_dataset():
    """Create dataset of hand gestures"""
    # Create directories if they don't exist
    if not os.path.exists("data/train"):
        os.makedirs("data/train")
    
    # Load hand histogram
    try:
        hand_hist = np.load("hist/hand_histogram.npy")
    except:
        print("Hand histogram not found. Creating new one...")
        create_hand_histogram()
        hand_hist = np.load("hist/hand_histogram.npy")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Get gesture name from user
    gesture_name = input("Enter gesture name (A-Z, space, nothing, del): ").upper()
    if not os.path.exists(f"data/train/{gesture_name}"):
        os.makedirs(f"data/train/{gesture_name}")
    
    # Initialize variables
    count = 0
    max_images = 1000  # Maximum number of images to capture
    
    print(f"Capturing images for gesture '{gesture_name}'")
    print("Press 'c' to capture, 'q' to quit")
    
    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply histogram backprojection
        dst = cv2.calcBackProject([hsv], [0, 1], hand_hist, [0, 180, 0, 256], 1)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cv2.filter2D(dst, -1, kernel, dst)
        
        # Threshold the image
        ret, thresh = cv2.threshold(dst, 50, 255, 0)
        thresh = cv2.merge((thresh, thresh, thresh))
        
        # Get the region of interest
        roi = frame[100:400, 100:400]
        roi_thresh = thresh[100:400, 100:400]
        
        # Draw rectangle
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
        
        # Add text
        cv2.putText(frame, f"Captured: {count}/{max_images}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'c' to capture", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show images
        cv2.imshow("Original", frame)
        cv2.imshow("Threshold", thresh)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Save the ROI
            cv2.imwrite(f"data/train/{gesture_name}/{count}.jpg", roi)
            count += 1
            print(f"Captured image {count}/{max_images}")
        elif key == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} images for gesture '{gesture_name}'")

if __name__ == "__main__":
    create_gesture_dataset() 