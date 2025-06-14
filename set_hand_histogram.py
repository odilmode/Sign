import cv2
import numpy as np
import os

def create_hand_histogram():
    """Create histogram for hand detection"""
    # Create directory if it doesn't exist
    if not os.path.exists("hist"):
        os.mkdir("hist")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Create window for ROI selection
    cv2.namedWindow("Set Hand Histogram")
    
    # Initialize variables
    roi_selected = False
    roi = None
    roi_hist = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Create a copy for drawing
        display = frame.copy()
        
        if not roi_selected:
            # Draw rectangle for ROI selection
            cv2.rectangle(display, (100, 100), (300, 300), (0, 255, 0), 2)
            cv2.putText(display, "Place hand in box and press 'c' to capture", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Draw the ROI
            cv2.rectangle(display, (100, 100), (300, 300), (0, 255, 0), 2)
            cv2.putText(display, "Press 's' to save histogram", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the ROI
            cv2.imshow("ROI", roi)
            
            # Calculate histogram
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            
            # Apply histogram backprojection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cv2.filter2D(dst, -1, kernel, dst)
            
            # Threshold the image
            ret, thresh = cv2.threshold(dst, 50, 255, 0)
            thresh = cv2.merge((thresh, thresh, thresh))
            
            # Show the thresholded image
            cv2.imshow("Threshold", thresh)
        
        # Show the main frame
        cv2.imshow("Set Hand Histogram", display)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and not roi_selected:
            # Capture ROI
            roi = frame[100:300, 100:300]
            roi_selected = True
        elif key == ord('s') and roi_selected:
            # Save histogram
            np.save("hist/hand_histogram.npy", roi_hist)
            print("Histogram saved successfully!")
            break
        elif key == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    create_hand_histogram() 