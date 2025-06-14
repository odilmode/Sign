import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

class SignLanguageInterpreter:
    def __init__(self):
        self.model = None
        self.cap = None
        self.labels = []
        self.confidence_threshold = 0.7
        
    def load_model(self, model_path='model.h5', labels_path='class_names.txt'):
        """Load the trained model and class labels"""
        try:
            self.model = load_model(model_path)
            with open(labels_path, 'r') as f:
                self.labels = f.read().splitlines()
            print(f"Model loaded successfully with {len(self.labels)} classes")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def start_camera(self):
        """Initialize the webcam"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return False
        return True

    def preprocess_frame(self, frame):
        """Preprocess the frame for model input"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize to model input size
        resized = cv2.resize(gray, (64, 64))
        # Normalize
        normalized = resized / 255.0
        # Reshape for model input
        reshaped = normalized.reshape(1, 64, 64, 1)
        return reshaped

    def draw_roi(self, frame):
        """Draw region of interest and instructions"""
        # Draw rectangle for ROI
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
        
        # Add instructions
        cv2.putText(frame, "Place hand in box", (100, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def run(self):
        """Main loop for sign language recognition"""
        if not self.start_camera():
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Draw ROI and instructions
            self.draw_roi(frame)
            
            # Get the region of interest (ROI)
            roi = frame[100:400, 100:400]
            
            if self.model is not None:
                # Preprocess the ROI
                processed_roi = self.preprocess_frame(roi)
                
                # Make prediction
                prediction = self.model.predict(processed_roi, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class]
                
                # Display prediction
                if confidence > self.confidence_threshold:
                    label = self.labels[predicted_class]
                    # Display prediction with different colors based on confidence
                    color = (0, 255, 0) if confidence > 0.9 else (0, 255, 255)
                    cv2.putText(frame, f"Sign: {label}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Display the frame
            cv2.imshow('Sign Language Interpreter', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    interpreter = SignLanguageInterpreter()
    if interpreter.load_model():
        interpreter.run() 