import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import time
import json
import os
from datetime import datetime
import threading
import platform
import subprocess
from collections import defaultdict

class SignLanguageInterpreter:
    def __init__(self):
        self.model = None
        self.cap = None
        self.labels = []
        self.confidence_threshold = 0.3
        self.current_word = []
        self.current_sentence = []
        self.last_prediction = None
        self.last_prediction_time = 0
        self.prediction_cooldown = 0.5
        self.waiting_for_new_word = True
        self.learning_mode = False
        self.practice_mode = False
        self.history = []
        self.speaking = False
        self.tts_enabled = False
        self.current_learning_letter = 0
        self.guide_images = {}
        self.show_letter_menu = False
        self.selected_letter = None
        
        # Practice mode variables
        self.practice_letters = []
        self.current_practice_letter = 0
        self.practice_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.practice_words = [
            "HELLO", "THANK", "PLEASE", "HELP", "YES", "NO",
            "GOOD", "BAD", "NAME", "FRIEND", "FAMILY", "LOVE"
        ]
        self.current_practice_word = ""
        self.word_progress = []
        
        # Try to enable text-to-speech based on platform
        if platform.system() == 'Darwin':  # macOS
            self.tts_enabled = True
            print("Text-to-speech enabled (using macOS say command)")
        else:
            print("Text-to-speech disabled (not supported on this platform)")
            
        # Load sign language guide images
        self.load_guide_images()
        
        # Load practice statistics if they exist
        self.load_practice_stats()

    def load_practice_stats(self):
        """Load practice statistics from file"""
        if os.path.exists('practice_stats.json'):
            with open('practice_stats.json', 'r') as f:
                self.practice_stats = defaultdict(lambda: {'correct': 0, 'total': 0},
                                                json.load(f))

    def save_practice_stats(self):
        """Save practice statistics to file"""
        with open('practice_stats.json', 'w') as f:
            json.dump(dict(self.practice_stats), f, indent=2)

    def start_practice_mode(self, mode='letters'):
        """Start practice mode with specified mode (letters or words)"""
        self.practice_mode = True
        self.learning_mode = False
        
        if mode == 'letters':
            # Select letters that need more practice
            self.practice_letters = []
            for letter in self.labels:
                if letter in ['space', 'del', 'nothing']:
                    continue
                if letter not in self.practice_stats or \
                   self.practice_stats[letter]['correct'] / max(1, self.practice_stats[letter]['total']) < 0.8:
                    self.practice_letters.append(letter)
            
            if not self.practice_letters:
                self.practice_letters = [l for l in self.labels if l not in ['space', 'del', 'nothing']]
            
            self.current_practice_letter = 0
            self.speak_text(f"Practice mode: {len(self.practice_letters)} letters to practice")
        else:  # words mode
            self.current_practice_word = self.practice_words[0]
            self.word_progress = []
            self.speak_text(f"Practice mode: Let's practice the word {self.current_practice_word}")

    def load_guide_images(self):
        """Load guide images for learning mode from test data"""
        test_dir = 'data/test'
        if not os.path.exists(test_dir):
            print(f"Test directory not found: {test_dir}")
            return
            
        for label in self.labels:
            if label in ['space', 'del', 'nothing']:
                continue
                
            # Look for test images in the corresponding label folder
            label_dir = os.path.join(test_dir, label)
            if os.path.exists(label_dir):
                # Look for the specific test image pattern (e.g., A_test.jpg)
                test_image = f"{label}_test.jpg"
                img_path = os.path.join(label_dir, test_image)
                if os.path.exists(img_path):
                    self.guide_images[label] = cv2.imread(img_path)
                    print(f"Loaded guide image for {label} from {img_path}")
                else:
                    print(f"Test image not found for {label} at {img_path}")
            else:
                print(f"Test directory not found for {label}")

    def load_model(self, model_path='model.h5', labels_path='class_names.txt'):
        """Load the trained model and class labels"""
        try:
            self.model = load_model(model_path)
            with open(labels_path, 'r') as f:
                self.labels = f.read().splitlines()
            print(f"Model loaded successfully with {len(self.labels)} classes")
            # Load guide images after labels are loaded
            self.load_guide_images()
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
        try:
            resized = cv2.resize(frame, (32, 32))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb / 255.0
            reshaped = normalized.reshape(1, 32, 32, 3)
            return reshaped
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None

    def speak_text(self, text):
        """Speak the given text using platform-specific methods"""
        if not self.tts_enabled or self.speaking:
            return
            
        self.speaking = True
        threading.Thread(target=self._speak_thread, args=(text,)).start()

    def _speak_thread(self, text):
        """Thread function for text-to-speech"""
        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['say', text])
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
        finally:
            self.speaking = False

    def save_history(self):
        """Save the current session history"""
        if not os.path.exists('history'):
            os.makedirs('history')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'history/session_{timestamp}.json'
        
        session_data = {
            'timestamp': timestamp,
            'sentences': self.history
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"History saved to {filename}")

    def load_history(self):
        """Load the most recent session history"""
        if not os.path.exists('history'):
            return
        
        history_files = [f for f in os.listdir('history') if f.endswith('.json')]
        if not history_files:
            return
        
        latest_file = max(history_files, key=lambda x: os.path.getctime(os.path.join('history', x)))
        with open(os.path.join('history', latest_file), 'r') as f:
            session_data = json.load(f)
            self.history = session_data.get('sentences', [])
            print(f"Loaded history from {latest_file}")

    def process_prediction(self, prediction, confidence):
        """Process the prediction and update words/sentences"""
        current_time = time.time()
        
        if current_time - self.last_prediction_time < self.prediction_cooldown:
            return
            
        self.last_prediction_time = current_time
        
        if self.practice_mode:
            if self.current_practice_word:  # Word practice mode
                if prediction == "del" and self.word_progress:
                    self.word_progress.pop()
                elif prediction not in ['space', 'del', 'nothing']:
                    self.word_progress.append(prediction)
                    current_word = ''.join(self.word_progress)
                    if current_word == self.current_practice_word:
                        self.speak_text("Excellent! Word completed correctly!")
                        # Move to next word
                        current_index = self.practice_words.index(self.current_practice_word)
                        if current_index < len(self.practice_words) - 1:
                            self.current_practice_word = self.practice_words[current_index + 1]
                            self.word_progress = []
                            self.speak_text(f"Next word: {self.current_practice_word}")
                        else:
                            self.speak_text("Congratulations! You've completed all words!")
                            self.practice_mode = False
            else:  # Letter practice mode
                current_letter = self.practice_letters[self.current_practice_letter]
                if prediction == current_letter and confidence > 0.7:
                    self.practice_stats[current_letter]['correct'] += 1
                    self.speak_text("Correct!")
                else:
                    self.speak_text("Try again")
                
                self.practice_stats[current_letter]['total'] += 1
                self.save_practice_stats()
                
                # Move to next letter
                self.current_practice_letter = (self.current_practice_letter + 1) % len(self.practice_letters)
                if self.current_practice_letter == 0:
                    self.speak_text("Practice session completed!")
                    self.practice_mode = False
                else:
                    self.speak_text(f"Next letter: {self.practice_letters[self.current_practice_letter]}")
            return
        
        if self.learning_mode:
            current_letter = self.labels[self.current_learning_letter]
            if prediction == current_letter and confidence > 0.7:
                self.current_learning_letter = (self.current_learning_letter + 1) % len(self.labels)
                if self.current_learning_letter == 0:
                    self.speak_text("Congratulations! You've completed all letters!")
                else:
                    self.speak_text(f"Good job! Now try {self.labels[self.current_learning_letter]}")
            return
        
        if prediction == "space":
            if self.current_word:
                word = ''.join(self.current_word)
                self.current_sentence.append(word)
                self.current_word = []
                self.waiting_for_new_word = True
                self.speak_text(word)
        elif prediction == "del":
            if self.current_word:
                self.current_word.pop()
        elif prediction == "nothing":
            pass
        else:
            if self.waiting_for_new_word:
                self.current_word = [prediction]
                self.waiting_for_new_word = False
            else:
                self.current_word.append(prediction)

    def create_mini_guide(self):
        """Create a mini guide image containing all available signs"""
        # Calculate grid dimensions
        n_signs = len([l for l in self.labels if l not in ['space', 'del', 'nothing']])
        grid_size = int(np.ceil(np.sqrt(n_signs)))
        
        # Create empty canvas
        cell_size = 64  # Size of each sign image
        padding = 2
        canvas_size = grid_size * (cell_size + padding)
        mini_guide = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        
        # Place each sign in the grid
        current_x = padding
        current_y = padding
        for label in self.labels:
            if label in ['space', 'del', 'nothing']:
                continue
                
            # Load and resize the test image
            test_image = f"{label}_test.jpg"
            img_path = os.path.join('data/test', label, test_image)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.resize(img, (cell_size, cell_size))
                
                # Add label
                cv2.putText(img, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Place in grid
                mini_guide[current_y:current_y+cell_size, current_x:current_x+cell_size] = img
                
                # Update position
                current_x += cell_size + padding
                if current_x + cell_size > canvas_size:
                    current_x = padding
                    current_y += cell_size + padding
        
        return mini_guide

    def draw_learning_mode(self, frame):
        """Draw learning mode interface"""
        if not self.learning_mode:
            return
            
        # Get current letter
        current_letter = self.labels[self.current_learning_letter]
        
        # Draw guide image if available
        if current_letter in self.guide_images:
            guide_img = self.guide_images[current_letter]
            # Resize guide image to fit in the right side of the frame
            h, w = guide_img.shape[:2]
            scale = min(300/h, 300/w)
            new_h, new_w = int(h*scale), int(w*scale)
            guide_img = cv2.resize(guide_img, (new_w, new_h))
            
            # Place guide image on the right side
            x_offset = frame.shape[1] - new_w - 20
            y_offset = 100
            frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = guide_img
            
            # Draw border around guide image
            cv2.rectangle(frame, (x_offset-2, y_offset-2), 
                         (x_offset+new_w+2, y_offset+new_h+2), (0, 255, 0), 2)
        
        # Draw mini guide in the bottom right
        mini_guide = self.create_mini_guide()
        h, w = mini_guide.shape[:2]
        scale = min(200/h, 200/w)
        new_h, new_w = int(h*scale), int(w*scale)
        mini_guide = cv2.resize(mini_guide, (new_w, new_h))
        
        # Place mini guide in bottom right
        x_offset = frame.shape[1] - new_w - 20
        y_offset = frame.shape[0] - new_h - 20
        frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = mini_guide
        
        # Draw border around mini guide
        cv2.rectangle(frame, (x_offset-2, y_offset-2), 
                     (x_offset+new_w+2, y_offset+new_h+2), (0, 255, 0), 2)
        
        # Draw current letter and instructions
        cv2.putText(frame, f"Learn: {current_letter}", (frame.shape[1] - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Match the hand position shown", (frame.shape[1] - 300, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'l' to exit learning mode", (frame.shape[1] - 300, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def draw_practice_mode(self, frame):
        """Draw practice mode interface"""
        if not self.practice_mode:
            return
            
        if self.current_practice_word:  # Word practice mode
            # Draw current word to practice
            cv2.putText(frame, f"Practice: {self.current_practice_word}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw current progress
            current_word = ''.join(self.word_progress)
            cv2.putText(frame, f"Your input: {current_word}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw instructions
            cv2.putText(frame, "Show letters to spell the word", (10, frame.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'p' to exit practice mode", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:  # Letter practice mode
            # Draw current letter to practice
            current_letter = self.practice_letters[self.current_practice_letter]
            cv2.putText(frame, f"Practice: {current_letter}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw statistics
            stats = self.practice_stats[current_letter]
            accuracy = stats['correct'] / max(1, stats['total']) * 100
            cv2.putText(frame, f"Accuracy: {accuracy:.1f}%", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw progress
            progress = f"Letter {self.current_practice_letter + 1} of {len(self.practice_letters)}"
            cv2.putText(frame, progress, (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw instructions
            cv2.putText(frame, "Match the letter shown", (10, frame.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'p' to exit practice mode", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw guide image
            if current_letter in self.guide_images:
                guide_img = self.guide_images[current_letter]
                h, w = guide_img.shape[:2]
                scale = min(300/h, 300/w)
                new_h, new_w = int(h*scale), int(w*scale)
                guide_img = cv2.resize(guide_img, (new_w, new_h))
                
                x_offset = frame.shape[1] - new_w - 20
                y_offset = 100
                frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = guide_img
                
                cv2.rectangle(frame, (x_offset-2, y_offset-2), 
                             (x_offset+new_w+2, y_offset+new_h+2), (0, 255, 0), 2)

    def draw_confidence_bar(self, frame, confidence):
        """Draw a confidence bar on the frame"""
        bar_width = 200
        bar_height = 20
        x = 10
        y = 180
        
        # Draw background
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (0, 0, 0), -1)
        
        # Draw confidence level
        confidence_width = int(bar_width * confidence)
        color = (0, 255, 0) if confidence > 0.9 else (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + confidence_width, y + bar_height), color, -1)
        
        # Draw border
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (255, 255, 255), 1)
        
        # Draw text
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_text(self, frame):
        """Draw the current word and sentence on the frame"""
        if self.learning_mode:
            return
            
        # Draw current word
        current_word = ''.join(self.current_word)
        word_status = "Waiting for new word..." if self.waiting_for_new_word else f"Current: {current_word}"
        cv2.putText(frame, word_status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw current sentence
        sentence = ' '.join(self.current_sentence)
        cv2.putText(frame, f"Sentence: {sentence}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw mode indicator
        mode_text = "Learning Mode" if self.learning_mode else "Recognition Mode"
        cv2.putText(frame, mode_text, (frame.shape[1] - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw TTS status
        tts_status = "TTS: Enabled" if self.tts_enabled else "TTS: Disabled"
        cv2.putText(frame, tts_status, (frame.shape[1] - 200, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw instructions
        cv2.putText(frame, "Show 'space' to complete word", (10, frame.shape[0] - 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Show 'del' to delete last letter", (10, frame.shape[0] - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'l' to toggle learning mode", (10, frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'c' to clear, 'q' to quit", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def draw_roi(self, frame):
        """Draw region of interest and instructions"""
        # Calculate ROI dimensions and position
        frame_height, frame_width = frame.shape[:2]
        
        # Make ROI larger and position it in the center
        roi_size = min(frame_width, frame_height) // 2  # Half of the smaller dimension
        x_start = (frame_width - roi_size) // 2
        y_start = (frame_height - roi_size) // 2
        
        # Draw the ROI box
        cv2.rectangle(frame, (x_start, y_start), 
                     (x_start + roi_size, y_start + roi_size), 
                     (0, 255, 0), 2)
        
        # Add a semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), 
                     (x_start + roi_size, y_start + roi_size), 
                     (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        
        # Draw instructions
        cv2.putText(frame, "Place hand in box", (x_start, y_start - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return (x_start, y_start, roi_size, roi_size)

    def draw_letter_menu(self, frame):
        """Draw the letter selection menu"""
        if not self.show_letter_menu:
            return
            
        # Create a semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Calculate grid dimensions
        letters = [l for l in self.labels if l not in ['space', 'del', 'nothing']]
        grid_size = int(np.ceil(np.sqrt(len(letters))))
        cell_size = min(frame.shape[1] // (grid_size + 2), frame.shape[0] // (grid_size + 2))
        
        # Draw title
        cv2.putText(frame, "Select a Letter to Learn", (frame.shape[1]//2 - 150, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw letters in a grid
        start_x = (frame.shape[1] - (grid_size * cell_size)) // 2
        start_y = 100
        
        for i, letter in enumerate(letters):
            row = i // grid_size
            col = i % grid_size
            x = start_x + (col * cell_size)
            y = start_y + (row * cell_size)
            
            # Draw letter box
            color = (0, 255, 0) if letter == self.selected_letter else (255, 255, 255)
            cv2.rectangle(frame, (x, y), (x + cell_size - 10, y + cell_size - 10), color, 2)
            
            # Draw letter
            cv2.putText(frame, letter, (x + cell_size//2 - 10, y + cell_size//2 + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Draw guide image if available
            if letter in self.guide_images:
                guide_img = self.guide_images[letter]
                guide_size = cell_size - 30
                guide_img = cv2.resize(guide_img, (guide_size, guide_size))
                frame[y + 5:y + 5 + guide_size, x + 5:x + 5 + guide_size] = guide_img
        
        # Draw instructions
        cv2.putText(frame, "Press number keys (1-9) to select letter", (frame.shape[1]//2 - 200, frame.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'ESC' to close menu", (frame.shape[1]//2 - 100, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def run(self):
        """Main loop for sign language recognition"""
        if not self.start_camera():
            return

        print("Starting sign language recognition...")
        print("Press 'q' to quit, 'c' to clear")
        print("Press 'l' for learning mode, 'p' for practice mode")
        print("Press 'w' to switch to word practice")
        print("Press 'm' to show letter selection menu")
        
        # Load previous history
        self.load_history()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            frame = cv2.flip(frame, 1)
            
            if self.show_letter_menu:
                self.draw_letter_menu(frame)
            else:
                x, y, w, h = self.draw_roi(frame)
                roi = frame[y:y+h, x:x+w]
                
                if self.model is not None:
                    processed_roi = self.preprocess_frame(roi)
                    
                    if processed_roi is not None:
                        prediction = self.model.predict(processed_roi, verbose=0)
                        predicted_class = np.argmax(prediction[0])
                        confidence = prediction[0][predicted_class]
                        
                        if confidence > self.confidence_threshold:
                            label = self.labels[predicted_class]
                            color = (0, 255, 0) if confidence > 0.9 else (0, 255, 255)
                            cv2.putText(frame, f"Sign: {label}", (x, y - 40),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            
                            # Draw confidence bar
                            self.draw_confidence_bar(frame, confidence)
                            
                            # Process the prediction
                            self.process_prediction(label, confidence)
                
                # Draw appropriate interface based on mode
                if self.practice_mode:
                    self.draw_practice_mode(frame)
                elif self.learning_mode:
                    self.draw_learning_mode(frame)
                else:
                    self.draw_text(frame)
            
            cv2.imshow('Sign Language Interpreter', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # Save history and stats before quitting
                if self.current_sentence:
                    self.history.append(' '.join(self.current_sentence))
                self.save_history()
                self.save_practice_stats()
                break
            elif key == ord('c'):
                if self.current_sentence:
                    self.history.append(' '.join(self.current_sentence))
                self.current_word = []
                self.current_sentence = []
                self.waiting_for_new_word = True
            elif key == ord('l'):
                self.learning_mode = not self.learning_mode
                self.practice_mode = False
                if self.learning_mode:
                    if self.selected_letter:
                        self.current_learning_letter = self.labels.index(self.selected_letter)
                        self.speak_text(f"Learning mode enabled. Let's practice {self.selected_letter}")
                    else:
                        self.current_learning_letter = 0
                        self.speak_text(f"Learning mode enabled. Let's start with {self.labels[0]}")
                else:
                    self.speak_text("Learning mode disabled")
            elif key == ord('p'):
                if not self.practice_mode:
                    self.start_practice_mode('letters')
                else:
                    self.practice_mode = False
                    self.speak_text("Practice mode disabled")
            elif key == ord('w'):
                if not self.practice_mode:
                    self.start_practice_mode('words')
                else:
                    self.practice_mode = False
                    self.speak_text("Practice mode disabled")
            elif key == ord('m'):
                self.show_letter_menu = not self.show_letter_menu
                if self.show_letter_menu:
                    self.selected_letter = None
            elif key == 27:  # ESC key
                self.show_letter_menu = False
            elif self.show_letter_menu and key >= ord('1') and key <= ord('9'):
                # Convert number key to index (1-9)
                index = key - ord('1')
                letters = [l for l in self.labels if l not in ['space', 'del', 'nothing']]
                if index < len(letters):
                    self.selected_letter = letters[index]
                    self.show_letter_menu = False
                    self.speak_text(f"Selected letter {self.selected_letter}")
                    if not self.learning_mode:
                        self.learning_mode = True
                        self.current_learning_letter = self.labels.index(self.selected_letter)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    interpreter = SignLanguageInterpreter()
    if interpreter.load_model():
        interpreter.run() 