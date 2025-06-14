# Sign Language Recognition System

A real-time sign language recognition system using deep learning and computer vision. This project enables real-time translation of American Sign Language (ASL) gestures into text, making communication more accessible for the deaf and hard-of-hearing community.

## Project Overview

### Core Features
- **Real-time Recognition**: Captures and processes video feed to recognize ASL gestures instantly
- **Comprehensive Sign Support**: Recognizes 29 different signs:
  - 26 letters (A-Z)
  - Space gesture
  - "Nothing" gesture (when no sign is shown)
  - Delete gesture (to remove mistakes)
- **Word & Sentence Building**: 
  - Constructs words from individual letters
  - Builds complete sentences with proper spacing
  - Supports word deletion and correction
- **Learning Mode**:
  - Visual guides for each sign
  - Interactive letter selection menu
  - Real-time feedback on hand positioning
- **Practice Mode**:
  - Focused practice on specific letters
  - Progress tracking
  - Performance statistics
- **User Experience**:
  - Text-to-speech output (on supported platforms)
  - Session history tracking
  - Clear visual feedback
  - Intuitive controls

### Technical Implementation
- **Computer Vision**:
  - Hand detection and tracking
  - Region of Interest (ROI) processing
  - Real-time frame processing
- **Deep Learning**:
  - Convolutional Neural Network (CNN) architecture
  - Optimized for real-time performance
  - High accuracy in gesture recognition
- **Data Processing**:
  - Image preprocessing
  - Data augmentation
  - Normalization and standardization

### Demo

![Demo GIF](img/demo.gif)

### Screenshots

![Main Interface](img/main_interface.png)
![Learning Mode](img/learning_interface.png)
![Practice Mode](img/practice_interface.png)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/odilmode/Sign.git
cd Sign
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main application:
```bash
python main.py
```

2. Place your hand in the designated area on the screen
3. Show signs to build words and sentences
4. Use the following controls:
   - Press 'l' to toggle learning mode
   - Press 'm' to open letter selection menu
   - Press 'c' to clear current word/sentence
   - Press 'q' to quit

## Project Structure

- `main.py` - Main application file
- `model.py` - Neural network model architecture
- `train.py` - Training script
- `data/` - Training and test data
- `history/` - Session history
- `guides/` - Guide images for learning mode
- `img/` - Demo and screenshot images

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 