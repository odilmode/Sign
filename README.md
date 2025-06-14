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

### Model Architecture
The system uses a custom CNN architecture optimized for real-time sign language recognition:

#### Input Layer
- Input shape: (32, 32, 3) - RGB images
- Normalized pixel values (0-1)

#### Convolutional Blocks
1. **First Block**:
   - Conv2D: 16 filters, (2x2) kernel
   - BatchNormalization
   - ReLU activation
   - MaxPooling2D: (2x2) pool size

2. **Second Block**:
   - Conv2D: 32 filters, (3x3) kernel
   - BatchNormalization
   - ReLU activation
   - MaxPooling2D: (3x3) pool size

3. **Third Block**:
   - Conv2D: 64 filters, (5x5) kernel
   - BatchNormalization
   - ReLU activation
   - MaxPooling2D: (5x5) pool size

#### Dense Layers
- Flatten layer
- Dense layer: 128 units with ReLU
- Dropout: 0.2
- Output layer: 29 units with softmax (one for each sign)

### Training Process
The model was trained using a comprehensive approach to ensure robust performance:

#### Data Preparation
- **Image Size**: 32x32 pixels
- **Data Augmentation**:
  - Rotation: ±20 degrees
  - Width/Height shift: ±0.2
  - Shear: ±0.2
  - Zoom: ±0.2
  - Horizontal flip
  - Fill mode: 'nearest'

#### Training Configuration
- **Optimizer**: Adam
  - Learning rate: 0.001
  - Beta1: 0.9
  - Beta2: 0.999
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 64
- **Epochs**: 50 (with early stopping)

#### Training Strategies
- **Early Stopping**:
  - Monitor: validation loss
  - Patience: 5 epochs
  - Restore best weights
- **Learning Rate Reduction**:
  - Monitor: validation loss
  - Factor: 0.2
  - Patience: 3 epochs
  - Min learning rate: 1e-6
- **Model Checkpointing**:
  - Save best model based on validation accuracy
  - Monitor: validation accuracy
  - Mode: max

### Model Performance
The model achieves excellent performance on both training and validation sets:

#### Training Metrics
- **Training Accuracy**: 92.19%
- **Validation Accuracy**: 100%
- **Training Loss**: 0.2452
- **Validation Loss**: 0.0283

#### Training Visualization
![Training Metrics](training_metrics.png)

The plots above show:
- **Training vs Validation Accuracy**: Demonstrates how well the model learns and generalizes
- **Training vs Validation Loss**: Shows the model's convergence and stability
- The model achieves very high accuracy on both training and validation sets, with perfect validation accuracy and very low validation loss

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