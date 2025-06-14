# Sign Language Recognition System

A real-time sign language recognition system using deep learning and computer vision.

## Features

- Real-time sign language recognition using webcam
- Support for 29 different signs (26 letters, space, nothing, and delete)
- Word and sentence building capabilities
- Learning mode with visual guides
- Practice mode for improving accuracy
- Text-to-speech output (on supported platforms)
- Session history tracking

## Demo

![Demo GIF](img/demo.gif)

## Screenshots

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