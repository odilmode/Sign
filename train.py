import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json

def load_dataset(data_dir, img_size=(64, 64)):
    """
    Load and preprocess the dataset
    
    Args:
        data_dir (str): Directory containing the dataset
        img_size (tuple): Size to resize images to
    
    Returns:
        X: Preprocessed images
        y: Labels
        class_names: List of class names
    """
    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    # Loop through each class directory
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        print(f"Loading images from {class_name}...")
        
        # Load images from class directory
        for img_name in os.listdir(class_dir):
            if img_name.startswith('.'):  # Skip hidden files
                continue
                
            img_path = os.path.join(class_dir, img_name)
            try:
                # Read and preprocess image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                    
                img = cv2.resize(img, img_size)
                img = img / 255.0  # Normalize
                
                images.append(img)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    
    return np.array(images), np.array(labels), class_names

def create_data_generator():
    """Create data generator for augmentation"""
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

def create_model(input_shape=(32, 32, 3), num_classes=29):
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Fourth Convolutional Block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    # Data directories
    train_dir = 'data/train'
    test_dir = 'data/test'
    
    # Image dimensions
    img_height, img_width = 32, 32
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load and augment training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=64,
        class_mode='categorical'
    )
    
    # Load validation data
    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=64,
        class_mode='categorical'
    )
    
    # Create and compile model
    model = create_model(input_shape=(img_height, img_width, 3), num_classes=29)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(history.history, f)
    
    return model

if __name__ == "__main__":
    train_model() 