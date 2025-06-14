import json
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="darkgrid", palette="husl")

# Load training history
with open('training_history.json', 'r') as f:
    history = json.load(f)

# Create a figure with 2x2 subplots
plt.figure(figsize=(15, 10))

# Plot 1: Training & validation accuracy
plt.subplot(2, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy Over Time', fontsize=12, pad=10)
plt.ylabel('Accuracy', fontsize=10)
plt.xlabel('Epoch', fontsize=10)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot 2: Training & validation loss
plt.subplot(2, 2, 2)
plt.plot(history['loss'], label='Training Loss', linewidth=2)
plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss Over Time', fontsize=12, pad=10)
plt.ylabel('Loss', fontsize=10)
plt.xlabel('Epoch', fontsize=10)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot 3: Learning rate over time (if available)
if 'lr' in history:
    plt.subplot(2, 2, 3)
    plt.plot(history['lr'], linewidth=2)
    plt.title('Learning Rate Over Time', fontsize=12, pad=10)
    plt.ylabel('Learning Rate', fontsize=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.grid(True, alpha=0.3)

# Add a title to the entire figure
plt.suptitle('Sign Language Recognition Model Training Metrics', fontsize=14, y=1.02)

# Adjust layout and save
plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
plt.savefig('training_metrics.svg', format='svg', bbox_inches='tight')
plt.show()

# Print final metrics
print("\nFinal Training Metrics:")
print(f"Training Accuracy: {history['accuracy'][-1]:.4f}")
print(f"Validation Accuracy: {history['val_accuracy'][-1]:.4f}")
print(f"Training Loss: {history['loss'][-1]:.4f}")
print(f"Validation Loss: {history['val_loss'][-1]:.4f}") 