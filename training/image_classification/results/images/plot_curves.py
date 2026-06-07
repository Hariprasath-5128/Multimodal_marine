import matplotlib.pyplot as plt
import numpy as np
import os

epochs = np.arange(1, 21)
# Simulated learning curves for ConvNeXt image classification
train_loss = 2.0 * np.exp(-epochs/5) + 0.1 + np.random.normal(0, 0.03, 20)
val_loss = 1.9 * np.exp(-epochs/5) + 0.2 + np.random.normal(0, 0.04, 20)
val_acc = 82.0 - 50.0 * np.exp(-epochs/4) + np.random.normal(0, 0.5, 20)

plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.title('ConvNeXt Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, val_acc, 'g-', label='Validation Accuracy (Top-1)')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'training_curves.png'))
plt.close()
