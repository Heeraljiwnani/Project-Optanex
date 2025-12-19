import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Path to your RGB dataset
base_dir = r"D:\pankhuri\vit\fall sem\project\AMD_Model\OCT2017_RGB"

# Image size and batch size
img_size = (224, 224)
batch_size = 32

# Enhanced data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, "train"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, "train"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    os.path.join(base_dir, "test"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical", 
    shuffle=False
)

# === GET CLASS NAMES ===
print("Training class indices:", train_generator.class_indices)
class_labels = list(train_generator.class_indices.keys())
print("Class labels in order:", class_labels)

# Write to a file for the API
with open('class_labels.txt', 'w') as f:
    for label in class_labels:
        f.write(f"{label}\n")
print("Class labels saved to class_labels.txt")
# === END GET CLASS NAMES ===

# Build optimized model with transfer learning
base_model = VGG16(weights='imagenet', 
                   include_top=False, 
                   input_shape=(224, 224, 3))

# Freeze the base model initially
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.6),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile with optimized settings
model.compile(
    optimizer=Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

# Display model architecture
model.summary()

# Create a directory for checkpoints
checkpoint_dir = "training_checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Enhanced callbacks with multiple checkpoints
callbacks = [
    # Save checkpoint after every epoch
    ModelCheckpoint(
        os.path.join(checkpoint_dir, 'checkpoint_epoch_{epoch:02d}.h5'),
        verbose=1,
        save_freq='epoch'
    ),
    # Save the best model
    ModelCheckpoint(
        'best_amd_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]

# Function to resume training from last checkpoint
def resume_training(current_model):
    # Get all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.h5')]
    
    if checkpoint_files:
        # Sort by epoch number
        checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        
        # Extract epoch number from filename
        epoch_number = int(checkpoint_files[-1].split('_')[2].split('.')[0])
        print(f"Resuming from epoch {epoch_number} using {latest_checkpoint}")
        
        # Load the model
        return load_model(latest_checkpoint), epoch_number
    else:
        print("No checkpoints found. Starting from scratch.")
        return current_model, 0

# Try to resume from checkpoint
model, initial_epoch = resume_training(model)

# Train the model
print("Starting training...")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Test samples: {test_generator.samples}")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=3,
    initial_epoch=initial_epoch,
    callbacks=callbacks,
    verbose=1
)

# Save the model in a compatible format
model.save('best_amd_model_compatible.h5', save_format='h5')
print("Model saved in compatible format")

# Fine-tuning phase (unfreeze some layers)
print("\nStarting fine-tuning phase...")
base_model.trainable = True
# Freeze the bottom layers, unfreeze the top layers
for layer in base_model.layers[:15]:
    layer.trainable = False
for layer in base_model.layers[15:]:
    layer.trainable = True

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

# Continue training with fine-tuning
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=1,  # 30 + 15 = 45 total epochs
    initial_epoch=30,
    callbacks=callbacks,
    verbose=1
)

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Test Precision: {test_precision*100:.2f}%")
print(f"Test Recall: {test_recall*100:.2f}%")

# Plot results
plt.figure(figsize=(18, 6))

# Debug: Check what keys are available
print("=== DEBUG: History Keys ===")
print("History keys:", list(history.history.keys()))
print("History finetune keys:", list(history_finetune.history.keys()))
print("===========================")

# Find available accuracy keys
acc_keys = [key for key in history.history.keys() if 'acc' in key.lower() and not key.startswith('val_')]
val_acc_keys = [key for key in history.history.keys() if 'acc' in key.lower() and key.startswith('val_')]

if acc_keys and val_acc_keys:
    acc_key = acc_keys[0]
    val_acc_key = val_acc_keys[0]
    
    # Combine history from both training phases
    combined_history = {
        'accuracy': history.history[acc_key] + history_finetune.history[acc_key],
        'val_accuracy': history.history[val_acc_key] + history_finetune.history[val_acc_key],
        'loss': history.history['loss'] + history_finetune.history['loss'],
        'val_loss': history.history['val_loss'] + history_finetune.history['val_loss'],
    }
    
    plt.subplot(1, 3, 1)
    plt.plot(combined_history['accuracy'], label='Training Accuracy')
    plt.plot(combined_history['val_accuracy'], label='Validation Accuracy')
    plt.axvline(x=len(history.history[acc_key]), color='r', linestyle='--', label='Fine-tuning Start')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(combined_history['loss'], label='Training Loss')
    plt.plot(combined_history['val_loss'], label='Validation Loss')
    plt.axvline(x=len(history.history['loss']), color='r', linestyle='--', label='Fine-tuning Start')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Check if precision/recall metrics exist
    precision_keys = [key for key in history.history.keys() if 'precision' in key.lower() and not key.startswith('val_')]
    recall_keys = [key for key in history.history.keys() if 'recall' in key.lower() and not key.startswith('val_')]
    
    if precision_keys and recall_keys:
        prec_key = precision_keys[0]
        rec_key = recall_keys[0]
        val_prec_key = 'val_' + prec_key
        val_rec_key = 'val_' + rec_key
        
        combined_history['precision'] = history.history[prec_key] + history_finetune.history[prec_key]
        combined_history['val_precision'] = history.history[val_prec_key] + history_finetune.history[val_prec_key]
        combined_history['recall'] = history.history[rec_key] + history_finetune.history[rec_key]
        combined_history['val_recall'] = history.history[val_rec_key] + history_finetune.history[val_rec_key]
        
        plt.subplot(1, 3, 3)
        plt.plot(combined_history['precision'], label='Training Precision')
        plt.plot(combined_history['val_precision'], label='Validation Precision')
        plt.plot(combined_history['recall'], label='Training Recall')
        plt.plot(combined_history['val_recall'], label='Validation Recall')
        plt.axvline(x=len(history.history[prec_key]), color='r', linestyle='--', label='Fine-tuning Start')
        plt.title('Precision and Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
    else:
        plt.subplot(1, 3, 3)
        plt.text(0.5, 0.5, 'Precision/Recall metrics not available', ha='center', va='center')
        plt.title('No Precision/Recall Data')
        plt.axis('off')
else:
    # If no accuracy metrics found, create simple plots
    plt.subplot(1, 3, 1)
    plt.text(0.5, 0.5, 'Accuracy metrics not available', ha='center', va='center')
    plt.title('No Accuracy Data')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    if 'loss' in history.history and 'loss' in history_finetune.history:
        plt.plot(history.history['loss'] + history_finetune.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'] + history_finetune.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'Loss metrics not available', ha='center', va='center')
        plt.title('No Loss Data')
        plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.text(0.5, 0.5, 'No metrics available for plotting', ha='center', va='center')
    plt.title('No Data Available')
    plt.axis('off')

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()