"""
Model Builder for Brain MRI Classification
Uses Transfer Learning with pre-trained models
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

class BrainMRIModel:
    """
    A class to build, train, and evaluate models for Brain MRI classification
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=4):
        """
        Initialize the model builder
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_cnn_model(self):
        """
        Build a custom CNN model from scratch
        
        Returns:
            TensorFlow model
        """
        print("=" * 60)
        print("BUILDING CUSTOM CNN MODEL")
        print("=" * 60)
        
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            # Global Average Pooling (replaces Flatten + Dense)
            layers.GlobalAveragePooling2D(),
            
            # Dense Layers
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            
            # Output Layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        model.summary()
        self.model = model
        return model
    
    def build_transfer_learning_model(self, base_model_name='VGG16'):
        """
        Build a model using transfer learning with pre-trained weights
        
        Args:
            base_model_name: Name of pre-trained model ('VGG16', 'ResNet50', 'EfficientNetB0')
            
        Returns:
            TensorFlow model
        """
        print("=" * 60)
        print(f"BUILDING TRANSFER LEARNING MODEL WITH {base_model_name}")
        print("=" * 60)
        
        # Choose base model
        if base_model_name == 'VGG16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif base_model_name == 'ResNet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif base_model_name == 'EfficientNetB0':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.input_shape)
        else:
            raise ValueError("Choose from: 'VGG16', 'ResNet50', 'EfficientNetB0'")
        
        # Freeze base model layers (don't train them initially)
        base_model.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        model.summary()
        self.model = model
        return model
    
    def train_model(self, train_generator, validation_generator, epochs=20):
        """
        Train the model
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        
        print("=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Callbacks for better training
        callbacks = [
            # Early stopping - stop if validation loss doesn't improve
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint - save best model
            ModelCheckpoint(
                'models/best_brain_mri_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate when plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        return history
    
    def plot_training_history(self):
        """
        Plot training history (accuracy and loss)
        """
        
        if self.history is None:
            print("❌ No training history found. Train the model first.")
            return
        
        os.makedirs('outputs/figures', exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epochs', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot Loss
        axes[1].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epochs', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('outputs/figures/training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print final metrics
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE - FINAL METRICS")
        print("=" * 60)
        print(f"Final Training Accuracy: {self.history.history['accuracy'][-1]:.4f}")
        print(f"Final Validation Accuracy: {self.history.history['val_accuracy'][-1]:.4f}")
        print(f"Best Validation Accuracy: {max(self.history.history['val_accuracy']):.4f}")

    def fine_tune_model(self, train_generator, validation_generator, epochs=10):
        """
        Fine-tune the model by unfreezing some layers of the base model
        This is for transfer learning models only
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of fine-tuning epochs
        """
        
        print("=" * 60)
        print("FINE-TUNING MODEL")
        print("=" * 60)
        
        # Unfreeze the top layers of the base model
        if hasattr(self.model.layers[0], 'trainable'):
            self.model.layers[0].trainable = True
            
            # Recompile with lower learning rate
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Continue training
            history_fine = self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=validation_generator,
                verbose=1
            )
            
            # Append to history
            for key in history_fine.history:
                self.history.history[key].extend(history_fine.history[key])
            
            print("✅ Fine-tuning complete!")
        else:
            print("⚠️ This model doesn't have a base model for fine-tuning")