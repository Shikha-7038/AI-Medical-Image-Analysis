"""
Data Loader and Preprocessor for Brain MRI Images
Handles loading, preprocessing, and augmentation of medical images
"""

import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

class BrainMRIDataLoader:
    """
    A class to load and preprocess Brain MRI images for classification
    """
    
    def __init__(self, img_size=(224, 224), batch_size=32):
        """
        Initialize the data loader
        
        Args:
            img_size: Tuple (height, width) to resize all images
            batch_size: Number of images per batch during training
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.num_classes = len(self.class_names)
        
        # Class mapping for reference
        self.class_mapping = {0: 'Glioma Tumor', 
                              1: 'Meningioma Tumor', 
                              2: 'No Tumor', 
                              3: 'Pituitary Tumor'}
        
    def create_data_generators(self, train_path="data/raw/Training", 
                               test_path="data/raw/Testing"):
        """
        Create data generators for training and testing
        
        Args:
            train_path: Path to training images
            test_path: Path to testing images
            
        Returns:
            train_generator, validation_generator, test_generator
        """
        
        print("=" * 60)
        print("CREATING DATA GENERATORS")
        print("=" * 60)
        
        # Data augmentation for training (helps prevent overfitting)
        train_datagen = ImageDataGenerator(
            rescale=1./255,           # Normalize pixel values to [0,1]
            rotation_range=20,        # Random rotation
            width_shift_range=0.1,    # Random horizontal shift
            height_shift_range=0.1,   # Random vertical shift
            shear_range=0.1,          # Random shear
            zoom_range=0.1,           # Random zoom
            horizontal_flip=True,     # Random flip
            fill_mode='nearest',      # Fill missing pixels
            validation_split=0.2      # Use 20% of training for validation
        )
        
        # For testing/validation - only rescale (no augmentation)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create training generator (80% of training data)
        print(f"\n📊 Loading training data from: {train_path}")
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            color_mode='rgb'
        )
        
        # Create validation generator (20% of training data)
        validation_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=True,
            color_mode='rgb'
        )
        
        # Create test generator (if Testing folder exists)
        test_generator = None
        if os.path.exists(test_path):
            print(f"\n📊 Loading test data from: {test_path}")
            test_generator = test_datagen.flow_from_directory(
                test_path,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False,
                color_mode='rgb'
            )
        
        # Print summary
        print("\n" + "=" * 60)
        print("DATA LOADING COMPLETE")
        print("=" * 60)
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {validation_generator.samples}")
        if test_generator:
            print(f"Test samples: {test_generator.samples}")
        print(f"Image size: {self.img_size}")
        print(f"Classes: {self.class_mapping}")
        
        return train_generator, validation_generator, test_generator
    
    def visualize_samples(self, generator, num_samples=9):
        """
        Visualize sample images from the dataset
        
        Args:
            generator: Data generator (train/val/test)
            num_samples: Number of samples to display
        """
        
        # Get one batch of images
        images, labels = next(generator)
        
        # Plot images
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            # Get image and label
            img = images[i]
            label_idx = np.argmax(labels[i])
            label_name = self.class_mapping[label_idx]
            
            # Display image
            axes[i].imshow(img)
            axes[i].set_title(f"{label_name}", fontsize=10, fontweight='bold')
            axes[i].axis('off')
        
        plt.suptitle("Sample Brain MRI Images", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        os.makedirs('outputs/figures', exist_ok=True)
        plt.savefig('outputs/figures/sample_brain_mri_images.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Sample images saved to: outputs/figures/sample_brain_mri_images.png")
    
    def get_class_distribution(self, train_path="data/raw/Training"):
        """
        Get distribution of classes in the dataset
        
        Args:
            train_path: Path to training images
        """
        
        print("\n" + "=" * 60)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("=" * 60)
        
        class_counts = {}
        
        for class_name in self.class_names:
            class_path = os.path.join(train_path, class_name)
            if os.path.exists(class_path):
                count = len([f for f in os.listdir(class_path) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))])
                class_counts[self.class_mapping[self.class_names.index(class_name)]] = count
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(class_counts.keys(), class_counts.values(), 
                       color=['#dc3545', '#fd7e14', '#28a745', '#6f42c1'])
        plt.title('Distribution of Brain MRI Images by Class', fontsize=14, fontweight='bold')
        plt.xlabel('Tumor Type', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        os.makedirs('outputs/figures', exist_ok=True)
        plt.savefig('outputs/figures/class_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Class Distribution:")
        for class_name, count in class_counts.items():
            print(f"   {class_name}: {count} images")
        
        return class_counts

# Quick test to verify the class works
if __name__ == "__main__":
    # Initialize loader
    loader = BrainMRIDataLoader(img_size=(224, 224), batch_size=32)
    
    # Get class distribution
    class_counts = loader.get_class_distribution()
    
    # Create generators
    train_gen, val_gen, test_gen = loader.create_data_generators()
    
    # Visualize samples
    loader.visualize_samples(train_gen, num_samples=9)