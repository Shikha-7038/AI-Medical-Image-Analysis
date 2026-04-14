"""
Prediction Module for Brain MRI Classification
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BrainMRIPredictor:
    """
    Class to make predictions on new Brain MRI images
    """
    
    def __init__(self, model_path='models/final_brain_mri_model.h5'):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
        
        # Colors for visualization
        self.colors = {
            'Glioma Tumor': '#dc3545',
            'Meningioma Tumor': '#fd7e14',
            'No Tumor': '#28a745',
            'Pituitary Tumor': '#6f42c1'
        }
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        if os.path.exists(self.model_path):
            print(f"✅ Loading model from: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            print(f"❌ Model not found at {self.model_path}")
            print("Please run 'python src/train.py' first to train the model")
            exit(1)
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """
        Preprocess a single image for prediction
        
        Args:
            image_path: Path to the image file
            target_size: Target size for the model
            
        Returns:
            Preprocessed image array
        """
        # Read image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Normalize pixel values
        image = image / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image_path, confidence_threshold=0.6):
        """
        Make prediction on a single image
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence for prediction
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        image = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(image, verbose=0)[0]
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[predicted_class_idx]
        
        # Get all class probabilities
        all_probabilities = {
            self.class_names[i]: predictions[i] for i in range(len(self.class_names))
        }
        
        # Determine if prediction is confident enough
        is_confident = confidence >= confidence_threshold
        
        result = {
            'image_path': image_path,
            'predicted_class': self.class_names[predicted_class_idx],
            'confidence': float(confidence),
            'all_probabilities': all_probabilities,
            'is_confident': is_confident,
            'prediction_successful': is_confident
        }
        
        return result
    
    def visualize_prediction(self, result):
        """
        Visualize the prediction result
        
        Args:
            result: Prediction result dictionary
        """
        # Read original image
        image = cv2.imread(result['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Display image with prediction
        axes[0].imshow(image)
        axes[0].axis('off')
        
        # Add prediction text
        color = self.colors.get(result['predicted_class'], 'black')
        title = f"Prediction: {result['predicted_class']}\nConfidence: {result['confidence']*100:.1f}%"
        axes[0].set_title(title, fontsize=14, fontweight='bold', color=color)
        
        # Create bar chart of probabilities
        classes = list(result['all_probabilities'].keys())
        probabilities = list(result['all_probabilities'].values())
        colors = [self.colors.get(c, 'gray') for c in classes]
        
        bars = axes[1].bar(classes, probabilities, color=colors)
        axes[1].set_ylim([0, 1])
        axes[1].set_ylabel('Probability', fontsize=12)
        axes[1].set_title('Class Probabilities', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{prob*100:.1f}%', ha='center', va='bottom')
        
        plt.suptitle("Brain MRI Classification Result", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/prediction_result.png', dpi=150, bbox_inches='tight')
        print(f"✅ Prediction visualization saved to: outputs/prediction_result.png")
        
        plt.show()

def main():
    """Main function for prediction"""
    
    print("=" * 70)
    print("🧠 BRAIN MRI PREDICTION SYSTEM")
    print("=" * 70)
    
    # Initialize predictor
    predictor = BrainMRIPredictor()
    
    print("\nChoose prediction mode:")
    print("1. Single image prediction")
    print("2. Batch prediction (all images in a folder)")
    print("3. Use sample from test dataset")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        # Single image
        image_path = input("Enter path to MRI image: ").strip()
        
        if os.path.exists(image_path):
            result = predictor.predict(image_path)
            predictor.visualize_prediction(result)
            
            print("\n" + "=" * 60)
            print("PREDICTION SUMMARY")
            print("=" * 60)
            print(f"Image: {result['image_path']}")
            print(f"Prediction: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']*100:.2f}%")
            
            if not result['is_confident']:
                print("\n⚠️ Warning: Low confidence prediction. Please consult a specialist.")
        else:
            print(f"❌ Image not found: {image_path}")
    
    elif choice == '2':
        # Batch prediction
        folder_path = input("Enter path to folder containing MRI images: ").strip()
        
        if os.path.exists(folder_path):
            results = []
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = [f for f in os.listdir(folder_path) 
                          if any(f.lower().endswith(ext) for ext in image_extensions)]
            
            print(f"\n📊 Processing {len(image_files)} images...")
            
            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                try:
                    result = predictor.predict(image_path)
                    results.append(result)
                    print(f"   {image_file}: {result['predicted_class']} ({result['confidence']*100:.1f}%)")
                except Exception as e:
                    print(f"   Error processing {image_file}: {e}")
            
            # Summary
            print("\n" + "=" * 60)
            print("BATCH PREDICTION SUMMARY")
            print("=" * 60)
            
            from collections import Counter
            predictions = [r['predicted_class'] for r in results]
            counts = Counter(predictions)
            
            for class_name, count in counts.items():
                print(f"{class_name}: {count} images ({count/len(results)*100:.1f}%)")
        else:
            print(f"❌ Folder not found: {folder_path}")
    
    elif choice == '3':
        # Use sample from test dataset
        test_path = "data/raw/Testing"
        
        if os.path.exists(test_path):
            # Get one image from each class
            classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
            class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
            
            for class_folder, class_name in zip(classes, class_names):
                class_path = os.path.join(test_path, class_folder)
                if os.path.exists(class_path):
                    images = [f for f in os.listdir(class_path) 
                             if f.endswith(('.jpg', '.jpeg', '.png'))]
                    if images:
                        image_path = os.path.join(class_path, images[0])
                        print(f"\n📷 Testing with {class_name} image: {images[0]}")
                        result = predictor.predict(image_path)
                        predictor.visualize_prediction(result)
                        
                        if choice == '3' and class_folder != 'pituitary':
                            input("\nPress Enter to continue to next image...")
        else:
            print(f"❌ Test dataset not found at {test_path}")

if __name__ == "__main__":
    main()