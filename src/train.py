"""
Training Module for Brain MRI Classification
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import BrainMRIDataLoader
from src.model_builder import BrainMRIModel

def main():
    """
    Main training pipeline
    """
    
    print("=" * 70)
    print("🧠 AI-POWERED BRAIN MRI CLASSIFICATION SYSTEM")
    print("=" * 70)
    print("\nThis system will classify Brain MRI images into 4 categories:")
    print("   1. Glioma Tumor")
    print("   2. Meningioma Tumor") 
    print("   3. No Tumor")
    print("   4. Pituitary Tumor")
    print("\n" + "=" * 70)
    
    # Create output directories if they don't exist
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Load and prepare data
    print("\n📊 STEP 1: Loading and Preparing Data...")
    loader = BrainMRIDataLoader(img_size=(224, 224), batch_size=32)
    
    # Check class distribution
    class_counts = loader.get_class_distribution()
    
    # Create data generators
    train_gen, val_gen, test_gen = loader.create_data_generators()
    
    # Visualize sample images
    print("\n🖼️ STEP 2: Visualizing Sample Images...")
    loader.visualize_samples(train_gen, num_samples=9)
    
    # Step 3: Build model
    print("\n🏗️ STEP 3: Building Model...")
    model_builder = BrainMRIModel(input_shape=(224, 224, 3), num_classes=4)
    
    # Choose model type (comment/uncomment as needed)
    print("\nChoose model type:")
    print("1. Custom CNN (from scratch)")
    print("2. Transfer Learning with VGG16 (Recommended)")
    print("3. Transfer Learning with ResNet50")
    print("4. Transfer Learning with EfficientNetB0")
    
    model_choice = input("\nEnter your choice (1-4, default is 2): ").strip()
    
    if model_choice == '1':
        model_builder.build_cnn_model()
    elif model_choice == '3':
        model_builder.build_transfer_learning_model('ResNet50')
    elif model_choice == '4':
        model_builder.build_transfer_learning_model('EfficientNetB0')
    else:
        model_builder.build_transfer_learning_model('VGG16')  # Default
    
    # Step 4: Train model
    print("\n🎯 STEP 4: Training Model...")
    epochs = int(input("\nEnter number of epochs (recommended: 20-30, default 20): ") or "20")
    
    history = model_builder.train_model(train_gen, val_gen, epochs=epochs)
    
    # Step 5: Plot training history
    print("\n📈 STEP 5: Plotting Training History...")
    model_builder.plot_training_history()
    
    # Step 6: Evaluate on test data
    if test_gen:
        print("\n🧪 STEP 6: Evaluating on Test Data...")
        test_loss, test_accuracy, test_precision, test_recall = model_builder.model.evaluate(test_gen)
        
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Test Loss:      {test_loss:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall:    {test_recall:.4f}")
    
    # Step 7: Save final model
    print("\n💾 STEP 7: Saving Model...")
    model_builder.model.save('models/final_brain_mri_model.h5')
    print("✅ Model saved to: models/final_brain_mri_model.h5")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run 'python src/predict.py' to test on new images")
    print("2. Run 'python main.py' to start the web application")
    print("3. Check 'outputs/figures/' for all visualizations")

if __name__ == "__main__":
    main()