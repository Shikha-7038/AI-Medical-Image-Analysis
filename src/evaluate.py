"""
Evaluation Module for Brain MRI Classification
Generates comprehensive evaluation metrics and visualizations
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import BrainMRIDataLoader

class ModelEvaluator:
    """
    Class to evaluate the trained model
    """
    
    def __init__(self, model_path='models/final_brain_mri_model.h5'):
        """
        Initialize the evaluator
        
        Args:
            model_path: Path to the trained model
        """
        self.model_path = model_path
        self.model = None
        self.class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        self.full_class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        if os.path.exists(self.model_path):
            print(f"✅ Loading model from: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            print(f"❌ Model not found at {self.model_path}")
            exit(1)
    
    def get_test_data(self):
        """Get test data generator"""
        loader = BrainMRIDataLoader(img_size=(224, 224), batch_size=32)
        _, _, test_gen = loader.create_data_generators()
        return test_gen
    
    def plot_confusion_matrix(self, test_generator):
        """
        Generate and plot confusion matrix
        """
        print("\n📊 Generating Confusion Matrix...")
        
        # Get predictions
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.full_class_names, 
                    yticklabels=self.full_class_names)
        plt.title('Confusion Matrix - Brain MRI Classification', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        os.makedirs('outputs/figures', exist_ok=True)
        plt.savefig('outputs/figures/confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Confusion matrix saved to: outputs/figures/confusion_matrix.png")
        
        return cm, true_classes, predicted_classes
    
    def plot_classification_report(self, true_classes, predicted_classes):
        """
        Generate and save classification report
        """
        print("\n📈 Generating Classification Report...")
        
        # Generate classification report
        report = classification_report(true_classes, predicted_classes, 
                                       target_names=self.full_class_names)
        
        # Save to file
        os.makedirs('outputs/results', exist_ok=True)
        with open('outputs/results/classification_report.txt', 'w') as f:
            f.write("CLASSIFICATION REPORT - BRAIN MRI CLASSIFICATION\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
        
        print(report)
        print("\n✅ Classification report saved to: outputs/results/classification_report.txt")
    
    def plot_roc_curves(self, test_generator):
        """
        Plot ROC curves for each class
        """
        print("\n📉 Generating ROC Curves...")
        
        # Get predictions
        predictions = self.model.predict(test_generator, verbose=1)
        true_classes = tf.keras.utils.to_categorical(test_generator.classes, num_classes=4)
        
        # Compute ROC curve and ROC area for each class
        plt.figure(figsize=(10, 8))
        
        for i in range(4):
            fpr, tpr, _ = roc_curve(true_classes[:, i], predictions[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{self.full_class_names[i]} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Brain MRI Classification', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('outputs/figures/roc_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ ROC curves saved to: outputs/figures/roc_curves.png")
    
    def calculate_metrics(self, test_generator):
        """
        Calculate and display comprehensive metrics
        """
        print("\n📊 Calculating Comprehensive Metrics...")
        
        # Get predictions
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Calculate per-class metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(true_classes, predicted_classes, average=None)
        recall = recall_score(true_classes, predicted_classes, average=None)
        f1 = f1_score(true_classes, predicted_classes, average=None)
        
        # Create metrics table
        print("\n" + "=" * 80)
        print("PER-CLASS METRICS")
        print("=" * 80)
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 80)
        
        for i, class_name in enumerate(self.full_class_names):
            print(f"{class_name:<20} {precision[i]:.4f}      {recall[i]:.4f}      {f1[i]:.4f}")
        
        # Overall metrics
        print("\n" + "=" * 80)
        print("OVERALL METRICS")
        print("=" * 80)
        print(f"Average Precision: {np.mean(precision):.4f}")
        print(f"Average Recall: {np.mean(recall):.4f}")
        print(f"Average F1-Score: {np.mean(f1):.4f}")
        
        # Save to file
        os.makedirs('outputs/results', exist_ok=True)
        with open('outputs/results/metrics_summary.txt', 'w') as f:
            f.write("METRICS SUMMARY - BRAIN MRI CLASSIFICATION\n")
            f.write("=" * 60 + "\n\n")
            f.write("Per-Class Metrics:\n")
            for i, class_name in enumerate(self.full_class_names):
                f.write(f"{class_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}\n")
            f.write(f"\nAverage Precision: {np.mean(precision):.4f}\n")
            f.write(f"Average Recall: {np.mean(recall):.4f}\n")
            f.write(f"Average F1-Score: {np.mean(f1):.4f}\n")
    
    def plot_misclassified_examples(self, test_generator, num_examples=8):
        """
        Display misclassified examples for analysis
        """
        print("\n🔍 Finding misclassified examples...")
        
        # Get predictions
        predictions = self.model.predict(test_generator, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Find misclassified indices
        misclassified_indices = np.where(predicted_classes != true_classes)[0]
        
        if len(misclassified_indices) == 0:
            print("🎉 No misclassified examples found! Perfect classification!")
            return
        
        # Get misclassified images
        num_to_show = min(num_examples, len(misclassified_indices))
        selected_indices = np.random.choice(misclassified_indices, num_to_show, replace=False)
        
        # Get the corresponding images
        test_generator.reset()
        all_images = []
        all_labels = []
        for i in range(len(test_generator)):
            images, labels = test_generator[i]
            all_images.extend(images)
            all_labels.extend(labels)
        
        # Plot misclassified examples
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for i, idx in enumerate(selected_indices):
            if i >= num_to_show:
                break
            
            image = all_images[idx]
            true_label = self.full_class_names[true_classes[idx]]
            pred_label = self.full_class_names[predicted_classes[idx]]
            confidence = predictions[idx][predicted_classes[idx]]
            
            axes[i].imshow(image)
            axes[i].axis('off')
            axes[i].set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}", 
                             fontsize=10, color='red')
        
        plt.suptitle("Misclassified Examples - Review for Improvement", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/figures/misclassified_examples.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Misclassified examples saved to: outputs/figures/misclassified_examples.png")

def main():
    """Main evaluation function"""
    
    print("=" * 70)
    print("🧠 BRAIN MRI MODEL EVALUATION")
    print("=" * 70)
    
    # Create output directories
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Get test data
    test_generator = evaluator.get_test_data()
    
    if test_generator is None:
        print("❌ Test data not found. Please ensure dataset is properly set up.")
        return
    
    # Generate all evaluation metrics
    cm, true_classes, predicted_classes = evaluator.plot_confusion_matrix(test_generator)
    evaluator.plot_classification_report(true_classes, predicted_classes)
    evaluator.plot_roc_curves(test_generator)
    evaluator.calculate_metrics(test_generator)
    evaluator.plot_misclassified_examples(test_generator)
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 70)
    print("\n📁 Results saved in:")
    print("   - outputs/figures/confusion_matrix.png")
    print("   - outputs/figures/roc_curves.png")
    print("   - outputs/figures/misclassified_examples.png")
    print("   - outputs/results/classification_report.txt")
    print("   - outputs/results/metrics_summary.txt")

if __name__ == "__main__":
    main()