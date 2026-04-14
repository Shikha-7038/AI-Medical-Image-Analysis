#!/usr/bin/env python3
"""
AI-Powered Medical Image Analysis System
Main entry point for the application

This script provides a menu-driven interface to:
1. Train the model
2. Make predictions
3. Evaluate the model
4. Run the web application
"""

import os
import sys
import subprocess

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🧠 AI-POWERED MEDICAL IMAGE ANALYSIS SYSTEM               ║
    ║                                                              ║
    ║   Brain MRI Classification for Tumor Detection              ║
    ║                                                              ║
    ║   Classes: Glioma | Meningioma | No Tumor | Pituitary       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_menu():
    """Print main menu"""
    menu = """
    ┌─────────────────────────────────────────────────────────────┐
    │                      MAIN MENU                              │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │   1. 🚀 Train Model                                         │
    │   2. 🔍 Make Predictions                                    │
    │   3. 📊 Evaluate Model                                      │
    │   4. 🌐 Run Web Application (Flask)                         │
    │   5. 📁 Check Dataset                                       │
    │   6. ❌ Exit                                                │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """
    print(menu)

def check_dataset():
    """Check if dataset exists and is properly structured"""
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)
    
    train_path = "data/raw/Training"
    test_path = "data/raw/Testing"
    
    if os.path.exists(train_path):
        print("✅ Training folder found")
        
        # Count images in each class
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        
        print("\n📊 Training Data Distribution:")
        for class_name, display_name in zip(classes, class_names):
            class_path = os.path.join(train_path, class_name)
            if os.path.exists(class_path):
                count = len([f for f in os.listdir(class_path) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))])
                print(f"   {display_name}: {count} images")
            else:
                print(f"   ❌ Missing: {display_name}")
    else:
        print("❌ Training folder not found!")
        print("\nPlease download the dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri")
        print("\nExtract it to: data/raw/")
        return False
    
    if os.path.exists(test_path):
        print("\n✅ Testing folder found")
    else:
        print("\n⚠️ Testing folder not found (optional for training)")
    
    print("\n" + "=" * 60)
    return True

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n▶️ {description}...")
    print("-" * 40)
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function to run the application"""
    
    while True:
        clear_screen()
        print_banner()
        print_menu()
        
        choice = input("\n👉 Enter your choice (1-6): ").strip()
        
        if choice == '1':
            # Train Model
            print("\n" + "=" * 60)
            print("TRAINING MODE")
            print("=" * 60)
            print("\nThis will train the CNN model on Brain MRI images.")
            print("Estimated time: 20-30 minutes (depending on your system)")
            print("\n⚠️ Make sure you have:")
            print("   - Dataset in 'data/raw/Training/' folder")
            print("   - At least 4GB free RAM")
            print("   - GPU recommended but not required")
            
            confirm = input("\nContinue with training? (y/n): ").strip().lower()
            if confirm == 'y':
                run_command("python src/train.py", "Starting training")
            
            input("\nPress Enter to continue...")
        
        elif choice == '2':
            # Make Predictions
            print("\n" + "=" * 60)
            print("PREDICTION MODE")
            print("=" * 60)
            
            # Check if model exists
            if not os.path.exists('models/final_brain_mri_model.h5'):
                print("\n⚠️ No trained model found!")
                print("Please train the model first (Option 1)")
            else:
                run_command("python src/predict.py", "Starting prediction")
            
            input("\nPress Enter to continue...")
        
        elif choice == '3':
            # Evaluate Model
            print("\n" + "=" * 60)
            print("EVALUATION MODE")
            print("=" * 60)
            
            # Check if model exists
            if not os.path.exists('models/final_brain_mri_model.h5'):
                print("\n⚠️ No trained model found!")
                print("Please train the model first (Option 1)")
            else:
                run_command("python src/evaluate.py", "Starting evaluation")
            
            input("\nPress Enter to continue...")
        
        elif choice == '4':
            # Run Web Application
            print("\n" + "=" * 60)
            print("WEB APPLICATION MODE")
            print("=" * 60)
            
            # Check if model exists
            if not os.path.exists('models/final_brain_mri_model.h5'):
                print("\n⚠️ No trained model found!")
                print("Please train the model first (Option 1)")
            else:
                print("\n🚀 Starting Flask web server...")
                print("📍 Access the application at: http://127.0.0.1:5000")
                print("📍 Press CTRL+C to stop the server")
                print("\n" + "=" * 60)
                
                # Check if templates exist
                if not os.path.exists('templates/index.html'):
                    print("\n⚠️ Warning: templates/index.html not found!")
                    print("Web interface may not work correctly.")
                
                run_command("python app.py", "Starting web application")
            
            input("\nPress Enter to continue...")
        
        elif choice == '5':
            # Check Dataset
            check_dataset()
            input("\nPress Enter to continue...")
        
        elif choice == '6':
            # Exit
            print("\n👋 Thank you for using AI-Powered Medical Image Analysis System!")
            print("   Goodbye!\n")
            sys.exit(0)
        
        else:
            print("\n❌ Invalid choice! Please enter 1-6")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()