#!/usr/bin/env python3
"""
Setup script for TensorFlow 2.20.0 Cyberbullying Detection System
This script helps set up the environment and verify TensorFlow installation
"""

import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is compatible with TensorFlow 2.20.0"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("❌ ERROR: TensorFlow 2.20.0 requires Python 3.8 or higher")
        print("   Current version:", f"{version.major}.{version.minor}.{version.micro}")
        print("   Please install Python 3.8+ and try again.")
        return False
    
    print("✅ Python version is compatible with TensorFlow 2.20.0")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def verify_tensorflow():
    """Verify TensorFlow installation"""
    print("\n🔍 Verifying TensorFlow installation...")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Test basic TensorFlow operations
        hello = tf.constant('Hello, TensorFlow 2.x!')
        print(f"✅ TensorFlow eager execution test: {hello.numpy().decode()}")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU devices available: {len(gpus)}")
        else:
            print("ℹ️  No GPU devices found (CPU-only mode)")
        
        return True
    except ImportError:
        print("❌ TensorFlow not found. Please install it manually:")
        print("   pip install tensorflow==2.20.0")
        return False
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✅ NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False

def test_model():
    """Test the cyberbullying detection model"""
    print("\n🧠 Testing cyberbullying detection model...")
    try:
        from models.cyberbully_detector import CyberbullyingDetector
        
        detector = CyberbullyingDetector()
        print("✅ Model class imported successfully")
        
        # Test with sample data
        import pandas as pd
        sample_data = {
            'text': [
                "You're doing great!",
                "I hate you so much",
                "Have a nice day",
                "You're worthless"
            ],
            'label': [0, 1, 0, 1]
        }
        
        df = pd.DataFrame(sample_data)
        X, y = df['text'], df['label']
        
        print("🏋️ Training small test model...")
        detector.train(X, y, epochs=10, batch_size=2)
        
        # Test prediction
        result = detector.predict("This is a test message")
        print(f"✅ Test prediction: {result['prediction']} (confidence: {result['confidence']:.2f})")
        
        # Display model summary
        print("\n📊 Model Architecture:")
        detector.get_model_summary()
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 TensorFlow 2.20.0 Cyberbullying Detection System Setup")
    print("=" * 60)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Install requirements
    if success and not install_requirements():
        success = False
    
    # Verify TensorFlow
    if success and not verify_tensorflow():
        success = False
    
    # Download NLTK data
    if success and not download_nltk_data():
        success = False
    
    # Test model
    if success and not test_model():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Setup completed successfully!")
        print("\nYou can now run the application:")
        print("   python app.py")
        print("\nThen open your browser to: http://localhost:5000")
    else:
        print("❌ Setup failed. Please check the errors above and try again.")
        print("\nFor help, check the README.md file or the troubleshooting section.")
    
    return success

if __name__ == "__main__":
    main()