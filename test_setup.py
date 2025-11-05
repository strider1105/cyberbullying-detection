#!/usr/bin/env python3
"""
Test script to verify all dependencies are working
"""

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import flask
        print(f"✅ Flask {flask.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        import nltk
        print(f"✅ NLTK {nltk.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ NLTK import failed: {e}")
        return False
    
    return True

def test_nltk_data():
    """Test if NLTK data is available"""
    print("\nTesting NLTK data...")
    
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        
        # Test stopwords
        stop_words = stopwords.words('english')
        print(f"✅ NLTK stopwords loaded ({len(stop_words)} words)")
        
        # Test lemmatizer
        lemmatizer = WordNetLemmatizer()
        test_word = lemmatizer.lemmatize('running')
        print(f"✅ NLTK lemmatizer working (running -> {test_word})")
        
        return True
    except Exception as e:
        print(f"❌ NLTK data test failed: {e}")
        return False

def test_tensorflow():
    """Test basic TensorFlow functionality"""
    print("\nTesting TensorFlow...")
    
    try:
        import tensorflow as tf
        
        # Test basic operation
        hello = tf.constant('Hello, TensorFlow!')
        print(f"✅ TensorFlow basic operation: {hello.numpy().decode()}")
        
        # Test Keras
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(1,))
        ])
        print("✅ Keras model creation successful")
        
        return True
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

def test_model_import():
    """Test if our cyberbullying detector can be imported"""
    print("\nTesting cyberbullying detector import...")
    
    try:
        from models.cyberbully_detector import CyberbullyingDetector
        detector = CyberbullyingDetector()
        print("✅ CyberbullyingDetector imported and initialized successfully")
        return True
    except Exception as e:
        print(f"❌ CyberbullyingDetector import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Cyberbullying Detection System Setup")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test NLTK data
    if not test_nltk_data():
        all_passed = False
    
    # Test TensorFlow
    if not test_tensorflow():
        all_passed = False
    
    # Test model import
    if not test_model_import():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! The system is ready to run.")
        print("\nYou can now start the application with:")
        print("   python app.py")
        print("\nThen open your browser to: http://localhost:5000")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTry running the installation script:")
        print("   install_and_run.bat")
    
    return all_passed

if __name__ == "__main__":
    main()