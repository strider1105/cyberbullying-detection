import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import os

# Image processing imports
try:
    from PIL import Image
    import pytesseract
    import cv2
    import platform
    
    # Configure Tesseract path for Windows
    if platform.system() == "Windows":
        # Common Tesseract installation paths on Windows
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.environ.get('USERNAME', '')),
            r'tesseract.exe'  # If in PATH
        ]
        
        tesseract_found = False
        for path in possible_paths:
            if os.path.exists(path) or path == 'tesseract.exe':
                try:
                    pytesseract.pytesseract.tesseract_cmd = path
                    # Test if it works
                    pytesseract.get_tesseract_version()
                    tesseract_found = True
                    print(f"✅ Tesseract found at: {path}")
                    break
                except:
                    continue
        
        if not tesseract_found:
            print("⚠️  Tesseract OCR not found. Image text extraction will not work.")
            print("   Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False
    print("Warning: Image processing libraries not available. Install Pillow, pytesseract, and opencv-python for image support.")

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure GPU memory growth (if available)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

class CyberbullyingDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words 
                 if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def load_data(self, filepath, text_column='text', label_column='label'):
        """
        Load dataset from CSV file
        Expected format:
        - text_column: contains the text messages
        - label_column: contains labels (0=non-offensive, 1=offensive)
        """
        df = pd.read_csv(filepath)
        
        # Preprocess all texts
        print("Preprocessing texts...")
        df['cleaned_text'] = df[text_column].apply(self.preprocess_text)
        
        X = df['cleaned_text']
        y = df[label_column]
        
        return X, y
    
    def _build_tensorflow_model(self, input_dim):
        """Build TensorFlow 2.x neural network model using Keras"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,), name='hidden_layer_1'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu', name='hidden_layer_2'),
            layers.Dropout(0.3),
            layers.Dense(2, activation='softmax', name='output_layer')
        ])
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self, X, y, model_type='tensorflow', test_size=0.2, epochs=100, batch_size=32):
        """
        Train the cyberbullying detection model using TensorFlow 2.x
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Vectorize text
        print("\nVectorizing text...")
        X_train_vec = self.vectorizer.fit_transform(X_train).toarray()
        X_test_vec = self.vectorizer.transform(X_test).toarray()
        
        # Convert to numpy arrays
        y_train_array = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
        y_test_array = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
        
        # Build TensorFlow model
        print(f"\nBuilding TensorFlow 2.x neural network...")
        input_dim = X_train_vec.shape[1]
        self.model = self._build_tensorflow_model(input_dim)
        
        print(f"\nTraining neural network for {epochs} epochs...")
        
        # Create callbacks for better training monitoring
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train the model
        history = self.model.fit(
            X_train_vec, y_train_array,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("\nEvaluating model...")
        test_loss, test_accuracy = self.model.evaluate(X_test_vec, y_test_array, verbose=0)
        
        # Get predictions
        test_predictions = self.model.predict(X_test_vec, verbose=0)
        y_pred = np.argmax(test_predictions, axis=1)
        
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test_array, y_pred, 
                                   target_names=['Non-Offensive', 'Offensive']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test_array, y_pred))
        
        return X_test, y_test_array, y_pred
    
    def predict(self, text):
        """Predict if a single text is offensive or non-offensive"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Preprocess
        cleaned = self.preprocess_text(text)
        
        # Vectorize
        vectorized = self.vectorizer.transform([cleaned]).toarray()
        
        # Predict using TensorFlow 2.x model
        prediction_probs = self.model.predict(vectorized, verbose=0)
        
        prediction = np.argmax(prediction_probs[0])
        confidence = prediction_probs[0][prediction]
        
        return {
            'text': text,
            'prediction': 'Offensive' if prediction == 1 else 'Non-Offensive',
            'label': int(prediction),
            'confidence': float(confidence)
        }
    
    def predict_batch(self, texts):
        """Predict multiple texts at once"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
    
    def save_model(self, filepath='cyberbully_model'):
        """Save trained TensorFlow 2.x model and vectorizer"""
        if self.model is None:
            raise ValueError("No trained model to save!")
        
        # Save TensorFlow 2.x model
        self.model.save(filepath + '.h5')
        
        # Save vectorizer
        with open(filepath + '_vectorizer.pkl', 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'input_dim': self.vectorizer.max_features
            }, f)
        
        print(f"TensorFlow 2.x model saved to {filepath}.h5")
        print(f"Vectorizer saved to {filepath}_vectorizer.pkl")
    
    def load_model(self, filepath='cyberbully_model'):
        """Load trained TensorFlow 2.x model and vectorizer"""
        try:
            # Load vectorizer first
            with open(filepath + '_vectorizer.pkl', 'rb') as f:
                data = pickle.load(f)
                self.vectorizer = data['vectorizer']
            
            # Load TensorFlow 2.x model
            self.model = keras.models.load_model(filepath + '.h5')
            
            print(f"TensorFlow 2.x model loaded from {filepath}.h5")
            print(f"Vectorizer loaded from {filepath}_vectorizer.pkl")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to creating a simple model for demo
            self._create_demo_model()
    
    def _create_demo_model(self):
        """Create a simple demo model for testing"""
        print("Creating demo model...")
        
        # Create simple vectorizer
        sample_texts = [
            "You're doing great! Keep it up!",
            "I hate you so much, you're worthless",
            "Let's meet for coffee tomorrow",
            "Nobody likes you, loser"
        ]
        self.vectorizer.fit(sample_texts)
        
        # Build simple TensorFlow model
        input_dim = len(self.vectorizer.get_feature_names_out())
        self.model = self._build_tensorflow_model(input_dim)
        
        print("Demo model created successfully!")
    
    def extract_text_from_image(self, image_path):
        """Extract text from image using OCR"""
        if not IMAGE_PROCESSING_AVAILABLE:
            raise ValueError("Image processing libraries not installed. Please install Pillow, pytesseract, and opencv-python.")
        
        try:
            # Check if Tesseract is available
            try:
                pytesseract.get_tesseract_version()
            except Exception as e:
                raise ValueError(f"Tesseract OCR not found. Please install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki. Error: {str(e)}")
            
            # Read image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image file. Please check the file format.")
            
            # Convert to RGB (OpenCV uses BGR by default)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Preprocess image for better OCR
            # Convert to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply noise reduction
            denoised = cv2.medianBlur(gray_image, 5)
            
            # Apply thresholding to get better text recognition
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to PIL for pytesseract
            processed_image = Image.fromarray(thresh)
            
            # Extract text using pytesseract with multiple configurations
            configs = [
                '--psm 6',  # Uniform block of text
                '--psm 8',  # Single word
                '--psm 13', # Raw line
                '--psm 3'   # Default
            ]
            
            extracted_text = ""
            for config in configs:
                try:
                    text = pytesseract.image_to_string(processed_image, config=config)
                    if text.strip():
                        extracted_text = text.strip()
                        break
                except:
                    continue
            
            # If no text found, try with original image
            if not extracted_text:
                try:
                    extracted_text = pytesseract.image_to_string(pil_image, config='--psm 6').strip()
                except:
                    pass
            
            return extracted_text
            
        except Exception as e:
            raise ValueError(f"Error extracting text from image: {str(e)}")
    
    def predict_from_image(self, image_path, additional_text=""):
        """Predict cyberbullying from image (extract text first) with optional additional text"""
        # Extract text from image
        extracted_text = self.extract_text_from_image(image_path)
        
        # Combine extracted text with additional text if provided
        combined_text = extracted_text
        if additional_text.strip():
            combined_text = f"{extracted_text} {additional_text}".strip()
        
        if not combined_text.strip():
            return {
                'text': combined_text,
                'extracted_text': extracted_text,
                'additional_text': additional_text,
                'prediction': 'No Text Found',
                'label': -1,
                'confidence': 0.0,
                'error': 'No text could be extracted from the image'
            }
        
        # Use existing predict method
        result = self.predict(combined_text)
        
        # Add image-specific information to result
        result['extracted_text'] = extracted_text
        result['additional_text'] = additional_text
        result['input_type'] = 'image_with_text' if additional_text.strip() else 'image_only'
        
        return result

    def extract_text_from_image(self, image_path):
        """Extract text from image using OCR (with fallback for missing Tesseract)"""
        if not IMAGE_PROCESSING_AVAILABLE:
            return "[Image processing not available - missing libraries]"
        
        try:
            # Check if Tesseract is available
            try:
                pytesseract.get_tesseract_version()
                tesseract_available = True
            except Exception:
                tesseract_available = False
            
            if not tesseract_available:
                # Return a message indicating OCR is not available
                return "[OCR not available - Tesseract not installed. Please enter text manually below.]"
            
            # Read image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                return "[Could not read image file]"
            
            # Convert to RGB (OpenCV uses BGR by default)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Preprocess image for better OCR
            # Convert to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply noise reduction
            denoised = cv2.medianBlur(gray_image, 5)
            
            # Apply thresholding to get better text recognition
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to PIL for pytesseract
            processed_image = Image.fromarray(thresh)
            
            # Extract text using pytesseract with multiple configurations
            configs = [
                '--psm 6',  # Uniform block of text
                '--psm 8',  # Single word
                '--psm 13', # Raw line
                '--psm 3'   # Default
            ]
            
            extracted_text = ""
            for config in configs:
                try:
                    text = pytesseract.image_to_string(processed_image, config=config)
                    if text.strip():
                        extracted_text = text.strip()
                        break
                except:
                    continue
            
            # If no text found, try with original image
            if not extracted_text:
                try:
                    extracted_text = pytesseract.image_to_string(pil_image, config='--psm 6').strip()
                except:
                    pass
            
            return extracted_text if extracted_text else "[No text detected in image]"
            
        except Exception as e:
            # Return a user-friendly message instead of raising an error
            return f"[Error reading image: {str(e)}]"
    
    def predict_from_image(self, image_path, additional_text=""):
        """Predict cyberbullying from image (extract text first) with optional additional text"""
        # Extract text from image
        extracted_text = self.extract_text_from_image(image_path)
        
        # Combine extracted text with additional text if provided
        combined_text = extracted_text
        if additional_text.strip():
            if extracted_text.startswith('[') and extracted_text.endswith(']'):
                # If extracted text is an error/info message, use only additional text
                combined_text = additional_text.strip()
            else:
                combined_text = f"{extracted_text} {additional_text}".strip()
        
        # Check if we have any actual text to analyze
        if not combined_text.strip() or (combined_text.startswith('[') and combined_text.endswith(']')):
            return {
                'text': combined_text,
                'extracted_text': extracted_text,
                'additional_text': additional_text,
                'prediction': 'No Text Available',
                'label': -1,
                'confidence': 0.0,
                'error': 'No text available for analysis. Please enter text manually or install Tesseract OCR for automatic text extraction.'
            }
        
        # Use existing predict method
        result = self.predict(combined_text)
        
        # Add image-specific information to result
        result['extracted_text'] = extracted_text
        result['additional_text'] = additional_text
        result['input_type'] = 'image_with_text' if additional_text.strip() else 'image_only'
        
        return result

    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is not None:
            return self.model.summary()
        else:
            return "No model available"