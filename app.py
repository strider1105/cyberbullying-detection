from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import pandas as pd
from models.cyberbully_detector import CyberbullyingDetector
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'cyberbully_detection_secret_key_2024'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['IMAGE_UPLOAD_FOLDER'] = 'uploads/images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGE_UPLOAD_FOLDER'], exist_ok=True)

# Initialize the ML detector
detector = CyberbullyingDetector()

# Simple user credentials (extend to database in production)
USERS = {
    'admin': 'password123',
    'user': 'demo123'
}

def get_enhanced_training_data():
    """Get enhanced training data with more examples for better detection"""
    return {
        'text': [
            # Safe/Non-offensive examples (0)
            "You're doing great! Keep it up!",
            "Let's meet for coffee tomorrow",
            "Great job on your presentation!",
            "Thanks for helping me today",
            "Have a wonderful day!",
            "I love spending time with friends",
            "What a beautiful sunny day!",
            "Thanks for being such a good friend",
            "You look nice today",
            "Good luck with your exam",
            "Hope you have a great weekend",
            "Congratulations on your achievement",
            "Nice work on the project",
            "See you later",
            "How are you doing",
            "That's really cool",
            "I appreciate your help",
            "You're very talented",
            "Thanks for listening",
            "Have a good day",
            "You're awesome",
            "Great to see you",
            "Keep up the good work",
            "You're so helpful",
            "I'm proud of you",
            
            # Offensive/Cyberbullying examples (1)
            "I hate you so much, you're worthless",
            "Nobody likes you, loser",
            "You're so stupid and ugly",
            "You're pathetic and worthless",
            "Go kill yourself, nobody cares",
            "You should just disappear forever",
            "You're a complete failure at everything",
            "I wish you would just die already",
            "You're disgusting and pathetic",
            "Everyone thinks you're a joke",
            "You're such a waste of space",
            "No one will ever love you",
            "You should be ashamed of yourself",
            "You're the worst person ever",
            "I hope you suffer",
            "You deserve to be alone",
            "You're absolutely useless",
            "Go away nobody wants you here",
            "You make me sick",
            "You're a total freak",
            "Kill yourself",
            "You're trash",
            "Nobody cares about you",
            "You're so dumb",
            "I hate your face",
            "You're disgusting",
            "Go die",
            "You suck",
            "You're annoying",
            "Shut up loser"
        ],
        'label': [
            # Labels: 0 = safe, 1 = offensive
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 25 Safe examples
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1   # 30 Offensive examples
        ]
    }

def login_required(f):
    """Decorator to require login for protected routes"""
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@app.route('/')
def index():
    """Redirect to dashboard if logged in, otherwise to login"""
    if 'logged_in' in session and session['logged_in']:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user authentication"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in USERS and USERS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            session['login_time'] = datetime.now().isoformat()
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Handle user logout"""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard page"""
    current_date = datetime.now().strftime('%b %d, %Y')
    return render_template('dashboard.html', current_date=current_date)

@app.route('/predict')
@login_required
def predict():
    """Text prediction interface"""
    return render_template('predict.html')

@app.route('/charts')
@login_required
def charts():
    """Data visualization page"""
    return render_template('charts.html')

@app.route('/dataset')
@login_required
def dataset():
    """Dataset management page"""
    return render_template('dataset.html')

# API Endpoints

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """API endpoint for text prediction"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Try to load existing model or create sample prediction
        try:
            if os.path.exists('cyberbully_model.h5'):
                detector.load_model('cyberbully_model')
            else:
                # Create and train with sample data for demo
                sample_data = {
                    'text': [
                        "You're doing great! Keep it up!",
                        "I hate you so much, you're worthless",
                        "Let's meet for coffee tomorrow",
                        "Nobody likes you, loser",
                        "Great job on your presentation!",
                        "You're so stupid and ugly",
                        "Thanks for helping me today",
                        "You're pathetic and worthless",
                        "Have a wonderful day!",
                        "Go kill yourself, nobody cares",
                        "I love spending time with friends",
                        "You should just disappear forever",
                        "What a beautiful sunny day!",
                        "You're a complete failure at everything",
                        "Thanks for being such a good friend",
                        "I wish you would just die already"
                    ],
                    'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                }
                df_sample = pd.DataFrame(sample_data)
                X, y = df_sample['text'], df_sample['label']
                
                # Train the TensorFlow 2.x model
                detector.train(X, y, epochs=50, batch_size=4)
                detector.save_model('cyberbully_model')
            
            result = detector.predict(text)
            return jsonify(result)
            
        except Exception as e:
            # Fallback to simple rule-based prediction for demo
            offensive_words = ['hate', 'stupid', 'ugly', 'loser', 'worthless', 'kill', 'die']
            is_offensive = any(word in text.lower() for word in offensive_words)
            
            return jsonify({
                'text': text,
                'prediction': 'Offensive' if is_offensive else 'Non-Offensive',
                'label': 1 if is_offensive else 0,
                'confidence': 0.85 if is_offensive else 0.92
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart-data')
@login_required
def api_chart_data():
    """API endpoint for chart data"""
    # Mock data for demonstration
    data = {
        'distribution': {
            'safe': 71.5,
            'offensive': 28.5
        },
        'categories': {
            'labels': ['Harassment', 'Hate Speech', 'Threats', 'Discrimination', 'Other'],
            'data': [89, 67, 45, 78, 76]
        },
        'trends': {
            'labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            'safe': [45, 52, 38, 67, 73, 29, 41],
            'offensive': [18, 23, 15, 27, 31, 12, 17]
        }
    }
    return jsonify(data)

@app.route('/api/upload-dataset', methods=['POST'])
@login_required
def api_upload_dataset():
    """API endpoint for dataset upload"""
    try:
        if 'dataset' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400
        
        file = request.files['dataset']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'success': False, 'message': 'File must be a CSV'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read and validate CSV
        try:
            df = pd.read_csv(filepath)
            
            # Check required columns
            if 'text' not in df.columns or 'label' not in df.columns:
                return jsonify({
                    'success': False, 
                    'message': 'CSV must contain "text" and "label" columns'
                }), 400
            
            # Calculate statistics
            total_rows = len(df)
            safe_count = len(df[df['label'] == 0])
            offensive_count = len(df[df['label'] == 1])
            
            # Get sample rows
            sample_rows = df.head(10).to_dict('records')
            
            return jsonify({
                'success': True,
                'file_path': filepath,
                'total_rows': total_rows,
                'safe_count': safe_count,
                'offensive_count': offensive_count,
                'sample_rows': sample_rows
            })
            
        except Exception as e:
            return jsonify({
                'success': False, 
                'message': f'Error reading CSV: {str(e)}'
            }), 400
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/train-model', methods=['POST'])
@login_required
def api_train_model():
    """API endpoint for model training"""
    try:
        data = request.get_json()
        dataset_path = data.get('dataset_path')
        model_type = data.get('model_type', 'logistic')
        
        if not dataset_path or not os.path.exists(dataset_path):
            return jsonify({'success': False, 'message': 'Dataset file not found'}), 400
        
        # Load and train model
        X, y = detector.load_data(dataset_path)
        X_test, y_test, y_pred = detector.train(X, y, model_type=model_type)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Save model
        detector.save_model('cyberbully_model')
        
        return jsonify({
            'success': True,
            'results': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/predict-image', methods=['POST'])
@login_required
def api_predict_image():
    """API endpoint for image + text prediction"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        additional_text = request.form.get('text', '').strip()
        
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Check file extension
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Invalid image format. Supported: PNG, JPG, JPEG, GIF, BMP, TIFF'}), 400
        
        # Save uploaded image
        filename = secure_filename(file.filename)
        # Add timestamp to avoid filename conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load existing model or create demo model
            if os.path.exists('cyberbully_model.h5'):
                detector.load_model('cyberbully_model')
            else:
                # Create and train with sample data for demo
                sample_data = {
                    'text': [
                        "You're doing great! Keep it up!",
                        "I hate you so much, you're worthless",
                        "Let's meet for coffee tomorrow",
                        "Nobody likes you, loser",
                        "Great job on your presentation!",
                        "You're so stupid and ugly",
                        "Thanks for helping me today",
                        "You're pathetic and worthless",
                        "Have a wonderful day!",
                        "Go kill yourself, nobody cares",
                        "I love spending time with friends",
                        "You should just disappear forever",
                        "What a beautiful sunny day!",
                        "You're a complete failure at everything",
                        "Thanks for being such a good friend",
                        "I wish you would just die already"
                    ],
                    'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                }
                df_sample = pd.DataFrame(sample_data)
                X, y = df_sample['text'], df_sample['label']
                
                # Train the TensorFlow 2.x model
                detector.train(X, y, epochs=50, batch_size=4)
                detector.save_model('cyberbully_model')
            
            # Predict from image
            result = detector.predict_from_image(filepath, additional_text)
            
            # Clean up uploaded file after processing
            try:
                os.remove(filepath)
            except:
                pass  # Don't fail if file cleanup fails
            
            return jsonify(result)
            
        except Exception as e:
            # Clean up uploaded file on error
            try:
                os.remove(filepath)
            except:
                pass
            
            # Fallback error response
            return jsonify({
                'error': f'Image processing failed: {str(e)}',
                'text': additional_text,
                'prediction': 'Error',
                'label': -1,
                'confidence': 0.0
            }), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)