# CyberGuard - Cyberbullying Detection System

A comprehensive web-based cyberbullying detection system powered by machine learning and built with Flask.

## 🚀 Features

- **TensorFlow 2.20.0 Neural Network**: Modern deep learning-powered cyberbullying detection
- **Interactive Dashboard**: Comprehensive analytics and statistics
- **Dataset Management**: Upload and train custom neural network models
- **Data Visualization**: Interactive charts showing detection patterns and trends
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Professional UI**: Cyberbullying awareness-themed interface
- **TensorFlow 2.x Compatibility**: Optimized for latest TensorFlow 2.20.0 with Keras integration

## 📋 Prerequisites

- Python 3.8 or higher (TensorFlow 2.20.0 compatibility)
- pip (Python package installer)
- **Note**: TensorFlow 2.20.0 supports Python 3.8+. Recommended: Python 3.9-3.11

## 🛠️ Installation

### 1. Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd cyberbullying-detection-app

# Or download and extract the ZIP file
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Required Packages

**For Windows users:**
```bash
# Use Windows-specific requirements (recommended)
pip install -r requirements-windows.txt

# Or install manually if above fails:
pip install --upgrade pip
pip install tensorflow pandas numpy scikit-learn nltk flask
```

**For Linux/macOS users:**
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data (First Time Only)
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

### 5. Verify TensorFlow Installation
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"
```

## 🚀 Running the Application

### 1. Start the Flask Server
```bash
python app.py
```

### 2. Access the Application
Open your web browser and navigate to:
```
http://localhost:5000
```

### 3. Login
Use one of the demo accounts:
- **Admin**: Username: `admin`, Password: `password123`
- **User**: Username: `user`, Password: `demo123`

## 📁 Project Structure

```
cyberbullying-detection-app/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies (TensorFlow 1.14.0)
├── README.md                       # This file
├── models/
│   └── cyberbully_detector.py      # TensorFlow 2.x Keras neural network model
├── static/
│   ├── css/
│   │   └── style.css               # Main stylesheet
│   ├── js/
│   │   └── main.js                 # JavaScript functionality
│   └── images/
│       └── cyberbullying-bg.jpg    # Background image
├── templates/
│   ├── base.html                   # Base template
│   ├── login.html                  # Login page
│   ├── dashboard.html              # Main dashboard
│   ├── predict.html                # Text analysis page
│   ├── charts.html                 # Analytics dashboard
│   └── dataset.html                # Dataset management
└── uploads/                        # Uploaded datasets (created automatically)
```

## 🎯 Usage Guide

### 1. **Dashboard**
- View system statistics and recent activity
- Navigate to different sections of the application
- Monitor model performance metrics

### 2. **Text Analysis**
- Enter text content to analyze for cyberbullying
- View real-time predictions with confidence scores
- Try sample texts to see the system in action

### 3. **Analytics**
- View interactive charts and statistics
- Analyze cyberbullying patterns and trends
- Export data in CSV or PDF format

### 4. **Dataset Management**
- Upload CSV files with training data
- Required format: columns named 'text' and 'label' (0=safe, 1=offensive)
- Train new models with your custom datasets
- Monitor training progress and results

## 📊 Dataset Format

When uploading datasets, ensure your CSV file has the following structure:

```csv
text,label
"Great job on your presentation!",0
"You're so stupid and worthless",1
"Thanks for helping me today",0
"Nobody likes you, loser",1
```

- **text**: The message content to analyze
- **label**: 0 for safe/non-offensive, 1 for offensive/cyberbullying

## 🔧 API Endpoints

The application provides REST API endpoints:

- `POST /api/predict` - Analyze text for cyberbullying
- `GET /api/chart-data` - Get analytics data for charts
- `POST /api/upload-dataset` - Upload training dataset
- `POST /api/train-model` - Train model with uploaded data

## 🛡️ Security Notes

- Change the `app.secret_key` in `app.py` for production use
- Implement proper user authentication for production deployment
- Consider using environment variables for sensitive configuration
- Add HTTPS in production environments

## 🐛 Troubleshooting

### Common Issues:

1. **Python Version Compatibility**
   - TensorFlow 2.x requires Python 3.8+
   - Use `python --version` to check your Python version
   - **Windows + Python 3.13**: Use `requirements-windows.txt` for better compatibility

2. **Windows Installation Issues**
   ```bash
   # If pandas fails to install, try:
   pip install --upgrade pip setuptools wheel
   pip install -r requirements-windows.txt
   
   # Or install pre-compiled wheels manually:
   pip install tensorflow pandas numpy scikit-learn nltk flask
   ```

2. **TensorFlow Installation Issues**
   ```bash
   # If TensorFlow fails to install, try:
   pip install --upgrade pip
   pip install tensorflow==2.20.0 --no-cache-dir
   ```

3. **GPU Support (Optional)**
   ```bash
   # For GPU support, ensure CUDA is installed
   python -c "import tensorflow as tf; print('GPU available:', tf.config.list_physical_devices('GPU'))"
   ```

4. **NLTK Data Missing**
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
   ```

5. **Memory Issues**
   - TensorFlow 2.x automatically manages GPU memory growth
   - Reduce batch size if encountering out-of-memory errors
   - Consider using mixed precision training for large models

6. **Port Already in Use**
   - Change the port in `app.py`: `app.run(port=5001)`
   - Or kill the process using port 5000

7. **Module Import Errors**
   - Ensure virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt`

8. **File Upload Issues**
   - Check file format (must be CSV)
   - Ensure file size is under 16MB
   - Verify CSV has 'text' and 'label' columns

9. **Model Training Issues**
   - TensorFlow 2.x uses eager execution by default
   - Early stopping prevents overfitting automatically
   - Check that uploaded dataset has balanced classes
   - Monitor training progress with built-in callbacks

## 📈 Model Performance

The TensorFlow 2.20.0 Keras neural network model achieves:
- **Architecture**: 3-layer neural network (128, 64, 2 neurons) with Dropout
- **Accuracy**: ~85-92%
- **Training**: Adam optimizer with early stopping and learning rate reduction
- **Features**: TF-IDF vectorization with 5000 features
- **Callbacks**: Early stopping, learning rate scheduling for optimal training

Performance can be improved by:
- Training with larger, more diverse datasets
- Using pre-trained embeddings (Word2Vec, GloVe, BERT)
- Implementing attention mechanisms
- Fine-tuning hyperparameters (learning rate, epochs, batch size)
- Using advanced NLP preprocessing techniques
- Leveraging GPU acceleration for faster training

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with applicable laws and regulations when using for cyberbullying detection.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the code comments and documentation
3. Create an issue in the repository

---

**Built with ❤️ for safer digital spaces**