@echo off
echo ========================================
echo Cyberbullying Detection App Setup
echo ========================================

echo.
echo Step 1: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 2: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 3: Installing TensorFlow...
pip install tensorflow==2.20.0

echo.
echo Step 4: Installing other packages...
pip install flask pandas numpy scikit-learn nltk

echo.
echo Step 5: Downloading NLTK data...
python -c "import nltk; print('Downloading NLTK data...'); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); print('NLTK data downloaded successfully!')"

echo.
echo Step 6: Testing imports...
python -c "import tensorflow as tf; import flask; import pandas; import numpy; import sklearn; import nltk; print('All packages imported successfully!')"

echo.
echo Step 7: Starting the Flask application...
echo The website will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python app.py

pause