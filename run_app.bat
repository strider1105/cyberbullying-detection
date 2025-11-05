@echo off
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing required packages...
pip install tensorflow==2.20.0 flask pandas numpy scikit-learn nltk

echo Downloading NLTK data...
python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"

echo Starting the application...
python app.py

pause