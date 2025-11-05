@echo off
echo ========================================
echo Installing Tesseract OCR for Windows
echo ========================================

echo.
echo Step 1: Downloading Tesseract OCR...
echo Please download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
echo.
echo Direct download link:
echo https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe
echo.
echo Step 2: Install Tesseract to the default location:
echo C:\Program Files\Tesseract-OCR\
echo.
echo Step 3: After installation, press any key to continue...
pause

echo.
echo Step 4: Testing Tesseract installation...
"C:\Program Files\Tesseract-OCR\tesseract.exe" --version
if %errorlevel% == 0 (
    echo ✅ Tesseract installed successfully!
) else (
    echo ❌ Tesseract not found. Please install it manually.
    echo Download from: https://github.com/UB-Mannheim/tesseract/wiki
)

echo.
echo Step 5: Installing Python packages...
pip install Pillow pytesseract opencv-python

echo.
echo Installation complete! You can now use image analysis.
pause