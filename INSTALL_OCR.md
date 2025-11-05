# Installing Tesseract OCR for Image Text Extraction

## Quick Installation (Windows)

### Option 1: Download and Install
1. **Download Tesseract OCR:**
   - Go to: https://github.com/UB-Mannheim/tesseract/wiki
   - Download the Windows installer (latest version)
   - Direct link: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe

2. **Install Tesseract:**
   - Run the downloaded installer
   - Install to the default location: `C:\Program Files\Tesseract-OCR\`
   - **Important:** Make sure to check "Add to PATH" during installation

3. **Restart your command prompt/PowerShell**

4. **Test the installation:**
   ```bash
   tesseract --version
   ```

5. **Restart your Flask app:**
   ```bash
   python app.py
   ```

### Option 2: Using Package Manager (if you have Chocolatey)
```bash
choco install tesseract
```

### Option 3: Using Windows Package Manager (if you have winget)
```bash
winget install UB-Mannheim.TesseractOCR
```

## What happens without Tesseract?

If Tesseract is not installed, the image analysis feature will still work, but:
- ✅ You can still upload images
- ✅ You can manually enter the text you see in the image
- ❌ Automatic text extraction from images won't work
- ℹ️ The system will show a message asking you to enter text manually

## Testing the Feature

1. **With Tesseract installed:**
   - Upload an image with text (screenshot, photo of text, etc.)
   - The system will automatically extract text from the image
   - You can add additional context in the text field
   - Both extracted and additional text will be analyzed

2. **Without Tesseract:**
   - Upload an image
   - The system will ask you to manually enter the text you see
   - Enter the text in the "Additional Text" field
   - Only the manually entered text will be analyzed

## Supported Image Formats
- PNG, JPG, JPEG, GIF, BMP, TIFF
- Maximum file size: 16MB
- Best results with clear, high-contrast text

## Troubleshooting

### "tesseract is not recognized as an internal or external command"
- Tesseract is not installed or not in PATH
- Follow the installation steps above
- Make sure to restart your command prompt after installation

### "Error extracting text from image"
- The image might be corrupted or in an unsupported format
- Try a different image or manually enter the text

### Poor text extraction quality
- Use images with clear, high-contrast text
- Avoid blurry or low-resolution images
- Screenshots usually work better than photos