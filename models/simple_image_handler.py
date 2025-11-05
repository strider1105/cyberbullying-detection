"""
Simple image handler that doesn't require Tesseract OCR
This is a fallback solution for when OCR is not available
"""

def handle_image_without_ocr(image_path, additional_text=""):
    """
    Handle image upload without OCR - just use additional text
    This is a fallback when Tesseract is not available
    """
    
    # Since we can't extract text from image, we'll use only the additional text
    if not additional_text.strip():
        return {
            'text': '',
            'extracted_text': '[OCR not available - Tesseract not installed]',
            'additional_text': additional_text,
            'prediction': 'No Text Available',
            'label': -1,
            'confidence': 0.0,
            'error': 'Cannot extract text from image. Please install Tesseract OCR or provide text manually.'
        }
    
    # Return the additional text for analysis
    return {
        'text': additional_text,
        'extracted_text': '[OCR not available - using manual text only]',
        'additional_text': additional_text,
        'needs_text_analysis': True
    }