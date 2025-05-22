import cv2
import pytesseract
import easyocr

# Initialize EasyOCR Reader for English characters (license plates use A-Z,0-9)
# Using GPU=False for compatibility; set True to use GPU if available for faster OCR.
reader = easyocr.Reader(['en'], gpu=False)

def ocr_easy(image):
    """
    Perform OCR using EasyOCR on the given plate image.
    Returns the recognized text (or empty string if nothing is detected).
    """
    # EasyOCR accepts a file path or an image array (OpenCV image).
    result = reader.readtext(image, detail=0)
    if len(result) == 0:
        return ""
    # Join all detected text parts (for plates, there is usually only one part)
    text = " ".join(result)
    return text.strip()

def ocr_tesseract(image):
    """
    Perform OCR using Tesseract on the given plate image.
    Returns the recognized text.
    """
    # Convert image to grayscale for better OCR results
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Otsu's threshold to binarize the image (black text on white background)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Configure Tesseract: assume a single line of text (--psm 6) and restrict char set to alphanumeric
    config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(thresh, config=config)
    return text.strip()
