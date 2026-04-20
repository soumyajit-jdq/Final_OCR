import cv2
import numpy as np
import io
import fitz  # PyMuPDF
from PIL import Image

def _downscale_if_needed(image, max_width=800):
    """Resizes the image if it's too large, to speed up processing."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / float(w)
    new_dim = (max_width, int(h * scale))
    return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

def is_blurry(image, threshold=100.0):
    """
    Detects if an image is blurry using the variance of the Laplacian method.
    """
    # Use a downscaled version for faster processing
    small_img = _downscale_if_needed(image)
    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance

def get_skew_angle(image):
    """
    Detects the skew angle of the document in the image.
    """
    # Significant speedup by working on a smaller image
    small_img = _downscale_if_needed(image)
    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]
    
    # Normalize the angle to be within [-45, 45)
    # This handles both old and new OpenCV angle conventions robustly.
    angle = (angle + 45) % 90 - 45
    return angle

def is_blank_or_black(image):
    """
    Checks if an image is completely black or blank (white with no content).
    """
    # Downscaling doesn't affect mean/stddev much, so we use a small version
    small_img = _downscale_if_needed(image, max_width=400)
    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    mean, stddev = cv2.meanStdDev(gray)
    
    if stddev[0][0] < 10:
        if mean[0][0] < 30:
            return True, "The photo is completely black or too dark."
        if mean[0][0] > 225:
            return True, "The photo appears to be blank white."
        return True, "The photo consists of a single solid color with no document content."
    
    if mean[0][0] < 15:
        return True, "The photo is too dark to read."
        
    return False, ""

def process_image_cv2(image_bytes):
    """Decodes image bytes into a CV2-compatible image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def pixmap_to_numpy(pix):
    """
    Directly converts a fitz.Pixmap to a numpy BGR array (zero-copy where possible).
    """
    # pix.samples is a bytearray of RGB pixels
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 3: # RGB
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif pix.n == 4: # RGBA
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img # Grayscale or other

def validate_image_quality(file_bytes: bytes, filename: str = "document"):
    """
    Validates quality of an image or all pages of a PDF.
    Returns (is_valid, message)
    """
    # Detect if PDF
    if file_bytes.startswith(b"%PDF"):
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if len(doc) == 0:
                return False, "The PDF file is empty."
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Lower matrix scale (1.0x) for faster rendering of quality checks
                pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0)) 
                image = pixmap_to_numpy(pix)
                
                if image is None:
                    return False, f"Could not process Page {page_num + 1} of the PDF."
                
                # Run Checks
                blank, blank_msg = is_blank_or_black(image)
                if blank:
                    return False, f"Page {page_num + 1}: {blank_msg}"
                
                blurry, score = is_blurry(image)
                if blurry:
                    return False, f"Page {page_num + 1} is too blurry (Score: {score:.1f}). Please upload a clear document."
                
                angle = get_skew_angle(image)
                if abs(angle) > 15.0:
                    return False, f"Page {page_num + 1} is slanted ({abs(angle):.1f}°). Please ensure the document is straight."
            
            page_count = len(doc)
            doc.close()
            return True, f"All {page_count} pages of the PDF passed quality checks."
            
        except Exception as e:
            return False, f"Error processing PDF: {str(e)}"
    
    # Handle as Image
    image = process_image_cv2(file_bytes)
    if image is None:
        return False, "Invalid image or PDF format. Please upload a valid file."
    
    blank, blank_msg = is_blank_or_black(image)
    if blank:
        return False, blank_msg
        
    blurry, score = is_blurry(image)
    if blurry:
        return False, f"The photo is too blurry (Score: {score:.1f}). Please upload a clear photo."
    
    angle = get_skew_angle(image)
    if abs(angle) > 15.0:
        return False, f"The photo is slanted ({abs(angle):.1f}°). Please capture the document straight from above."
    
    return True, "Photo quality is excellent."

if __name__ == "__main__":
    import sys
    import os
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <file_path>")
    else:
        path = sys.argv[1]
        if not os.path.exists(path):
            print(f"File not found: {path}")
            sys.exit(1)
            
        with open(path, "rb") as f:
            b = f.read()
            valid, msg = validate_image_quality(b, os.path.basename(path))
            print(f"Result: {'PASS' if valid else 'FAIL'}")
            print(f"Message: {msg}")
