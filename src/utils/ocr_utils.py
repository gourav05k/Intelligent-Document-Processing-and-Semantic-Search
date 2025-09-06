import pytesseract
from PIL import Image
import pdf2image
import cv2
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class OCRProcessor:
    def __init__(self, language: str = 'eng'):
        self.language = language
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR accuracy"""
        try:
            # Convert PIL to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply threshold to get image with only black and white
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to PIL
            return Image.fromarray(thresh)
        
        except Exception as e:
            logger.warning(f"Error preprocessing image: {e}. Using original image.")
            return image
    
    def extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from a single image"""
        try:
            preprocessed = self.preprocess_image(image)
            
            # Configure tesseract for better table recognition
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/$%-() '
            
            text = pytesseract.image_to_string(
                preprocessed, 
                lang=self.language,
                config=custom_config
            )
            
            return text.strip()
        
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[Image.Image]:
        """Convert PDF pages to images"""
        try:
            images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
            logger.info(f"Converted PDF to {len(images)} images")
            return images
        
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            return []
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from scanned PDF using OCR"""
        try:
            images = self.pdf_to_images(pdf_path)
            
            if not images:
                logger.error("No images extracted from PDF")
                return ""
            
            extracted_text = []
            
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1}/{len(images)}")
                text = self.extract_text_from_image(image)
                if text:
                    extracted_text.append(f"--- Page {i+1} ---\n{text}")
                else:
                    logger.warning(f"No text extracted from page {i+1}")
            
            result = "\n\n".join(extracted_text)
            logger.info(f"OCR extraction completed. Total text length: {len(result)}")
            return result
        
        except Exception as e:
            logger.error(f"Error in OCR processing: {e}")
            return ""
    
    def extract_tables_from_image(self, image: Image.Image) -> List[List[str]]:
        """Extract table structure from image (basic implementation)"""
        try:
            # This is a basic implementation - for production, consider using
            # specialized table detection libraries like table-transformer
            
            # Use tesseract with table-specific configuration
            custom_config = r'--oem 3 --psm 6'
            
            # Get bounding boxes for text
            data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Group text by approximate rows (y-coordinate)
            rows = {}
            for i, text in enumerate(data['text']):
                if text.strip():
                    y = data['top'][i]
                    x = data['left'][i]
                    
                    # Group by approximate y-coordinate (allowing some variance)
                    row_key = round(y / 10) * 10
                    
                    if row_key not in rows:
                        rows[row_key] = []
                    
                    rows[row_key].append((x, text.strip()))
            
            # Sort rows by y-coordinate and cells by x-coordinate
            table = []
            for y in sorted(rows.keys()):
                row_cells = sorted(rows[y], key=lambda x: x[0])
                row = [cell[1] for cell in row_cells]
                if row:  # Only add non-empty rows
                    table.append(row)
            
            return table
        
        except Exception as e:
            logger.error(f"Error extracting tables from image: {e}")
            return []
    
    def is_tesseract_available(self) -> bool:
        """Check if Tesseract is properly installed and accessible"""
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            logger.error(f"Tesseract not available: {e}")
            return False
