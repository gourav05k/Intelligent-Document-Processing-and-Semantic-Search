import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

from src.utils.ocr_utils import OCRProcessor
from src.utils.text_processing import FinancialDataExtractor
from src.models.schemas import DocumentCreate, UnitCreate, LeaseCreate

logger = logging.getLogger(__name__)


class DocumentParser:
    def __init__(self):
        self.ocr_processor = OCRProcessor()
        self.data_extractor = FinancialDataExtractor()
        logger.info("DocumentParser initialized")
    
    def is_machine_readable(self, pdf_path: str) -> bool:
        """Determine if PDF is machine-readable or requires OCR"""
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            text_length = 0
            
            # Handle edge case of empty PDF
            if total_pages == 0:
                doc.close()
                logger.warning("PDF has no pages")
                return False
            
            # Check first few pages for text content (max 3 pages or all pages if fewer)
            pages_to_check = min(3, total_pages)
            logger.info(f"Checking {pages_to_check} page(s) out of {total_pages} total pages")
            
            for page_num in range(pages_to_check):
                page = doc[page_num]
                text = page.get_text()
                page_text_length = len(text.strip())
                text_length += page_text_length
                logger.debug(f"Page {page_num + 1}: {page_text_length} characters")
            
            doc.close()
            
            # If we got substantial text, it's machine-readable
            # Threshold: 20 characters total across checked pages (lowered for documents with less text)
            is_readable = text_length > 20
            logger.info(f"PDF readability check: {text_length} characters across {pages_to_check} page(s), machine-readable: {is_readable}")
            return is_readable
        
        except Exception as e:
            logger.error(f"Error checking PDF readability: {e}")
            return False
    
    def extract_text_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF for machine-readable PDFs"""
        try:
            doc = fitz.open(pdf_path)
            text_content = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():  # Only add pages with content
                    text_content.append(f"--- Page {page_num + 1} ---\n{text}")
            
            doc.close()
            
            result = "\n\n".join(text_content)
            logger.info(f"PyMuPDF extraction completed. Text length: {len(result)}")
            return result
        
        except Exception as e:
            logger.error(f"Error extracting text with PyMuPDF: {e}")
            return ""
    
    def extract_text_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (better for tables)"""
        try:
            text_content = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_content = f"--- Page {i + 1} ---\n"
                    
                    # Extract regular text
                    text = page.extract_text()
                    if text:
                        page_content += text + "\n"
                    
                    # Try to extract tables
                    tables = page.extract_tables()
                    if tables:
                        page_content += "\n=== TABLES ===\n"
                        for table_idx, table in enumerate(tables):
                            page_content += f"\nTable {table_idx + 1}:\n"
                            for row in table:
                                if row and any(cell for cell in row if cell):  # Skip empty rows
                                    row_text = " | ".join(str(cell) if cell else "" for cell in row)
                                    page_content += row_text + "\n"
                    
                    if page_content.strip() != f"--- Page {i + 1} ---":
                        text_content.append(page_content)
            
            result = "\n\n".join(text_content)
            logger.info(f"pdfplumber extraction completed. Text length: {len(result)}")
            return result
        
        except Exception as e:
            logger.error(f"Error extracting text with pdfplumber: {e}")
            return ""
    
    def process_document(self, pdf_path: str) -> Tuple[str, List[Dict]]:
        """Main document processing pipeline"""
        logger.info(f"Starting document processing: {pdf_path}")
        
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Determine processing method
        is_readable = self.is_machine_readable(pdf_path)
        
        if is_readable:
            logger.info("Document is machine-readable, using direct text extraction")
            
            # Try pdfplumber first for better table handling
            text = self.extract_text_pdfplumber(pdf_path)
            
            # Fallback to PyMuPDF if pdfplumber fails or returns empty
            if not text.strip():
                logger.info("pdfplumber returned empty, trying PyMuPDF")
                text = self.extract_text_pymupdf(pdf_path)
        else:
            logger.info("Document requires OCR processing")
            
            # Check if Tesseract is available
            if not self.ocr_processor.is_tesseract_available():
                raise RuntimeError("Tesseract OCR is not available. Please install Tesseract.")
            
            text = self.ocr_processor.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        # Extract structured data using our financial data extractor
        logger.info("Extracting structured data from text")
        structured_data = self.data_extractor.extract_structured_data(text)
        
        # Validate the extracted data
        validated_data = self.data_extractor.validate_extracted_data(structured_data)
        
        logger.info(f"Document processing completed. Extracted {len(validated_data)} records")
        
        # Log summary statistics
        if validated_data:
            summary = self.data_extractor.extract_summary_statistics(validated_data)
            logger.info(f"Summary: {summary.get('total_units', 0)} units, "
                       f"{summary.get('occupied_units', 0)} occupied, "
                       f"${summary.get('total_rent', 0):,.2f} total rent")
        
        return text, validated_data
    
    def create_document_record(self, filename: str, text: str) -> DocumentCreate:
        """Create document record for database storage"""
        return DocumentCreate(
            filename=filename,
            document_type="financial_pdf",
            content_text=text
        )
    
    def create_unit_records(self, structured_data: List[Dict], property_id: int) -> List[UnitCreate]:
        """Create unit records from structured data"""
        units = []
        
        for data in structured_data:
            if data.get('unit_number'):
                # Determine unit type from the data or use a default
                unit_type = data.get('unit_type', 'Unknown')
                
                # If unit_type is still generic, try to infer from unit_number or other data
                if unit_type == 'Unknown' and data.get('unit_number'):
                    # You could add logic here to infer unit type from unit number patterns
                    # For now, we'll use a generic classification
                    unit_type = self._infer_unit_type(data)
                
                unit = UnitCreate(
                    property_id=property_id,
                    unit_number=data['unit_number'],
                    unit_type=unit_type,
                    area_sqft=data.get('area_sqft'),
                    rent_amount=data.get('rent_amount'),
                    status=data.get('status', 'vacant')
                )
                units.append(unit)
        
        logger.info(f"Created {len(units)} unit records")
        return units
    
    def create_lease_records(self, structured_data: List[Dict], unit_mapping: Dict[str, int]) -> List[LeaseCreate]:
        """Create lease records from structured data"""
        leases = []
        
        for data in structured_data:
            unit_number = data.get('unit_number')
            tenant_name = data.get('tenant_name')
            
            # Only create lease if we have both unit and tenant
            if unit_number and tenant_name and unit_number in unit_mapping:
                lease = LeaseCreate(
                    unit_id=unit_mapping[unit_number],
                    tenant_name=tenant_name,
                    lease_start=data.get('lease_start'),
                    lease_end=data.get('lease_end'),
                    move_in_date=data.get('move_in_date'),
                    move_out_date=data.get('move_out_date'),
                    total_amount=data.get('total_amount'),
                    status='active'  # Default to active for new leases
                )
                leases.append(lease)
        
        logger.info(f"Created {len(leases)} lease records")
        return leases
    
    def _infer_unit_type(self, data: Dict) -> str:
        """Infer unit type from available data"""
        unit_number = data.get('unit_number', '')
        unit_type_raw = data.get('unit_type', '')
        rent_amount = data.get('rent_amount', 0)
        area_sqft = data.get('area_sqft', 0)
        
        # If we have explicit unit type from extraction, clean it up
        if unit_type_raw and unit_type_raw != 'Unknown':
            # Extract meaningful part from codes like MBL2AC60
            if 'MBL' in unit_type_raw:
                if '2' in unit_type_raw:
                    return '2BR'
                elif '3' in unit_type_raw:
                    return '3BR'
                elif '1' in unit_type_raw:
                    return '1BR'
            return unit_type_raw
        
        # Infer from rent amount (rough estimates)
        if isinstance(rent_amount, (int, float)):
            if rent_amount > 2000:
                return '3BR'
            elif rent_amount > 1500:
                return '2BR'
            elif rent_amount > 1000:
                return '1BR'
            elif rent_amount > 0:
                return 'Studio'
        
        # Infer from area
        if isinstance(area_sqft, (int, float)):
            if area_sqft > 1200:
                return '3BR'
            elif area_sqft > 800:
                return '2BR'
            elif area_sqft > 500:
                return '1BR'
            elif area_sqft > 0:
                return 'Studio'
        
        # Default fallback
        return 'Unknown'
    
    def get_document_metadata(self, pdf_path: str) -> Dict:
        """Extract metadata from PDF document"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            doc.close()
            
            # Add file information
            file_path = Path(pdf_path)
            metadata.update({
                'file_size': file_path.stat().st_size,
                'file_name': file_path.name,
                'file_extension': file_path.suffix
            })
            
            return metadata
        
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
    
    def validate_pdf_file(self, pdf_path: str) -> bool:
        """Validate that the file is a valid PDF"""
        try:
            if not Path(pdf_path).exists():
                return False
            
            # Try to open with PyMuPDF
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
            
            return page_count > 0
        
        except Exception as e:
            logger.error(f"PDF validation failed: {e}")
            return False
