import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date
import pandas as pd
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)


class FinancialDataExtractor:
    def __init__(self):
        # Enhanced patterns based on your PDF samples
        self.patterns = {
            # Unit patterns: 01-101, 01-102, etc.
            'unit_number': r'(\d{2}-\d{3}|\d{1,3}[A-Z]?\d{0,3})',
            
            # Unit types: MBL2AC60, MBL3AC60, etc.
            'unit_type': r'(MBL\d+AC\d+|[A-Z]{2,4}\d+[A-Z]*\d*)',
            
            # Rent amounts: $1,511.00, $1,306.00, etc.
            'rent_amount': r'\$?\s*([0-9,]+\.?\d{0,2})',
            
            # Square feet: various formats
            'square_feet': r'(\d+(?:,\d{3})*)\s*(?:sq\.?\s*ft\.?|sqft|square\s*feet)',
            
            # Dates: MM/DD/YYYY, M/D/YYYY, etc.
            'date': r'(\d{1,2}\/\d{1,2}\/\d{2,4})',
            
            # Tenant names: proper case names
            'tenant_name': r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z][a-z]+)*)',
            
            # Status indicators
            'occupancy_status': r'(Occupied|Vacant|Available|Rented|Notice)',
            
            # Lease status
            'lease_status': r'(No Notice|Notice|Unrented|Rented)',
        }
        
        # Compiled patterns for better performance
        self.compiled_patterns = {
            key: re.compile(pattern, re.IGNORECASE) 
            for key, pattern in self.patterns.items()
        }
    
    def extract_structured_data(self, text: str) -> List[Dict]:
        """Extract structured data from document text"""
        logger.info("Starting structured data extraction")
        
        # Split text into lines and process
        lines = text.split('\n')
        extracted_data = []
        
        # Try to identify table structure first
        table_data = self._extract_table_data(text)
        if table_data:
            logger.info(f"Extracted {len(table_data)} records from table structure")
            return table_data
        
        # Fallback to line-by-line processing
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 10:  # Skip empty or very short lines
                continue
            
            # Try to extract data from each line
            data_point = self._extract_from_line(line)
            if data_point:
                data_point['source_line'] = line_num + 1
                data_point['raw_text'] = line
                extracted_data.append(data_point)
        
        # Consolidate and clean data
        consolidated_data = self._consolidate_data(extracted_data)
        logger.info(f"Extracted {len(consolidated_data)} consolidated records")
        
        return consolidated_data
    
    def _extract_table_data(self, text: str) -> List[Dict]:
        """Extract data assuming tabular structure like your PDF samples"""
        lines = text.split('\n')
        table_records = []
        
        # Look for lines that contain unit numbers and rent amounts
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line contains unit pattern and rent pattern
            unit_match = self.compiled_patterns['unit_number'].search(line)
            rent_matches = self.compiled_patterns['rent_amount'].findall(line)
            
            if unit_match and rent_matches:
                record = self._parse_table_row(line)
                if record:
                    table_records.append(record)
        
        return table_records
    
    def _parse_table_row(self, line: str) -> Optional[Dict]:
        """Parse a single table row from your PDF format"""
        try:
            record = {}
            
            # Extract unit number
            unit_match = self.compiled_patterns['unit_number'].search(line)
            if unit_match:
                record['unit_number'] = unit_match.group(1)
            
            # Extract unit type
            unit_type_match = self.compiled_patterns['unit_type'].search(line)
            if unit_type_match:
                record['unit_type'] = unit_type_match.group(1)
            
            # Extract rent amounts (there might be multiple)
            rent_matches = self.compiled_patterns['rent_amount'].findall(line)
            if rent_matches:
                # Convert to decimal and find the main rent amount
                amounts = []
                for match in rent_matches:
                    try:
                        amount = float(match.replace(',', ''))
                        if amount > 100:  # Filter out small amounts that might be fees
                            amounts.append(amount)
                    except ValueError:
                        continue
                
                if amounts:
                    record['rent_amount'] = max(amounts)  # Take the largest as main rent
            
            # Extract occupancy status
            status_match = self.compiled_patterns['occupancy_status'].search(line)
            lease_status_match = self.compiled_patterns['lease_status'].search(line)
            
            if 'Vacant' in line or 'Available' in line:
                record['status'] = 'vacant'
            elif 'Occupied' in line or 'Rented' in line:
                record['status'] = 'occupied'
            else:
                record['status'] = 'unknown'
            
            # Extract tenant name (look for proper case names)
            tenant_match = self.compiled_patterns['tenant_name'].search(line)
            if tenant_match and record.get('status') == 'occupied':
                # Clean up tenant name
                tenant_name = tenant_match.group(1).strip()
                # Filter out common non-name words
                non_names = ['Unit', 'Rent', 'Total', 'Notice', 'Occupied', 'Vacant', 'Status']
                if not any(word in tenant_name for word in non_names):
                    record['tenant_name'] = tenant_name
            
            # Extract dates
            date_matches = self.compiled_patterns['date'].findall(line)
            if date_matches:
                # Try to identify which dates are which based on position/context
                parsed_dates = []
                for date_str in date_matches:
                    parsed_date = self.parse_date(date_str)
                    if parsed_date:
                        parsed_dates.append(parsed_date)
                
                # Assign dates based on typical lease document structure
                if len(parsed_dates) >= 2:
                    record['lease_start'] = parsed_dates[0]
                    record['lease_end'] = parsed_dates[1]
                elif len(parsed_dates) == 1:
                    record['lease_start'] = parsed_dates[0]
            
            # Only return record if it has essential data
            if record.get('unit_number') and (record.get('rent_amount') or record.get('status')):
                return record
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing table row: {e}")
            return None
    
    def _extract_from_line(self, line: str) -> Optional[Dict]:
        """Extract data from a single line (fallback method)"""
        data = {}
        
        # Extract unit number
        unit_match = self.compiled_patterns['unit_number'].search(line)
        if unit_match:
            data['unit_number'] = unit_match.group(1)
        
        # Extract rent amount
        rent_matches = self.compiled_patterns['rent_amount'].findall(line)
        if rent_matches:
            # Take the largest amount as rent
            amounts = []
            for match in rent_matches:
                try:
                    amount = float(match.replace(',', ''))
                    amounts.append(amount)
                except ValueError:
                    continue
            
            if amounts:
                data['rent_amount'] = max(amounts)
        
        # Extract square feet
        sqft_match = self.compiled_patterns['square_feet'].search(line)
        if sqft_match:
            try:
                data['area_sqft'] = int(sqft_match.group(1).replace(',', ''))
            except ValueError:
                pass
        
        # Extract dates
        date_matches = self.compiled_patterns['date'].findall(line)
        if date_matches:
            data['dates'] = date_matches
        
        # Extract tenant name
        tenant_match = self.compiled_patterns['tenant_name'].search(line)
        if tenant_match:
            data['tenant_name'] = tenant_match.group(1)
        
        # Determine occupancy status
        if 'vacant' in line.lower() or 'available' in line.lower():
            data['status'] = 'vacant'
        elif 'occupied' in line.lower() or 'rented' in line.lower():
            data['status'] = 'occupied'
        
        return data if data else None
    
    def _consolidate_data(self, data_points: List[Dict]) -> List[Dict]:
        """Consolidate extracted data points into structured records"""
        # Group data by unit number
        units = {}
        
        for point in data_points:
            unit_num = point.get('unit_number')
            if unit_num:
                if unit_num not in units:
                    units[unit_num] = {}
                
                # Merge data, prioritizing non-empty values
                for key, value in point.items():
                    if value and (key not in units[unit_num] or not units[unit_num][key]):
                        units[unit_num][key] = value
        
        # Convert to list and clean up
        consolidated = []
        for unit_data in units.values():
            # Ensure we have minimum required data
            if unit_data.get('unit_number'):
                # Set default status if not determined
                if 'status' not in unit_data:
                    unit_data['status'] = 'occupied' if unit_data.get('tenant_name') else 'vacant'
                
                consolidated.append(unit_data)
        
        return consolidated
    
    def parse_date(self, date_str: str) -> Optional[date]:
        """Parse date string to date object"""
        if not date_str:
            return None
        
        date_formats = [
            '%m/%d/%Y',   # 12/31/2024
            '%m-%d-%Y',   # 12-31-2024
            '%m/%d/%y',   # 12/31/24
            '%m-%d-%y',   # 12-31-24
            '%Y-%m-%d',   # 2024-12-31
            '%d/%m/%Y',   # 31/12/2024
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str.strip(), fmt).date()
                # Validate reasonable date range
                if 2000 <= parsed_date.year <= 2030:
                    return parsed_date
            except ValueError:
                continue
        
        logger.warning(f"Could not parse date: {date_str}")
        return None
    
    def validate_extracted_data(self, data: List[Dict]) -> List[Dict]:
        """Validate and clean extracted data"""
        validated_data = []
        
        for record in data:
            # Validate unit number
            if not record.get('unit_number'):
                continue
            
            # Validate rent amount
            if record.get('rent_amount'):
                try:
                    rent = float(record['rent_amount'])
                    if rent < 0 or rent > 50000:  # Reasonable rent range
                        logger.warning(f"Unusual rent amount: {rent} for unit {record['unit_number']}")
                    record['rent_amount'] = Decimal(str(rent))
                except (ValueError, TypeError):
                    record['rent_amount'] = None
            
            # Validate area
            if record.get('area_sqft'):
                try:
                    area = int(record['area_sqft'])
                    if area < 100 or area > 10000:  # Reasonable area range
                        logger.warning(f"Unusual area: {area} sqft for unit {record['unit_number']}")
                    record['area_sqft'] = area
                except (ValueError, TypeError):
                    record['area_sqft'] = None
            
            # Clean tenant name
            if record.get('tenant_name'):
                tenant_name = str(record['tenant_name']).strip()
                # Remove common artifacts
                artifacts = ['$', ',', 'rent', 'total', 'unit']
                for artifact in artifacts:
                    tenant_name = tenant_name.replace(artifact, '').strip()
                
                if len(tenant_name) > 2 and tenant_name.replace(' ', '').isalpha():
                    record['tenant_name'] = tenant_name
                else:
                    record['tenant_name'] = None
            
            # Ensure status is set
            if not record.get('status'):
                record['status'] = 'occupied' if record.get('tenant_name') else 'vacant'
            
            validated_data.append(record)
        
        return validated_data
    
    def extract_summary_statistics(self, data: List[Dict]) -> Dict[str, Any]:
        """Extract summary statistics from the data"""
        if not data:
            return {}
        
        total_units = len(data)
        occupied_units = sum(1 for record in data if record.get('status') == 'occupied')
        vacant_units = total_units - occupied_units
        
        rent_amounts = [
            float(record['rent_amount']) 
            for record in data 
            if record.get('rent_amount')
        ]
        
        areas = [
            record['area_sqft'] 
            for record in data 
            if record.get('area_sqft')
        ]
        
        summary = {
            'total_units': total_units,
            'occupied_units': occupied_units,
            'vacant_units': vacant_units,
            'occupancy_rate': (occupied_units / total_units * 100) if total_units > 0 else 0,
            'total_rent': sum(rent_amounts) if rent_amounts else 0,
            'average_rent': sum(rent_amounts) / len(rent_amounts) if rent_amounts else 0,
            'total_area': sum(areas) if areas else 0,
            'average_area': sum(areas) / len(areas) if areas else 0,
        }
        
        return summary
