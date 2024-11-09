# utils/field_extraction.py

from typing import Dict, Tuple, Optional
import re
from fuzzywuzzy import fuzz

class FieldExtractor:
    """Field extraction with enhanced accuracy and validation"""
    
    def __init__(self, field_patterns: Dict, confidence_threshold: float = 0.6):
        self.field_patterns = field_patterns
        self.confidence_threshold = confidence_threshold
        self.context_cache = {}

    def extract_with_context(self, text: str, field_name: str, context_window: int = 3) -> Tuple[Optional[str], float]:
        """Extract field value using contextual information"""
        if not text:
            return None, 0.0

        lines = text.split('\n')
        pattern = self.field_patterns.get(field_name, {}).get('pattern', '')
        
        if not pattern:
            return None, 0.0

        for i, line in enumerate(lines):
            if re.search(pattern, line, re.IGNORECASE):
                # Get surrounding context
                start = max(0, i - context_window)
                end = min(len(lines), i + context_window + 1)
                context = ' '.join(lines[start:end])
                
                # Extract value using context
                value, confidence = self._extract_value(context, field_name)
                if value and confidence >= self.confidence_threshold:
                    return value, confidence
        
        return None, 0.0

    def _extract_value(self, text: str, field_name: str) -> Tuple[Optional[str], float]:
        """Extract and validate field value"""
        field_info = self.field_patterns.get(field_name, {})
        pattern = field_info.get('pattern', '')
        
        if not pattern:
            return None, 0.0

        # Remove the field label
        value_text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
        
        # Apply validation if available
        validation_pattern = field_info.get('validation')
        if validation_pattern:
            match = re.search(validation_pattern, value_text)
            if match:
                return match.group(0), 1.0
        
        # Apply fuzzy matching
        if 'example' in field_info:
            confidence = self._fuzzy_match(value_text, field_info['example'])
            if confidence >= self.confidence_threshold:
                return value_text, confidence
        
        # Basic extraction
        words = value_text.split()
        if words:
            return ' '.join(words[:5]), 0.6
        
        return None, 0.0

    def _fuzzy_match(self, text: str, example: str) -> float:
        """Calculate fuzzy match confidence score"""
        return fuzz.ratio(text.lower(), example.lower()) / 100.0


