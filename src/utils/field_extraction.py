# utils/field_extraction.py

from typing import Dict, List, Tuple, Optional
import re
from fuzzywuzzy import fuzz
import numpy as np
from collections import defaultdict

class FieldExtractor:
    """Enhanced field extraction with context awareness and validation"""
    
    def __init__(self, field_patterns: Dict, confidence_threshold: float = 0.6):
        self.field_patterns = field_patterns
        self.confidence_threshold = confidence_threshold
        self.context_cache = {}

    def extract_with_context(self, text: str, field_name: str, context_window: int = 3) -> Tuple[Optional[str], float]:
        """Extract field value using contextual information"""
        lines = text.split('\n')
        field_pattern = self.field_patterns[field_name]['pattern']
        
        for i, line in enumerate(lines):
            if re.search(field_pattern, line, re.IGNORECASE):
                # Get surrounding context
                start = max(0, i - context_window)
                end = min(len(lines), i + context_window + 1)
                context = ' '.join(lines[start:end])
                
                # Extract value using context
                value, confidence = self._extract_value_from_context(context, field_name)
                if value and confidence >= self.confidence_threshold:
                    return value, confidence
        
        return None, 0.0

    def _extract_value_from_context(self, context: str, field_name: str) -> Tuple[Optional[str], float]:
        """Extract value from context using pattern matching and validation"""
        field_info = self.field_patterns[field_name]
        pattern = field_info['pattern']
        
        # Remove the field label from context
        value_text = re.sub(pattern, '', context, flags=re.IGNORECASE).strip()
        
        # Apply validation if available
        if 'validation' in field_info and field_info['validation']:
            match = re.search(field_info['validation'], value_text)
            if match:
                return match.group(0), 1.0
        
        # Apply fuzzy matching if no exact validation
        if 'example' in field_info:
            confidence = self._fuzzy_match_confidence(value_text, field_info['example'])
            if confidence >= self.confidence_threshold:
                return value_text, confidence
        
        # Default text processing
        words = value_text.split()
        if words:
            return ' '.join(words[:5]), 0.6
        
        return None, 0.0

    def _fuzzy_match_confidence(self, value: str, example: str) -> float:
        """Calculate confidence score using fuzzy matching"""
        return fuzz.ratio(value.lower(), example.lower()) / 100.0

    def validate_field_value(self, field_name: str, value: str) -> Tuple[bool, float]:
        """Validate field value and return confidence score"""
        field_info = self.field_patterns[field_name]
        
        # Exact pattern matching
        if 'validation' in field_info and field_info['validation']:
            if re.match(field_info['validation'], value):
                return True, 1.0
        
        # Fuzzy matching for non-strict fields
        if 'example' in field_info:
            confidence = self._fuzzy_match_confidence(value, field_info['example'])
            return confidence >= self.confidence_threshold, confidence
        
        return False, 0.0

    def group_related_fields(self, extracted_fields: Dict) -> Dict:
        """Group related fields based on their categories"""
        grouped_fields = defaultdict(dict)
        
        for field_name, (value, confidence) in extracted_fields.items():
            for category, fields in self.field_patterns.items():
                if field_name in fields:
                    grouped_fields[category][field_name] = {
                        'value': value,
                        'confidence': confidence
                    }
                    break
        
        return dict(grouped_fields)

    def enhance_extraction_accuracy(self, text: str, field_name: str) -> Tuple[Optional[str], float]:
        """Enhance extraction accuracy using multiple techniques"""
        # Try context-based extraction first
        value, confidence = self.extract_with_context(text, field_name)
        if value and confidence >= self.confidence_threshold:
            return value, confidence
        
        # Try pattern-based extraction
        value, confidence = self._extract_value_from_context(text, field_name)
        if value and confidence >= self.confidence_threshold:
            return value, confidence
        
        # Try fuzzy matching as last resort
        field_info = self.field_patterns[field_name]
        if 'example' in field_info:
            words = text.split()
            best_match = None
            best_confidence = 0
            
            for word in words:
                confidence = self._fuzzy_match_confidence(word, field_info['example'])
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = word
            
            if best_match and best_confidence >= self.confidence_threshold:
                return best_match, best_confidence
        
        return None, 0.0