"""
Enhanced text extraction using production-grade field extraction engine.

This module replaces legacy regex-based extraction with the advanced
multi-strategy extraction pipeline from field_extraction.py.
"""

import logging
from typing import Dict, Tuple, Optional
from utils.field_extraction import FieldExtractor
from config.field_patterns import FIELD_CATEGORIES

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  SINGLETON EXTRACTOR INSTANCE
# ════════════════════════════════════════════════════════════════════

_extractor_instance = None


def get_field_extractor(confidence_threshold: float = 0.6) -> FieldExtractor:
    """Get or create a singleton FieldExtractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = FieldExtractor(
            FIELD_CATEGORIES,
            confidence_threshold=confidence_threshold
        )
    return _extractor_instance


# ════════════════════════════════════════════════════════════════════
#  HIGH-LEVEL EXTRACTION INTERFACE
# ════════════════════════════════════════════════════════════════════

def extract_value(
    text: str,
    field: str,
    confidence_threshold: float = 0.6
) -> Tuple[Optional[str], float]:
    """
    Extract a field value from text using the advanced extraction pipeline.
    
    Args:
        text: OCR-extracted text from document
        field: Field name (e.g., "PAN", "Email", "Mobile Number")
        confidence_threshold: Minimum confidence to accept extraction
    
    Returns:
        (value, confidence) tuple, or (None, 0.0) if extraction failed
    """
    if not text or not field:
        return None, 0.0
    
    extractor = get_field_extractor(confidence_threshold)
    return extractor.extract_with_context(text, field)


def extract_all_fields(
    text: str,
    selected_fields: Optional[Dict[str, bool]] = None,
    confidence_threshold: float = 0.6
) -> Dict[str, dict]:
    """
    Extract multiple fields from text at once.
    
    Args:
        text: OCR-extracted text
        selected_fields: Optional {field_name: include_flag} dict
        confidence_threshold: Minimum confidence threshold
    
    Returns:
        Dictionary of successfully extracted fields with values and confidences
    """
    if not text:
        return {}
    
    extractor = get_field_extractor(confidence_threshold)
    return extractor.extract_all_fields(text, selected_fields)


def extract_field_with_location(
    text: str,
    field: str,
) -> Tuple[Optional[str], float, Optional[int], Optional[int]]:
    """
    Extract field value and return its location in the text.
    
    Args:
        text: OCR-extracted text
        field: Field name
    
    Returns:
        (value, confidence, start_pos, end_pos) or (None, 0.0, None, None)
    """
    value, confidence = extract_value(text, field)
    if value:
        start = text.find(value)
        end = start + len(value) if start >= 0 else None
        return value, confidence, start, end
    return None, 0.0, None, None


# ════════════════════════════════════════════════════════════════════
#  FIELD CLASSIFICATION
# ════════════════════════════════════════════════════════════════════

def determine_label(field: str) -> str:
    """
    Classify a field as 'question', 'answer', or 'other'.
    
    Used by visualization/labeling systems to categorize extracted fields.
    """
    question_fields = {
        "Scheme Name", "Folio Number", "Number of Units", "PAN",
        "Tax Status", "Mobile Number", "Email", "Address", "Date",
        "Bank Account Details", "Date of Journey", "Aadhaar",
        "Bank Account Number", "IFSC Code", "Name", "Amount",
        "NAV", "ISIN", "Date of Birth", "Father's Name",
        "PIN Code", "City", "State", "Gender", "Nationality",
        "Bank Name", "Document Number",
    }
    
    if field in question_fields:
        return "question"
    elif field == "Signature":
        return "other"
    return "answer"


def get_field_info(field: str) -> Optional[Dict]:
    """Get metadata about a field (description, example, validation rules)."""
    extractor = get_field_extractor()
    return extractor.fields.get(field)


def is_valid_field(field: str) -> bool:
    """Check if a field is recognized in the field patterns."""
    extractor = get_field_extractor()
    return field in extractor.fields
