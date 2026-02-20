# Helper function to generate annotated image with bounding boxes

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple

def get_confidence_color(confidence: float) -> Tuple[int, int, int]:
    """Get BGR color based on confidence score."""
    if confidence >= 0.9:
        return (0, 255, 0)  # Green (HIGH)
    elif confidence >= 0.7:
        return (0, 255, 255)  # Yellow (MEDIUM)
    elif confidence >= 0.5:
        return (0, 165, 255)  # Orange (LOW)
    else:
        return (0, 0, 255)  # Red (VERY LOW)


def create_annotated_image(
    original_image: np.ndarray,
    extracted_fields: Dict[str, dict],
    raw_text: str
) -> np.ndarray:
    """
    Create an annotated image with bounding boxes for detected fields.
    
    Args:
        original_image: numpy array of the original image
        extracted_fields: Dictionary with field names and confidence scores
        raw_text: Raw OCR text
    
    Returns:
        Annotated image with bounding boxes
    """
    annotated = original_image.copy()
    if len(annotated.shape) == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    
    # Create a mapping of text to confidence scores
    field_confidences = {}
    for field_name, field_data in extracted_fields.items():
        value = str(field_data.get('value', ''))
        confidence = field_data.get('confidence', 0.5)
        field_confidences[value.lower()] = (field_name, confidence)
    
    # Find and draw boxes for detected text
    text_lower = raw_text.lower()
    height, width = annotated.shape[:2]
    
    # Simple approach: find extracted values in the image and draw boxes
    for value_text, (field_name, confidence) in field_confidences.items():
        if len(value_text) < 2:
            continue
        
        # Find position in raw_text
        pos = text_lower.find(value_text)
        if pos >= 0:
            # Estimate bounding box (simplified - in practice you'd use OCR coordinates)
            # For now, just add text annotation at multiple positions
            lines = raw_text.split('\n')
            y_pos = 30
            
            for line in lines:
                if value_text in line.lower():
                    # Find x position in line
                    x_pos = line.lower().find(value_text)
                    if x_pos >= 0:
                        # Draw box
                        color = get_confidence_color(confidence)
                        
                        # Estimate text width (rough)
                        text_width = len(value_text) * 7  # ~7 pixels per char
                        
                        # Draw rectangle
                        cv2.rectangle(
                            annotated,
                            (max(0, x_pos * 8), max(0, y_pos - 20)),
                            (min(width, x_pos * 8 + text_width), min(height, y_pos + 5)),
                            color,
                            2
                        )
                        
                        # Draw label
                        label = f"{field_name} ({confidence:.0%})"
                        cv2.putText(
                            annotated,
                            label,
                            (max(0, x_pos * 8), max(15, y_pos - 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1
                        )
                
                y_pos += 20
    
    return annotated


def create_annotated_image_simple(
    original_image: np.ndarray,
    extracted_fields: Dict[str, dict]
) -> np.ndarray:
    """
    Create annotated image with colored text labels for each extracted field.
    Colors indicate confidence levels.
    
    Green  (>= 90%): High confidence
    Yellow (70-90%): Medium confidence  
    Orange (50-70%): Low confidence
    Red    (< 50%): Very low confidence
    """
    annotated = original_image.copy()
    if len(annotated.shape) == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    
    height, width = annotated.shape[:2]
    y_offset = 25
    
    # Sort fields by confidence (highest first) for better visibility
    sorted_fields = sorted(
        extracted_fields.items(),
        key=lambda x: x[1].get('confidence', 0.0),
        reverse=True
    )
    
    for idx, (field_name, field_data) in enumerate(sorted_fields):
        confidence = field_data.get('confidence', 0.5)
        value = str(field_data.get('value', ''))[:40]  # Truncate long values
        
        color = get_confidence_color(confidence)
        
        # Format: Field Name: value (confidence)
        text = f"✓ {field_name}: {value} ({confidence:.0%})"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        text_color = color
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Draw semi-transparent dark background
        padding = 5
        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (5, y_offset - text_height - padding),
            (15 + text_width, y_offset + baseline + padding),
            (0, 0, 0),
            -1
        )
        # Blend overlay with original (40% opacity)
        cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
        
        # Draw colored border rectangle
        cv2.rectangle(
            annotated,
            (5, y_offset - text_height - padding),
            (15 + text_width, y_offset + baseline + padding),
            color,
            2  # Border thickness
        )
        
        # Draw text
        cv2.putText(
            annotated,
            text,
            (10, y_offset),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA  # Anti-aliased text
        )
        
        y_offset += text_height + baseline + 15
        
        # Break if we run out of vertical space
        if y_offset > height - 30:
            # Add indicator if more fields exist
            remaining = len(sorted_fields) - idx - 1
            if remaining > 0:
                text = f"... and {remaining} more field{'s' if remaining > 1 else ''}"
                cv2.putText(
                    annotated,
                    text,
                    (10, y_offset),
                    font,
                    0.4,
                    (128, 128, 128),
                    1,
                    cv2.LINE_AA
                )
            break
    
    return annotated
