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


def create_annotated_image_with_exact_boxes(
    original_image: np.ndarray,
    extracted_fields: Dict[str, dict],
    words_data: list = None
) -> np.ndarray:
    """
    Create annotated image with EXACT bounding boxes from OCR detection.
    Draws colored rectangles around the actual locations where fields are detected.
    
    Colors indicate confidence levels:
    - Green  (>= 90%): High confidence
    - Yellow (70-90%): Medium confidence  
    - Orange (50-70%): Low confidence
    - Red    (< 50%): Very low confidence
    
    Args:
        original_image: Numpy array of the original image
        extracted_fields: Dict with field names, values, and confidence scores
        words_data: List of word objects from OCR with bounding boxes
                   Each word: {"text": str, "confidence": float, "bbox": [[x1,y1], [x2,y2], ...]}
    
    Returns:
        Annotated image with colored bounding boxes
    """
    annotated = original_image.copy()
    if len(annotated.shape) == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    
    height, width = annotated.shape[:2]
    
    if not words_data:
        # Fallback to text overlay if no OCR bbox data available
        return create_annotated_image_simple(original_image, extracted_fields)
    
    # Build mapping of extracted field values to field info
    field_text_map = {}
    for field_name, field_data in extracted_fields.items():
        value = str(field_data.get('value', '')).strip().lower()
        if value:
            field_text_map[value] = {
                'name': field_name,
                'confidence': field_data.get('confidence', 0.5),
                'original_value': field_data.get('value', '')
            }
    
    # Find and draw boxes for extracted field values
    drawn_boxes = []
    
    for word in words_data:
        word_text = word.get('text', '').strip().lower()
        word_confidence = word.get('confidence', 0.0)
        bbox = word.get('bbox', [])
        
        if not bbox or not word_text:
            continue
        
        # Check if this word matches any extracted field value (or is part of it)
        matched_field = None
        for field_value, field_info in field_text_map.items():
            if word_text in field_value or field_value in word_text or word_text == field_value:
                matched_field = field_info
                break
        
        if not matched_field:
            continue
        
        try:
            # Parse bbox: typically [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] for quad
            # Convert to rectangle coordinates
            x_coords = [pt[0] for pt in bbox]
            y_coords = [pt[1] for pt in bbox]
            
            x_min = max(0, int(min(x_coords)))
            y_min = max(0, int(min(y_coords)))
            x_max = min(width, int(max(x_coords)))
            y_max = min(height, int(max(y_coords)))
            
            if x_max <= x_min or y_max <= y_min:
                continue
            
            color = get_confidence_color(matched_field['confidence'])
            
            # Draw semi-transparent background overlay (slight highlight)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
            
            # Draw colored border box (thick)
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 3)
            
            # Draw field name and confidence above the box
            label_text = f"{matched_field['name']}: {matched_field['confidence']:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, font, font_scale, thickness
            )
            
            # Background for label
            label_y = max(15, y_min - 5)
            label_x = x_min
            
            label_overlay = annotated.copy()
            cv2.rectangle(
                label_overlay,
                (label_x, label_y - text_h - 3),
                (label_x + text_w + 6, label_y + baseline + 2),
                color,
                -1
            )
            cv2.addWeighted(label_overlay, 0.8, annotated, 0.2, 0, annotated)
            
            # Draw label text
            cv2.putText(
                annotated,
                label_text,
                (label_x + 3, label_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )
            
            drawn_boxes.append({
                'field': matched_field['name'],
                'bbox': (x_min, y_min, x_max, y_max)
            })
            
        except Exception as e:
            # Skip malformed bbox
            continue
    
    # Add text summary on top-left if space available
    if drawn_boxes:
        summary_text = f"✓ {len(set(b['field'] for b in drawn_boxes))} fields detected"
        cv2.putText(
            annotated,
            summary_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )
    
    return annotated


def create_annotated_image_simple(
    original_image: np.ndarray,
    extracted_fields: Dict[str, dict]
) -> np.ndarray:
    """
    Fallback: Create annotated image with colored text labels for each extracted field.
    Used when OCR bounding box data is not available.
    
    Colors indicate confidence levels:
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
