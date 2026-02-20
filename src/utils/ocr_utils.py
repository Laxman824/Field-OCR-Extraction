"""
OCR processing pipeline with advanced field extraction and visualization.

Integrates:
- DocTR for OCR
- Field extraction engine for intelligent field recognition
- Advanced image processing for document enhancement
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from utils.text_extraction import extract_all_fields, determine_label, get_field_info
from utils.image_processing import ImageProcessor
from config.field_patterns import FIELD_CATEGORIES, get_all_field_names

logger = logging.getLogger(__name__)


class OCRProcessor:
    """Enterprise OCR processing with field extraction and visualization."""

    def __init__(self, ocr_model, confidence_threshold: float = 0.6):
        """
        Initialize OCR processor.
        
        Args:
            ocr_model: DocTR OCR model instance
            confidence_threshold: Confidence threshold for extraction (0.0-1.0)
        """
        self.ocr_model = ocr_model
        self.confidence_threshold = confidence_threshold
        self.all_field_names = get_all_field_names()

    def process_image(
        self,
        image: np.ndarray,
        extract_fields: bool = True,
        visualize: bool = True,
        enhance_image: bool = True,
    ) -> Dict:
        """
        Complete OCR processing pipeline.
        
        Args:
            image: Input image (numpy array or PIL)
            extract_fields: Extract structured fields
            visualize: Generate annotated visualizations
            enhance_image: Apply image preprocessing
        
        Returns:
            Dictionary with raw_text, extracted_fields, bounding_boxes, visualizations
        """
        try:
            # Convert PIL to numpy if needed
            from PIL import Image as PILImage
            if isinstance(image, PILImage):
                image = np.array(image)
            
            # Enhance image if requested
            if enhance_image:
                pil_image = PILImage.fromarray(image)
                enhanced = ImageProcessor.preprocess_image(pil_image)
                image = np.array(enhanced)
            
            # Run OCR
            logger.info("Running OCR on image...")
            ocr_result = self.ocr_model([image])
            
            # Extract raw text and bounding boxes
            raw_text, bounding_boxes, text_with_confidence = self._extract_ocr_data(
                ocr_result, image
            )
            
            logger.info(f"OCR extracted {len(bounding_boxes)} words from image")
            
            # Extract fields
            extracted_fields = {}
            if extract_fields and raw_text:
                logger.info("Extracting structured fields...")
                extracted_fields = extract_all_fields(
                    raw_text,
                    confidence_threshold=self.confidence_threshold
                )
                logger.info(f"Extracted {len(extracted_fields)} fields with sufficient confidence")
            
            # Generate visualizations
            visualizations = {}
            if visualize:
                visualizations = self._create_visualizations(
                    image, bounding_boxes, extracted_fields
                )
            
            return {
                "raw_text": raw_text,
                "text_with_confidence": text_with_confidence,
                "extracted_fields": extracted_fields,
                "bounding_boxes": bounding_boxes,
                "visualizations": visualizations,
                "processing_metadata": {
                    "word_count": len(bounding_boxes),
                    "field_count": len(extracted_fields),
                    "confidence_threshold": self.confidence_threshold,
                }
            }
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}", exc_info=True)
            return {
                "raw_text": "",
                "text_with_confidence": "",
                "extracted_fields": {},
                "bounding_boxes": [],
                "visualizations": {},
                "error": str(e)
            }

    def _extract_ocr_data(
        self,
        ocr_result,
        image: np.ndarray
    ) -> Tuple[str, List[Dict], str]:
        """Extract text, bounding boxes, and confidence scores from OCR result."""
        bounding_boxes = []
        text_lines = []
        text_with_conf = []
        height, width = image.shape[:2]
        
        word_id = 0
        for page in ocr_result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = []
                    line_conf = []
                    
                    for word in line.words:
                        text_val = word.value
                        confidence = word.confidence if hasattr(word, 'confidence') else 0.95
                        
                        # Extract geometry (normalized coordinates)
                        x1, y1 = word.geometry[0]  # top-left
                        x2, y2 = word.geometry[1]  # bottom-right
                        
                        # Convert to pixel coordinates
                        box_pixel = [
                            int(x1 * width),
                            int(y1 * height),
                            int(x2 * width),
                            int(y2 * height)
                        ]
                        
                        # Create bounding box entry
                        bbox_entry = {
                            "id": word_id,
                            "text": text_val,
                            "confidence": confidence,
                            "box": box_pixel,
                            "label": "other",
                            "category": None,
                        }
                        bounding_boxes.append(bbox_entry)
                        
                        line_text.append(text_val)
                        line_conf.append(f"{text_val}({confidence:.2f})")
                        word_id += 1
                    
                    # Combine line
                    if line_text:
                        text_lines.append(" ".join(line_text))
                        text_with_conf.append(" ".join(line_conf))
        
        raw_text = "\n".join(text_lines)
        text_with_confidence = "\n".join(text_with_conf)
        
        return raw_text, bounding_boxes, text_with_confidence

    def _create_visualizations(
        self,
        image: np.ndarray,
        bounding_boxes: List[Dict],
        extracted_fields: Dict[str, dict]
    ) -> Dict[str, np.ndarray]:
        """Generate visualization images with bounding boxes and field labels."""
        visualizations = {}
        
        try:
            # Visualization 1: All text with bounding boxes
            img_all_text = image.copy()
            for bbox in bounding_boxes:
                x1, y1, x2, y2 = bbox["box"]
                cv2.rectangle(img_all_text, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(
                    img_all_text,
                    bbox["text"],
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )
            visualizations["all_text"] = img_all_text
            
            # Visualization 2: Only extracted fields highlighted
            img_fields = image.copy()
            for field_name, field_data in extracted_fields.items():
                value = field_data.get("value", "")
                confidence = field_data.get("confidence", 0.0)
                
                # Find bounding boxes matching this field
                for bbox in bounding_boxes:
                    if value and bbox["text"] in value:
                        x1, y1, x2, y2 = bbox["box"]
                        # Color code by confidence
                        color = self._get_confidence_color(confidence)
                        cv2.rectangle(img_fields, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            img_fields,
                            f"{field_name}:{confidence:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                            cv2.LINE_AA
                        )
            visualizations["fields_highlighted"] = img_fields
            
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
        
        return visualizations

    @staticmethod
    def _get_confidence_color(confidence: float) -> Tuple[int, int, int]:
        """Get RGB color based on confidence score."""
        if confidence >= 0.9:
            return (0, 255, 0)  # Green (high confidence)
        elif confidence >= 0.7:
            return (0, 255, 255)  # Yellow (medium confidence)
        elif confidence >= 0.5:
            return (0, 165, 255)  # Orange (low confidence)
        else:
            return (0, 0, 255)  # Red (very low confidence)

    def batch_process_images(
        self,
        images: List[np.ndarray],
        batch_size: int = 4
    ) -> List[Dict]:
        """Process multiple images in batches."""
        results = []
        total_images = len(images)
        
        for i in range(0, total_images, batch_size):
            batch = images[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_images + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            for j, image in enumerate(batch):
                try:
                    result = self.process_image(image)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process image {i+j}: {e}")
                    results.append({"error": str(e)})
        
        logger.info(f"Batch processing complete: {len(results)}/{total_images} images processed")
        return results


# ════════════════════════════════════════════════════════════════════
#  BACKWARD COMPATIBILITY FUNCTION
# ════════════════════════════════════════════════════════════════════

def process_image(image: np.ndarray, ocr_model, **kwargs) -> Tuple:
    """
    Legacy function for backward compatibility.
    
    Returns: (img_all_text, img_fields, extracted_fields, bounding_boxes)
    """
    processor = OCRProcessor(ocr_model)
    result = processor.process_image(image, **kwargs)
    
    return (
        result["visualizations"].get("all_text"),
        result["visualizations"].get("fields_highlighted"),
        result["extracted_fields"],
        result["bounding_boxes"]
    )
