"""
Advanced results formatting and export for OCR extraction.

Provides:
- JSON/CSV/Excel export
- Confidence-based filtering and sorting
- Detailed extraction reports
- Batch processing results aggregation
"""

import json
import csv
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
from utils.text_extraction import get_field_info, determine_label

logger = logging.getLogger(__name__)


class ResultsFormatter:
    """Format and export OCR extraction results."""

    @staticmethod
    def format_extraction_result(
        extracted_fields: Dict[str, dict],
        raw_text: Optional[str] = None,
        include_metadata: bool = True,
        confidence_threshold: float = 0.0,
    ) -> Dict:
        """
        Format extraction results with metadata and filtering.
        
        Args:
            extracted_fields: {field_name: {value, confidence}}
            raw_text: Original OCR text
            include_metadata: Include field descriptions and examples
            confidence_threshold: Only include fields above this confidence
        
        Returns:
            Formatted result dictionary
        """
        result = {
            "extraction_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_fields": len(extracted_fields),
                "high_confidence": 0,
                "medium_confidence": 0,
                "low_confidence": 0,
                "average_confidence": 0.0,
            },
            "fields": {},
            "text_preview": raw_text[:500] if raw_text else None,
        }
        
        # Confidence distribution
        confidences = []
        
        # Process each field
        for field_name, field_data in extracted_fields.items():
            confidence = field_data.get("confidence", 0.0)
            
            # Apply confidence filter
            if confidence < confidence_threshold:
                continue
            
            # Build field entry
            entry = {
                "value": field_data.get("value"),
                "confidence": confidence,
                "label_type": determine_label(field_name),
            }
            
            # Add metadata if requested
            if include_metadata:
                info = get_field_info(field_name)
                if info:
                    entry["description"] = info.get("description", "")
                    entry["validation_pattern"] = info.get("validation")
                    entry["example"] = info.get("example")
            
            result["fields"][field_name] = entry
            confidences.append(confidence)
            
            # Update confidence distribution
            if confidence >= 0.9:
                result["summary"]["high_confidence"] += 1
            elif confidence >= 0.7:
                result["summary"]["medium_confidence"] += 1
            else:
                result["summary"]["low_confidence"] += 1
        
        # Calculate average confidence
        if confidences:
            result["summary"]["average_confidence"] = sum(confidences) / len(confidences)
        
        return result

    @staticmethod
    def export_to_json(
        extracted_fields: Dict[str, dict],
        output_path: str,
        pretty_print: bool = True,
        **kwargs
    ) -> bool:
        """Export extraction results to JSON file."""
        try:
            formatted = ResultsFormatter.format_extraction_result(
                extracted_fields, **kwargs
            )
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(
                    formatted,
                    f,
                    indent=2 if pretty_print else None,
                    ensure_ascii=False
                )
            
            logger.info(f"Results exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return False

    @staticmethod
    def export_to_csv(
        extracted_fields: Dict[str, dict],
        output_path: str,
        include_confidence: bool = True,
        **kwargs
    ) -> bool:
        """Export extraction results to CSV file."""
        try:
            rows = []
            
            for field_name, field_data in extracted_fields.items():
                row = {"Field": field_name, "Value": field_data.get("value")}
                if include_confidence:
                    row["Confidence"] = f"{field_data.get('confidence', 0.0):.2%}"
                rows.append(row)
            
            if not rows:
                logger.warning("No fields to export")
                return False
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            logger.info(f"Results exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return False

    @staticmethod
    def export_to_excel(
        extracted_fields: Dict[str, dict],
        output_path: str,
        include_metadata: bool = True,
        **kwargs
    ) -> bool:
        """Export extraction results to Excel file with multiple sheets."""
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Sheet 1: Summary
                summary_data = {
                    "Metric": ["Total Fields", "Average Confidence", "Export Time"],
                    "Value": [
                        len(extracted_fields),
                        f"{sum(f.get('confidence', 0) for f in extracted_fields.values()) / len(extracted_fields) if extracted_fields else 0:.2%}",
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Sheet 2: Extracted Fields
                field_rows = []
                for field_name, field_data in extracted_fields.items():
                    row = {
                        "Field Name": field_name,
                        "Value": field_data.get("value"),
                        "Confidence": f"{field_data.get('confidence', 0.0):.2%}",
                        "Label Type": determine_label(field_name),
                    }
                    
                    if include_metadata:
                        info = get_field_info(field_name)
                        if info:
                            row["Description"] = info.get("description", "")
                            row["Example"] = info.get("example", "")
                    
                    field_rows.append(row)
                
                if field_rows:
                    pd.DataFrame(field_rows).to_excel(writer, sheet_name='Fields', index=False)
            
            logger.info(f"Results exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Excel export failed: {e}")
            return False

    @staticmethod
    def generate_extraction_report(
        extracted_fields: Dict[str, dict],
        raw_text: Optional[str] = None,
        include_examples: bool = True,
    ) -> str:
        """Generate a human-readable text report."""
        lines = [
            "=" * 80,
            "OCR FIELD EXTRACTION REPORT",
            "=" * 80,
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total Fields Extracted: {len(extracted_fields)}")
        
        if extracted_fields:
            confidences = [f.get('confidence', 0.0) for f in extracted_fields.values()]
            lines.append(f"Average Confidence: {sum(confidences)/len(confidences):.2%}")
            lines.append(f"Highest Confidence: {max(confidences):.2%}")
            lines.append(f"Lowest Confidence: {min(confidences):.2%}")
        
        lines.append("")
        
        # Extracted Fields
        lines.append("EXTRACTED FIELDS")
        lines.append("-" * 40)
        
        # Sort by confidence
        sorted_fields = sorted(
            extracted_fields.items(),
            key=lambda x: x[1].get('confidence', 0.0),
            reverse=True
        )
        
        for field_name, field_data in sorted_fields:
            value = field_data.get('value', '')
            confidence = field_data.get('confidence', 0.0)
            label_type = determine_label(field_name)
            
            lines.append(f"\n{field_name}")
            lines.append(f"  Value: {value}")
            lines.append(f"  Confidence: {confidence:.2%}")
            lines.append(f"  Type: {label_type}")
            
            if include_examples:
                info = get_field_info(field_name)
                if info and info.get('example'):
                    lines.append(f"  Example Format: {info['example']}")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)

    @staticmethod
    def get_confidence_summary(extracted_fields: Dict[str, dict]) -> Dict:
        """Generate confidence statistics."""
        if not extracted_fields:
            return {
                "count": 0,
                "avg_confidence": 0.0,
                "max_confidence": 0.0,
                "min_confidence": 0.0,
                "high_confidence_count": 0,
                "medium_confidence_count": 0,
                "low_confidence_count": 0,
            }
        
        confidences = [f.get('confidence', 0.0) for f in extracted_fields.values()]
        
        return {
            "count": len(confidences),
            "avg_confidence": sum(confidences) / len(confidences),
            "max_confidence": max(confidences),
            "min_confidence": min(confidences),
            "high_confidence_count": sum(1 for c in confidences if c >= 0.9),
            "medium_confidence_count": sum(1 for c in confidences if 0.7 <= c < 0.9),
            "low_confidence_count": sum(1 for c in confidences if c < 0.7),
        }

    @staticmethod
    def filter_by_confidence(
        extracted_fields: Dict[str, dict],
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
    ) -> Dict[str, dict]:
        """Filter extracted fields by confidence range."""
        return {
            field: data
            for field, data in extracted_fields.items()
            if min_confidence <= data.get('confidence', 0.0) <= max_confidence
        }

    @staticmethod
    def sort_by_confidence(
        extracted_fields: Dict[str, dict],
        reverse: bool = True,
    ) -> Dict[str, dict]:
        """Sort extracted fields by confidence score."""
        sorted_items = sorted(
            extracted_fields.items(),
            key=lambda x: x[1].get('confidence', 0.0),
            reverse=reverse
        )
        return dict(sorted_items)

    @staticmethod
    def compare_results(
        results1: Dict[str, dict],
        results2: Dict[str, dict],
    ) -> Dict:
        """Compare two extraction results."""
        common_fields = set(results1.keys()) & set(results2.keys())
        only_in_1 = set(results1.keys()) - set(results2.keys())
        only_in_2 = set(results2.keys()) - set(results1.keys())
        
        differences = {}
        for field in common_fields:
            val1 = results1[field].get('value')
            val2 = results2[field].get('value')
            if val1 != val2:
                differences[field] = {
                    "result1": val1,
                    "result2": val2,
                    "confidence1": results1[field].get('confidence', 0.0),
                    "confidence2": results2[field].get('confidence', 0.0),
                }
        
        return {
            "common_fields": list(common_fields),
            "only_in_first": list(only_in_1),
            "only_in_second": list(only_in_2),
            "differences": differences,
        }
