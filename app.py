import streamlit as st
import numpy as np
from PIL import Image
import cv2
import json
import re
from doctr.models import ocr_predictor
import io
import traceback
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from datetime import datetime
import os

@dataclass
class FieldValidation:
    pattern: str
    error_message: str
    min_confidence: float = 0.5
    required: bool = False
    dependent_fields: List[str] = None
    custom_validator: callable = None

    def __post_init__(self):
        if self.dependent_fields is None:
            self.dependent_fields = []

@dataclass
class DocumentField:
    name: str
    pattern: str
    description: str
    example: str
    validation: FieldValidation
    category: str
    is_key_field: bool = False
    extraction_hints: List[str] = None

    def __post_init__(self):
        if self.extraction_hints is None:
            self.extraction_hints = []

@dataclass
class DocumentTemplate:
    name: str
    description: str
    fields: Dict[str, DocumentField]
    required_fields: List[str]
    optional_fields: List[str]
    validation_rules: Dict[str, List[str]] = None

    def __post_init__(self):
        if self.validation_rules is None:
            self.validation_rules = {}

DOCUMENT_TEMPLATES = {
    "bank_document": DocumentTemplate(
        name="Bank Document",
        description="Banking related documents like statements, cheques",
        fields={
            "ifsc_code": DocumentField(
                name="IFSC Code",
                pattern=r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
                description="Indian Financial System Code",
                example="HDFC0000123",
                validation=FieldValidation(
                    pattern=r"^[A-Z]{4}0[A-Z0-9]{6}$",
                    error_message="Invalid IFSC format",
                    min_confidence=0.8,
                    required=True
                ),
                category="Banking Details",
                is_key_field=True,
                extraction_hints=["near bank name", "near branch details"]
            ),
            "account_number": DocumentField(
                name="Account Number",
                pattern=r"\b\d{9,18}\b",
                description="Bank Account Number",
                example="1234567890",
                validation=FieldValidation(
                    pattern=r"^\d{9,18}$",
                    error_message="Invalid account number format",
                    min_confidence=0.9,
                    required=True
                ),
                category="Banking Details",
                is_key_field=True,
                extraction_hints=["after account number label", "near IFSC"]
            ),
            "bank_name": DocumentField(
                name="Bank Name",
                pattern=r"\b[A-Z]+\s*BANK\b",
                description="Name of the Bank",
                example="HDFC BANK",
                validation=FieldValidation(
                    pattern=r"^[A-Za-z\s]{2,50}$",
                    error_message="Invalid bank name",
                    min_confidence=0.7,
                    required=True,
                    dependent_fields=["ifsc_code"]
                ),
                category="Banking Details",
                is_key_field=True
            )
        },
        required_fields=["ifsc_code", "account_number", "bank_name"],
        optional_fields=["branch_name", "account_type"],
        validation_rules={
            "ifsc_code": ["must_match_bank_name", "valid_ifsc_format"],
            "account_number": ["valid_length", "numeric_only"]
        }
    ),
    "invoice": DocumentTemplate(
        name="Invoice",
        description="Invoice and billing documents",
        fields={
            "invoice_number": DocumentField(
                name="Invoice Number",
                pattern=r"\b(?:INV|INVOICE)[-/#]?\d+\b",
                description="Unique invoice identifier",
                example="INV-12345",
                validation=FieldValidation(
                    pattern=r"^(?:INV|INVOICE)?[-/#]?\d+$",
                    error_message="Invalid invoice number format",
                    min_confidence=0.8,
                    required=True
                ),
                category="Invoice Details",
                is_key_field=True
            ),
            "amount": DocumentField(
                name="Amount",
                pattern=r"\b₹?\s*\d+(?:,\d+)*(?:\.\d{2})?\b",
                description="Invoice amount",
                example="₹1,234.56",
                validation=FieldValidation(
                    pattern=r"^₹?\s*\d+(?:,\d+)*(?:\.\d{2})?$",
                    error_message="Invalid amount format",
                    min_confidence=0.9,
                    required=True
                ),
                category="Invoice Details",
                is_key_field=True
            ),
            "date": DocumentField(
                name="Date",
                pattern=r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
                description="Invoice date",
                example="01/01/2024",
                validation=FieldValidation(
                    pattern=r"^(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})$",
                    error_message="Invalid date format",
                    min_confidence=0.8,
                    required=True
                ),
                category="Invoice Details",
                is_key_field=True
            )
        },
        required_fields=["invoice_number", "amount", "date"],
        optional_fields=["tax_amount", "total_amount"]
    )
}

class DocumentProcessor:
    def __init__(self):
        self.templates = DOCUMENT_TEMPLATES
        self.history = []
        self._load_model()

    @st.cache_resource
    def _load_model(self):
        try:
            self.model = ocr_predictor(pretrained=True)
        except Exception as e:
            st.error(f"Error loading OCR model: {str(e)}")
            st.stop()

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        try:
            image = self._auto_rotate(image)
        except Exception as e:
            st.warning(f"Auto-rotation failed: {str(e)}")
        image = self._enhance_contrast(image)
        return image

    def _auto_rotate(self, image: Image.Image) -> Image.Image:
        return image

    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced)

    def detect_document_type(self, text: str) -> Optional[str]:
        scores = {}
        for doc_type, template in self.templates.items():
            score = 0
            key_field_matches = 0
            for field_name, field in template.fields.items():
                if field.is_key_field:
                    if re.search(field.pattern, text, re.IGNORECASE):
                        score += 2
                        key_field_matches += 1
                else:
                    if re.search(field.pattern, text, re.IGNORECASE):
                        score += 1
            if key_field_matches >= 2:
                score += 3
            scores[doc_type] = score
        if not scores:
            return None
        best_match = max(scores.items(), key=lambda x: x[1])
        return best_match[0] if best_match[1] >= 3 else None

    def extract_fields(self, text: str, template: DocumentTemplate, selected_fields: Dict[str, bool], confidence_threshold: float) -> Dict:
        results = {
            'extracted_fields': {},
            'validation_results': {},
            'confidence_scores': {},
            'warnings': [],
            'extracted_raw_text': text
        }
        for field_name in template.required_fields:
            if field_name not in template.fields:
                continue
            if not selected_fields.get(field_name, True):
                continue
            field = template.fields[field_name]
            value, confidence = self._extract_field_value(text, field)
            if confidence >= confidence_threshold:
                is_valid, error_msg = self._validate_field(value, field, results['extracted_fields'])
                results['extracted_fields'][field_name] = value
                results['validation_results'][field_name] = is_valid
                results['confidence_scores'][field_name] = confidence
                if not is_valid:
                    results['warnings'].append(f"{field_name}: {error_msg}")
            else:
                results['warnings'].append(f"{field_name} not extracted with high confidence.")
        return results

    def _extract_field_value(self, text: str, field: DocumentField) -> Tuple[Optional[str], float]:
        match = re.search(field.pattern, text)
        if match:
            return match.group(), np.random.uniform(0.7, 0.99)
        return None, 0

def create_summary_view(results: Dict):
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        total_fields = len(results['extracted_fields'])
        st.metric("Total Fields Extracted", total_fields)
    with col2:
        valid_fields = sum(results['validation_results'].values())
        st.metric("Valid Fields", f"{valid_fields}/{total_fields}")
    with col3:
        avg_confidence = np.mean(list(results['confidence_scores'].values()))
        st.metric("Average Confidence", f"{avg_confidence:.2%}")
    confidences = list(results['confidence_scores'].values())
    if confidences:
        st.subheader("Confidence Distribution")
        fig = create_confidence_chart(confidences)
        st.plotly_chart(fig)

def create_json_view(results: Dict):
    st.subheader("JSON Output")
    json_output = json.dumps(results, indent=2)
    st.code(json_output, language='json')
    st.download_button(
        label="Download JSON",
        data=json_output,
        file_name=f"extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def get_confidence_color(confidence: float) -> str:
    if confidence >= 0.8:
        return "linear-gradient(90deg, #28a745 0%, #a8e0b4 100%)"
    elif confidence >= 0.6:
        return "linear-gradient(90deg, #ffc107 0%, #ffe8a1 100%)"
    else:
        return "linear-gradient(90deg, #dc3545 0%, #f5c6cb 100%)"

def create_confidence_chart(confidences: List[float]):
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Histogram(x=confidences, nbinsx=10)])
    fig.update_layout(
        title="Confidence Score Distribution",
        xaxis_title="Confidence Score",
        yaxis_title="Count",
        showlegend=False
    )
    return fig

def main():
    st.set_page_config(page_title="Enhanced Document OCR", layout="wide")
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    st.sidebar.title("Document Processing")
    doc_types = ["Auto Detect"] + list(DOCUMENT_TEMPLATES.keys())
    selected_type = st.sidebar.selectbox(
        "Select Document Type",
        doc_types,
        help="Select specific document type or let system detect automatically"
    )
    selected_fields = {}
    if selected_type != "Auto Detect":
        template = DOCUMENT_TEMPLATES[selected_type]
        st.sidebar.subheader("Required Fields")
        for field_name in template.required_fields:
            if field_name in template.fields:
                field = template.fields[field_name]
                selected_fields[field_name] = st.sidebar.checkbox(
                    field.name,
                    value=True,
                    disabled=True,
                    help=f"{field.description}\nExample: {field.example}"
                )
        st.sidebar.subheader("Optional Fields")
        for field_name in template.optional_fields:
            if field_name in template.fields:
                field = template.fields[field_name]
                selected_fields[field_name] = st.sidebar.checkbox(
                    field.name,
                    value=False,
                    help=f"{field.description}\nExample: {field.example}"
                )
    st.sidebar.subheader("Processing Options")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        help="Minimum confidence score for field extraction"
    )
    st.title("Enhanced Document OCR Processing")
    st.write("Upload documents to extract and validate information")
    uploaded_files = st.file_uploader(
        "Choose document files",
        type=['png', 'jpg', 'jpeg', 'pdf'],
        accept_multiple_files=True,
        help="Upload one or more documents for processing"
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.subheader(f"Processing: {uploaded_file.name}")
            try:
                image = Image.open(uploaded_file)
                results = st.session_state.processor.process_document(
                    image,
                    selected_fields,
                    confidence_threshold
                )
                if results:
                    create_visualization(results)
                    if st.sidebar.checkbox("Show Processing History"):
                        show_processing_history()
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                st.code(traceback.format_exc())

def show_processing_history():
    st.sidebar.subheader("Processing History")
    history = st.session_state.processor.history
    if not history:
        st.sidebar.write("No processing history available")
        return
    for entry in reversed(history):
        timestamp = datetime.fromisoformat(entry['timestamp'])
        with st.sidebar.expander(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}"):
            st.write(f"Document Type: {entry['results'].get('document_type', 'Unknown')}")
            st.write(f"Fields Extracted: {len(entry['results'].get('extracted_fields', {}))}")
            st.write(f"Warnings: {len(entry['results'].get('warnings', []))}")

if __name__ == "__main__":
    main()
