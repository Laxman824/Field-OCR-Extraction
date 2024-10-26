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
import tempfile
import os

# Document Templates and Field Definitions
@dataclass
class FieldValidation:
    pattern: str
    error_message: str
    min_confidence: float = 0.5
    required: bool = False
    dependent_fields: List[str] = None

@dataclass
class DocumentField:
    name: str
    pattern: str
    description: str
    example: str
    validation: FieldValidation
    category: str
    is_key_field: bool = False

@dataclass
class DocumentTemplate:
    name: str
    description: str
    fields: Dict[str, DocumentField]
    required_fields: List[str]
    optional_fields: List[str]

# Define document templates
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
                is_key_field=True
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
                is_key_field=True
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
                    required=True
                ),
                category="Banking Details",
                is_key_field=True
            )
        },
        required_fields=["ifsc_code", "account_number", "bank_name"],
        optional_fields=["branch_name", "account_type"]
    ),
    "invoice": DocumentTemplate(
        name="Invoice",
        description="Invoice and billing documents",
        fields={
            "invoice_number": DocumentField(
                name="Invoice Number",
                pattern=r"\b(INV|INVOICE)[-/#]?\d+\b",
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
            )
        },
        required_fields=["invoice_number", "amount", "date"],
        optional_fields=["tax_amount", "total_amount"]
    )
}

class DocumentProcessor:
    def __init__(self):
        self.templates = DOCUMENT_TEMPLATES
        self.model = self._load_model()
        self.batch_results = []

    @st.cache_resource
    def _load_model(self):
        try:
            return ocr_predictor(pretrained=True)
        except Exception as e:
            st.error(f"Error loading OCR model: {str(e)}")
            st.stop()

    def detect_document_type(self, text: str) -> Optional[str]:
        """Detect document type based on key fields"""
        scores = {}
        for doc_type, template in self.templates.items():
            score = 0
            for field_name, field in template.fields.items():
                if field.is_key_field and re.search(field.pattern, text, re.IGNORECASE):
                    score += 1
            scores[doc_type] = score
        
        if not scores:
            return None
        return max(scores.items(), key=lambda x: x[1])[0]

    def process_document(self, image: np.ndarray, selected_fields: Dict[str, bool],
                        confidence_threshold: float) -> Dict:
        """Process single document"""
        try:
            # Extract text
            result = self.model([image])
            extracted_text = self._get_text_from_result(result)
            
            # Detect document type
            doc_type = self.detect_document_type(extracted_text)
            
            # Process fields
            processed_result = self._process_fields(
                extracted_text,
                selected_fields,
                doc_type,
                confidence_threshold,
                image
            )
            
            return processed_result

        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            st.code(traceback.format_exc())
            return None

    def process_batch(self, images: List[np.ndarray], selected_fields: Dict[str, bool],
                     confidence_threshold: float) -> List[Dict]:
        """Process multiple documents"""
        results = []
        progress_bar = st.progress(0)
        
        for idx, image in enumerate(images):
            result = self.process_document(image, selected_fields, confidence_threshold)
            if result:
                results.append(result)
            progress_bar.progress((idx + 1) / len(images))
        
        self.batch_results = results
        return results

    def _get_text_from_result(self, result) -> str:
        """Extract text from OCR result"""
        text = ""
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    text += " ".join([word.value for word in line.words]) + "\n"
        return text

    def _process_fields(self, text: str, selected_fields: Dict[str, bool],
                       doc_type: str, confidence_threshold: float,
                       image: np.ndarray) -> Dict:
        """Process and validate fields"""
        results = {
            'extracted_fields': {},
            'validation_results': {},
            'confidence_scores': {},
            'warnings': [],
            'document_type': doc_type,
            'original_image': image,
            'processed_image': image.copy()
        }

        if doc_type and doc_type in self.templates:
            template = self.templates[doc_type]
            
            for field_name, field in template.fields.items():
                if not selected_fields.get(field_name, False):
                    continue

                # Extract value
                value = self._extract_field_value(text, field)
                if value:
                    # Validate
                    is_valid, error_msg, confidence = self._validate_field(
                        field, value, text
                    )
                    
                    if confidence >= confidence_threshold:
                        results['extracted_fields'][field_name] = value
                        results['validation_results'][field_name] = is_valid
                        results['confidence_scores'][field_name] = confidence
                        
                        if not is_valid:
                            results['warnings'].append(f"{field_name}: {error_msg}")
                
                elif field.validation.required:
                    results['warnings'].append(f"Required field {field_name} not found")

        return results

    def _extract_field_value(self, text: str, field: DocumentField) -> Optional[str]:
        """Extract value for a specific field"""
        match = re.search(field.pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
        return None

    def _validate_field(self, field: DocumentField, value: str, text: str) -> Tuple[bool, str, float]:
        """Validate field value"""
        if not value:
            return False, "Value is missing", 0.0

        # Pattern validation
        if not re.match(field.validation.pattern, value):
            return False, field.validation.error_message, 0.0

        # Context validation
        confidence = self._calculate_confidence(value, text)
        
        # Check dependent fields
        if field.validation.dependent_fields:
            for dep_field in field.validation.dependent_fields:
                if dep_field not in text:
                    return False, f"Missing dependent field: {dep_field}", confidence

        return True, "", confidence

    def _calculate_confidence(self, value: str, text: str) -> float:
        """Calculate confidence score"""
        # Simple confidence calculation based on surrounding context
        words_around = 5
        value_pos = text.find(value)
        if value_pos == -1:
            return 0.0
        
        context = text[max(0, value_pos - 50):min(len(text), value_pos + len(value) + 50)]
        relevant_words = len([w for w in context.split() if len(w) > 2])
        
        return min(1.0, relevant_words / words_around)

def create_ui():
    """Create enhanced UI"""
    st.set_page_config(page_title="Enhanced Document OCR", layout="wide")
    st.title("Enhanced Document OCR Processing")

    # Sidebar
    st.sidebar.title("Processing Options")
    
    # Document type selection
    doc_types = list(DOCUMENT_TEMPLATES.keys())
    selected_type = st.sidebar.selectbox(
        "Select Document Type",
        ["Auto Detect"] + doc_types,
        help="Select specific document type or let system detect automatically"
    )

    # Field selection
    selected_fields = {}
    if selected_type != "Auto Detect":
        template = DOCUMENT_TEMPLATES[selected_type]
        
        st.sidebar.subheader("Required Fields")
        for field_name in template.required_fields:
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

    # Processing options
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        help="Minimum confidence score for field extraction"
    )

    # Batch processing option
    is_batch = st.sidebar.checkbox("Enable Batch Processing", value=False)

    return selected_type, selected_fields, confidence_threshold, is_batch

def display_results(results: Dict):
    """Display processing results"""
    if not results:
        return

    # Display document type
    if results['document_type']:
        st.success(f"Detected Document Type: {results['document_type'].replace('_', ' ').title()}")

    # Display images
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Document")
        st.image(results['original_image'])
    
    with col2:
        st.subheader("Processed Document")
        st.image(results['processed_image'])

    # Display extracted information
    st.subheader("Extracted Information")
    
    # Create DataFrame for better visualization
    data = []
    for field_name, value in results['extracted_fields'].items():
        confidence = results['confidence_scores'].get(field_name, 0)
        is_valid = results['validation_results'].get(field_name, False)
        
        data.append({
            'Field': field_name,
            'Value': value,
            'Confidence': confidence,
            'Valid': '✅' if is_valid else '❌'
        })
    
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df)

        # Show warnings
        if results['warnings']:
            st.warning("Warnings:")
            for warning in results['warnings']:
                st.write(f"- {warning}")

def display_batch_results(results: List[Dict]):
    """Display batch processing results"""
    st.subheader("Batch Processing Results")
    
    # Summary statistics
    total_docs = len(results)
    successful_docs = len([r for r in results if r['extracted_fields']])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Documents", total_docs)
    with col2:
        st.metric("Successfully Processed", successful_docs)

    # Detailed results
    if st.checkbox("Show Detailed Results"):
        for idx, result in enumerate(results):
            with st.expander(f"Document {idx + 1}"):
                display_results(result)

    # Export options
    if st.button("Export Results"):
        export_results(results)



def export_results(results: List[Dict]):
    """Export processing results to various formats"""
    
    # Create temporary directory for exports
    with tempfile.TemporaryDirectory() as temp_dir:
        # Excel Export
        excel_path = os.path.join(temp_dir, 'ocr_results.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for idx, result in enumerate(results):
                summary_data.append({
                    'Document': f"Doc {idx + 1}",
                    'Type': result['document_type'],
                    'Fields Found': len(result['extracted_fields']),
                    'Warnings': len(result['warnings'])
                })
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed results sheet
            detailed_data = []
            for idx, result in enumerate(results):
                for field_name, value in result['extracted_fields'].items():
                    detailed_data.append({
                        'Document': f"Doc {idx + 1}",
                        'Field': field_name,
                        'Value': value,
                        'Confidence': result['confidence_scores'].get(field_name, 0),
                        'Valid': result['validation_results'].get(field_name, False)
                    })
            pd.DataFrame(detailed_data).to_excel(writer, sheet_name='Detailed Results', index=False)

            # Warnings sheet
            warnings_data = []
            for idx, result in enumerate(results):
                for warning in result['warnings']:
                    warnings_data.append({
                        'Document': f"Doc {idx + 1}",
                        'Warning': warning
                    })
            pd.DataFrame(warnings_data).to_excel(writer, sheet_name='Warnings', index=False)

        # JSON Export
        json_path = os.path.join(temp_dir, 'ocr_results.json')
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'total_documents': len(results),
            'results': [
                {
                    'document_id': idx,
                    'document_type': result['document_type'],
                    'extracted_fields': result['extracted_fields'],
                    'confidence_scores': result['confidence_scores'],
                    'validation_results': result['validation_results'],
                    'warnings': result['warnings']
                }
                for idx, result in enumerate(results)
            ]
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        # Create ZIP file containing all exports
        zip_path = os.path.join(temp_dir, 'ocr_results.zip')
        import zipfile
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(excel_path, 'ocr_results.xlsx')
            zipf.write(json_path, 'ocr_results.json')

        # Offer download through Streamlit
        with open(zip_path, 'rb') as f:
            st.download_button(
                label="Download All Results",
                data=f,
                file_name="ocr_results.zip",
                mime="application/zip",
                help="Download all results including Excel and JSON formats"
            )

def main():
    """Main application function"""
    # Create UI
    selected_type, selected_fields, confidence_threshold, is_batch = create_ui()
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # File upload section
    if is_batch:
        uploaded_files = st.file_uploader(
            "Upload multiple documents",
            type=['png', 'jpg', 'jpeg', 'pdf'],
            accept_multiple_files=True,
            help="Select multiple files to process in batch"
        )
        
        if uploaded_files:
            with st.spinner('Processing batch of documents...'):
                images = []
                for uploaded_file in uploaded_files:
                    try:
                        image = Image.open(uploaded_file)
                        images.append(np.array(image))
                    except Exception as e:
                        st.error(f"Error loading {uploaded_file.name}: {str(e)}")
                
                if images:
                    # Process batch
                    results = processor.process_batch(images, selected_fields, confidence_threshold)
                    display_batch_results(results)
                    
                    # Export options
                    if st.button("Export All Results"):
                        export_results(results)
    else:
        # Single file processing
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=['png', 'jpg', 'jpeg'],
            help="Select a single document to process"
        )
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                with st.spinner('Processing document...'):
                    result = processor.process_document(
                        np.array(image),
                        selected_fields,
                        confidence_threshold
                    )
                    display_results(result)
                    
                    # Single document export
                    if st.button("Export Results"):
                        export_results([result])
                        
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                st.code(traceback.format_exc())

    # Help section
    with st.expander("Help & Instructions"):
        st.markdown("""
        ### How to use this application:
        1. Select document type or use Auto Detect
        2. Choose fields to extract (required fields are pre-selected)
        3. Adjust confidence threshold if needed
        4. Upload document(s)
        5. Review results and download if needed
        
        ### Document Types Supported:
        - Bank Documents (statements, checks)
        - Invoices
        - More document types coming soon!
        
        ### Tips:
        - Use clear, well-lit images
        - Ensure text is clearly visible
        - Higher confidence threshold = more accurate but fewer results
        - Lower confidence threshold = more results but potential errors
        """)

if __name__ == "__main__":
    main()

# Add CSS for better styling
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .stAlert > div {
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
    .stDataFrame {
        margin: 1rem 0;
    }
    .stExpander {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)
