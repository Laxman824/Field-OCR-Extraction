# from src.utils.field_extraction import FieldExtractor
# from src.utils.file_processing import DocumentProcessor
# from src.models.model_loader import load_model
# from src.config.field_patterns import FIELD_CATEGORIES
# from utils.field_extraction import FieldExtractor
# from utils.file_processing import DocumentProcessor
# from models.model_loader import load_model
# from config.field_patterns import FIELD_CATEGORIES
# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
import os 
import sys
# Add the src directory to the Python path
src_dir = os.path.join(current_dir, 'src')
print(src_dir)
sys.path.insert(0, src_dir)
import streamlit as st
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import traceback
from typing import Dict, List, Tuple, Optional
import yaml
# Add the src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.append(str(src_path))

# Import custom modules

from utils.field_extraction import FieldExtractor
from models.model_loader import load_model
from config.field_patterns import FIELD_CATEGORIES
from utils.file_processing import DocumentProcessor
class OCREnhancedApp:
    def __init__(self):
        """Initialize the enhanced OCR application"""
        self.setup_page_config()
        self.initialize_components()
        self.initialize_session_state()
        
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Enhanced Document OCR",
            page_icon="📄",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def initialize_components(self):
        """Initialize main components and load models"""
        try:
            # Load configuration
            with open("config.yaml", 'r') as file:
                self.config = yaml.safe_load(file)
            
            # Initialize processors
            self.doc_processor = DocumentProcessor()
            self.model = load_model()
            self.field_extractor = FieldExtractor(FIELD_CATEGORIES)
            
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
            st.stop()
            
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = []
        if 'batch_results' not in st.session_state:
            st.session_state.batch_results = {}

    def create_sidebar(self) -> Dict:
        """Create sidebar with enhanced options"""
        st.sidebar.title("📋 Processing Options")

        # Document Type Selection
        doc_type = st.sidebar.selectbox(
            "Select Document Type",
            ["Auto Detect", "Investment Document", "Bank Statement", "ID Card", "Custom"]
        )

        # Advanced Settings
        with st.sidebar.expander("🛠️ Advanced Settings"):
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.6
            )
            
            enable_preprocessing = st.checkbox(
                "Enable Enhanced Preprocessing",
                value=True
            )
            
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=10,
                value=4
            )

        # Field Selection
        st.sidebar.subheader("Select Fields to Extract")
        selected_fields = {}
        
        for category, fields in FIELD_CATEGORIES.items():
            st.sidebar.subheader(f"📌 {category}")
            for field_name, field_info in fields.items():
                help_text = f"""
                Description: {field_info['description']}
                Example: {field_info['example']}
                """
                selected_fields[field_name] = st.sidebar.checkbox(
                    field_name,
                    value=True,
                    help=help_text
                )

        return {
            'doc_type': doc_type,
            'confidence_threshold': confidence_threshold,
            'enable_preprocessing': enable_preprocessing,
            'batch_size': batch_size,
            'selected_fields': selected_fields
        }

    def file_uploader_section(self):
        """Enhanced file uploader section"""
        st.header("📤 Upload Documents")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_files = st.file_uploader(
                "Drop files here or click to upload",
                type=["pdf", "png", "jpg", "jpeg"],
                accept_multiple_files=True,
                help="Supports PDF and images (PNG, JPG)"
            )

        with col2:
            if st.button("🔄 Clear All"):
                st.session_state.processed_files = []
                st.session_state.batch_results = {}
                st.experimental_rerun()

            if st.button("📊 Generate Report"):
                self.generate_report()

        return uploaded_files

    def process_files(self, files: List, settings: Dict):
        """Process multiple files with progress tracking"""
        if not files:
            return []

        progress_bar = st.progress(0)
        status_text = st.empty()
        processed_files = []

        for idx, file in enumerate(files):
            try:
                status_text.text(f"Processing {file.name}...")
                progress_bar.progress((idx + 1) / len(files))

                # Handle PDF files
                if file.name.lower().endswith('.pdf'):
                    images = self.doc_processor.convert_pdf_to_images(file)
                else:
                    images = [Image.open(file)]

                # Process each image
                for page_idx, image in enumerate(images):
                    if settings['enable_preprocessing']:
                        image = self.doc_processor.enhance_image(image)

                    # Process image and extract fields
                    result = self.process_single_image(
                        image, 
                        settings['selected_fields'],
                        settings['confidence_threshold']
                    )

                    processed_files.append({
                        'filename': f"{file.name}_page{page_idx + 1}",
                        'result': result
                    })

            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                st.code(traceback.format_exc())

        progress_bar.empty()
        status_text.empty()
        return processed_files

    def process_single_image(self, image: Image.Image, selected_fields: Dict, confidence_threshold: float):
        """Process a single image and extract fields"""
        try:
            # Convert image to array
            img_np = np.array(image)
            
            # OCR processing
            result = self.model([img_np])
            
            # Extract fields
            extracted_fields = {}
            for field_name, is_selected in selected_fields.items():
                if is_selected:
                    value, confidence = self.field_extractor.extract_with_context(
                        result.pages[0].get_text(),
                        field_name
                    )
                    if value and confidence >= confidence_threshold:
                        extracted_fields[field_name] = (value, confidence)

            return {
                'extracted_fields': extracted_fields,
                'ocr_result': result
            }

        except Exception as e:
            st.error(f"Error in image processing: {str(e)}")
            return None

    def display_results(self, processed_files: List):
        """Display processed results with enhanced visualization"""
        if not processed_files:
            return

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Summary", "📄 Details", "🔍 Raw Data"])

        with tab1:
            self.display_summary(processed_files)

        with tab2:
            self.display_detailed_results(processed_files)

        with tab3:
            self.display_raw_data(processed_files)

    def display_summary(self, processed_files: List):
        """Display summary statistics and charts"""
        st.subheader("Processing Summary")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Files Processed", len(processed_files))
        
        with col2:
            avg_confidence = self.calculate_average_confidence(processed_files)
            st.metric("Average Confidence", f"{avg_confidence:.2f}%")
        
        with col3:
            success_rate = self.calculate_success_rate(processed_files)
            st.metric("Success Rate", f"{success_rate:.2f}%")

        # Confidence distribution chart
        fig = self.create_confidence_chart(processed_files)
        st.plotly_chart(fig, use_container_width=True)

    def display_detailed_results(self, processed_files: List):
        """Display detailed results for each file"""
        for file_data in processed_files:
            with st.expander(f"📄 {file_data['filename']}", expanded=False):
                if file_data['result']:
                    # Display extracted fields
                    df = pd.DataFrame([
                        {
                            'Field': field,
                            'Value': value,
                            'Confidence': f"{conf:.2%}"
                        }
                        for field, (value, conf) in file_data['result']['extracted_fields'].items()
                    ])
                    st.dataframe(df)

    def display_raw_data(self, processed_files: List):
        """Display raw JSON data with download option"""
        st.json(processed_files)
        
        # Add download button
        json_str = json.dumps(processed_files, indent=2)
        st.download_button(
            "Download JSON Results",
            json_str,
            file_name=f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    def generate_report(self):
        """Generate downloadable report"""
        if not st.session_state.processed_files:
            st.warning("No processed files to generate report from.")
            return

        # Create report content
        report_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_files': len(st.session_state.processed_files),
                'average_confidence': self.calculate_average_confidence(st.session_state.processed_files),
                'success_rate': self.calculate_success_rate(st.session_state.processed_files)
            },
            'detailed_results': st.session_state.processed_files
        }

        # Convert to JSON and offer download
        report_json = json.dumps(report_data, indent=2)
        st.download_button(
            "Download Detailed Report",
            report_json,
            file_name=f"ocr_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    def calculate_average_confidence(self, processed_files: List) -> float:
        """Calculate average confidence across all extracted fields"""
        confidences = []
        for file_data in processed_files:
            if file_data['result'] and file_data['result']['extracted_fields']:
                confidences.extend([conf for _, conf in file_data['result']['extracted_fields'].values()])
        return sum(confidences) / len(confidences) * 100 if confidences else 0

    def calculate_success_rate(self, processed_files: List) -> float:
        """Calculate success rate of field extraction"""
        total_fields = 0
        successful_fields = 0
        for file_data in processed_files:
            if file_data['result'] and file_data['result']['extracted_fields']:
                for _, conf in file_data['result']['extracted_fields'].values():
                    total_fields += 1
                    if conf >= 0.6:
                        successful_fields += 1
        return (successful_fields / total_fields * 100) if total_fields > 0 else 0

    def create_confidence_chart(self, processed_files: List):
        """Create confidence distribution chart"""
        confidences = []
        for file_data in processed_files:
            if file_data['result'] and file_data['result']['extracted_fields']:
                confidences.extend([conf for _, conf in file_data['result']['extracted_fields'].values()])

        fig = go.Figure(data=[go.Histogram(x=confidences, nbinsx=20)])
        fig.update_layout(
            title="Confidence Score Distribution",
            xaxis_title="Confidence Score",
            yaxis_title="Frequency"
        )
        return fig

    def run(self):
        """Run the enhanced OCR application"""
        st.title("📚 Advanced Document OCR System")
        st.markdown("""
        Upload your documents and extract information with advanced processing capabilities.
        Supports PDF and image formats with batch processing.
        """)

        # Create sidebar and get settings
        settings = self.create_sidebar()
        
        # File upload section
        uploaded_files = self.file_uploader_section()
        
        # Process files if uploaded
        if uploaded_files:
            with st.spinner('Processing documents...'):
                processed_files = self.process_files(uploaded_files, settings)
                if processed_files:
                    st.session_state.processed_files.extend(processed_files)
            
            # Display results
            self.display_results(st.session_state.processed_files)

if __name__ == "__main__":
    app = OCREnhancedApp()
    app.run()