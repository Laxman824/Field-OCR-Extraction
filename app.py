
import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

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

# class OCREnhancedApp:
#     def __init__(self):
#         """Initialize the enhanced OCR application"""
#         self.setup_page_config()
#         self.initialize_components()
#         self.initialize_session_state()
        
#     def setup_page_config(self):
#         """Configure Streamlit page settings"""
#         st.set_page_config(
#             page_title="Enhanced Document OCR",
#             page_icon="📄",
#             layout="wide",
#             initial_sidebar_state="expanded"
#         )
        
#     def initialize_components(self):
#         """Initialize main components and load models"""
#         try:
#             # Load configuration
#             with open("config.yaml", 'r') as file:
#                 self.config = yaml.safe_load(file)
            
#             # Initialize processors
#             self.doc_processor = DocumentProcessor()
#             self.model = load_model()
#             self.field_extractor = FieldExtractor(FIELD_CATEGORIES)
            
#         except Exception as e:
#             st.error(f"Error initializing components: {str(e)}")
#             st.stop()
            
#     def initialize_session_state(self):
#         """Initialize Streamlit session state variables"""
#         if 'processed_files' not in st.session_state:
#             st.session_state.processed_files = []
#         if 'batch_results' not in st.session_state:
#             st.session_state.batch_results = {}

#     def create_sidebar(self) -> Dict:
#         """Create sidebar with enhanced options"""
#         st.sidebar.title("📋 Processing Options")

#         # Document Type Selection
#         doc_type = st.sidebar.selectbox(
#             "Select Document Type",
#             ["Auto Detect", "Investment Document", "Bank Statement", "ID Card", "Custom"]
#         )

#         # Advanced Settings
#         with st.sidebar.expander("🛠️ Advanced Settings"):
#             confidence_threshold = st.slider(
#                 "Confidence Threshold",
#                 min_value=0.0,
#                 max_value=1.0,
#                 value=0.6
#             )
            
#             enable_preprocessing = st.checkbox(
#                 "Enable Enhanced Preprocessing",
#                 value=True
#             )
            
#             batch_size = st.number_input(
#                 "Batch Size",
#                 min_value=1,
#                 max_value=10,
#                 value=4
#             )

#         # Field Selection
#         st.sidebar.subheader("Select Fields to Extract")
#         selected_fields = {}
        
#         for category, fields in FIELD_CATEGORIES.items():
#             st.sidebar.subheader(f"📌 {category}")
#             for field_name, field_info in fields.items():
#                 help_text = f"""
#                 Description: {field_info['description']}
#                 Example: {field_info['example']}
#                 """
#                 selected_fields[field_name] = st.sidebar.checkbox(
#                     field_name,
#                     value=True,
#                     help=help_text
#                 )

#         return {
#             'doc_type': doc_type,
#             'confidence_threshold': confidence_threshold,
#             'enable_preprocessing': enable_preprocessing,
#             'batch_size': batch_size,
#             'selected_fields': selected_fields
#         }

#     def file_uploader_section(self):
#         """Enhanced file uploader section"""
#         st.header("📤 Upload Documents")
        
#         col1, col2 = st.columns([2, 1])
#         with col1:
#             uploaded_files = st.file_uploader(
#                 "Drop files here or click to upload",
#                 type=["pdf", "png", "jpg", "jpeg"],
#                 accept_multiple_files=True,
#                 help="Supports PDF and images (PNG, JPG)"
#             )

#         with col2:
#             if st.button("🔄 Clear All"):
#                 st.session_state.processed_files = []
#                 st.session_state.batch_results = {}
#                 st.rerun()  # Updated from experimental_rerun
                
#             if st.button("📊 Generate Report"):
#                 self.generate_report()

#         return uploaded_files

#     def process_files(self, files: List, settings: Dict):
#         """Process multiple files with progress tracking"""
#         if not files:
#             return []

#         progress_bar = st.progress(0)
#         status_text = st.empty()
#         processed_files = []

#         for idx, file in enumerate(files):
#             try:
#                 status_text.text(f"Processing {file.name}...")
#                 progress_bar.progress((idx + 1) / len(files))

#                 # Handle PDF files
#                 if file.name.lower().endswith('.pdf'):
#                     images = self.doc_processor.convert_pdf_to_images(file)
#                 else:
#                     # Open image and ensure it's RGB
#                     image = Image.open(file)
#                     if image.mode != 'RGB':
#                         image = image.convert('RGB')
#                     images = [image]

#                 # Process each image
#                 for page_idx, image in enumerate(images):
#                     try:
#                         if settings['enable_preprocessing']:
#                             image = self.doc_processor.enhance_image(image)

#                         # Process image and extract fields
#                         result = self.process_single_image(
#                             image, 
#                             settings['selected_fields'],
#                             settings['confidence_threshold']
#                         )

#                         processed_files.append({
#                             'filename': f"{file.name}_page{page_idx + 1}" if len(images) > 1 else file.name,
#                             'result': result
#                         })

#                     except Exception as e:
#                         st.error(f"Error processing page {page_idx + 1} of {file.name}: {str(e)}")
#                         st.code(traceback.format_exc())

#             except Exception as e:
#                 st.error(f"Error processing file {file.name}: {str(e)}")
#                 st.code(traceback.format_exc())

#         progress_bar.empty()
#         status_text.empty()
#         return processed_files
        

#     def process_single_image(self, image: Image.Image, selected_fields: Dict, confidence_threshold: float):
#         """Process a single image and extract fields with JSON-serializable results"""
#         try:
#             # Ensure image is in RGB mode
#             if image.mode != 'RGB':
#                 image = image.convert('RGB')
            
#             # Convert image to numpy array
#             img_np = np.array(image)
            
#             # Ensure image is 3-channel RGB
#             if len(img_np.shape) == 2:  # If grayscale
#                 img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
#             elif len(img_np.shape) == 3 and img_np.shape[2] == 4:  # If RGBA
#                 img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
                
#             # OCR processing
#             doc = self.model([img_np])
            
#             # Initialize result dictionary with only serializable data
#             result_dict = {
#                 'extracted_fields': {},
#                 'words': [],
#                 'full_text': '',
#                 'image_size': {
#                     'width': img_np.shape[1],
#                     'height': img_np.shape[0]
#                 }
#             }
            
#             if doc and doc.pages:
#                 page = doc.pages[0]
#                 full_text = []
                
#                 # Process blocks, lines, and words
#                 for block in page.blocks:
#                     for line in block.lines:
#                         line_words = []
#                         for word in line.words:
#                             word_dict = {
#                                 'text': word.value,
#                                 'confidence': float(word.confidence),
#                                 'bbox': [float(coord) for point in word.geometry for coord in point]
#                             }
#                             line_words.append(word_dict['text'])
#                             result_dict['words'].append(word_dict)
#                         full_text.append(' '.join(line_words))
                
#                 # Join all text
#                 result_dict['full_text'] = '\n'.join(full_text)
                
#                 # Extract fields
#                 for field_name, is_selected in selected_fields.items():
#                     if is_selected:
#                         value, confidence = self.field_extractor.extract_with_context(
#                             result_dict['full_text'],
#                             field_name
#                         )
#                         if value and confidence >= confidence_threshold:
#                             result_dict['extracted_fields'][field_name] = {
#                                 'value': str(value),
#                                 'confidence': float(confidence)
#                             }

#             return result_dict

#         except Exception as e:
#             st.error(f"Error in image processing: {str(e)}")
#             st.error(f"Full error: {traceback.format_exc()}")
#             return {
#                 'extracted_fields': {},
#                 'words': [],
#                 'full_text': '',
#                 'image_size': {'width': 0, 'height': 0}
#             }


#     def display_summary(self, processed_files: List):
#         """Display summary statistics"""
#         st.subheader("Processing Summary")
        
#         # Summary metrics
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             st.metric(
#                 label="Files Processed",
#                 value=len(processed_files)
#             )
        
#         with col2:
#             avg_confidence = self.calculate_average_confidence(processed_files)
#             st.metric(
#                 label="Average Confidence",
#                 value=f"{avg_confidence:.2f}%"
#             )
        
#         with col3:
#             success_rate = self.calculate_success_rate(processed_files)
#             st.metric(
#                 label="Success Rate",
#                 value=f"{success_rate:.2f}%"
#             )

#         # Add confidence distribution if there are results
#         if processed_files:
#             st.subheader("Confidence Distribution")
#             confidence_data = []
#             for file_data in processed_files:
#                 if file_data['result'].get('extracted_fields'):
#                     for field_info in file_data['result']['extracted_fields'].values():
#                         confidence_data.append(field_info['confidence'])
            
#             if confidence_data:
#                 fig = go.Figure(data=[go.Histogram(x=confidence_data, nbinsx=20)])
#                 fig.update_layout(
#                     title="Field Confidence Distribution",
#                     xaxis_title="Confidence Score",
#                     yaxis_title="Frequency",
#                     xaxis=dict(range=[0, 1])
#                 )
#                 st.plotly_chart(fig, use_container_width=True)

#     def display_detailed_results(self, processed_files: List):
#         """Display detailed results with safe JSON handling"""
#         for idx, file_data in enumerate(processed_files):
#             with st.expander(f"📄 {file_data['filename']}", expanded=idx == 0):
#                 result = file_data.get('result', {})
#                 if result:
#                     # Create two columns
#                     col1, col2 = st.columns([2, 1])
                    
#                     with col1:
#                         # Display extracted fields
#                         st.subheader("Extracted Fields")
#                         if result.get('extracted_fields'):
#                             df_data = [
#                                 {
#                                     'Field': field,
#                                     'Value': info['value'],
#                                     'Confidence': f"{info['confidence']:.2%}"
#                                 }
#                                 for field, info in result['extracted_fields'].items()
#                             ]
#                             if df_data:
#                                 df = pd.DataFrame(df_data)
#                                 st.dataframe(
#                                     df,
#                                     use_container_width=True,
#                                     key=f"df_{idx}"
#                                 )
#                         else:
#                             st.warning(
#                                 "No fields were extracted",
#                                 key=f"warn_{idx}"
#                             )

#                         # Display full text
#                         st.subheader("Full Extracted Text")
#                         st.text_area(
#                             label="Extracted Text",
#                             value=result.get('full_text', ''),
#                             height=200,
#                             key=f"text_{idx}"
#                         )

#                     with col2:
#                         # Display confidence visualization
#                         if result.get('extracted_fields'):
#                             st.subheader("Confidence Scores")
#                             scores = [
#                                 (field, info['confidence'])
#                                 for field, info in result['extracted_fields'].items()
#                             ]
#                             if scores:
#                                 fig = go.Figure(go.Bar(
#                                     x=[score[1] for score in scores],
#                                     y=[score[0] for score in scores],
#                                     orientation='h'
#                                 ))
#                                 fig.update_layout(
#                                     title="Field Confidence Scores",
#                                     xaxis_title="Confidence",
#                                     yaxis_title="Field",
#                                     xaxis=dict(range=[0, 1])
#                                 )
#                                 st.plotly_chart(
#                                     fig,
#                                     use_container_width=True,
#                                     key=f"plot_{idx}"
#                                 )

#                         try:
#                             # Safely create JSON for download
#                             json_str = json.dumps(result, indent=2)
#                             st.download_button(
#                                 label="Download Results",
#                                 data=json_str,
#                                 file_name=f"{file_data['filename']}_results.json",
#                                 mime="application/json",
#                                 key=f"download_{idx}"
#                             )
#                         except TypeError as e:
#                             st.error(
#                                 f"Error creating JSON: {str(e)}",
#                                 key=f"error_json_{idx}"
#                             )

#     def display_raw_data(self, processed_files: List):
#         """Display raw JSON data"""
#         try:
#             # Create a clean version of the data for JSON display
#             clean_data = []
#             for file_data in processed_files:
#                 clean_file_data = {
#                     'filename': file_data['filename'],
#                     'result': {
#                         'extracted_fields': file_data['result'].get('extracted_fields', {}),
#                         'full_text': file_data['result'].get('full_text', ''),
#                         'words': file_data['result'].get('words', [])
#                     }
#                 }
#                 clean_data.append(clean_file_data)

#             # Display JSON
#             st.json(clean_data)
            
#             # Download button for complete data
#             json_str = json.dumps(clean_data, indent=2)
#             st.download_button(
#                 label="Download Complete Results",
#                 data=json_str,
#                 file_name="all_results.json",
#                 mime="application/json",
#                 key="download_all"
#             )
#         except Exception as e:
#             st.error(
#                 f"Error displaying raw data: {str(e)}",
#                 key="raw_data_error"
#             )

#     def generate_report(self):
#         """Generate downloadable report"""
#         if not st.session_state.processed_files:
#             st.warning("No processed files to generate report from.")
#             return

#         # Create report content
#         report_data = {
#             'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             'summary': {
#                 'total_files': len(st.session_state.processed_files),
#                 'average_confidence': self.calculate_average_confidence(st.session_state.processed_files),
#                 'success_rate': self.calculate_success_rate(st.session_state.processed_files)
#             },
#             'detailed_results': st.session_state.processed_files
#         }

#         # Convert to JSON and offer download
#         report_json = json.dumps(report_data, indent=2)
#         st.download_button(
#             "Download Detailed Report",
#             report_json,
#             file_name=f"ocr_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
#             mime="application/json"
#         )

#     def calculate_average_confidence(self, processed_files: List) -> float:
#         """Calculate average confidence across all extracted fields"""
#         confidences = []
#         for file_data in processed_files:
#             if file_data['result'] and file_data['result']['extracted_fields']:
#                 confidences.extend([conf for _, conf in file_data['result']['extracted_fields'].values()])
#         return sum(confidences) / len(confidences) * 100 if confidences else 0

#     def calculate_success_rate(self, processed_files: List) -> float:
#         """Calculate success rate of field extraction"""
#         total_fields = 0
#         successful_fields = 0
#         for file_data in processed_files:
#             if file_data['result'] and file_data['result']['extracted_fields']:
#                 for _, conf in file_data['result']['extracted_fields'].values():
#                     total_fields += 1
#                     if conf >= 0.6:
#                         successful_fields += 1
#         return (successful_fields / total_fields * 100) if total_fields > 0 else 0

#     def create_confidence_chart(self, processed_files: List):
#         """Create confidence distribution chart"""
#         confidences = []
#         for file_data in processed_files:
#             if file_data['result'] and file_data['result']['extracted_fields']:
#                 confidences.extend([conf for _, conf in file_data['result']['extracted_fields'].values()])

#         fig = go.Figure(data=[go.Histogram(x=confidences, nbinsx=20)])
#         fig.update_layout(
#             title="Confidence Score Distribution",
#             xaxis_title="Confidence Score",
#             yaxis_title="Frequency"
#         )
#         return fig

#     def run(self):
#         """Run the enhanced OCR application"""
#         st.title("📚 Advanced Document OCR System")
#         st.markdown("""
#         Upload your documents and extract information with advanced processing capabilities.
#         Supports PDF and image formats with batch processing.
#         """)

#         # Create sidebar and get settings
#         settings = self.create_sidebar()
        
#         # File upload section
#         uploaded_files = self.file_uploader_section()
        
#         # Process files if uploaded
#         if uploaded_files:
#             with st.spinner('Processing documents...'):
#                 processed_files = self.process_files(uploaded_files, settings)
#                 if processed_files:
#                     st.session_state.processed_files.extend(processed_files)
            
#             # Display results
#             self.display_results(st.session_state.processed_files)

# if __name__ == "__main__":
#     app = OCREnhancedApp()
#     app.run()


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
                selected_fields[field_name] = st.sidebar.checkbox(
                    field_name,
                    value=True,
                    help=f"Description: {field_info.get('description', '')}\nExample: {field_info.get('example', '')}"
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
            if st.button("🔄 Clear All", key="clear_btn"):
                st.session_state.processed_files = []
                st.session_state.batch_results = {}
                st.rerun()

            if st.button("📊 Generate Report", key="report_btn"):
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
                    image = Image.open(file)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    images = [image]

                # Process each image
                for page_idx, image in enumerate(images):
                    try:
                        if settings['enable_preprocessing']:
                            image = self.doc_processor.enhance_image(image)

                        result = self.process_single_image(
                            image, 
                            settings['selected_fields'],
                            settings['confidence_threshold']
                        )

                        processed_files.append({
                            'filename': f"{file.name}_page{page_idx + 1}" if len(images) > 1 else file.name,
                            'result': result
                        })

                    except Exception as e:
                        st.error(f"Error processing page {page_idx + 1} of {file.name}: {str(e)}")
                        st.code(traceback.format_exc())

            except Exception as e:
                st.error(f"Error processing file {file.name}: {str(e)}")
                st.code(traceback.format_exc())

        progress_bar.empty()
        status_text.empty()
        return processed_files

    def process_single_image(self, image: Image.Image, selected_fields: Dict, confidence_threshold: float):
        """Process a single image and extract fields"""
        try:
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert image to numpy array
            img_np = np.array(image)
            
            # OCR processing
            doc = self.model([img_np])
            
            # Initialize result dictionary with only serializable data
            result_dict = {
                'extracted_fields': {},
                'words': [],
                'full_text': '',
                'image_size': {
                    'width': img_np.shape[1],
                    'height': img_np.shape[0]
                }
            }
            
            if doc and doc.pages:
                page = doc.pages[0]
                full_text = []
                
                # Process blocks, lines, and words
                for block in page.blocks:
                    for line in block.lines:
                        line_words = []
                        for word in line.words:
                            word_dict = {
                                'text': word.value,
                                'confidence': float(word.confidence),
                                'bbox': [float(coord) for point in word.geometry for coord in point]
                            }
                            line_words.append(word_dict['text'])
                            result_dict['words'].append(word_dict)
                        full_text.append(' '.join(line_words))
                
                # Join all text
                result_dict['full_text'] = '\n'.join(full_text)
                
                # Extract fields
                for field_name, is_selected in selected_fields.items():
                    if is_selected:
                        value, confidence = self.field_extractor.extract_with_context(
                            result_dict['full_text'],
                            field_name
                        )
                        if value and confidence >= confidence_threshold:
                            result_dict['extracted_fields'][field_name] = {
                                'value': str(value),
                                'confidence': float(confidence)
                            }

            return result_dict

        except Exception as e:
            st.error(f"Error in image processing: {str(e)}")
            st.error(f"Full error: {traceback.format_exc()}")
            return {
                'extracted_fields': {},
                'words': [],
                'full_text': '',
                'image_size': {'width': 0, 'height': 0}
            }

    def display_results(self, processed_files: List):
        """Display results with tabs"""
        if not processed_files:
            st.warning("No files have been processed yet.")
            return

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Summary", "📄 Detailed View", "🔍 JSON View"])

        with tab1:
            self.display_summary(processed_files)

        with tab2:
            self.display_detailed_results(processed_files)

        with tab3:
            self.display_raw_data(processed_files)

    def display_summary(self, processed_files: List):
        """Display summary statistics"""
        st.subheader("Processing Summary")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Files Processed",
                value=len(processed_files)
            )
        
        with col2:
            avg_confidence = self.calculate_average_confidence(processed_files)
            st.metric(
                label="Average Confidence",
                value=f"{avg_confidence:.2f}%"
            )
        
        with col3:
            success_rate = self.calculate_success_rate(processed_files)
            st.metric(
                label="Success Rate",
                value=f"{success_rate:.2f}%"
            )

    def display_detailed_results(self, processed_files: List):
        """Display detailed results for each file"""
        for idx, file_data in enumerate(processed_files):
            with st.expander(f"📄 {file_data['filename']}", expanded=idx == 0):
                result = file_data.get('result', {})
                if result:
                    # Create two columns
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Display extracted fields
                        st.subheader("Extracted Fields")
                        if result.get('extracted_fields'):
                            df_data = [
                                {
                                    'Field': field,
                                    'Value': info['value'],
                                    'Confidence': f"{info['confidence']:.2%}"
                                }
                                for field, info in result['extracted_fields'].items()
                            ]
                            if df_data:
                                df = pd.DataFrame(df_data)
                                st.dataframe(df, use_container_width=True, key=f"df_{idx}")
                        else:
                            st.warning("No fields were extracted", key=f"warn_{idx}")

                        # Display full text
                        st.subheader("Full Extracted Text")
                        st.text_area(
                            label="Extracted Text",
                            value=result.get('full_text', ''),
                            height=200,
                            key=f"text_{idx}"
                        )

                    with col2:
                        # Display confidence visualization
                        if result.get('extracted_fields'):
                            st.subheader("Confidence Scores")
                            scores = [
                                (field, info['confidence'])
                                for field, info in result['extracted_fields'].items()
                            ]
                            if scores:
                                fig = go.Figure(go.Bar(
                                    x=[score[1] for score in scores],
                                    y=[score[0] for score in scores],
                                    orientation='h'
                                ))
                                fig.update_layout(
                                    title="Field Confidence Scores",
                                    xaxis_title="Confidence",
                                    yaxis_title="Field",
                                    xaxis=dict(range=[0, 1])
                                )
                                st.plotly_chart(fig, use_container_width=True, key=f"plot_{idx}")

    def display_raw_data(self, processed_files: List):
        """Display raw JSON data"""
        try:
            # Create a clean version of the data for JSON display
            clean_data = []
            for file_data in processed_files:
                clean_file_data = {
                    'filename': file_data['filename'],
                    'result': {
                        'extracted_fields': file_data['result'].get('extracted_fields', {}),
                        'full_text': file_data['result'].get('full_text', ''),
                        'words': file_data['result'].get('words', [])
                    }
                }
                clean_data.append(clean_file_data)

            # Display JSON
            st.json(clean_data)
            
            # Download button for complete data
            json_str = json.dumps(clean_data, indent=2)
            st.download_button(
                label="Download Complete Results",
                data=json_str,
                file_name="all_results.json",
                mime="application/json",
                key="download_all"
            )
        except Exception as e:
            st.error(f"Error displaying raw data: {str(e)}", key="raw_data_error")

    def calculate_average_confidence(self, processed_files: List) -> float:
        """Calculate average confidence across all extracted fields"""
        confidences = []
        for file_data in processed_files:
            if file_data['result'].get('extracted_fields'):
                confidences.extend([
                    info['confidence']
                    for info in file_data['result']['extracted_fields'].values()
                ])
        return sum(confidences) / len(confidences) * 100 if confidences else 0

    def calculate_success_rate(self, processed_files: List) -> float:
        """Calculate success rate of field extraction"""
        total_fields = 0
        successful_fields = 0
        for file_data in processed_files:
            if file_data['result'].get('extracted_fields'):
                for info in file_data['result']['extracted_fields'].values():
                    total_fields += 1
                    if info['confidence'] >= 0.6:
                        successful_fields += 1
        return (successful_fields / total_fields * 100) if total_fields > 0 else 0

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

    def generate_report(self):
        """Generate a comprehensive report of processed documents"""
        if not st.session_state.processed_files:
            st.warning("No processed files available for report generation.")
            return

        try:
            report_data = {
                'summary': {
                    'total_files': len(st.session_state.processed_files),
                    'average_confidence': self.calculate_average_confidence(st.session_state.processed_files),
                    'success_rate': self.calculate_success_rate(st.session_state.processed_files)
                },
                'files': []
            }

            # Process each file
            for file_data in st.session_state.processed_files:
                file_report = {
                    'filename': file_data['filename'],
                    'extracted_fields': {},
                    'statistics': {
                        'field_count': 0,
                        'average_confidence': 0.0
                    }
                }

                if file_data['result'].get('extracted_fields'):
                    field_confidences = []
                    for field, info in file_data['result']['extracted_fields'].items():
                        file_report['extracted_fields'][field] = {
                            'value': info['value'],
                            'confidence': info['confidence']
                        }
                        field_confidences.append(info['confidence'])

                    file_report['statistics']['field_count'] = len(field_confidences)
                    file_report['statistics']['average_confidence'] = (
                        sum(field_confidences) / len(field_confidences)
                        if field_confidences else 0.0
                    )

                report_data['files'].append(file_report)

            # Create JSON for download
            json_str = json.dumps(report_data, indent=2)
            
            # Create download button
            st.download_button(
                label="📥 Download Report",
                data=json_str,
                file_name=f"ocr_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_report"
            )

            # Display report summary
            st.subheader("Report Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Total Files",
                    value=report_data['summary']['total_files']
                )
            with col2:
                st.metric(
                    label="Average Confidence",
                    value=f"{report_data['summary']['average_confidence']:.2f}%"
                )
            with col3:
                st.metric(
                    label="Success Rate",
                    value=f"{report_data['summary']['success_rate']:.2f}%"
                )

            # Display detailed results
            st.subheader("Detailed Results")
            for file_data in report_data['files']:
                with st.expander(f"📄 {file_data['filename']}", expanded=False):
                    if file_data['extracted_fields']:
                        df_data = [
                            {
                                'Field': field,
                                'Value': info['value'],
                                'Confidence': f"{info['confidence']:.2%}"
                            }
                            for field, info in file_data['extracted_fields'].items()
                        ]
                        st.dataframe(
                            pd.DataFrame(df_data),
                            use_container_width=True,
                        )
                    else:
                        st.warning("No fields extracted from this document")

        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            st.error(traceback.format_exc())

if __name__ == "__main__":
    app = OCREnhancedApp()
    app.run()