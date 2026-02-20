
# import os 
# import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Add the src directory to the Python path
# src_dir = os.path.join(current_dir, 'src')
# print(src_dir)
# sys.path.insert(0, src_dir)
# import streamlit as st
# import sys
# from pathlib import Path
# import numpy as np
# from PIL import Image
# import cv2
# import json
# import pandas as pd
# import plotly.graph_objects as go
# from datetime import datetime
# import traceback
# from typing import Dict, List, Tuple, Optional
# import yaml
# # Add the src directory to Python path
# src_path = Path(__file__).parent / 'src'
# sys.path.append(str(src_path))

# # Import custom modules

# from utils.field_extraction import FieldExtractor
# from models.model_loader import load_model
# from config.field_patterns import FIELD_CATEGORIES
# from utils.file_processing import DocumentProcessor

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
#                 selected_fields[field_name] = st.sidebar.checkbox(
#                     field_name,
#                     value=True,
#                     help=f"Description: {field_info.get('description', '')}\nExample: {field_info.get('example', '')}"
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
#             if st.button("🔄 Clear All", key="clear_btn"):
#                 st.session_state.processed_files = []
#                 st.session_state.batch_results = {}
#                 st.rerun()

#             if st.button("📊 Generate Report", key="report_btn"):
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
#                     image = Image.open(file)
#                     if image.mode != 'RGB':
#                         image = image.convert('RGB')
#                     images = [image]

#                 # Process each image
#                 for page_idx, image in enumerate(images):
#                     try:
#                         if settings['enable_preprocessing']:
#                             image = self.doc_processor.enhance_image(image)

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
#         """Process a single image and extract fields"""
#         try:
#             # Ensure image is in RGB mode
#             if image.mode != 'RGB':
#                 image = image.convert('RGB')
            
#             # Convert image to numpy array
#             img_np = np.array(image)
            
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

#     def display_detailed_results(self, processed_files: List):
#         """Display detailed results for each file"""
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
#                                 st.dataframe(df, use_container_width=True)
#                             else:
#                                 st.warning("No fields were successfully extracted")
#                         else:
#                             st.warning("No fields were extracted")

#                         # Display full text
#                         st.subheader("Full Extracted Text")
#                         st.text_area(
#                             label="",
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
#                                 st.plotly_chart(fig, use_container_width=True)

#                         # Add download button
#                         try:
#                             json_str = json.dumps(result, indent=2)
#                             st.download_button(
#                                 label="Download Results",
#                                 data=json_str,
#                                 file_name=f"{file_data['filename']}_results.json",
#                                 mime="application/json",
#                                 key=f"download_{idx}"
#                             )
#                         except Exception as e:
#                             st.error(f"Error creating download: {str(e)}")

#     def display_results(self, processed_files: List):
#         """Display results with tabs"""
#         if not processed_files:
#             st.warning("No files have been processed yet.")
#             return

#         # Create tabs for different views
#         tab1, tab2, tab3 = st.tabs(["📊 Summary", "📄 Detailed View", "🔍 JSON View"])

#         with tab1:
#             self.display_summary(processed_files)

#         with tab2:
#             self.display_detailed_results(processed_files)

#         with tab3:
#             self.display_raw_data(processed_files)

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
#             st.error(f"Error displaying raw data: {str(e)}")

#     def generate_report(self):
#         """Generate a comprehensive report"""
#         if not st.session_state.processed_files:
#             st.warning("No processed files available for report generation.")
#             return

#         try:
#             report_data = {
#                 'summary': {
#                     'total_files': len(st.session_state.processed_files),
#                     'average_confidence': self.calculate_average_confidence(st.session_state.processed_files),
#                     'success_rate': self.calculate_success_rate(st.session_state.processed_files)
#                 },
#                 'files': []
#             }

#             # Process each file
#             for file_data in st.session_state.processed_files:
#                 file_report = {
#                     'filename': file_data['filename'],
#                     'extracted_fields': file_data['result'].get('extracted_fields', {}),
#                     'statistics': {
#                         'field_count': len(file_data['result'].get('extracted_fields', {})),
#                         'average_confidence': self.calculate_average_confidence([file_data])
#                     }
#                 }
#                 report_data['files'].append(file_report)

#             # Display report summary
#             st.subheader("Report Summary")
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.metric(label="Total Files", value=report_data['summary']['total_files'])
#             with col2:
#                 st.metric(
#                     label="Average Confidence",
#                     value=f"{report_data['summary']['average_confidence']:.2f}%"
#                 )
#             with col3:
#                 st.metric(
#                     label="Success Rate",
#                     value=f"{report_data['summary']['success_rate']:.2f}%"
#                 )

#             # Add download button
#             json_str = json.dumps(report_data, indent=2)
#             st.download_button(
#                 label="Download Report",
#                 data=json_str,
#                 file_name=f"ocr_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
#                 mime="application/json",
#                 key="download_report"
#             )

#         except Exception as e:
#             st.error(f"Error generating report: {str(e)}")
#             st.error(traceback.format_exc())

            

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



#     def calculate_average_confidence(self, processed_files: List) -> float:
#         """Calculate average confidence across all extracted fields"""
#         confidences = []
#         for file_data in processed_files:
#             if file_data['result'].get('extracted_fields'):
#                 confidences.extend([
#                     info['confidence']
#                     for info in file_data['result']['extracted_fields'].values()
#                 ])
#         return sum(confidences) / len(confidences) * 100 if confidences else 0

#     def calculate_success_rate(self, processed_files: List) -> float:
#         """Calculate success rate of field extraction"""
#         total_fields = 0
#         successful_fields = 0
#         for file_data in processed_files:
#             if file_data['result'].get('extracted_fields'):
#                 for info in file_data['result']['extracted_fields'].values():
#                     total_fields += 1
#                     if info['confidence'] >= 0.6:
#                         successful_fields += 1
#         return (successful_fields / total_fields * 100) if total_fields > 0 else 0

#     def run(self):
#         """Run the enhanced OCR application"""
#         st.title("📚 Advanced Document OCR System")
#         st.markdown("""
#             Upload your documents and extract information with advanced processing capabilities.
#             Supports PDF and image formats with batch processing.
#             """)

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

# main.py
import os
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ── Path setup ──────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# ── Custom module imports ───────────────────────────────────────────
from utils.field_extraction import FieldExtractor
from models.model_loader import load_model
from config.field_patterns import FIELD_CATEGORIES
from utils.file_processing import DocumentProcessor


# ═══════════════════════════════════════════════════════════════════
#  CSS THEME
# ═══════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown(
        """
    <style>
    /* ── Font ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Hide defaults ────────────────────────────── */
    #MainMenu, header, footer, .stDeployButton { display: none !important; }

    /* ── Hero banner ──────────────────────────────── */
    .hero {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 20px;
        padding: 2.5rem 2rem 2rem;
        text-align: center;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(255,255,255,.06);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute; top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(99,102,241,.1) 0%, transparent 60%);
        animation: pulse 6s ease-in-out infinite;
    }
    @keyframes pulse {
        0%,100% { transform: scale(1); opacity:.6; }
        50%     { transform: scale(1.1); opacity:1; }
    }
    .hero h1 {
        font-size: 2.6rem; font-weight: 900;
        background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0; position: relative;
    }
    .hero .subtitle {
        color: #94a3b8; font-size: 1.1rem; margin-top: .4rem; position: relative;
    }

    /* ── Stat cards ───────────────────────────────── */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem; margin: 1.5rem 0;
    }
    .stat-card {
        background: linear-gradient(145deg, #1e1b4b, #1a1a2e);
        border: 1px solid rgba(255,255,255,.06);
        border-radius: 16px; padding: 1.4rem; text-align: center;
        transition: transform .2s, box-shadow .2s;
    }
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(99,102,241,.15);
    }
    .stat-icon  { font-size: 1.6rem; margin-bottom: .3rem; }
    .stat-value {
        font-size: 1.9rem; font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stat-label { color: #94a3b8; font-size: .82rem; margin-top: .15rem; }

    /* ── Section headers ──────────────────────────── */
    .section-title {
        font-size: 1.3rem; font-weight: 700; color: #e2e8f0;
        margin: 1.8rem 0 1rem;
        display: flex; align-items: center; gap: .5rem;
    }
    .section-title .line {
        flex: 1; height: 1px;
        background: linear-gradient(90deg, rgba(99,102,241,.4), transparent);
    }

    /* ── Upload zone ──────────────────────────────── */
    .upload-zone {
        background: linear-gradient(145deg, #1e1b4b, #1a1a2e);
        border: 2px dashed rgba(99,102,241,.3);
        border-radius: 16px; padding: 2rem; text-align: center;
        transition: border-color .3s;
        margin-bottom: 1.5rem;
    }
    .upload-zone:hover { border-color: rgba(99,102,241,.6); }

    /* ── Result cards ─────────────────────────────── */
    .result-card {
        background: linear-gradient(145deg, #1e1b4b, #1a1a2e);
        border: 1px solid rgba(255,255,255,.06);
        border-radius: 14px; padding: 1.2rem 1.5rem;
        margin-bottom: .6rem;
        transition: all .2s;
    }
    .result-card:hover {
        border-color: rgba(99,102,241,.35);
        transform: translateX(4px);
    }

    /* ── Field row ────────────────────────────────── */
    .field-row {
        display: flex; align-items: center;
        justify-content: space-between;
        padding: .6rem 0;
        border-bottom: 1px solid rgba(255,255,255,.04);
    }
    .field-row:last-child { border-bottom: none; }
    .field-name  { color: #94a3b8; font-size: .88rem; font-weight: 500; }
    .field-value { color: #e2e8f0; font-weight: 600; font-size: .95rem; }

    /* ── Confidence pill ──────────────────────────── */
    .conf-pill {
        font-size: .75rem; font-weight: 700;
        padding: 3px 10px; border-radius: 20px;
        display: inline-block;
    }
    .conf-high   { background: rgba(34,197,94,.12); color: #22c55e; }
    .conf-medium { background: rgba(234,179,8,.12); color: #eab308; }
    .conf-low    { background: rgba(239,68,68,.12); color: #ef4444; }

    /* ── Progress file item ───────────────────────── */
    .file-item {
        background: rgba(99,102,241,.06);
        border: 1px solid rgba(99,102,241,.15);
        border-radius: 10px; padding: .8rem 1.2rem;
        margin-bottom: .5rem;
        display: flex; align-items: center; gap: .8rem;
    }
    .file-item .file-icon { font-size: 1.3rem; }
    .file-item .file-name { color: #e2e8f0; font-weight: 500; flex: 1; }
    .file-item .file-status { font-size: .8rem; color: #22c55e; font-weight: 600; }

    /* ── Sidebar ──────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
    }

    /* ── Footer ──────────────────────────────────── */
    .app-footer {
        text-align: center; color: #475569; font-size: .78rem;
        padding: 2rem 0 1rem; margin-top: 3rem;
        border-top: 1px solid rgba(255,255,255,.04);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
#  PLOTLY THEME CONSTANTS
# ═══════════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Inter"),
    title_font=dict(size=16, color="#e2e8f0"),
    title_x=0.5,
    margin=dict(l=30, r=30, t=50, b=30),
    xaxis=dict(gridcolor="rgba(255,255,255,.04)"),
    yaxis=dict(gridcolor="rgba(255,255,255,.04)"),
)

COLORS = {
    "primary": "#a78bfa",
    "secondary": "#60a5fa",
    "success": "#22c55e",
    "warning": "#eab308",
    "danger": "#ef4444",
    "accent": "#34d399",
    "bar_gradient": ["#ef4444", "#eab308", "#22c55e"],
}


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════
def confidence_class(conf: float) -> str:
    if conf >= 0.8:
        return "conf-high"
    if conf >= 0.5:
        return "conf-medium"
    return "conf-low"


def confidence_label(conf: float) -> str:
    return f"{conf:.0%}"


# ═══════════════════════════════════════════════════════════════════
#  MAIN APP CLASS
# ═══════════════════════════════════════════════════════════════════
class OCREnhancedApp:
    """Enhanced Document OCR application with modern UI."""

    # ── Init ────────────────────────────────────────────────────
    def __init__(self):
        self._setup_page_config()
        inject_css()
        self._init_components()
        self._init_session_state()

    def _setup_page_config(self):
        st.set_page_config(
            page_title="Enhanced Document OCR",
            page_icon="📄",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def _init_components(self):
        """Load models and processors once."""
        try:
            self.doc_processor = DocumentProcessor()
            self.model = load_model()
            self.field_extractor = FieldExtractor(FIELD_CATEGORIES)
        except Exception as e:
            st.error(f"⚠️ Component initialization failed: {e}")
            if st.checkbox("Show traceback", key="init_tb"):
                st.code(traceback.format_exc())
            st.stop()

    def _init_session_state(self):
        defaults = {
            "processed_files": [],
            "batch_results": {},
            "processing_complete": False,
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    # ── Sidebar ─────────────────────────────────────────────────
    def _create_sidebar(self) -> Dict:
        with st.sidebar:
            st.markdown("## 📄 Document OCR")
            st.caption("Advanced extraction & analysis")
            st.markdown("---")

            # Document type
            st.markdown("### 📂 Document Type")
            doc_type = st.selectbox(
                "Type",
                ["Auto Detect", "Investment Document", "Bank Statement", "ID Card", "Custom"],
                label_visibility="collapsed",
            )

            st.markdown("---")

            # Settings
            st.markdown("### ⚙️ Settings")
            with st.expander("🛠️ Processing Options", expanded=False):
                confidence_threshold = st.slider(
                    "Confidence Threshold", 0.0, 1.0, 0.6, 0.05,
                    help="Minimum confidence score to accept a field",
                )
                enable_preprocessing = st.toggle(
                    "Enhanced Preprocessing", value=True,
                    help="Apply image enhancement before OCR",
                )
                batch_size = st.number_input(
                    "Batch Size", 1, 10, 4,
                    help="Number of pages to process in parallel",
                )

            st.markdown("---")

            # Field selection
            st.markdown("### 🏷️ Fields to Extract")
            selected_fields: Dict[str, bool] = {}

            for category, fields in FIELD_CATEGORIES.items():
                with st.expander(f"📌 {category}", expanded=False):
                    # Select all / none toggle
                    all_key = f"all_{category}"
                    select_all = st.checkbox(
                        "Select all", value=True, key=all_key,
                    )
                    for field_name, field_info in fields.items():
                        desc = field_info.get("description", "")
                        example = field_info.get("example", "")
                        selected_fields[field_name] = st.checkbox(
                            field_name,
                            value=select_all,
                            help=f"{desc}\nExample: {example}" if desc else None,
                            key=f"field_{category}_{field_name}",
                        )

            st.markdown("---")

            # Stats
            processed_count = len(st.session_state.processed_files)
            if processed_count:
                st.markdown(f"**✅ {processed_count}** files processed")
            else:
                st.caption("No files processed yet")

            st.markdown("---")
            st.caption(f"v2.0 • {datetime.now().strftime('%d %b %Y')}")

        return {
            "doc_type": doc_type,
            "confidence_threshold": confidence_threshold,
            "enable_preprocessing": enable_preprocessing,
            "batch_size": batch_size,
            "selected_fields": selected_fields,
        }

    # ── Hero ────────────────────────────────────────────────────
    @staticmethod
    def _render_hero():
        st.markdown(
            """
        <div class="hero">
            <h1>📄 Advanced Document OCR</h1>
            <div class="subtitle">
                Upload documents • Extract fields • Analyze with confidence
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ── Upload section ──────────────────────────────────────────
    def _render_upload_section(self) -> list:
        st.markdown(
            '<div class="section-title">📤 Upload Documents '
            '<span class="line"></span></div>',
            unsafe_allow_html=True,
        )

        col_upload, col_actions = st.columns([3, 1])

        with col_upload:
            uploaded_files = st.file_uploader(
                "Drop files here or click to upload",
                type=["pdf", "png", "jpg", "jpeg"],
                accept_multiple_files=True,
                help="Supports PDF and images (PNG, JPG, JPEG)",
                label_visibility="collapsed",
            )

        with col_actions:
            st.markdown("")  # spacer
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🗑️ Clear All", use_container_width=True, key="clear_btn"):
                    st.session_state.processed_files = []
                    st.session_state.batch_results = {}
                    st.session_state.processing_complete = False
                    st.rerun()
            with c2:
                if st.button("📊 Report", use_container_width=True, key="report_btn"):
                    self._generate_report()

        # Show file preview chips
        if uploaded_files:
            # ⚠️ AUTO-CLEAR on new upload (detects if files changed)
            if "previous_file_count" not in st.session_state:
                st.session_state.previous_file_count = 0
            
            current_file_count = len(uploaded_files)
            if (current_file_count != st.session_state.previous_file_count and 
                st.session_state.previous_file_count > 0):
                # User uploaded new files - auto-clear old results
                st.session_state.processed_files = []
                st.session_state.batch_results = {}
                st.session_state.processing_complete = False
                st.warning("📢 Previous results cleared. Processing new files...")
            
            st.session_state.previous_file_count = current_file_count
            
            chips_html = ""
            for f in uploaded_files:
                ext = f.name.split(".")[-1].upper()
                icon = "📄" if ext == "PDF" else "🖼️"
                size_kb = f.size / 1024
                chips_html += (
                    f'<div class="file-item">'
                    f'<span class="file-icon">{icon}</span>'
                    f'<span class="file-name">{f.name}</span>'
                    f'<span class="file-status">{size_kb:.0f} KB • {ext}</span>'
                    f"</div>"
                )
            st.markdown(chips_html, unsafe_allow_html=True)

        return uploaded_files or []

    # ── Processing ──────────────────────────────────────────────
    def _process_files(self, files: list, settings: Dict) -> list:
        if not files:
            return []

        progress_bar = st.progress(0, text="Starting…")
        status_container = st.empty()
        processed: list = []

        total = len(files)
        for idx, file in enumerate(files):
            try:
                progress_bar.progress(
                    (idx) / total,
                    text=f"Processing {file.name} ({idx + 1}/{total})…",
                )

                # Convert to image(s)
                if file.name.lower().endswith(".pdf"):
                    images = self.doc_processor.convert_pdf_to_images(file)
                else:
                    image = Image.open(file)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    images = [image]

                for page_idx, image in enumerate(images):
                    try:
                        if settings["enable_preprocessing"]:
                            image = self.doc_processor.enhance_image(image)

                        result = self._process_single_image(
                            image,
                            settings["selected_fields"],
                            settings["confidence_threshold"],
                        )

                        filename = (
                            f"{file.name} (p.{page_idx + 1})"
                            if len(images) > 1
                            else file.name
                        )
                        # processed.append({"filename": filename, "result": result})
                        processed.append({
    "filename": filename,
    "result": result,
    "image": image,  # ← store the PIL image
})

                    except Exception as e:
                        st.error(
                            f"❌ Error on page {page_idx + 1} of {file.name}: {e}"
                        )

            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {e}")

        progress_bar.progress(1.0, text="✅ Processing complete!")
        return processed

    def _process_single_image(
        self,
        image: Image.Image,
        selected_fields: Dict,
        confidence_threshold: float,
    ) -> Dict:
        """Process one image → return structured result dict."""
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            img_np = np.array(image)
            doc = self.model([img_np])

            result = {
                "extracted_fields": {},
                "words": [],
                "full_text": "",
                "image_size": {
                    "width": img_np.shape[1],
                    "height": img_np.shape[0],
                },
            }

            if doc and doc.pages:
                page = doc.pages[0]
                full_text_lines: list = []

                for block in page.blocks:
                    for line in block.lines:
                        line_words: list = []
                        for word in line.words:
                            word_dict = {
                                "text": word.value,
                                "confidence": float(word.confidence),
                                "bbox": [
                                    float(coord)
                                    for point in word.geometry
                                    for coord in point
                                ],
                            }
                            line_words.append(word_dict["text"])
                            result["words"].append(word_dict)
                        full_text_lines.append(" ".join(line_words))

                result["full_text"] = "\n".join(full_text_lines)

                # Field extraction
                for field_name, is_selected in selected_fields.items():
                    if not is_selected:
                        continue
                    value, confidence = self.field_extractor.extract_with_context(
                        result["full_text"], field_name
                    )
                    if value and confidence >= confidence_threshold:
                        result["extracted_fields"][field_name] = {
                            "value": str(value),
                            "confidence": float(confidence),
                        }

            return result

        except Exception as e:
            st.error(f"Image processing error: {e}")
            return {
                "extracted_fields": {},
                "words": [],
                "full_text": "",
                "image_size": {"width": 0, "height": 0},
            }

    # ── Display: Summary metrics ────────────────────────────────
    def _render_summary(self, processed_files: list):
        st.markdown(
            '<div class="section-title">📊 Processing Summary '
            '<span class="line"></span></div>',
            unsafe_allow_html=True,
        )

        total = len(processed_files)
        avg_conf = self._calc_avg_confidence(processed_files)
        success = self._calc_success_rate(processed_files)
        total_fields = sum(
            len(f["result"].get("extracted_fields", {})) for f in processed_files
        )

        st.markdown(
            f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-icon">📁</div>
                <div class="stat-value">{total}</div>
                <div class="stat-label">Files Processed</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🏷️</div>
                <div class="stat-value">{total_fields}</div>
                <div class="stat-label">Fields Extracted</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🎯</div>
                <div class="stat-value">{avg_conf:.1f}%</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">✅</div>
                <div class="stat-value">{success:.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # # ── Display: Detailed results ───────────────────────────────
    # def _render_detailed(self, processed_files: list):
    #     st.markdown(
    #         '<div class="section-title">📄 Detailed Results '
    #         '<span class="line"></span></div>',
    #         unsafe_allow_html=True,
    #     )

    #     for idx, file_data in enumerate(processed_files):
    #         result = file_data.get("result", {})
    #         fields = result.get("extracted_fields", {})
    #         field_count = len(fields)

    #         with st.expander(
    #             f"{'📄' if file_data['filename'].endswith('.pdf') else '🖼️'} "
    #             f"{file_data['filename']}  —  {field_count} fields extracted",
    #             expanded=(idx == 0),
    #         ):
    #             col_left, col_right = st.columns([3, 2])

    #             with col_left:
    #                 # Extracted fields as styled cards
    #                 if fields:
    #                     st.markdown("**🏷️ Extracted Fields**")
    #                     fields_html = ""
    #                     for fname, finfo in fields.items():
    #                         conf = finfo["confidence"]
    #                         cls = confidence_class(conf)
    #                         fields_html += (
    #                             f'<div class="field-row">'
    #                             f'<span class="field-name">{fname}</span>'
    #                             f'<span>'
    #                             f'<span class="field-value">{finfo["value"]}</span> '
    #                             f'<span class="conf-pill {cls}">'
    #                             f"{confidence_label(conf)}</span>"
    #                             f"</span></div>"
    #                         )
    #                     st.markdown(
    #                         f'<div class="result-card">{fields_html}</div>',
    #                         unsafe_allow_html=True,
    #                     )

    #                     # Also show as a table
    #                     with st.popover("📋 View as table"):
    #                         df = pd.DataFrame(
    #                             [
    #                                 {
    #                                     "Field": f,
    #                                     "Value": i["value"],
    #                                     "Confidence": f"{i['confidence']:.0%}",
    #                                 }
    #                                 for f, i in fields.items()
    #                             ]
    #                         )
    #                         st.dataframe(df, use_container_width=True, hide_index=True)
    #                 else:
    #                     st.warning("No fields extracted from this document.")

    #                 # Full text
    #                 st.markdown("**📝 Full Text**")
    #                 st.text_area(
    #                     "Extracted text",
    #                     value=result.get("full_text", ""),
    #                     height=180,
    #                     key=f"text_{idx}",
    #                     label_visibility="collapsed",
    #                 )

    #             with col_right:
    #                 # Confidence chart
    #                 if fields:
    #                     st.markdown("**📊 Confidence Scores**")
    #                     names = list(fields.keys())
    #                     confs = [fields[n]["confidence"] for n in names]

    #                     colors = [
    #                         COLORS["success"] if c >= 0.8
    #                         else COLORS["warning"] if c >= 0.5
    #                         else COLORS["danger"]
    #                         for c in confs
    #                     ]

    #                     fig = go.Figure(
    #                         go.Bar(
    #                             x=confs,
    #                             y=names,
    #                             orientation="h",
    #                             marker=dict(color=colors, cornerradius=6),
    #                             text=[f"{c:.0%}" for c in confs],
    #                             textposition="outside",
    #                             textfont=dict(color="#94a3b8", size=11),
    #                         )
    #                     )
    #                     fig.update_layout(
    #                         **PLOTLY_LAYOUT,
    #                         height=max(200, len(names) * 40 + 80),
    #                         xaxis=dict(
    #                             range=[0, 1.15],
    #                             gridcolor="rgba(255,255,255,.04)",
    #                         ),
    #                         yaxis=dict(autorange="reversed"),
    #                     )
    #                     st.plotly_chart(fig, use_container_width=True)

    #                 # Download button
    #                 try:
    #                     json_str = json.dumps(result, indent=2)
    #                     st.download_button(
    #                         "⬇️ Download JSON",
    #                         data=json_str,
    #                         file_name=f"{file_data['filename']}_results.json",
    #                         mime="application/json",
    #                         key=f"dl_{idx}",
    #                         use_container_width=True,
    #                     )
    #                 except Exception as e:
    #                     st.error(f"Download error: {e}")
    def _render_detailed(self, processed_files: list):
        st.markdown(
            '<div class="section-title">📄 Detailed Results '
            '<span class="line"></span></div>',
            unsafe_allow_html=True,
        )

        for idx, file_data in enumerate(processed_files):
            result = file_data.get("result", {})
            fields = result.get("extracted_fields", {})
            field_count = len(fields)
            has_image = "image" in file_data and file_data["image"] is not None

            with st.expander(
                f"{'📄' if file_data['filename'].endswith('.pdf') else '🖼️'} "
                f"{file_data['filename']}  —  {field_count} fields extracted",
                expanded=(idx == 0),
            ):
                # ── TOP: Original image + Annotated ─────────────────────────
                if has_image:
                    st.markdown("### 🖼️ Document Visualization")
                    
                    # InfoBox explaining the visualization
                    st.info(
                        "**📍 Exact Field Detection:** Each colored box shows the precise location where a field was detected. "
                        "Box colors indicate confidence level (🟢Green≥90% | 🟡Yellow 70-90% | 🟠Orange 50-70% | 🔴Red<50%)"
                    )
                    
                    # Create two columns for side-by-side view
                    orig_col, anno_col = st.columns(2)
                    
                    with orig_col:
                        st.markdown("**📄 Original Document**")
                        st.image(
                            file_data["image"],
                            use_container_width=True,
                            caption="Original",
                        )
                    
                    with anno_col:
                        st.markdown("**🏷️ Extracted Fields (Annotated with Exact Boxes)**")
                        
                        # Generate annotated image with exact bounding boxes
                        from utils.annotation_helper import create_annotated_image_with_exact_boxes
                        
                        img_array = np.array(file_data["image"])
                        words_data = result.get("words", [])
                        
                        # Use exact boxes if word data available, fallback to text labels
                        if words_data:
                            annotated_img = create_annotated_image_with_exact_boxes(
                                img_array, fields, words_data
                            )
                        else:
                            from utils.annotation_helper import create_annotated_image_simple
                            annotated_img = create_annotated_image_simple(img_array, fields)
                        
                        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
                        
                        st.image(
                            annotated_pil,
                            use_container_width=True,
                            caption="Exact field detection locations with colored boxes",
                        )
                    
                    # Legend for confidence colors
                    st.markdown("---")
                    st.markdown("**Color Legend:**")
                    legend_cols = st.columns(4)
                    legend_items = [
                        ("🟢", "High", ">= 90%"),
                        ("🟡", "Medium", "70-90%"),
                        ("🟠", "Low", "50-70%"),
                        ("🔴", "Very Low", "< 50%"),
                    ]
                    for col, (emoji, label, range_str) in zip(legend_cols, legend_items):
                        with col:
                            st.caption(f"{emoji} {label} ({range_str})")
                    
                    st.markdown("---")

                # ── BOTTOM: Fields + Confidence (same as before) ─
                col_left, col_right = st.columns([3, 2])

                with col_left:
                    if fields:
                        st.markdown("**🏷️ Extracted Fields**")
                        fields_html = ""
                        for fname, finfo in fields.items():
                            conf = finfo["confidence"]
                            cls = confidence_class(conf)
                            fields_html += (
                                f'<div class="field-row">'
                                f'<span class="field-name">{fname}</span>'
                                f'<span>'
                                f'<span class="field-value">{finfo["value"]}</span> '
                                f'<span class="conf-pill {cls}">'
                                f"{confidence_label(conf)}</span>"
                                f"</span></div>"
                            )
                        st.markdown(
                            f'<div class="result-card">{fields_html}</div>',
                            unsafe_allow_html=True,
                        )

                        with st.popover("📋 View as table"):
                            df = pd.DataFrame(
                                [
                                    {
                                        "Field": f,
                                        "Value": i["value"],
                                        "Confidence": f"{i['confidence']:.0%}",
                                    }
                                    for f, i in fields.items()
                                ]
                            )
                            st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No fields extracted from this document.")

                    st.markdown("**📝 Full Text**")
                    st.text_area(
                        "Extracted text",
                        value=result.get("full_text", ""),
                        height=180,
                        key=f"text_{idx}",
                        label_visibility="collapsed",
                    )

                with col_right:
                    if fields:
                        st.markdown("**📊 Confidence Scores**")
                        names = list(fields.keys())
                        confs = [fields[n]["confidence"] for n in names]
                        colors = [
                            COLORS["success"] if c >= 0.8
                            else COLORS["warning"] if c >= 0.5
                            else COLORS["danger"]
                            for c in confs
                        ]
                        fig = go.Figure(
                            go.Bar(
                                x=confs, y=names, orientation="h",
                                marker=dict(color=colors, cornerradius=6),
                                text=[f"{c:.0%}" for c in confs],
                                textposition="outside",
                                textfont=dict(color="#94a3b8", size=11),
                            )
                        )
                        layout_config = dict(PLOTLY_LAYOUT)
                        layout_config.update({
                            "height": max(200, len(names) * 40 + 80),
                            "xaxis": dict(range=[0, 1.15], gridcolor="rgba(255,255,255,.04)"),
                            "yaxis": dict(autorange="reversed"),
                        })
                        fig.update_layout(**layout_config)
                        st.plotly_chart(fig, use_container_width=True)

                    try:
                        json_str = json.dumps(result, indent=2)
                        st.download_button(
                            "⬇️ Download JSON",
                            data=json_str,
                            file_name=f"{file_data['filename']}_results.json",
                            mime="application/json",
                            key=f"dl_{idx}",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"Download error: {e}")
    # ── Display: JSON view ──────────────────────────────────────
    @staticmethod
    def _render_json(processed_files: list):
        st.markdown(
            '<div class="section-title">🔍 Raw JSON Data '
            '<span class="line"></span></div>',
            unsafe_allow_html=True,
        )

        try:
            clean_data = [
                {
                    "filename": fd["filename"],
                    "result": {
                        "extracted_fields": fd["result"].get("extracted_fields", {}),
                        "full_text": fd["result"].get("full_text", ""),
                        "words": fd["result"].get("words", []),
                    },
                }
                for fd in processed_files
            ]

            st.json(clean_data, expanded=False)

            json_str = json.dumps(clean_data, indent=2)
            st.download_button(
                "⬇️ Download All Results",
                data=json_str,
                file_name=f"ocr_results_{datetime.now():%Y%m%d_%H%M%S}.json",
                mime="application/json",
                key="dl_all",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"JSON display error: {e}")

    # ── Display: Tabbed results ─────────────────────────────────
    def _render_results(self, processed_files: list):
        if not processed_files:
            st.info("Upload documents above to see results here.")
            return

        tab1, tab2, tab3 = st.tabs(
            ["📊 Summary", "📄 Detailed View", "🔍 JSON"]
        )
        with tab1:
            self._render_summary(processed_files)
            self._render_confidence_overview(processed_files)
        with tab2:
            self._render_detailed(processed_files)
        with tab3:
            self._render_json(processed_files)

    # ── Confidence overview chart ───────────────────────────────
    @staticmethod
    def _render_confidence_overview(processed_files: list):
        """Aggregated confidence chart across all files."""
        all_fields: Dict[str, list] = {}
        for fd in processed_files:
            for fname, finfo in fd["result"].get("extracted_fields", {}).items():
                all_fields.setdefault(fname, []).append(finfo["confidence"])

        if not all_fields:
            return

        st.markdown(
            '<div class="section-title">🎯 Confidence Overview '
            '<span class="line"></span></div>',
            unsafe_allow_html=True,
        )

        names = list(all_fields.keys())
        avgs = [sum(v) / len(v) for v in all_fields.values()]
        counts = [len(v) for v in all_fields.values()]

        col1, col2 = st.columns(2)

        with col1:
            colors = [
                COLORS["success"] if a >= 0.8
                else COLORS["warning"] if a >= 0.5
                else COLORS["danger"]
                for a in avgs
            ]
            fig = go.Figure(
                go.Bar(
                    x=avgs,
                    y=names,
                    orientation="h",
                    marker=dict(color=colors, cornerradius=6),
                    text=[f"{a:.0%}" for a in avgs],
                    textposition="outside",
                    textfont=dict(color="#94a3b8", size=11),
                )
            )
            layout_config = dict(PLOTLY_LAYOUT)
            layout_config.update({
                "title": "Average Confidence per Field",
                "height": max(280, len(names) * 35 + 80),
                "xaxis": dict(range=[0, 1.15], gridcolor="rgba(255,255,255,.04)"),
                "yaxis": dict(autorange="reversed"),
            })
            fig.update_layout(**layout_config)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure(
                go.Bar(
                    x=counts,
                    y=names,
                    orientation="h",
                    marker=dict(color=COLORS["secondary"], cornerradius=6),
                    text=counts,
                    textposition="outside",
                    textfont=dict(color="#94a3b8", size=11),
                )
            )
            layout_config = dict(PLOTLY_LAYOUT)
            layout_config.update({
                "title": "Extraction Count per Field",
                "height": max(280, len(names) * 35 + 80),
                "yaxis": dict(autorange="reversed"),
            })
            fig2.update_layout(**layout_config)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Report generator ────────────────────────────────────────
    def _generate_report(self):
        pf = st.session_state.processed_files
        if not pf:
            st.warning("No processed files for report generation.")
            return

        try:
            report = {
                "generated_at": datetime.now().isoformat(),
                "summary": {
                    "total_files": len(pf),
                    "average_confidence": round(self._calc_avg_confidence(pf), 2),
                    "success_rate": round(self._calc_success_rate(pf), 2),
                    "total_fields_extracted": sum(
                        len(f["result"].get("extracted_fields", {})) for f in pf
                    ),
                },
                "files": [
                    {
                        "filename": fd["filename"],
                        "fields": fd["result"].get("extracted_fields", {}),
                        "field_count": len(fd["result"].get("extracted_fields", {})),
                        "avg_confidence": round(
                            self._calc_avg_confidence([fd]), 2
                        ),
                    }
                    for fd in pf
                ],
            }

            # Display
            st.markdown(
                '<div class="section-title">📊 Generated Report '
                '<span class="line"></span></div>',
                unsafe_allow_html=True,
            )

            s = report["summary"]
            st.markdown(
                f"""
            <div class="stat-grid">
                <div class="stat-card">
                    <div class="stat-icon">📁</div>
                    <div class="stat-value">{s['total_files']}</div>
                    <div class="stat-label">Total Files</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">🎯</div>
                    <div class="stat-value">{s['average_confidence']:.1f}%</div>
                    <div class="stat-label">Avg Confidence</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">✅</div>
                    <div class="stat-value">{s['success_rate']:.1f}%</div>
                    <div class="stat-label">Success Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon">🏷️</div>
                    <div class="stat-value">{s['total_fields_extracted']}</div>
                    <div class="stat-label">Fields Extracted</div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            json_str = json.dumps(report, indent=2)
            st.download_button(
                "⬇️ Download Report",
                data=json_str,
                file_name=f"ocr_report_{datetime.now():%Y%m%d_%H%M%S}.json",
                mime="application/json",
                key="dl_report",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"Report generation error: {e}")
            if st.checkbox("Show traceback", key="report_tb"):
                st.code(traceback.format_exc())

    # ── Calculation helpers ─────────────────────────────────────
    @staticmethod
    def _calc_avg_confidence(processed_files: list) -> float:
        confs = []
        for fd in processed_files:
            for info in fd["result"].get("extracted_fields", {}).values():
                confs.append(info["confidence"])
        return (sum(confs) / len(confs) * 100) if confs else 0.0

    @staticmethod
    def _calc_success_rate(processed_files: list) -> float:
        total = 0
        success = 0
        for fd in processed_files:
            for info in fd["result"].get("extracted_fields", {}).values():
                total += 1
                if info["confidence"] >= 0.6:
                    success += 1
        return (success / total * 100) if total > 0 else 0.0

    # ── Main run ────────────────────────────────────────────────
    def run(self):
        """Entry point — orchestrate the full app."""
        settings = self._create_sidebar()
        self._render_hero()

        # Upload
        uploaded_files = self._render_upload_section()

        # Process
        if uploaded_files:
            with st.spinner(""):
                processed = self._process_files(uploaded_files, settings)
                if processed:
                    st.session_state.processed_files = processed
                    st.session_state.processing_complete = True
                    st.toast(
                        f"✅ {len(processed)} document(s) processed!",
                        icon="📄",
                    )

        # Results
        self._render_results(st.session_state.processed_files)

        # Footer
        st.markdown(
            """
        <div class="app-footer">
            Advanced Document OCR System v2.0 &nbsp;•&nbsp;
            Built with ❤️ using Streamlit &amp; docTR
        </div>
        """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = OCREnhancedApp()
    app.run()