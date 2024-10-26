import streamlit as st
import numpy as np
from PIL import Image
import cv2
import json
import re
from doctr.models import ocr_predictor
import io
import traceback
import pandas as pd
from PIL import ImageEnhance

# Custom error handling for OpenCV import
try:
    import cv2
except ImportError as e:
    st.error("""
    Error importing OpenCV. Please run the following commands:
    ```bash
    apt-get update
    apt-get install -y libgl1-mesa-glx libglib2.0-0
    pip uninstall -y opencv-python cv2 opencv-python-headless
    pip install opencv-python-headless==4.7.0.72
    ```
    """)
    st.stop()

# Try importing doctr
try:
    from doctr.models import ocr_predictor
except ImportError as e:
    st.error("""
    Error importing doctr. Please install it using:
    ```bash
    pip install python-doctr[torch]
    ```
    """)
    st.stop()

# Configuration and setup


# Set page config
st.set_page_config(page_title="Document OCR Processing", layout="wide")

# Define fields with regular expressions for flexible matching
FIELDS = {
    "Scheme Name": r"\b(Scheme|Plan)\s+Name\b",
    "Folio Number": r"\b(Folio|Account)\s+(Number|No\.?)\b",
    "Number of Units": r"\b(Number\s+of\s+Units|Units|Quantity)\b",
    "PAN": r"\b(PAN|Permanent\s+Account\s+Number)\b",
    "Signature": r"\bSignature\b",
    "Tax Status": r"\bTax\s+Status\b",
    "Mobile Number": r"\b(Mobile|Phone|Cell)\s+(Number|No\.?)\b",
    "Email": r"\b(Email|E-mail)\b",
    "Address": r"\bAddress\b",
    "Bank Account Details": r"\b(Bank\s+Account|Account)\s+(Details|Information)\b",
    "Date of Journey": r"\bDate\s+of\s+Journey\b",
    "Date": r"\b[Dd]ate\b"
}

# Add sidebar with preprocessing options
# Add this at the sidebar section of your code, replace the existing sidebar code

with st.sidebar:
    st.header("Preprocessing Options")
    resize_factor = st.slider("Resize Factor", 0.1, 2.0, 1.0)
    enhance_contrast = st.checkbox("Enhance Contrast", False)
    denoise = st.checkbox("Remove Noise", False)

    
@st.cache_resource(ttl=3600)
def load_model():
    try:
        return ocr_predictor(pretrained=True)
    except Exception as e:
        st.error(f"Error loading OCR model: {str(e)}")
        st.stop()

def preprocess_image(image, resize_factor=1.0, enhance_contrast=1.0, denoise=False):
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        if resize_factor != 1.0:
            new_size = tuple(int(dim * resize_factor) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Enhance contrast
        if enhance_contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(enhance_contrast)
        
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        
        # Denoise
        if denoise:
            img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        
        return Image.fromarray(img_array)
    except Exception as e:
        st.error(f"Error during image preprocessing: {str(e)}")
        return image

def process_image(image, model, resize_factor=1.0, enhance_contrast=1.0, denoise=False):
    try:
        with st.spinner('Processing document...'):
            progress_bar = st.progress(0)
            
            # Preprocess image
            image = preprocess_image(image, resize_factor, enhance_contrast, denoise)
            progress_bar.progress(20)
            
            # Convert to numpy array
            img_np = np.array(image)
            result = model([img_np])
            progress_bar.progress(40)

            extracted_fields = {}
            bounding_boxes = []
            height, width = img_np.shape[:2]

            img_all_text = img_np.copy()
            img_fields = img_np.copy()

            # Process detected text
            progress_bar.progress(60)
            word_id = 0
            for block in result.pages[0].blocks:
                for line in block.lines:
                    for word in line.words:
                        # Process each word
                        process_word(word, width, height, bounding_boxes, word_id)
                        word_id += 1

            # Extract fields
            progress_bar.progress(80)
            extracted_fields = extract_fields(bounding_boxes)
            
            progress_bar.progress(100)
            st.success('Processing complete!')
            
            return img_all_text, img_fields, extracted_fields, bounding_boxes
            
    except Exception as e:
        st.error(f"Error during image processing: {str(e)}")
        st.code(traceback.format_exc())
        return None, None, None, None

def process_word(word, width, height, bounding_boxes, word_id):
    x, y, w, h = (word.geometry[0][0], word.geometry[0][1], 
                  word.geometry[1][0] - word.geometry[0][0],
                  word.geometry[1][1] - word.geometry[0][1])
    
    box = {
        "text": word.value,
        "confidence": float(word.confidence),
        "box": [
            int(x * width),
            int(y * height),
            int((x + w) * width),
            int((y + h) * height)
        ],
        "label": "other",
        "id": word_id
    }
    bounding_boxes.append(box)

def extract_fields(bounding_boxes):
    extracted_fields = {}
    # Your existing field extraction logic here
    return extracted_fields

def convert_to_csv(field_values):
    df = pd.DataFrame(list(field_values.items()), columns=['Field', 'Value'])
    return df.to_csv(index=False)

def main():
    st.title("Document OCR Processing")
    st.write("Upload an image to extract text and identify fields")

    # Load model
    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        try:
            # Read image
            image = Image.open(uploaded_file)
            
            # Process image with selected options
            img_all_text, img_fields, field_values, bounding_boxes = process_image(
                image, 
                model,
                resize_factor,
                enhance_contrast,
                denoise
            )

            if img_all_text is not None and img_fields is not None:
                # Display results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("All Detected Text")
                    st.image(img_all_text)

                with col2:
                    st.subheader("Extracted Fields")
                    st.image(img_fields)

                # Display extracted fields with confidence scores
                st.subheader("Extracted Field-Value Pairs")
                if show_confidence:
                    for field, (value, confidence) in field_values.items():
                        st.write(f"{field}: {value} (Confidence: {confidence:.2f})")
                else:
                    st.json({k: v[0] for k, v in field_values.items()})

                # Download buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Download JSON
                    json_data = json.dumps(field_values, indent=4)
                    st.download_button(
                        "Download JSON",
                        json_data,
                        "extracted_fields.json",
                        "application/json"
                    )
                
                with col2:
                    # Download CSV
                    csv_data = convert_to_csv(field_values)
                    st.download_button(
                        "Download CSV",
                        csv_data,
                        "extracted_fields.csv",
                        "text/csv"
                    )
                
                with col3:
                    # Download bounding boxes
                    boxes_json = json.dumps({"form": bounding_boxes}, indent=4)
                    st.download_button(
                        "Download Bounding Boxes",
                        boxes_json,
                        "bounding_boxes.json",
                        "application/json"
                    )

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
