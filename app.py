import streamlit as st
import numpy as np
from PIL import Image
import cv2
import json
import re
from doctr.models import ocr_predictor
import io
import traceback

# Set page config
st.set_page_config(page_title="Document OCR App", layout="wide")

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

@st.cache_resource
def load_model():
    return ocr_predictor(pretrained=True)

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    max_size = 1000
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))
    return image

def find_field(text, fields):
    matches = []
    for field, pattern in fields.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            matches.append((field, match.start(), match.end()))
    return sorted(matches, key=lambda x: x[1])

def extract_value(text, field):
    if field == "Email":
        match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        return (match.group(0), 1.0) if match else (None, 0.0)

    elif field == "Date of Journey" or field == "Date":
        patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b\d{2}/\d{2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{2}-\d{2}-\d{4}\b',  # DD-MM-YYYY
            r'\b\d{1,2}\s+[A-Za-z]+\s*,\s*\d{4}\b'  # DD MMMM, YYYY
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return (match.group(0), 1.0)
        return (None, 0.0)

    elif field == "Mobile Number":
        match = re.search(r'\b(\+\d{1,3}[-.\s]?)?\d{10,14}\b', text)
        return (match.group(0), 1.0) if match else (None, 0.0)

    elif field == "Folio Number":
        match = re.search(r'\b[A-Za-z0-9]+\b', text)
        return (match.group(0), 0.8) if match else (None, 0.0)

    elif field == "PAN":
        match = re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', text)
        return (match.group(0), 1.0) if match else (None, 0.0)

    elif field == "Number of Units":
        match = re.search(r'\b\d+(\.\d+)?\b', text)
        return (match.group(0), 0.9) if match else (None, 0.0)

    elif field == "Tax Status":
        statuses = ["individual", "company", "huf", "nri", "trust"]
        for status in statuses:
            if status in text.lower():
                return (status.capitalize(), 0.9)
        return (text.split()[0], 0.5) if text else (None, 0.0)

    elif field == "Address":
        lines = text.split('\n')
        address_lines = []
        for line in lines:
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in FIELDS.values()):
                break
            address_lines.append(line.strip())
        return (' '.join(address_lines), 0.7)

    elif field == "Bank Account Details":
        account_match = re.search(r'\b\d{9,18}\b', text)
        ifsc_match = re.search(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', text)
        if account_match and ifsc_match:
            return (f"A/C: {account_match.group(0)}, IFSC: {ifsc_match.group(0)}", 1.0)
        elif account_match:
            return (f"A/C: {account_match.group(0)}", 0.8)
        return (text.strip(), 0.5)

    else:
        words = text.split()
        return (' '.join(words[:5]), 0.6)

def determine_label(field):
    question_fields = ["Scheme Name", "Folio Number", "Number of Units", "PAN", "Tax Status", 
                      "Mobile Number", "Email", "Address", "Date", "Bank Account Details", 
                      "Date of Journey"]
    if field in question_fields:
        return "question"
    elif field == "Signature":
        return "other"
    else:
        return "answer"

def process_image(image, ocr_model):
    try:
        image = preprocess_image(image)
        img_np = np.array(image)
        result = ocr_model([img_np])

        extracted_fields = {}
        bounding_boxes = []
        height, width = img_np.shape[:2]

        img_all_text = img_np.copy()
        img_fields = img_np.copy()

        # Create bounding boxes for all detected words
        word_id = 0
        for block in result.pages[0].blocks:
            for line in block.lines:
                for word in line.words:
                    x, y, w, h = (word.geometry[0][0], word.geometry[0][1], 
                                word.geometry[1][0] - word.geometry[0][0],
                                word.geometry[1][1] - word.geometry[0][1])
                    box = {
                        "text": word.value,
                        "box": [
                            int(x * width),
                            int(y * height),
                            int((x + w) * width),
                            int((y + h) * height)
                        ],
                        "label": "other",
                        "linking": [],
                        "words": [{
                            "text": word.value,
                            "box": [
                                int(x * width),
                                int(y * height),
                                int((x + w) * width),
                                int((y + h) * height)
                            ]
                        }],
                        "id": word_id
                    }
                    bounding_boxes.append(box)
                    word_id += 1

        # Combine words into lines
        lines = []
        current_line = []
        for word in bounding_boxes:
            if not current_line or abs(word['box'][1] - current_line[-1]['box'][1]) < 20:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
        if current_line:
            lines.append(current_line)

        # Extract fields
        for line in lines:
            line_text = ' '.join([word['text'] for word in line])
            field_matches = find_field(line_text, FIELDS)

            for field, start, end in field_matches:
                value_text = line_text[end:].strip()
                next_field_match = find_field(value_text, FIELDS)
                if next_field_match:
                    value_text = value_text[:next_field_match[0][1]].strip()

                value, confidence = extract_value(value_text, field)

                if value and (field not in extracted_fields or confidence > extracted_fields[field][1]):
                    extracted_fields[field] = (value, confidence)

                    field_words = [word for word in line if field.lower() in word['text'].lower()]
                    if field_words:
                        field_words[0]['label'] = determine_label(field)

                    value_words = [word for word in line if word['text'] in value]
                    if value_words:
                        for word in value_words:
                            word['label'] = 'answer'

        # Draw bounding boxes
        for box in bounding_boxes:
            x1, y1, x2, y2 = box['box']
            cv2.rectangle(img_all_text, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_all_text, box['text'], (x1, y1 - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            if box['label'] in ['question', 'answer']:
                cv2.rectangle(img_fields, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_fields, f"{box['text']} ({box['label']})",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1, cv2.LINE_AA)

        return (img_all_text, img_fields, 
                {k: v[0] for k, v in extracted_fields.items()}, 
                bounding_boxes)

    except Exception as e:
        st.error(f"An error occurred during image processing: {str(e)}")
        st.code(traceback.format_exc())
        return None, None, None, None

def main():
    st.title("Document OCR Processing")
    st.write("Upload an image to extract text and identify fields")

    # Load model
    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        
        # Process image
        img_all_text, img_fields, field_values, bounding_boxes = process_image(image, model)

        if img_all_text is not None and img_fields is not None:
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("All Detected Text")
                st.image(img_all_text)

            with col2:
                st.subheader("Extracted Fields")
                st.image(img_fields)

            # Display extracted fields
            st.subheader("Extracted Field-Value Pairs")
            st.json(field_values)

            # Download buttons for JSON files
            field_json = json.dumps(field_values, indent=4)
            box_json = json.dumps({"form": bounding_boxes}, indent=4)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Field Values JSON",
                    data=field_json,
                    file_name="field_values.json",
                    mime="application/json"
                )
            with col2:
                st.download_button(
                    label="Download Bounding Boxes JSON",
                    data=box_json,
                    file_name="bounding_boxes.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
