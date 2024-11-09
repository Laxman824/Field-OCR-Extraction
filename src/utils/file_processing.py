# utils/file_processing.py

import pdf2image
from PIL import Image
import numpy as np
import cv2
from typing import List, Union
import io
import fitz  # PyMuPDF
import tempfile
import os
import streamlit as st

class DocumentProcessor:
    """Document processing with support for multiple file types"""
    
    SUPPORTED_FORMATS = {
        'image': ['.jpg', '.jpeg', '.png', '.tiff', '.bmp'],
        'pdf': ['.pdf']
    }

    @staticmethod
    def is_supported_format(file_extension: str) -> bool:
        """Check if file format is supported"""
        all_formats = []
        for formats in DocumentProcessor.SUPPORTED_FORMATS.values():
            all_formats.extend(formats)
        return file_extension.lower() in all_formats

    @staticmethod
    @st.cache_data  # Cache the conversion results
    def convert_pdf_to_images(_pdf_file: Union[str, io.BytesIO]) -> List[Image.Image]:
        """Convert PDF file to list of PIL Images"""
        try:
            if isinstance(_pdf_file, str):
                pdf_document = fitz.open(_pdf_file)
            else:
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                    tmp.write(_pdf_file.getvalue())
                    tmp_path = tmp.name
                pdf_document = fitz.open(tmp_path)
                os.unlink(tmp_path)

            images = []
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            
            return images
            
        except Exception as e:
            st.error(f"Error converting PDF: {str(e)}")
            return []

    @staticmethod
    @st.cache_data  # Cache preprocessing results
    def enhance_image(_image: Image.Image) -> Image.Image:
        """Enhanced image preprocessing"""
        try:
            # Convert to numpy array
            img_array = np.array(_image)
            
            # Convert to grayscale if RGB
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )

            # Denoise
            denoised = cv2.fastNlMeansDenoising(binary)

            # Deskew if needed
            angle = DocumentProcessor.determine_skew(denoised)
            if abs(angle) > 0.5:
                rotated = DocumentProcessor.rotate_image(denoised, angle)
                return Image.fromarray(rotated)

            return Image.fromarray(denoised)
            
        except Exception as e:
            st.error(f"Error enhancing image: {str(e)}")
            return _image

    @staticmethod
    def determine_skew(image: np.ndarray) -> float:
        """Determine skew angle of the image"""
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        return -angle

    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle"""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    @staticmethod
    def batch_process_images(images: List[Image.Image], process_func, batch_size: int = 4) -> List:
        """Process multiple images in batches"""
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = process_func(batch)
            results.extend(batch_results)
        return results