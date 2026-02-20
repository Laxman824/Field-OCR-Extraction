"""
Advanced image preprocessing for document OCR.

Implements multi-stage enhancement pipeline:
1. Format conversion (RGB/Grayscale)
2. Noise reduction (NLM denoising)
3. Adaptive thresholding (CLAHE)
4. Deskewing
5. Contrast enhancement
"""

import cv2
import numpy as np
from PIL import Image
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Enterprise-grade image preprocessing for OCR"""

    # Configuration parameters
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_SIZE = (8, 8)
    ADAPTIVE_THRESHOLD_BLOCK_SIZE = 11
    ADAPTIVE_THRESHOLD_C = 2
    NLM_DENOISE_H = 10
    SKEW_THRESHOLD = 0.5
    MAX_IMAGE_SIZE = 2048
    MIN_IMAGE_SIZE = 100

    @staticmethod
    def preprocess_image(
        image: Image.Image,
        enhance_contrast: bool = True,
        denoise: bool = True,
        deskew: bool = True,
    ) -> Image.Image:
        """
        Complete preprocessing pipeline for OCR.
        
        Args:
            image: PIL Image object
            enhance_contrast: Apply CLAHE contrast enhancement
            denoise: Apply NLM denoising
            deskew: Auto-correct skewed images
        
        Returns:
            Enhanced PIL Image ready for OCR
        """
        try:
            # Step 1: Ensure RGB and resize if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image = ImageProcessor._resize_image(image)
            
            # Step 2: Convert to grayscale
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Step 3: Denoise (optional)
            if denoise:
                gray = ImageProcessor._denoise_image(gray)
            
            # Step 4: Contrast enhancement (optional)
            if enhance_contrast:
                gray = ImageProcessor._enhance_contrast(gray)
            
            # Step 5: Adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                ImageProcessor.ADAPTIVE_THRESHOLD_BLOCK_SIZE,
                ImageProcessor.ADAPTIVE_THRESHOLD_C
            )
            
            # Step 6: Deskew (optional)
            if deskew:
                binary = ImageProcessor._deskew_image(binary)
            
            return Image.fromarray(binary)
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image

    @staticmethod
    def preprocess_image_advanced(
        image: Image.Image,
        bilateral_filter: bool = True,
        morphological_ops: bool = True,
    ) -> Image.Image:
        """
        Advanced preprocessing with bilateral filtering and morphology.
        
        Useful for documents with varying lighting conditions.
        """
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image = ImageProcessor._resize_image(image)
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Bilateral filter (preserves edges while denoising)
            if bilateral_filter:
                gray = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Enhance contrast
            gray = ImageProcessor._enhance_contrast(gray)
            
            # Morphological operations
            if morphological_ops:
                gray = ImageProcessor._apply_morphology(gray)
            
            # Adaptive threshold
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                15, 2
            )
            
            return Image.fromarray(binary)
            
        except Exception as e:
            logger.error(f"Advanced preprocessing failed: {e}")
            return image

    @staticmethod
    def _resize_image(image: Image.Image) -> Image.Image:
        """Resize image to optimal size for OCR."""
        width, height = image.size
        max_dim = max(width, height)
        
        if max_dim > ImageProcessor.MAX_IMAGE_SIZE:
            scale = ImageProcessor.MAX_IMAGE_SIZE / max_dim
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        elif max_dim < ImageProcessor.MIN_IMAGE_SIZE:
            scale = ImageProcessor.MIN_IMAGE_SIZE / max_dim
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image

    @staticmethod
    def _denoise_image(gray: np.ndarray) -> np.ndarray:
        """Apply NLM (Non-Local Means) denoising."""
        return cv2.fastNlMeansDenoising(
            gray,
            h=ImageProcessor.NLM_DENOISE_H,
            templateWindowSize=7,
            searchWindowSize=21
        )

    @staticmethod
    def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        clahe = cv2.createCLAHE(
            clipLimit=ImageProcessor.CLAHE_CLIP_LIMIT,
            tileGridSize=ImageProcessor.CLAHE_TILE_SIZE
        )
        return clahe.apply(gray)

    @staticmethod
    def _apply_morphology(gray: np.ndarray) -> np.ndarray:
        """Apply morphological operations to strengthen text."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        return gray

    @staticmethod
    def _deskew_image(binary: np.ndarray) -> np.ndarray:
        """Auto-correct skewed images."""
        try:
            # Find contours
            contours, _ = cv2.findContours(
                binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return binary
            
            # Get the largest contour
            largest = max(contours, key=cv2.contourArea)
            angle = cv2.minAreaRect(largest)[-1]
            
            # Normalize angle
            if angle < -45:
                angle = 90 + angle
            
            # Only rotate if skew is significant
            if abs(angle) > ImageProcessor.SKEW_THRESHOLD:
                h, w = binary.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, -angle, 1.0)
                return cv2.warpAffine(
                    binary, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
            
            return binary
            
        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            return binary

    @staticmethod
    def estimate_text_orientation(image: np.ndarray) -> float:
        """Estimate text orientation angle."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Use Hough Transform to find line orientation
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
            
            if lines is not None:
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = np.degrees(theta)
                    if angle > 90:
                        angle -= 180
                    angles.append(angle)
                return np.median(angles)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Orientation estimation failed: {e}")
            return 0.0

    @staticmethod
    def get_image_quality_score(image: np.ndarray) -> float:
        """
        Calculate image quality score (0-1).
        
        Considers: brightness, contrast, sharpness
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Brightness score
            mean_brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 if 0.3 <= mean_brightness <= 0.7 else 0.5
            
            # Contrast score
            contrast = np.std(gray) / 128.0
            contrast_score = min(contrast / 0.5, 1.0)
            
            # Sharpness score (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian)
            sharpness_score = min(sharpness / 100.0, 1.0)
            
            # Combined score
            overall_score = (brightness_score + contrast_score + sharpness_score) / 3.0
            return overall_score
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5
