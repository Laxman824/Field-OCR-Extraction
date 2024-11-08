from PIL import Image
import numpy as np

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Enhanced preprocessing
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(binary)
    
    max_size = 1000
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))
    return image
