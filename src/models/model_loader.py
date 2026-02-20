"""
Optimized model loading for DocTR OCR with caching and error handling.

Features:
- Singleton pattern for efficient resource usage
- Stream cache for concurrent access
- Automatic fallback to CPU if CUDA unavailable
- Model telemetry and health checks
"""

import logging
import torch
from typing import Optional
import streamlit as st

logger = logging.getLogger(__name__)


class ModelManager:
    """Unified model loading and management."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.device = None
        self.model = None
        self.model_metadata = {}
    
    @staticmethod
    def _get_device() -> str:
        """Determine optimal compute device."""
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("MPS (Metal Performance Shaders) available")
            return "mps"
        else:
            logger.info("Using CPU for inference")
            return "cpu"
    
    @staticmethod
    def _load_model_unsafe(device: str):
        """Load DocTR model with specified device."""
        try:
            from doctr.models import ocr_predictor
            
            logger.info(f"Loading doctr OCR model on {device}...")
            
            # Load based on device
            if device == "cuda":
                model = ocr_predictor(
                    pretrained=True,
                    device=device,
                    pretrained_backbone=True
                )
            else:
                model = ocr_predictor(pretrained=True)
            
            logger.info("DocTR model loaded successfully")
            return model
            
        except ImportError as e:
            logger.error(f"doctr library not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model on {device}: {e}")
            raise
    
    def load_model(self) -> Optional[object]:
        """
        Load OCR model with automatic fallback.
        
        Returns:
            Loaded OCR model or None if loading fails
        """
        if self.model is not None:
            logger.debug("Returning cached model")
            return self.model
        
        # Try to determine best device
        device = self._get_device()
        
        try:
            self.model = self._load_model_unsafe(device)
            self.device = device
            self.model_metadata = {
                "device": device,
                "cuda_available": torch.cuda.is_available(),
                "model_type": "doctr_ocr",
                "status": "healthy"
            }
            return self.model
            
        except Exception as e:
            logger.error(f"Model loading failed on {device}: {e}")
            
            # Try CPU fallback if not already on CPU
            if device != "cpu":
                logger.info("Attempting CPU fallback...")
                try:
                    self.model = self._load_model_unsafe("cpu")
                    self.device = "cpu"
                    self.model_metadata = {
                        "device": "cpu",
                        "cuda_available": False,
                        "model_type": "doctr_ocr",
                        "status": "degraded"
                    }
                    return self.model
                except Exception as cpu_error:
                    logger.error(f"CPU fallback also failed: {cpu_error}")
            
            self.model_metadata = {
                "device": None,
                "model_type": "doctr_ocr",
                "status": "failed",
                "error": str(e)
            }
            return None
    
    def get_model_metadata(self) -> dict:
        """Get model load status and metadata."""
        return self.model_metadata
    
    def is_model_healthy(self) -> bool:
        """Check if model is loaded and healthy."""
        return self.model is not None and self.model_metadata.get("status") != "failed"


# ════════════════════════════════════════════════════════════════════
#  STREAMLIT-OPTIMIZED CACHING
# ════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load OCR model with Streamlit caching.
    
    Uses @cache_resource to ensure model is loaded only once per session
    and reused across all reruns.
    
    Returns:
        Loaded OCR model instance
    """
    with st.spinner("Loading OCR model..."):
        manager = ModelManager()
        model = manager.load_model()
        
        if model is None:
            st.error("❌ Failed to load OCR model. Please check dependencies.")
            st.stop()
        
        st.success("✅ OCR model loaded successfully")
        return model


@st.cache_resource
def get_model_manager() -> ModelManager:
    """Get the model manager singleton instance."""
    return ModelManager()


@st.cache_data
def get_model_info():
    """Cache model metadata."""
    manager = get_model_manager()
    return manager.get_model_metadata()


# ════════════════════════════════════════════════════════════════════
#  BACKWARD COMPATIBILITY
# ════════════════════════════════════════════════════════════════════

def get_ocr_model():
    """Alias for load_model() for backward compatibility."""
    return load_model()
