"""
Image processing functions for PNN analysis.
"""

import numpy as np


def load_image(filepath):
    """
    Load an image file for analysis.
    
    Parameters
    ----------
    filepath : str
        Path to the image file
        
    Returns
    -------
    image : numpy.ndarray
        Loaded image as numpy array
        
    Examples
    --------
    >>> image = load_image('data/raw/sample_image.tif')
    """
    try:
        from skimage import io
        image = io.imread(filepath)
        return image
    except ImportError:
        print("Warning: scikit-image not installed. Install with: pip install scikit-image")
        return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def preprocess_image(image, remove_noise=True, normalize=True):
    """
    Preprocess image for analysis.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    remove_noise : bool, optional
        Apply noise removal filter (default: True)
    normalize : bool, optional
        Normalize image intensities (default: True)
        
    Returns
    -------
    processed_image : numpy.ndarray
        Preprocessed image
        
    Examples
    --------
    >>> processed = preprocess_image(raw_image, remove_noise=True, normalize=True)
    """
    if image is None:
        return None
    
    processed = image.copy()
    
    if remove_noise:
        try:
            from skimage import filters
            processed = filters.gaussian(processed, sigma=1)
        except ImportError:
            print("Warning: scikit-image not installed for noise removal")
    
    if normalize:
        processed = processed - processed.min()
        max_val = processed.max()
        if max_val > 0:
            processed = processed / max_val
    
    return processed


def segment_pnn(image, threshold_method='otsu'):
    """
    Segment perineuronal nets from preprocessed image.
    
    Parameters
    ----------
    image : numpy.ndarray
        Preprocessed image
    threshold_method : str, optional
        Thresholding method ('otsu', 'adaptive', or 'manual')
        Default: 'otsu'
        
    Returns
    -------
    mask : numpy.ndarray
        Binary mask of segmented PNNs
        
    Examples
    --------
    >>> mask = segment_pnn(processed_image, threshold_method='otsu')
    """
    if image is None:
        return None
    
    try:
        from skimage import filters
        
        if threshold_method == 'otsu':
            threshold = filters.threshold_otsu(image)
            mask = image > threshold
        elif threshold_method == 'adaptive':
            # Placeholder for adaptive thresholding
            threshold = filters.threshold_otsu(image)
            mask = image > threshold
        else:
            # Default to Otsu if method not recognized
            threshold = filters.threshold_otsu(image)
            mask = image > threshold
            
        return mask.astype(np.uint8)
    except ImportError:
        print("Warning: scikit-image not installed for segmentation")
        return None
