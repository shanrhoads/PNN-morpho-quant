"""
Morphology analysis functions for PNN structures.
"""

import numpy as np


def extract_morphology_features(mask):
    """
    Extract morphological features from segmented PNN mask.
    
    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask of segmented PNNs
        
    Returns
    -------
    features : dict
        Dictionary containing morphological features:
        - area: Total area of PNNs
        - perimeter: Total perimeter
        - circularity: Measure of roundness
        - num_objects: Number of detected PNN structures
        
    Examples
    --------
    >>> features = extract_morphology_features(pnn_mask)
    >>> print(f"Total area: {features['area']}")
    """
    if mask is None:
        return None
    
    try:
        from skimage import measure
        
        # Label connected components
        labeled_mask = measure.label(mask)
        props = measure.regionprops(labeled_mask)
        
        features = {
            'num_objects': len(props),
            'total_area': sum([p.area for p in props]),
            'mean_area': np.mean([p.area for p in props]) if props else 0,
            'total_perimeter': sum([p.perimeter for p in props]),
            'mean_perimeter': np.mean([p.perimeter for p in props]) if props else 0,
        }
        
        # Calculate circularity for each object
        circularities = []
        for p in props:
            if p.perimeter > 0:
                circularity = 4 * np.pi * p.area / (p.perimeter ** 2)
                circularities.append(circularity)
        
        features['mean_circularity'] = np.mean(circularities) if circularities else 0
        
        return features
    except ImportError:
        print("Warning: scikit-image not installed for morphology analysis")
        return None


def calculate_intensity_features(image, mask):
    """
    Calculate intensity-based features within PNN regions.
    
    Parameters
    ----------
    image : numpy.ndarray
        Original or preprocessed image
    mask : numpy.ndarray
        Binary mask of segmented PNNs
        
    Returns
    -------
    intensity_features : dict
        Dictionary containing intensity features:
        - mean_intensity: Mean intensity in PNN regions
        - std_intensity: Standard deviation of intensity
        - max_intensity: Maximum intensity value
        
    Examples
    --------
    >>> intensity_features = calculate_intensity_features(image, pnn_mask)
    """
    if image is None or mask is None:
        return None
    
    # Extract intensities within masked regions
    masked_intensities = image[mask > 0]
    
    if len(masked_intensities) == 0:
        return {
            'mean_intensity': 0,
            'std_intensity': 0,
            'max_intensity': 0,
            'min_intensity': 0
        }
    
    intensity_features = {
        'mean_intensity': np.mean(masked_intensities),
        'std_intensity': np.std(masked_intensities),
        'max_intensity': np.max(masked_intensities),
        'min_intensity': np.min(masked_intensities)
    }
    
    return intensity_features


def analyze_mesh_structure(mask):
    """
    Analyze the mesh-like structure of PNNs.
    
    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask of segmented PNNs
        
    Returns
    -------
    mesh_features : dict
        Dictionary containing mesh-related features:
        - density: Overall density of the mesh structure
        - connectivity: Measure of how connected the mesh is
        
    Examples
    --------
    >>> mesh_features = analyze_mesh_structure(pnn_mask)
    """
    if mask is None:
        return None
    
    mesh_features = {
        'density': np.sum(mask) / mask.size,
        'coverage': np.sum(mask > 0) / mask.size
    }
    
    # Add more sophisticated mesh analysis here
    # (e.g., skeleton analysis, branching points)
    
    return mesh_features
