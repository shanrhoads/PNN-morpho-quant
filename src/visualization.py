"""
Visualization functions for PNN analysis results.
"""

import numpy as np


def plot_image(image, title="Image", cmap='gray', figsize=(10, 8)):
    """
    Display an image.
    
    Parameters
    ----------
    image : numpy.ndarray
        Image to display
    title : str, optional
        Title for the plot
    cmap : str, optional
        Colormap to use (default: 'gray')
    figsize : tuple, optional
        Figure size (default: (10, 8))
        
    Examples
    --------
    >>> plot_image(image, title="Raw PNN Image")
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Warning: matplotlib not installed. Install with: pip install matplotlib")


def plot_overlay(image, mask, alpha=0.5, figsize=(10, 8)):
    """
    Display image with segmentation mask overlay.
    
    Parameters
    ----------
    image : numpy.ndarray
        Original image
    mask : numpy.ndarray
        Binary segmentation mask
    alpha : float, optional
        Transparency of overlay (default: 0.5)
    figsize : tuple, optional
        Figure size (default: (10, 8))
        
    Examples
    --------
    >>> plot_overlay(image, pnn_mask, alpha=0.5)
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image, cmap='gray')
        ax.imshow(mask, cmap='Reds', alpha=alpha)
        ax.set_title('Segmentation Overlay')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Warning: matplotlib not installed")


def plot_features_comparison(features_list, labels, figsize=(12, 6)):
    """
    Create comparison plots for multiple feature sets.
    
    Parameters
    ----------
    features_list : list of dict
        List of feature dictionaries to compare
    labels : list of str
        Labels for each feature set
    figsize : tuple, optional
        Figure size (default: (12, 6))
        
    Examples
    --------
    >>> features1 = extract_morphology_features(mask1)
    >>> features2 = extract_morphology_features(mask2)
    >>> plot_features_comparison([features1, features2], ['Condition 1', 'Condition 2'])
    """
    try:
        import matplotlib.pyplot as plt
        
        if not features_list or not labels:
            print("No features to plot")
            return
        
        # Extract common keys
        common_keys = set(features_list[0].keys())
        for f in features_list[1:]:
            common_keys = common_keys.intersection(set(f.keys()))
        
        common_keys = sorted(list(common_keys))
        
        if not common_keys:
            print("No common features to plot")
            return
        
        fig, axes = plt.subplots(1, len(common_keys), figsize=figsize)
        if len(common_keys) == 1:
            axes = [axes]
        
        for idx, key in enumerate(common_keys):
            values = [f[key] for f in features_list]
            axes[idx].bar(labels, values)
            axes[idx].set_title(key.replace('_', ' ').title())
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Warning: matplotlib not installed")


def create_summary_figure(image, mask, features, figsize=(15, 5)):
    """
    Create a comprehensive summary figure with image, mask, and features.
    
    Parameters
    ----------
    image : numpy.ndarray
        Original image
    mask : numpy.ndarray
        Segmentation mask
    features : dict
        Extracted features
    figsize : tuple, optional
        Figure size (default: (15, 5))
        
    Examples
    --------
    >>> create_summary_figure(image, mask, features)
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmentation mask
        axes[1].imshow(mask, cmap='Reds')
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        # Features text
        axes[2].axis('off')
        axes[2].set_title('Extracted Features')
        
        if features:
            text_str = '\n'.join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" 
                                 for k, v in features.items()])
            axes[2].text(0.1, 0.5, text_str, fontsize=10, verticalalignment='center')
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Warning: matplotlib not installed")
