"""
Utility functions for PNN analysis.
"""

import os
import json


def save_features_to_json(features, filepath):
    """
    Save extracted features to a JSON file.
    
    Parameters
    ----------
    features : dict
        Dictionary of features to save
    filepath : str
        Path where to save the JSON file
        
    Examples
    --------
    >>> save_features_to_json(features, 'results/features.json')
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(features, f, indent=4)
        print(f"Features saved to {filepath}")
    except Exception as e:
        print(f"Error saving features: {e}")


def load_features_from_json(filepath):
    """
    Load features from a JSON file.
    
    Parameters
    ----------
    filepath : str
        Path to the JSON file
        
    Returns
    -------
    features : dict
        Dictionary of loaded features
        
    Examples
    --------
    >>> features = load_features_from_json('results/features.json')
    """
    try:
        with open(filepath, 'r') as f:
            features = json.load(f)
        return features
    except Exception as e:
        print(f"Error loading features: {e}")
        return None


def batch_process_images(image_folder, output_folder, processing_func):
    """
    Process multiple images in a folder.
    
    Parameters
    ----------
    image_folder : str
        Path to folder containing images
    output_folder : str
        Path to folder for saving results
    processing_func : callable
        Function to apply to each image
        
    Returns
    -------
    results : dict
        Dictionary mapping filenames to processing results
        
    Examples
    --------
    >>> def my_process(img_path):
    ...     return extract_features(img_path)
    >>> results = batch_process_images('data/raw', 'results', my_process)
    """
    if not os.path.exists(image_folder):
        print(f"Folder not found: {image_folder}")
        return None
    
    os.makedirs(output_folder, exist_ok=True)
    
    results = {}
    image_files = [f for f in os.listdir(image_folder) 
                   if f.endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        try:
            result = processing_func(img_path)
            results[img_file] = result
            print(f"Processed: {img_file}")
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            results[img_file] = None
    
    return results


def create_report(features_dict, output_filepath):
    """
    Create a summary report from analysis results.
    
    Parameters
    ----------
    features_dict : dict
        Dictionary containing analysis results
    output_filepath : str
        Path where to save the report
        
    Examples
    --------
    >>> create_report(all_features, 'results/analysis_report.txt')
    """
    try:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w') as f:
            f.write("PNN Morphology Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            for key, value in features_dict.items():
                f.write(f"{key}:\n")
                if isinstance(value, dict):
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"  {value}\n")
                f.write("\n")
        
        print(f"Report saved to {output_filepath}")
    except Exception as e:
        print(f"Error creating report: {e}")
