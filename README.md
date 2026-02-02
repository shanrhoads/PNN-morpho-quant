# PNN-morpho-quant

Quantitative analysis of perineuronal net (PNN) morphology from STED super-resolution images.

## Overview

This repository provides a complete analysis pipeline for quantifying morphological features of perineuronal nets from STED microscopy images. The pipeline is implemented in Python with support for R integration for statistical analysis.

## Repository Structure

```
PNN-morpho-quant/
├── src/                          # Python modules with analysis functions
│   ├── __init__.py              # Package initialization
│   ├── image_processing.py      # Image loading, preprocessing, and segmentation
│   ├── morphology_analysis.py   # Morphological feature extraction
│   ├── visualization.py         # Plotting and visualization functions
│   └── utils.py                 # Utility functions (I/O, batch processing)
├── notebooks/                    # Jupyter notebooks
│   └── PNN_Analysis_Pipeline.ipynb  # Main analysis pipeline notebook
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## Features

- **Image Processing**: Load and preprocess STED super-resolution images
- **Segmentation**: Automated PNN structure segmentation
- **Morphological Analysis**: Extract quantitative features (area, perimeter, circularity, etc.)
- **Intensity Analysis**: Measure intensity-based properties within PNN regions
- **Mesh Structure Analysis**: Quantify mesh-like properties of PNN networks
- **Visualization**: Create publication-ready figures and overlays
- **Batch Processing**: Process multiple images efficiently
- **R Integration**: Support for R code execution for statistical analysis

## Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook
- (Optional) R 4.0+ for R integration

### Setup

1. Clone the repository:
```bash
git clone https://github.com/shanrhoads/PNN-morpho-quant.git
cd PNN-morpho-quant
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For R integration, install rpy2:
```bash
pip install rpy2
```

## Usage

### Using the Jupyter Notebook

The main analysis pipeline is provided in an easy-to-use Jupyter notebook:

1. Navigate to the notebooks directory:
```bash
cd notebooks
```

2. Launch Jupyter:
```bash
jupyter notebook
```

3. Open `PNN_Analysis_Pipeline.ipynb` and follow the step-by-step instructions.

### Using Python Modules Directly

You can also use the Python modules directly in your own scripts:

```python
import sys
sys.path.insert(0, '../src')

from src import image_processing as ip
from src import morphology_analysis as ma
from src import visualization as vis

# Load and process image
image = ip.load_image('data/raw/sample.tif')
processed = ip.preprocess_image(image)
mask = ip.segment_pnn(processed)

# Extract features
features = ma.extract_morphology_features(mask)
intensity_features = ma.calculate_intensity_features(processed, mask)

# Visualize results
vis.plot_overlay(processed, mask)
vis.create_summary_figure(processed, mask, features)
```

## Analysis Pipeline

The typical analysis workflow includes:

1. **Load Image**: Import STED super-resolution images
2. **Preprocess**: Apply noise reduction and normalization
3. **Segment**: Identify PNN structures using automated thresholding
4. **Extract Features**: Quantify morphological and intensity properties
5. **Visualize**: Create figures and overlays
6. **Export**: Save results as JSON and text reports

## Module Documentation

### image_processing.py

- `load_image(filepath)`: Load image files
- `preprocess_image(image, remove_noise, normalize)`: Preprocess images
- `segment_pnn(image, threshold_method)`: Segment PNN structures

### morphology_analysis.py

- `extract_morphology_features(mask)`: Extract morphological features
- `calculate_intensity_features(image, mask)`: Calculate intensity metrics
- `analyze_mesh_structure(mask)`: Analyze mesh-like properties

### visualization.py

- `plot_image(image, title, cmap)`: Display images
- `plot_overlay(image, mask, alpha)`: Show segmentation overlays
- `create_summary_figure(image, mask, features)`: Generate summary figures
- `plot_features_comparison(features_list, labels)`: Compare multiple datasets

### utils.py

- `save_features_to_json(features, filepath)`: Save results to JSON
- `load_features_from_json(filepath)`: Load saved results
- `batch_process_images(image_folder, output_folder, processing_func)`: Batch processing
- `create_report(features_dict, output_filepath)`: Generate text reports

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```
Rhoads, S. (2026). PNN-morpho-quant: Quantitative analysis of perineuronal net morphology 
from STED super-resolution images. GitHub repository. 
https://github.com/shanrhoads/PNN-morpho-quant
```

## Contact

For questions or issues, please open an issue on GitHub.
