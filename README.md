# ThermoOmniFlux

A Python-based analysis pipeline for thermal digital PCR (dPCR) fluorescence microscopy data, customized for ThermoOmniFlux systems. This repository contains Jupyter notebooks and a utility library to process raw image data, extract melting temperatures (Tm) for probes, and perform clustering analysis.

## Project Structure

```
├── README.md                                      # This file
├── utils.py                                       # Utility library
└── jupyter_notebooks/
    ├── EvaGreen_Based_Processing_Pipeline.ipynb   # Image processing workflow
    └── Pooling_Clustering.ipynb                   # Clustering and analysis
```

## Files

**`utils.py`**: Core library providing image I/O, preprocessing, well/spot detection, keypoint tracking, fluorescence extraction, melting curve computation, clustering, and data assembly. All functions called by the notebooks are sourced here.

**`EvaGreen_Based_Processing_Pipeline.ipynb`**: Step-by-step workflow demonstrating the complete analysis from raw microscopy images to per-well melting profiles. Includes interactive parameter tuning and visualizations at each stage.

**`Pooling_Clustering.ipynb`**: Secondary analysis notebook for clustering analysis of molecule melting signal profiles, Tm distribution exploration, and result visualization.

## Workflow

### Pipeline (EvaGreen_Based_Processing_Pipeline.ipynb)

1. **Load & Preprocess**
   - Import TIFF/CZI images
   - Apply background correction

2. **Detect & Track**
   - Identify microplate wells
   - Track fluorescent spots across frames
   - Extract intensities over time

3. **Compute Tm**
   - Smooth fluorescence curves
   - Detect peaks and inflection points
   - Calculate melting temperatures
   - Filter by signal quality and Tm range

4. **Consolidate Results**
   - Combine Tm, position, and confidence data
   - Export to DataFrame

### Analysis (Pooling_Clustering.ipynb)

1. **Cluster Signals**
   - Group probe signals by similarity
   - Visualize cluster assignments

2. **Explore Distributions**
   - Plot Tm histograms and distributions
   - Compare against expected Tm values

3. **Generate Figures**
   - Create publication-ready plots
   - Export results for reporting

## Dependencies

- **Image I/O**: PIL, CziFile
- **Numerical**: NumPy, SciPy, Scikit-learn
- **ML/Torch**: PyTorch
- **Viz**: Matplotlib, Seaborn, Plotly, Dash
- **Data**: Pandas
- **Widgets**: IPywidgets
- **Progress**: tqdm

## Quick Start

1. Open `EvaGreen_Based_Processing_Pipeline.ipynb` in Jupyter
2. Adjust image file path and parameters (heating rate, exposure time, etc.)
3. Run cells sequentially to process images and extract Tm values
4. Use `Pooling_Clustering.ipynb` for downstream analysis and visualization

## License



## Contact


