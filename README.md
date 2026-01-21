# ThermoOmniFlux

 Python-based analysis pipeline for thermal digital PCR (dPCR) fluorescence microscopy data, tailored to ThermoOmniFlux systems. The repository includes Jupyter notebooks and a supporting utility library for processing raw image data, extracting melting temperatures (Tm) from molecule melting profiles, and performing downstream clustering analyses.

## Project Structure

```
├── README.md                                      # This file
├── utils.py                                       # Utility library
└── jupyter_notebooks/
    ├── EvaGreen_Based_Processing_Pipeline.ipynb   # Image processing workflow
    └── Pooling_Clustering.ipynb                   # Clustering and analysis
```

## Files

**`utils.py`**: Core library providing image I/O, preprocessing, well localization, well location tracking, fluorescence extraction, melting curve signal processing, clustering, and data assembly. All functions called by the notebooks are sourced here.

**`EvaGreen_Based_Processing_Pipeline.ipynb`**: Step-by-step workflow demonstrating the complete analysis from raw microscopy images to per-well melting profiles. Includes interactive parameter tuning and visualizations at each stage.

**`Pooling_Clustering.ipynb`**: Secondary analysis notebook for clustering analysis of molecule melting signal profiles, Tm distribution exploration, molecule identification, and result visualization.

## General Workflow

### Pipeline (EvaGreen_Based_Processing_Pipeline.ipynb)

1. **Load & Preprocess**
   - Import TIFF/CZI images
   - Apply background correction

2. **Detect & Track**
   - Identify microplate wells
   - Track fluorescent spots across frames
   - Extract intensities over time

3. **Compute Global Tm**
   - Smooth fluorescence curves
   - Detect peaks and inflection points
   - Calculate melting temperatures
   - Filter by signal quality and Tm range

4. **Color Channel Splitting/Compute Local Tm**
    - Separate color channels if applicable
    - Repeat Tm calculation per channel if applicable

5. **Results Compilation and Saving**
   - Combine Tm, position, and confidence data
   - Export to DataFrame

### Analysis (Pooling_Clustering.ipynb)

1. **Load Data**
   - Import Tm results from previous pipeline

2. **Pooling**
   - Aggregate data across chips
   - Align data accounting for experimental batch effects
   - Prepare for clustering

3. **Clustering**  
   - Match with predicted melting profiles
   - Apply Greedy Surjective Matching algorithm
   - Identify moelcules based on Tm profiles

4. **Statistics and Visualization** 
    - Plot Tm distributions
    - Visualize clustering results
    - Generate summary statistics

## Dependencies

- **Image I/O**: PIL, CziFile
- **Numerical**: NumPy, SciPy, Scikit-learn
- **ML/Torch**: PyTorch
- **Viz**: Matplotlib, Seaborn, Plotly, Dash
- **Data**: Pandas
- **Widgets**: IPywidgets
- **Progress**: tqdm

### Setting Up the Environment

An `environment.yml` file is included to set up a conda environment with all required dependencies:

```bash
conda env create -f environment.yml
conda activate thermoomniflux
```

This will create and activate a conda environment named `thermoomniflux` with all necessary packages pre-configured.

## Quick Start

1. Open `EvaGreen_Based_Processing_Pipeline.ipynb` in Jupyter
2. Adjust image file path and parameters (heating rate, exposure time, etc.)
3. Run cells sequentially to process images and extract Tm values
4. Use `Pooling_Clustering.ipynb` for downstream analysis and visualization

## License



## Contact


