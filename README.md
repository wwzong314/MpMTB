# MpMTB

## Overview
MpMTB is a comprehensive Python-based analysis pipeline for processing and analyzing thermal fluorescence microscopy data, with a focus on EvaGreen-based qPCR applications. The project provides tools for image processing, well detection, fluorescence tracking, melting temperature (Tm) calculation, and hierarchical clustering of PCR probes.

## Features

### Core Processing Pipeline (`EvaGreen_Based_Processing_Pipeline.ipynb`)
- **Image I/O**: Convert TIFF and CZI microscopy files to NumPy arrays with metadata extraction
- **Well Detection**: Automated identification of microplate wells in fluorescence images
- **Background Correction**: Gaussian background subtraction and image normalization
- **Keypoint Tracking**: Track fluorescent spots across image sequences with multi-channel support
- **Fluorescence Analysis**: Extract and smooth fluorescence time-series data
- **Melting Curve Processing**: Calculate melting temperatures with peak detection and filtering
- **Probe Classification**: Distinguish positive signals from background noise

### Advanced Analysis (`Pooling_Clustering.ipynb`)
- **Hierarchical Clustering**: Group similar PCR probes using scipy linkage methods
- **Signal Clustering**: K-means clustering of fluorescence signals
- **Tm Distribution Analysis**: Visualize and analyze melting temperature distributions
- **Multi-probe Analysis**: Handle multiple amplicon targets with expected Tm values
- **Interactive Filtering**: Threshold-based filtering with interactive widgets

## Project Structure

```
├── README.md                               # This file
├── utils.py                                # Core utility functions (4700+ lines)
└── jupyter_notebooks/
    ├── EvaGreen_Based_Processing_Pipeline.ipynb   # Main processing workflow
    └── Pooling_Clustering.ipynb                   # Clustering and analysis
```

## Dependencies

### Core Libraries
- **Image Processing**: PIL, CziFile
- **Numerical Computing**: NumPy, SciPy, Scikit-learn
- **Machine Learning**: PyTorch, scikit-learn (clustering, outlier detection)
- **Visualization**: Matplotlib, Seaborn, Plotly, Dash
- **Data Analysis**: Pandas
- **Progress Tracking**: tqdm
- **Interactive Widgets**: IPywidgets

See imports in `utils.py` for complete dependency list.

## Key Modules in utils.py

### Image Processing
- `tiff_to_arr()`: Convert TIFF files to numpy arrays
- `retrieve_czi_metadata()`: Extract CZI file metadata
- `invert_image()`, `gaussian_background_correction()`: Image preprocessing
- `find_well_new_no_tile()`: Well detection in microplates

### Keypoint Tracking
- `track_keypoints()`, `track_keypoints_multi_channel()`: Track fluorescent spots across frames
- `filter_keypoints()`: Remove spurious detections
- `plot_keypoint_tracking()`: Visualize tracking results

### Fluorescence Analysis
- `generate_fluorescence_vs_time()`: Extract intensity time-series
- `get_average_pixel_values_circ()`: Calculate circular region intensities with Gaussian weighting
- `savgol()`: Savitzky-Golay filtering of signals
- `snr_moving_avg()`: Signal-to-noise ratio calculation

### Melting Temperature Calculation
- `compute_Tm()`: Calculate Tm from frame index and heating rate
- `get_Tm()`: Advanced Tm extraction with peak detection
- `get_Tm_lvl2()`: Multi-peak Tm detection
- `compute_local_tms()`: Per-probe Tm calculation
- `get_noise_floor()`: Determine signal threshold

### Clustering & Filtering
- `cluster_signals()`: Hierarchical clustering of fluorescence profiles
- `probe_filter_by_shape()`: Filter probes based on signal shape
- `filter_local_tms()`: Resolve ambiguous Tm values
- `visualize_probe_clusters()`: Cluster visualization

### Data Assembly
- `join_all_tms()`: Combine local and global Tm data
- `join_meta_data()`: Create output DataFrame with position and confidence metrics

## Usage

### Basic Workflow
1. **Load Images**: Use `tiff_to_arr()` or CziFile to load microscopy data
2. **Preprocess**: Apply background correction with `gaussian_background_correction()`
3. **Detect Wells**: Find well positions using `find_well_new_no_tile()`
4. **Track Spots**: Monitor fluorescent spots with `track_keypoints()`
5. **Extract Tm**: Calculate melting temperatures with `get_Tm()` or `compute_local_tms()`
6. **Cluster & Filter**: Group similar probes with `cluster_signals()` and filter with probe shape criteria
7. **Export Results**: Combine metadata and Tm values with `join_meta_data()`

### Interactive Analysis
Use the Jupyter notebooks for step-by-step exploration:
- Adjust parameters interactively with IPywidgets sliders
- Visualize intermediate results (images, tracking overlays, curves)
- Export processed data to CSV/DataFrame format

## Configuration Parameters

Common tuning parameters:
- `pix_range`: Circular region size for fluorescence extraction
- `n_SD`: Number of standard deviations for noise floor threshold
- `heating_rate_per_min`: Temperature increase rate (°C/min)
- `exposure_in_sec`: Time between consecutive frames
- `num_clusters`: Number of clusters for K-means analysis
- `savgol_window`: Smoothing window length for Savitzky-Golay filter

## Output

The pipeline generates:
- **Melting Temperatures (Tm)**: Per-probe calculated Tm values with confidence scores
- **Fluorescence Profiles**: Time-series intensity data per well/probe
- **Position Data**: (x, y) coordinates of detected wells
- **Cluster Assignments**: Grouping of similar probe signals
- **Visualizations**: PNG plots of processing steps and final results
- **DataFrame**: Consolidated output with all metadata

## Performance Notes

- Processes multi-channel fluorescence data
- Handles multi-peak detection for multiplexed PCR assays
- Supports hierarchical filtering for ambiguous Tm values
- Interactive parameter optimization via Jupyter widgets
- Visualization tools for result validation

## Citation & Attribution

Part of the Weitz Lab research on thermal analysis and microfluidic systems.

## License

[Add appropriate license information]

## Contact

Will [Add contact information]
