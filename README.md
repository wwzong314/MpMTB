# ThermoOmniFlux

A Python-based analysis pipeline for **thermal digital PCR (dPCR) fluorescence microscopy data**, tailored to ThermoOmniFlux systems. The repository provides Jupyter notebooks and a supporting utility library for processing raw CZI/TIFF microscopy images, extracting melting temperatures (Tm) from single-molecule melting profiles, and performing downstream molecule identification via clustering.

---

## Project Structure

```
ThermoOmniFlux/
├── README.md
├── utils.py                                        # Core utility library (all processing functions)
└── jupyter_notebooks/
    ├── Image_Processing_Pipeline.ipynb              # End-to-end image processing workflow
    ├── Pooling_Clustering.ipynb                     # Multi-area pooled clustering and analysis
    └── Direct_Clustering.ipynb                      # Single-area clustering (no pooling)
```

---

## Notebooks

### `Image_Processing_Pipeline.ipynb`

The primary notebook. Walks through the full analysis from raw microscopy images to per-molecule melting temperature profiles. Supports multi-channel CZI image stacks (EvaGreen, HEX, Cy5, ROX, etc.) with interactive parameter tuning and visualizations at each stage.

**Sections include:**
- File loading and CZI metadata extraction (channels, exposure times, laser intensities)
- Well detection via GPU-accelerated convolution (PyTorch, with MPS/CPU fallback)
- Keypoint tracking and alignment across frames
- Positive / rain / negative well partitioning (KMeans-based fluorescence thresholding)
- Background subtraction (neighborhood-based and Wittwer exponential methods)
- Savitzky-Golay smoothing and -dF/dT derivative computation
- Multi-level melting temperature (Tm) determination with adaptive noise floors
- Per-channel Tm probe signal processing (shape filtering, anomaly detection, local Tm extraction)
- Final results compilation and export to CSV / NumPy formats

### `Pooling_Clustering.ipynb`

Secondary analysis notebook for experiments with **multiple imaging areas** (e.g., multiple chips or fields of view). Aggregates Tm results from the image processing pipeline, aligns datasets across areas to correct for batch effects, and performs molecule identification.

**Key steps:**
1. Load per-area Tm result CSVs from the processing pipeline
2. Align Tm distributions across imaging areas using grid-based registration (`align_datasets`, `apply_global_shift`)
3. Align observed Tm values with predicted Tm barcodes
4. Molecule identification via Greedy Surjective Matching
5. Cluster refinement with Gaussian Mixture Models (GMM) or KMeans
6. Interactive cluster visualization and quantification (Dash-powered scatter plots)
7. Export aligned data and per-molecule statistics

### `Direct_Clustering.ipynb`

A streamlined variant of `Pooling_Clustering.ipynb` for experiments with a **single imaging area**. Skips the multi-area pooling and alignment steps and proceeds directly to matching and clustering against predicted Tm barcodes.

---

## `utils.py` — Core Library

All functions called by the notebooks are defined here. The library is organized into the following functional groups:

| Category | Key Functions | Description |
|---|---|---|
| **Image I/O** | `tiff_to_arr`, `retrieve_czi_metadata`, `parse_filename` | Load TIFF/CZI images, extract metadata (channels, filters, exposure times) |
| **Preprocessing** | `gaussian_background_correction`, `invert_image`, `gaussian_background_correction_div` | Gaussian background correction and image inversion |
| **Well Detection** | `_find_well`, `find_well_new_no_tile`, `generate_pos_seq_new_no_tile` | GPU-accelerated well localization using Gaussian convolution + non-max suppression (PyTorch) |
| **Keypoint Tracking** | `track_keypoints`, `track_keypoints_multi_channel`, `filter_keypoints` | Track well positions across frames; multi-channel support |
| **Partitioning** | `define_the_rain`, `define_the_rain_kmeans` | Classify wells as positive / rain / negative using KMeans fluorescence thresholding (with optional adaptive windowed mode) |
| **Signal Processing** | `savgol`, `generate_fluorescence_vs_time`, `get_average_pixel_values_circ`, `gaussian_smooth`, `min_max_normalize` | Extract fluorescence time series, Savitzky-Golay filtering, Gaussian smoothing, normalization |
| **Background Subtraction** | `subtract_background`, `wittwer_background_subtract` | Neighborhood-based median subtraction; Wittwer exponential background model |
| **Tm Extraction** | `get_Tm`, `get_Tm_lvl2`, `compute_Tm`, `get_noise_floor`, `compute_local_tms` | Multi-level Tm determination from -dF/dT curves with adaptive noise floors and peak finding |
| **Probe Analysis** | `probe_filter_by_shape`, `interactive_probe_filtering`, `interactive_probe_clustering_thresholding` | Shape-based probe signal filtering, interactive clustering with ipywidgets |
| **Anomaly Detection** | `interactive_anomaly_filtering`, `anomaly_filter_by_isolation_forest`, `anomaly_filter_by_lof`, `anomaly_filter_by_one_class_svm`, `train_autoencoder` | Isolation Forest, LOF, One-Class SVM, and autoencoder-based outlier removal |
| **Dataset Alignment** | `align_datasets`, `apply_global_shift`, `create_grid`, `grid_transform` | Cross-area Tm distribution alignment via grid-based registration and global shift correction |
| **Clustering** | `greedy_surjective_constrained_matching`, `refine_clusters` | Greedy surjective matching against predicted Tm barcodes; GMM / KMeans cluster refinement |
| **Visualization** | `plot_matching`, `plot_matching_interactive`, `scatter_plot_dfs`, `plot_tm_levels`, `interactive_visual_QC` | Static and interactive (Dash / ipywidgets) plots for QC, matching results, and Tm distributions |
| **Data Assembly** | `join_all_tms`, `join_meta_data`, `update_cluster_assignment`, `save_data`, `load_data` | Combine Tm, position, fluorescence, and confidence data; save/load pickle archives |

> **Tip:** Run `help(utils.<function_name>)` in any notebook cell to view a function's full docstring and parameter descriptions.

---

## General Workflow

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                  Image_Processing_Pipeline.ipynb                    │
 │                                                                     │
 │  Raw CZI/TIFF ─► Well Detection ─► Tracking ─► Partitioning         │
 │       │                                              │              │
 │       │         Background Subtraction ◄─────────────┘              │
 │       │              │                                              │
 │       │         -dF/dT Derivative                                   │
 │       │              │                                              │
 │       │         Tm Determination (Multi-Level)                      │
 │       │              │                                              │
 │       │         Per-Channel Probe Tm Extraction                     │
 │       │              │                                              │
 │       └──────── Results CSV ────────────────────────────────────►   │
 └─────────────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              ▼                            ▼
 ┌───────────────────────┐    ┌───────────────────────────┐
 │  Direct_Clustering    │    │  Pooling_Clustering       │
 │  (single area)        │    │  (multiple areas)         │
 │                       │    │                           │
 │  Load Tm results      │    │  Load per-area results    │
 │  Align w/ predictions │    │  Align across areas       │
 │  Greedy Matching      │    │  Align w/ predictions     │
 │  GMM/KMeans Refine    │    │  Greedy Matching          │
 │  Quantify & Export    │    │  GMM/KMeans Refine        │
 └───────────────────────┘    │  Quantify & Export        │
                              └───────────────────────────┘
```


## Configurable Parameters

### Image Processing

| Parameter | Description | Guidance |
|---|---|---|
| `k0` | Kernel size for first Gaussian convolution (pixels, odd) | Set based on microscope magnification and well size. Reference values provided in notebook. |
| `k1` | Kernel size for second convolution / locality (pixels, odd) | Controls non-max suppression neighborhood. Typically similar to or larger than `k0`. |
| `pixel_range` | Radius for fluorescence extraction (pixels) | Set according to well diameter. Generally ≥ `k0`. |
| `merge_threshold` | Max distance to merge detections across rounds (pixels) | Controls duplicate suppression in multi-round detection. |
| `eps` | Search radius for frame-to-frame alignment (pixels) | Max displacement between consecutive frames. |
| `n_SD` | Standard deviations for positive/rain threshold | Higher values = stricter partitioning. Adjust based on image SNR. |

### Tm Computation

| Parameter | Description | Guidance |
|---|---|---|
| `heating_rate_per_min` | Temperature ramp rate (C/min) | Must match experimental protocol. |
| `exposure_in_sec` | Time between consecutive frames (seconds) | Must match camera acquisition settings. |
| `window_length` | Savitzky-Golay smoothing window (frames, odd) | Wider = smoother curves but lower resolution. Adjust based on melt transition width and noise. |
| `initial_T` / `final_T` | Temperature range of the melt ramp (C) | Set to match heating plate start and end temperatures. |
| `weight` | Noise floor blending weight (0-1) | Balances empirical vs. fitted noise floor. Default 0.5. |
| `height_tolerance` | Peak height comparison tolerance | Controls strictness of two-peak validation. |

### Clustering

| Parameter | Description | Guidance |
|---|---|---|
| `max_cost_threshold` | Maximum distance for greedy matching | Controls how far an observation can be from a predicted barcode and still be assigned. |
| `max_shift_x` / `max_shift_y` | Maximum alignment shift (Tm units) | Bounds on cross-area Tm correction. |
| `std_x` / `std_y` | Gaussian ellipse standard deviations for interactive clustering | Controls cluster boundary visualization. |

---

## Dependencies

| Category | Packages |
|---|---|
| **Image I/O** | PIL (Pillow), czifile |
| **Numerical** | NumPy, SciPy |
| **Data** | Pandas |
| **Machine Learning** | Scikit-learn (KMeans, GMM, Isolation Forest, LOF, One-Class SVM) |
| **Deep Learning** | PyTorch (GPU-accelerated well detection, autoencoder anomaly filtering) |
| **Visualization** | Matplotlib, Seaborn, Plotly, Dash |
| **Interactive Widgets** | ipywidgets, IPython |
| **Progress** | tqdm |

### Setting Up the Environment

```bash
conda env create -f environment.yml
conda activate thermoomniflux
```

This creates a conda environment named `thermoomniflux` with all necessary packages pre-configured.

---

## Quick Start

1. **Set up the environment** using the conda instructions above
2. **Open** `Image_Processing_Pipeline.ipynb` in Jupyter
3. **Configure** the image file paths and experimental parameters (heating rate, exposure time, temperature range, kernel sizes)
4. **Run cells sequentially** to process images and extract Tm values — interactive widgets allow real-time parameter tuning
5. **Open** `Pooling_Clustering.ipynb` (multi-area) or `Direct_Clustering.ipynb` (single area) to perform molecule identification and visualization

---

## License

## Contact
