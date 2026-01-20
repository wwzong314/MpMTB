import os
import math
import time
import ast
import itertools
from typing import List, Union
from ast import literal_eval
import copy
from datetime import datetime

import numpy as np
import numpy.ma as ma

from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import euclidean
from scipy.stats import linregress
from scipy.stats import norm 
from scipy.stats import multivariate_normal
from scipy.interpolate import griddata
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.spatial import KDTree


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from tqdm import tqdm
from PIL import Image
from czifile import CziFile
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import ipywidgets as widgets
from IPython.display import display, clear_output
from ipywidgets import VBox, HBox, interact, FloatSlider, interactive_output, Layout

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


import dash
from dash import dcc, html, Input, Output, State
import socket
import plotly.express as px
import plotly.graph_objs as go
import pickle
import glob
import seaborn as sns




def tiff_to_arr(file_path:str) -> np.array:
    """Function to convert a tiff file to a numpy array
    Args:
    file_path (str): The path to the tiff file

    Returns:
    np.array: The numpy array of the tiff file"""

    dataset = Image.open(file_path)
    h,w = np.shape(dataset)
    tiffarray = np.zeros((dataset.n_frames,h,w))
    for i in range(dataset.n_frames):
        dataset.seek(i)
        tiffarray[i,:,:] = np.array(dataset)
    return tiffarray

def retrieve_czi_metadata(file_path:str, get_filters:bool=False):

    """Function to retrieve metadata from a CZI file. If get_filters is set to True, it will also return the filter cut-in and cut-out wavelengths for each channel.
    Args:
    file_path (str): The path to the CZI file
    get_filters (bool): Whether to retrieve the filter cut-in and cut-out wavelengths for each channel. Default is False.

    Returns:
    The metadata dictionary, and other parsed metadata values """

    if not file_path.endswith("czi"):
        # throw exception
        print("Error! Please provide a CZI file")
        return
    else:

        with CziFile(file_path) as czi:
            metadata = czi.metadata(raw=False)
            
            dimensions = czi.shape
            pixel_type = czi.dtype
            # pixel_size = metadata['ImageDocument']['Metadata']['Scaling']['Items']['Distance']

            n_channels = metadata["ImageDocument"]["Metadata"]["Information"]["Image"]["SizeC"]
            
            print("\nCZI Image Dimensions:")
            print(dimensions)
            
            print("\nCZI Pixel Color Depth:")
            print(pixel_type)

            print("\nNumber of Color Channels:")
            print(n_channels)

            if get_filters:

                filters = metadata["ImageDocument"]["Metadata"]["Information"]["Instrument"]["Filters"]["Filter"]

                # list of 2-tuples (CutIn, CutOut)
                # cutin_cutout_list = [(filter['TransmittanceRange']['CutIn'], filter['TransmittanceRange']['CutOut']) for filter in filters]

                # dictionary with channel id being keys and 2-tuples (CutIn, CutOut) being values. tuple 1: lightsource excitation filter, tuple 2: fluorophore emission filter
                cutin_cutout_dict = {}
                channel = 0

                for i in range(0, len(filters), 2):
                    cutin_cutout_dict[channel] = [
                        (filters[i]['TransmittanceRange']['CutIn'], filters[i]['TransmittanceRange']['CutOut']),
                        (filters[i + 1]['TransmittanceRange']['CutIn'], filters[i + 1]['TransmittanceRange']['CutOut'])
                    ]
                    channel += 1

                print("\nFilter Cut-In and Cut-Out Wavelengths (nm):")

                for channel, (cutin, cutout) in cutin_cutout_dict.items():
                    print(f"Channel {channel}:")
                    print(f"Laser Excitation Filter: {cutin}")
                    print(f"Dye Emission Filter: {cutout}")

            
            if n_channels >1:
            
                channel_names = [channel["Name"] for channel in metadata["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]]
    
                excitation_wavelengths = [channel["ExcitationWavelength"] for channel in metadata["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["Channels"]["Channel"]]
                
                exposure_times = [channel["ExposureTime"]/1e9 for channel in metadata["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["Channels"]["Channel"]]

                # laser_intensity = [channel["LightSourcesSettings"]["LightSourceSettings"]["Intensity"] for channel in metadata["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["Channels"]["Channel"]]

                laser_intensity = []

                for channel in metadata["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["Channels"]["Channel"]:
                    light_source_settings = channel["LightSourcesSettings"]["LightSourceSettings"]
                    if isinstance(light_source_settings, dict):
                        # If it's a single dictionary, add its intensity
                        laser_intensity.append(light_source_settings["Intensity"])
                    elif isinstance(light_source_settings, list):
                        # If it's a list of dictionaries, add the first valid intensity that's not 'n/a'
                        for item in light_source_settings:
                            if item["Intensity"] != 'n/a':
                                laser_intensity.append(item["Intensity"])
                                break
                        else:
                            laser_intensity.append('n/a')  # If all entries are 'n/a', add 'n/a'


                colors = [channel["Color"] for channel in metadata["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["Channels"]["Channel"]]
            else:
                channel_names =[metadata["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]["Name"]]

                excitation_wavelengths = [metadata["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["Channels"]["Channel"]["ExcitationWavelength"]]

                exposure_times = [metadata["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["Channels"]["Channel"]["ExposureTime"]/1e9]

                n_light_sources = len(metadata["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["Channels"]["Channel"]["LightSourcesSettings"]["LightSourceSettings"])
                # print(n_light_sources)
                if n_light_sources > 1 and type(metadata["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["Channels"]["Channel"]["LightSourcesSettings"]["LightSourceSettings"]) == list:
                    laser_intensity = [channel["Intensity"] for channel in metadata["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["Channels"]["Channel"]["LightSourcesSettings"]["LightSourceSettings"] if channel["Intensity"] != "n/a"]

                else:
                    laser_intensity = [metadata["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["Channels"]["Channel"]["LightSourcesSettings"]["LightSourceSettings"]["Intensity"]]

                colors = [metadata["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["Channels"]["Channel"]["Color"]]
            print("\nColor Channels Found:")
            print(channel_names)
            
            print("\nExcitation Wavelengths (nm):")
            print(excitation_wavelengths)                

            print("\nExposure Time (seconds):")
            print(exposure_times)

            print("\nLaser Intensity:")
            print(laser_intensity)

    if get_filters:
        return metadata, n_channels, channel_names, excitation_wavelengths, exposure_times, laser_intensity, cutin_cutout_dict, colors
    else:
        return metadata, n_channels, channel_names, excitation_wavelengths, exposure_times, laser_intensity, colors
    

def parse_filename(file_path:str):
    """Function to parse the filename from a file path
    Args:
    file_path (str): The path to the file
    Returns:
    str: The parsed filename"""
    file_name = file_path.split('/')[-1]
    parsed_part = file_name.replace('.czi', '')
    return parsed_part


def determine_processing_order(all_file_paths, override_name_list = None):
    """Function to determine the order of processing based on the time of image acquisition.
    
    Args:
    all_file_paths (list): List of file paths

    Returns:
    np.array: The sorted image array stack from latest to earliest acquisition
    np.array: The divider indices for each file
    list: The sorted channel names
    dict: A dictionary with final_names as keys and index pairs as values
    """
    img_arrs, times, all_channel_names = [], [], []
    for file_path in all_file_paths:
        if file_path.endswith("tif"):
            arr = tiff_to_arr(file_path)
        else:
            with CziFile(file_path) as czi:
                arr = czi.asarray().squeeze()

        if len(arr.shape) == 2:
            arr = np.expand_dims(arr, axis=0)
        arr = arr[:, :1200, :1200]
        arr = arr.astype("int16")
        
        metadata, n_channels, channel_names, excitation_wavelengths, exposure_times, laser_intensity, _ = retrieve_czi_metadata(file_path)
        datetime = np.datetime64(metadata["ImageDocument"]['Metadata']['Information']['Image']['AcquisitionDateAndTime'])
        times.append(datetime)
        img_arrs.append(arr)
        all_channel_names.append(channel_names)
    if override_name_list is not None:
        all_channel_names = override_name_list

    result = [x for y, x in sorted(zip(times, img_arrs), reverse=True)]
    sorted_names = [x for y, x in sorted(zip(times, all_channel_names), reverse=True)]

    final_result = []
    final_names = []
    index_dict = {}
    current_idx = 0
    for arr, channel in zip(result, sorted_names):
        if len(arr.shape) >= 4 and len(channel) > 1:
            # print("HHHHHHHHHHHHHHHHHHHHHHHHH")
            for k in range(arr.shape[1]):
                sub_arr = arr[:, k:k+1, :, :].squeeze()
                sub_arr = sub_arr[:1200, :1200] if len(sub_arr.shape) == 2 else sub_arr[:, :1200, :1200]
                sub_channel = channel[k]
                final_names.append(sub_channel)
                final_result.append(sub_arr)
                index_dict[sub_channel] = (current_idx, current_idx + sub_arr.shape[0])
                current_idx += sub_arr.shape[0]
        else:
            # print("+++++++++++++++++++++++++")
            if type(channel) == list:
                channel = channel[0]
            final_names.append(channel)
            final_result.append(arr)
            index_dict[channel] = (current_idx, current_idx + arr.shape[0])
            current_idx += arr.shape[0]

    divider_idxs = [len(arr) for arr in final_result]
    divider_idxs = np.cumsum(divider_idxs)
    print(f"The determined order of analysis based on time of image acquisition is: {final_names}")

    return np.vstack(final_result), divider_idxs, final_names, index_dict




def plot_labeled_img_multi(image1:np.ndarray, positive_pos:np.ndarray, rain_pos:np.ndarray, 
                           channel_names:list, excitation_wavelengths:list, exposure_times:list, laser_intensity:list,
                             def_the_rain_sd:float, gamma:float=3, marker:str = "circ",
                               i:int=None, show_index:bool=False, radius = 8):
    """Function to plot a multi-channel image with labeled points
    
    Args:
    image1 (np.array): The image array
    positive_pos (np.array): The positive positions
    rain_pos (np.array): The rain positions
    channel_names (list): The channel names
    excitation_wavelengths (list): The excitation wavelengths
    exposure_times (list): The exposure times
    laser_intensity (list): The laser intensities
    def_the_rain_sd (int): Number of standard deviations used in define_the_rain
    gamma (int): The gamma value for contrast enhancement
    marker (str): The marker type for the points. Default is "circ"
    i (int): The channel index. Default is None --> Automatically set to 0
    show_index (bool): Whether to show the index of the points. Default is False

    Returns:
    None"""
    if i is None:
        i = 0
    lower_percentile = 0.1
    upper_percentile = 99.9
    vmin = np.percentile(image1, lower_percentile)
    vmax = np.percentile(image1, upper_percentile)

    stretched_image = np.clip((image1 - vmin) / (vmax - vmin), 0, 1)
    gamma = gamma
    contrast_enhanced_image = np.power(stretched_image, gamma)

    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['black', 'white'], N=128)

    fig, ax = plt.subplots(figsize=(13, 13))
    plt.imshow(contrast_enhanced_image, cmap=cmap)

    if marker == "circ":
        for x, y in positive_pos:
            circle = Circle((y, x), radius=radius, color='blue', fill=False)  
            ax.add_patch(circle)
        for x, y in rain_pos:
            circle = Circle((y, x), radius=radius, color='red', fill=False)  
            ax.add_patch(circle)
    else:
        plt.scatter(positive_pos[:, 1], positive_pos[:, 0], color='blue', s=3)
        plt.scatter(rain_pos[:, 1], rain_pos[:, 0], color='red', s=3)
        # label each point with index
        if show_index == True:
            for j, (x, y) in enumerate(positive_pos):
                plt.text(y, x, str(j), fontsize=9, color='lime')

    plt.title(f"Channel: {channel_names[i]}, Excitation Wavelength: {excitation_wavelengths[i]} (nm), Exposure Time: {exposure_times[i]} (s), Laser Intensity: {laser_intensity[i]}, Def The Rain Used: {def_the_rain_sd} Stdev(s)")

    plt.show()


    
from sklearn.cluster import MiniBatchKMeans


def define_the_rain_kmeans(fluorescence_vals: np.ndarray, n_SD: float = 3.0, rain: bool = True):
    """Function to define the rain using KMeans clustering with robustness to outliers

    Args:
    fluorescence_vals (np.array): The raw fluorescence values extracted
    n_SD (float): The number of standard deviations to use for thresholding. Default is 3.0
    rain (bool): Whether to consider rain in the clustering. Default is True.

    Returns:
    float: The positive threshold
    float: The rain threshold (if rain is True, otherwise None)
    """
    scaler = RobustScaler()
    reshaped_vals = fluorescence_vals.reshape(-1, 1)
    scaled_vals = scaler.fit_transform(reshaped_vals)

    kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, n_init=10, max_iter=800, init='k-means++').fit(scaled_vals)
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    labels = kmeans.labels_

    negative_cluster_mean = cluster_centers.min()
    positive_cluster_mean = cluster_centers.max()

    positive_cluster_std = np.std(reshaped_vals[labels == labels.max()])
    negative_cluster_std = np.std(reshaped_vals[labels == labels.min()])

    weighted_std = 0.5 * (positive_cluster_std + negative_cluster_std)

    positive_threshold = positive_cluster_mean - n_SD * weighted_std

    if rain:
        rain_threshold = negative_cluster_mean + n_SD * negative_cluster_std
        return positive_threshold, rain_threshold
    else:
        rain_threshold = negative_cluster_mean + n_SD * negative_cluster_std
        return rain_threshold, None



from sklearn.preprocessing import RobustScaler, StandardScaler

def define_the_rain(first_frame_img: np.ndarray, all_pos_seq: list, n_SD: float = 3.0, rain: bool = True, return_positive_fluorescence: bool = False, pix_range: int = 21):
    """Function that creates positive, rain, and negative partitions based on Global Tm Channel Start Intensity

    Args:
    first_frame_img (np.array): The first frame image
    all_pos_seq (list): The list of all positions
    n_SD (float): The number of standard deviations to use for thresholding. Default is 3.0
    rain (bool): Whether to consider rain in the clustering. Default is True.

    Returns:
    np.array: The positive indices
    np.array: The rain indices (if rain is True)
    np.array: The negative indices
    np.array: The positive positions
    np.array: The rain positions (if rain is True)
    np.array: The negative positions
    """
    image = gaussian_background_correction(first_frame_img)
    # image = first_frame_img
    raw_fluorescence_vals = generate_fluorescence_vs_time(
        img_arr=np.expand_dims(image, 0),
        pts_seq=np.expand_dims(all_pos_seq[0], 1),
        pix_range=pix_range,
        filter="None", sigma=12, gaussian=True)
    raw_fluorescence_vals = min_max_normalize(raw_fluorescence_vals, use_global_min_max=True)


    positive_threshold, rain_threshold = define_the_rain_kmeans(raw_fluorescence_vals, n_SD, rain=rain)

    if rain and positive_threshold < rain_threshold:
        positive_threshold = rain_threshold

    positives = raw_fluorescence_vals >= positive_threshold
    negatives = raw_fluorescence_vals < (rain_threshold if rain else positive_threshold)
    
    if rain:
        rain_mask = (raw_fluorescence_vals > rain_threshold) & (raw_fluorescence_vals <= positive_threshold)
    else:
        rain_mask = np.array([])

    positive_count = np.sum(positives)
    negative_count = np.sum(negatives)
    rain_count = np.sum(rain_mask) if rain else 0

    total_count = len(raw_fluorescence_vals)
    positive_percentage = positive_count / total_count * 100
    negative_percentage = negative_count / total_count * 100
    rain_percentage = rain_count / total_count * 100 if rain else 0

    plt.figure(figsize=(13, 10))
    plt.scatter(np.where(positives)[0], raw_fluorescence_vals[positives], color='blue', label='Positives', s=20, marker="*", linewidths=1)
    plt.scatter(np.where(negatives)[0], raw_fluorescence_vals[negatives], color='grey', label='Negatives', s=20, marker="x", linewidths=1)
    if rain:
        plt.scatter(np.where(rain_mask)[0], raw_fluorescence_vals[rain_mask], color='red', label='Rain', s=20, marker="+", linewidths=1)
        plt.axhline(y=rain_threshold, color='orange', linestyle='dotted', linewidth=2, label='Rain Threshold')
        plt.text(0.95, rain_threshold + (positive_threshold - rain_threshold) / 2, 
                 f'Rain: {rain_count} ({rain_percentage:.2f}%)', 
                 transform=plt.gca().transAxes, fontsize=12, color='red', verticalalignment='center', horizontalalignment='right')

    plt.axhline(y=positive_threshold, color='green', linestyle='dotted', linewidth=2, label='Positive Threshold')
    plt.text(0.95, positive_threshold + (max(raw_fluorescence_vals) - positive_threshold) * 0.05, 
             f'Positives: {positive_count} ({positive_percentage:.2f}%)', 
             transform=plt.gca().transAxes, fontsize=12, color='blue', verticalalignment='bottom', horizontalalignment='right')

    plt.text(0.95, (rain_threshold - (rain_threshold - min(raw_fluorescence_vals)) * 0.05) if rain else 0, 
             f'Negatives: {negative_count} ({negative_percentage:.2f}%)', 
             transform=plt.gca().transAxes, fontsize=12, color='grey', verticalalignment='top', horizontalalignment='right')

    plt.xlabel('Well Number')
    plt.ylabel('Normalized Start Intensity')
    plt.legend(loc='upper left')
    plt.title('Fluorescence Intensity Distribution')
    plt.show()

    positive_idxs = np.argwhere(positives.squeeze())
    negative_idxs = np.argwhere(negatives.squeeze())
    positive_pos = all_pos_seq[0][positive_idxs].squeeze()
    negative_pos = all_pos_seq[0][negative_idxs].squeeze()


    if rain:
        rain_idxs = np.argwhere(rain_mask.squeeze())
        rain_pos = all_pos_seq[0][rain_idxs].squeeze()
        if return_positive_fluorescence:
            return positive_idxs.squeeze(), rain_idxs.squeeze(), negative_idxs.squeeze(), positive_pos, rain_pos, negative_pos, raw_fluorescence_vals
        else:
            return positive_idxs.squeeze(), rain_idxs.squeeze(), negative_idxs.squeeze(), positive_pos, rain_pos, negative_pos
    else:
        dummy_rain_idx = np.array([], dtype="bool")
        dummy_rain_pos = np.array([])
        if return_positive_fluorescence:
            return positive_idxs.squeeze(), dummy_rain_idx, negative_idxs.squeeze(), positive_pos, dummy_rain_pos, negative_pos, raw_fluorescence_vals
        else:
            return positive_idxs.squeeze(), dummy_rain_idx, negative_idxs.squeeze(), positive_pos, dummy_rain_pos, negative_pos




def _find_well(image: np.ndarray, k0: int, k1: int, threshold: int = 0, var: float = 1.0,
               plot_hist: bool = False, first_frame: bool = False):

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    x = torch.as_tensor(image, dtype=torch.float32, device=device)     
    x4 = x.unsqueeze(0).unsqueeze(0)                                  

    g = gaussian_kernel(shape=(k0, k0), mean=0, cov=[[var, 0.0], [0.0, var]])
    if isinstance(g, np.ndarray):
        g_t = torch.from_numpy(g.astype(np.float32))
    else:
        g_t = g.to(torch.float32)
    g_t = g_t.to(device).unsqueeze(0).unsqueeze(0)                         

    y = F.conv2d(x4, g_t, padding=0).squeeze(0).squeeze(0)               

    pad = (k1 - 1) // 2
    y_max = F.max_pool2d(y.unsqueeze(0).unsqueeze(0), kernel_size=k1, stride=1, padding=pad)
    y_max = y_max.squeeze(0).squeeze(0)

    if first_frame:
        y_norm = (y - y.mean()) / (y.std().clamp_min(1e-8))
        mask = (y_norm > float(threshold)) & (y >= y_max)  
    else:
        mask = (y >= y_max)

    if plot_hist:
        import matplotlib.pyplot as plt
        y_norm = (y - y.mean()) / (y.std().clamp_min(1e-8))
        hist_data, _, _ = plt.hist(y_norm.detach().cpu().numpy().reshape(-1), bins=30, alpha=0.5, density=False)
        ylim = hist_data.max() if len(hist_data) else 1
        plt.vlines(float(threshold) if first_frame else 0.0, 0, ylim, color='red', label='Threshold')
        plt.title('Normalized Intensity Distribution')
        plt.xlabel('Intensity Z-Score')
        plt.ylabel('Sampled Frequency')
        plt.legend()
        plt.show()

    coords = torch.nonzero(mask, as_tuple=False)           # (N, 2), int64 on device
    if coords.numel() == 0:
        return np.empty((0, 2), dtype=np.int32)

    # Work on CPU for the selection step
    coords_cpu_i64 = coords.to('cpu', dtype=torch.int64).contiguous()

    # Use float only for distance computation
    coords_cpu_f = coords_cpu_i64.to(torch.float32)
    dists = torch.cdist(coords_cpu_f.unsqueeze(0), coords_cpu_f.unsqueeze(0), p=2).squeeze(0)

    keep = torch.zeros(coords_cpu_f.shape[0], dtype=torch.bool)
    for i in range(coords_cpu_f.shape[0]):
        if not (dists[i, keep] < k1).any():
            keep[i] = True

    # Avoid boolean indexing on MPS; do it on CPU.
    coords_kept_cpu = coords_cpu_i64[keep].contiguous()

    # Offset by kernel center (assumes odd k0; consider assert k0%2==1)
    peak_pos = (coords_kept_cpu + (k0 // 2)).to(torch.int32).contiguous().numpy()
    return peak_pos




def invert_image(img:np.ndarray):
    """Function to invert an image

    Args:
    img (np.array): The image array

    Returns:
    np.array: The inverted image"""
    max_val = np.max(img)
    inverted_image = max_val - img
    return inverted_image

def gaussian_background_correction(img:np.ndarray, sigma:int=50, radius:int=100):
    """Function to perform gaussian background correction on an image
    
    Args:
    img (np.array): The image array
    sigma (int): The sigma value for the gaussian filter. Default is 50
    radius (int): The radius value for the gaussian filter. Default is 100
    
    Returns:
    np.array: The background corrected image
    """

    blurred = gaussian_filter(img, sigma = sigma, radius=radius)
    background_mask = invert_image(blurred)
    result = background_mask+img
    return result
    
def gaussian_background_correction_div(img, sigma=20, radius=30):
    blurred = gaussian_filter(img, sigma = sigma, radius=radius)
    # background_mask = invert_image(blurred)
    result = img/blurred
    return result

def filter_wells_not_within_epsilon(all_wells:list, positives:np.ndarray, epsilon:int):
    """Function to filter wells that are not within epsilon of any point in the positives

    Args:
    all_wells (list): The list of all wells
    positives (np.array): The positive positions
    epsilon (int): The epsilon value

    Returns:
    list: The filtered wells
    """
    all_wells_np = np.array(all_wells)
    positives_np = np.array(positives)

    distances_squared = np.sum((all_wells_np[:, np.newaxis, :] - positives_np) ** 2, axis=2)

    outside_epsilon = distances_squared > epsilon ** 2
    wells_outside_epsilon = np.all(outside_epsilon, axis=1)
    return all_wells_np[wells_outside_epsilon].tolist()



def generate_pos_seq_new_no_tile(img_seq:np.ndarray, k0:int, k1:int, hist_threshold:int=None, 
                                 var:float=1.0, gamma:float = 2, enhance:bool = False, get_negative:bool=False,
                                   plot_hist:bool = False, first_frame:bool = True, 
                                   bkg_correct_radius:int=100, bkg_correct_sigma:int=50):
    """Wrapper function to generate the position sequence for a given image sequence
    
    Args:
    img_seq (np.array): The image sequence
    k0 (int): The kernel size for the first convolution
    k1 (int): The kernel size for the second convolution
    hist_threshold (int): The histogram threshold
    var (float): The variance value for the gaussian kernel. Default is 1.0
    gamma (float): The gamma value for contrast enhancement. Default is 2
    enhance (bool): Whether to enhance the image. Default is False
    get_negative (bool): Whether to get the negative. Default is False
    plot_hist (bool): Whether to plot the histogram. Default is False
    first_frame (bool): Whether it is the first frame. Default is True
    """

    if hist_threshold is None:
        threshold = 0
    else:
        threshold = hist_threshold
    seq_len = img_seq.shape[0]
    pos_seq = [None]*seq_len
    # for i in range(seq_len):
    for i in tqdm(range(seq_len), desc="Processing"):
        image = img_seq[i]
        # image = upsample_frame_pil(image, factor=upsampling_factor)
        if i>0 and get_negative==False:
            threshold = threshold-0.1
        if i == 0:
            first_frame = True
        else:
            first_frame = False
        if enhance == True:
            stretched_image = min_max_normalize(image, use_global_min_max=True)
            gamma = gamma
            image = np.power(stretched_image, gamma)

        if get_negative:
            # print(threshold)
            positive = find_well_new_no_tile(image, k0, k1, threshold=threshold, var=var, first_frame=True)
            all = find_well_new_no_tile(image, k0, k1, threshold=-10000, var=var, correct=True, first_frame=True)
            negative = filter_wells_not_within_epsilon(all, positive, epsilon=k1)
            # print(positive.shape, all.shape)
            if len(negative)==0:
                negative = all
            negative = np.stack(negative)
            pos_seq[i] = negative
        else:
            pos_seq[i] = find_well_new_no_tile(image, k0, k1, threshold=threshold, var=var, 
                                               plot_hist=plot_hist, first_frame=first_frame,
                                                 bkg_correct_radius=bkg_correct_radius, bkg_correct_sigma=bkg_correct_sigma)
    return pos_seq





def find_well_new_no_tile(image:np.ndarray, k0:int, k1:int, threshold:float, var:float=1.0, correct:bool = True, 
                          plot_hist:bool = False, first_frame:bool = False, 
                          bkg_correct_radius:int=100, bkg_correct_sigma:int=50):
    """Wrapper function to find the well positions in an image
    
    Args:
    image (np.array): The image array
    k0 (int): The kernel size for the first convolution
    k1 (int): The kernel size for the second convolution
    threshold (int): The threshold value
    var (float): The variance value for the gaussian kernel. Default is 1.0
    correct (bool): Whether to correct the image. Default is True
    plot_hist (bool): Whether to plot the histogram. Default is False
    first_frame (bool): Whether it is the first frame. Default is False
    
    Returns:
    np.array: The peak positions
    """
    if correct:
        # print("got here")
        corrected_img = gaussian_background_correction(image, radius=bkg_correct_radius, sigma=bkg_correct_sigma)
        # corrected_img = gaussian_background_correction_div(image, radius = bkg_correct_radius, sigma=bkg_correct_sigma)
    else:
        corrected_img = image   
    pos1 = _find_well(corrected_img, k0, k1, threshold=threshold, var=var, plot_hist = plot_hist, first_frame=first_frame)

    return pos1


def gaussian_kernel(shape:tuple, mean:float=0.0, cov:list=[[1.0, 0.0], [0.0, 1.0]]):
    """Function that generates a 2D Gaussian kernel
    
    Args:
    shape (tuple): The shape of the kernel
    mean (float): The mean value. Default is 0.0
    cov (list): The covariance matrix. Default is [[1.0, 0.0], [0.0, 1.0]]
    
    Returns:
    np.array: The 2D Gaussian kernel
    """
    center = (math.ceil(shape[0] / 2), math.ceil(shape[1] / 2))
    x = torch.linspace(-1, 1, steps=shape[0])
    y = torch.linspace(-1, 1, steps=shape[1])
    xv, yv = torch.meshgrid(x, y)
    pos = torch.stack([xv, yv], dim=2)
    mean = mean
    cov = torch.tensor(cov)
    cov_inv = torch.inverse(cov)
    det_cov = torch.det(cov)
    norm_factor = 1 / (2 * math.pi * torch.sqrt(det_cov))
    exponent = -0.5 * torch.einsum('...k,kl,...l->...', pos - mean, cov_inv, pos - mean)
    tensor = norm_factor * torch.exp(exponent)
    return (tensor*10)



def track_keypoints_multi_channel(keypoints_list:list, epsilon:float, divider_idxs:list):
    """Wrappeer Function to track keypoints across multiple channels
    
    Args:
    keypoints_list (list): The list of keypoints
    epsilon (int): The size of the search range
    divider_idxs (list): The indices for the begin/end of different channels, which serve as dividers
    
    Returns:
    np.array: The tracked position sequence
    """

    global_pos_seq = keypoints_list[:divider_idxs[0]]
    local_pos_seqs = [keypoints_list[divider_idxs[i]:divider_idxs[i+1]] for i in range(len(divider_idxs)-1)]
    
    reversed_local_pos_seqs = [local_pos_seq[::-1] for local_pos_seq in local_pos_seqs]
    

    merged_local_pos_seqs = [global_pos_seq[0]] + list(itertools.chain(*reversed_local_pos_seqs))
    
    tracked_global_pos_seq = track_keypoints(global_pos_seq, epsilon)
    tracked_local_pos_seq = track_keypoints(merged_local_pos_seqs, epsilon)

    tracked_local_pos_seq = tracked_local_pos_seq[:, 1:, :]

    # Reverse the local pos seqs back to their original order by revsersing its second axis; keep the result as an array of shape (num_kps, num_frames, 2)
    local_seqs_lengths = [divider_idxs[i+1] - divider_idxs[i] for i in range(len(divider_idxs)-1)]
    start_idx = 0
    corrected_tracked_local_pos_seq = []
    for length in local_seqs_lengths:
        end_idx = start_idx + length
        corrected_tracked_local_pos_seq.append(tracked_local_pos_seq[:, start_idx:end_idx, :][:, ::-1, :])
        start_idx = end_idx
    
    if len(corrected_tracked_local_pos_seq) > 0:
        corrected_tracked_local_pos_seq = np.concatenate(corrected_tracked_local_pos_seq, axis=1)
    if len(divider_idxs) > 1:
        tracked_pos_seq = np.concatenate([tracked_global_pos_seq, corrected_tracked_local_pos_seq], axis=1)
    else:
        tracked_pos_seq = tracked_global_pos_seq
    return tracked_pos_seq

def track_keypoints(keypoints_list:list, epsilon:float):
    """Internal function to track keypoints across frames
    
    Args:
    keypoints_list (list): The list of keypoints
    epsilon (float): The size fo the search range
    
    Returns:
    np.array: The tracked position sequence"""

    num_frames = len(keypoints_list)

    num_kps = len(keypoints_list[0])
    

    # Shape = (num_frames, num_kps, 2), filled initially with the first frame keypoints
    tracked_kps = np.full((num_frames, num_kps, 2), keypoints_list[0])
    
    start_kdtree_time = time.time()
    #  KD-Trees for each frame
    kdtrees = [KDTree(keypoints) for keypoints in keypoints_list]
    end_kdtree_time = time.time()
    print(f"KD-Trees creation time: {end_kdtree_time - start_kdtree_time:.4f} seconds")
    
    start_tracking_time = time.time()
    for i in tqdm(range(1, num_frames), desc="Processing Frames"):
        # Get the keypoints from the previous frame
        prev_kps = tracked_kps[i - 1]
        
        # KDTree of the current frame
        curr_tree = kdtrees[i]
    
        for j, prev_kp in enumerate(prev_kps):
            # Query the KDTree for the closest keypoint within the epsilon distance
            dist, index = curr_tree.query(prev_kp, distance_upper_bound=epsilon)
            
            # If there's a keypoint within epsilon distance, update it in the tracked keypoints
            if dist < epsilon:
                tracked_kps[i, j] = keypoints_list[i][index]
            else:
                tracked_kps[i, j] = tracked_kps[i - 1, j]
    end_tracking_time = time.time()
    print(f"Searching time: {end_tracking_time - start_tracking_time:.4f} seconds")
    
    return np.transpose(tracked_kps, (1, 0, 2))



def filter_keypoints(image_array:np.ndarray, pts_seq:np.ndarray, R:int):
    """
    Funciton that filters the edge points in the key center points. A point is filter if it goes over the edge of the image at any frame
    
    Args:
    - image_array (np.array): The image array of shape (N, M, M), where N is the nunber of frames and M is the image width/height
    - pts_seq (np.array): The key points sequence of shape (K, N, 2), where K is the number of key center points, N is the number of frames, and 2 is the x-y coordinates
    - R (int): The radius of tolarance
    
    Returns:
    - np.array: The filtered key points sequence
    - np.array: The invalid indices
    - np.array: The valid indices
    """
    N, height, width = image_array.shape
    K = pts_seq.shape[0]
    
    valid_points = np.ones(K, dtype=bool)
    
    for k in tqdm(range(K), desc="Filtering Edge Points"):
        for n in range(N):
            x, y = pts_seq[k, n]
            if (x - R < 0 or x + R >= width or y - R < 0 or y + R >= height):
                valid_points[k] = False
                break

    masked_pts_seq = ma.masked_array(pts_seq, mask=np.broadcast_to(~valid_points[:, None, None], pts_seq.shape))
    
    filtered_pts_seq = masked_pts_seq.compressed().reshape(-1, N, 2)
    invalid_indices = np.where(valid_points == 0)[0]
    valid_indices = np.where(valid_points == 1)[0]
    
    return filtered_pts_seq, invalid_indices, valid_indices


def get_cropped_image_and_adjusted_pos(img, kp_x, kp_y, m):
    half_m = m // 2
    img_height, img_width = img.shape[:2]

    # Ensure the top-left corner doesn't go out of bounds
    top_left_x = max(0, kp_x - half_m)
    top_left_y = max(0, kp_y - half_m)

    # Ensure the bottom-right corner doesn't exceed the image size
    bottom_right_x = min(img_width, top_left_x + m)
    bottom_right_y = min(img_height, top_left_y + m)

    # Adjust crop size if it exceeds image boundaries
    cropped_img = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # Adjust keypoints relative to the top-left corner of the cropped image
    adjusted_x = kp_x - top_left_x
    adjusted_y = kp_y - top_left_y

    return cropped_img, adjusted_x, adjusted_y

def plot_keypoint_tracking(image_array:np.ndarray, pts_seq:np.ndarray, keypoint_index:int, m:int, alignment_correction:bool =None, 
                           save_video:bool=False, video_filename:str='tracking.mp4', radius=8):
    """
    Visualizes the key point tracking before and after alignment with circles around the actual keypoints.
    Optionally saves the entire series as an mp4 video.

    Args:
    - image_array (np.array): The image array of shape (N, M, M), where N is the nunber of frames and M is the image width/height
    - pts_seq (np.array): The key points sequence of shape (K, N, 2), where K is the number of key center points, N is the number of frames, and 2 is the x-y coordinates
    - keypoint_index (int): The index of the key point to visualize.
    - m (int): Size of the zoom area (m x m).
    - alignment_correction (np.array or None): The corrected positions of key points, same shape as pts_seq.
    - save_video (bool): Whether to save the series as a video. Default is False.
    - video_filename (str): Filename for the saved video. Default is 'keypoint_tracking.mp4'.
    
    Displays:
    - The (m x m) area around the key point at each frame, before and after alignment.
    """
    N = image_array.shape[0]

    def get_cropped_image_and_adjusted_pos(img, kp_x, kp_y, m):
        half_m = m // 2
        top_left_x = max(0, kp_x - half_m)
        top_left_y = max(0, kp_y - half_m)
        cropped_img = img[top_left_y:top_left_y+m, top_left_x:top_left_x+m]
        adjusted_x = kp_x - top_left_x
        adjusted_y = kp_y - top_left_y
        return cropped_img, adjusted_x, adjusted_y

    def plot_patches(ax, frame, keypoints, color, title):
        y, x = int(keypoints[frame, 0]), int(keypoints[frame, 1])
        cropped_img, adj_x, adj_y = get_cropped_image_and_adjusted_pos(image_array[frame], x, y, m)
        
        ax.imshow(cropped_img, cmap='gray')
        circ = Circle((adj_x, adj_y), radius=radius, edgecolor=color, facecolor='none')
        ax.add_patch(circ)
        ax.set_title(f"{title} - Frame {frame}")

    if save_video:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        with writer.saving(fig, video_filename, 100):
            for frame in tqdm(range(N), desc="Saving video"):
                ax1.clear()
                ax2.clear()
                plot_patches(ax1, frame, pts_seq[keypoint_index], 'red', "Before Alignment")
                plot_patches(ax2, frame, alignment_correction[keypoint_index], 'lime', "After Alignment")
                writer.grab_frame()
        plt.close(fig)
        print(f"Video saved as {video_filename}")
    else:
        output_before = widgets.Output()
        output_after = widgets.Output()

        def update_image(change):
            frame = change['new']
            with output_before:
                output_before.clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(4, 4))
                plot_patches(ax, frame, pts_seq[keypoint_index], 'red', "Before Alignment")
                plt.show()
            with output_after:
                output_after.clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(4, 4))
                plot_patches(ax, frame, alignment_correction[keypoint_index], 'lime', "After Alignment")
                plt.show()

        slider = widgets.IntSlider(min=0, max=N-1, step=1, description='Frame:', layout=widgets.Layout(width='800px'))
        slider.observe(update_image, names='value')
        display(VBox([slider, HBox([output_before, output_after])]))
        with output_before:
            fig, ax = plt.subplots(figsize=(4, 4))
            plot_patches(ax, 0, pts_seq[keypoint_index], 'red', "Before Alignment")
            plt.show()
        with output_after:
            fig, ax = plt.subplots(figsize=(4, 4))
            plot_patches(ax, 0, alignment_correction[keypoint_index], 'lime', "After Alignment")
            plt.show()





def savgol(pixel_values:np.ndarray, window_length:int, polyorder:int, deriv:int=0, delta:int=1, mode:str="nearest"):
    """Wrapper function that applies the Savitzky-Golay filter to a 1D or 2D array of pixel values
    
    Args:
    - pixel_values (np.array): The pixel values to be smoothed
    - window_length (int): The length of the filter window
    - polyorder (int): The polynomial order
    - deriv (int): The order of the derivative to compute. Default is 0, which means no derivative
    - delta (int): The spacing of the samples. Default is 1
    - mode (str): The boundary mode. Default is 'nearest'
    
    Returns:
    - np.array: The smoothed pixel values"""

    # Ensure the window length is odd and greater than the polynomial order
    assert window_length % 2 == 1 and window_length > polyorder

    # Check if the input is 1D or 2D
    if pixel_values.ndim == 1:
        # Apply the Savitzky-Golay filter directly to the 1D pixel values
        return savgol_filter(pixel_values, window_length, polyorder, deriv, delta)
    elif pixel_values.ndim == 2:
        smoothed_pixel_values = np.zeros_like(pixel_values)
        for i in range(pixel_values.shape[0]):
            smoothed_pixel_values[i, :] = savgol_filter(pixel_values[i, :], window_length, polyorder, deriv, delta, mode=mode)
        return smoothed_pixel_values
    else:
        raise ValueError("Input array must be 1D or 2D.")



def generate_fluorescence_vs_time(img_arr:np.ndarray, pts_seq:np.ndarray, pix_range:int, window_length:int = 9, polyorder:int=3, filter:str=None, gaussian:bool =False, sigma:int = 12):
    """Wrapper function to generate the fluorescence vs time plot
    
    Args:
    - img_arr (np.array): The image array of shape (N, M, M), where N is the nunber of frames and M is the image width/height
    - pts_seq (np.array): The key points sequence of shape (K, N, 2), where K is the number of key center points, N is the number of frames, and 2 is the x-y coordinates
    - pix_range (int): The radial pixel range to extract the average pixel values

    Returns:
    - np.array: The smoothed pixel values
    """
    pix_vals = get_average_pixel_values_circ(images = img_arr, keypoint_positions = pts_seq, pix_range=pix_range, sigma=sigma, gaussian=gaussian)
    if filter == "sav_gol":
        smoothed_pix_vals = savgol(pix_vals, window_length, polyorder)
    else:
        smoothed_pix_vals = pix_vals
    return smoothed_pix_vals


def get_circular_offsets_and_weights(radius:int, sigma:int, gaussian:bool=True):
    """Internal function to get the circular offsets and weights for a given radius and sigma
    
    Args:
    - radius (int): The radius value for the gaussian kernel if gaussian is True
    - sigma (int): The sigma value for the gaussian kernel if gaussian is True
    - gaussian (bool): Whether to use a gaussian kernel. Default is True
    
    Returns:
    - np.array: The circular offsets
    - np.array: The weights
    """
    offsets = []
    weights = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if i**2 + j**2 <= radius**2:
                offsets.append((i, j))
                if gaussian:
                    weight = np.exp(-(i**2 + j**2) / (2 * sigma**2))
                else:
                    weight = 1.0  # Ensure weight is a float
                weights.append(weight)
    offsets = np.array(offsets)
    weights = np.array(weights, dtype=float)  # Ensure weights are float for proper division
    weights /= np.sum(weights) 
    return offsets, weights

def get_average_pixel_values_circ(images:np.ndarray, keypoint_positions:np.ndarray, pix_range:int, sigma:int=12, gaussian:bool=True):
    """Internal wrapper function to get the average pixel values for a given circular range
    
    Args:
    - images (np.array): The image array of shape (N, M, M), where N is the number of frames and M is the image width/height
    - keypoint_positions (np.array): The key points sequence of shape (K, N, 2), where K is the number of key center points, N is the number of frames, and 2 is the x-y coordinates
    - pix_range (int): The radial pixel range to extract the average pixel values
    - sigma (int): The sigma value for the gaussian kernel if gaussian is True
    - gaussian (bool): Whether to use a gaussian kernel. Default is True
    
    Returns:
    - np.array: The average pixel values
    """
    radius = pix_range // 2

    offsets, weights = get_circular_offsets_and_weights(radius, sigma, gaussian)

    y_positions = keypoint_positions[:, :, 0].astype(int)
    x_positions = keypoint_positions[:, :, 1].astype(int)

    average_pixel_values = np.zeros((keypoint_positions.shape[0], images.shape[0]))

    y_positions_exp = np.expand_dims(y_positions, -1)
    x_positions_exp = np.expand_dims(x_positions, -1)
    
    y_neighbors = y_positions_exp + offsets[:, 0]
    x_neighbors = x_positions_exp + offsets[:, 1]
    
    mask_y = (y_neighbors >= 0) & (y_neighbors < images.shape[1])
    mask_x = (x_neighbors >= 0) & (x_neighbors < images.shape[2])
    mask = mask_y & mask_x
    
    y_neighbors = np.clip(y_neighbors, 0, images.shape[1]-1)
    x_neighbors = np.clip(x_neighbors, 0, images.shape[2]-1)

    for j in tqdm(range(images.shape[0]), desc="Processing images"):
        gathered_pixels = images[j, y_neighbors[:, j], x_neighbors[:, j]].astype(float)
        
        gathered_pixels[~mask[:, j]] = np.nan

        valid_weights = weights.reshape(-1, 1)
        valid_weights = np.tile(valid_weights, (1, keypoint_positions.shape[0]))
        valid_weights = valid_weights.T
        
        valid_weights[~mask[:, j]] = np.nan

        weighted_sum = np.nansum(gathered_pixels * valid_weights, axis=-1)
        weight_sum = np.nansum(valid_weights, axis=-1)
        average_pixel_values[:, j] = weighted_sum / weight_sum

    return average_pixel_values



def min_max_normalize(arr: np.ndarray, use_global_min_max: bool = False, return_min_max: bool = False, 
                      use_predefined_min_max_param: bool = False, predefined_min: float = None, predefined_max: float = None):
    """Function to perform min-max normalization on a 1D or 2D array
    
    Args:
    - arr (np.array): The input array
    - use_global_min_max (bool): Whether to use global min and max statistics when doing normalization. Default is False
    - return_min_max (bool): Whether to return the min and max values used for normalization. Default is False
    - use_predefined_min_max_param (bool): Whether to use predefined min and max values for normalization. Default is False
    - predefined_min (float): The predefined minimum value for normalization. Default is None
    - predefined_max (float): The predefined maximum value for normalization. Default is None
    
    Returns:
    - np.array: The normalized array
    - tuple: The min and max values (only if return_min_max is True)
    """
    arr = np.array(arr)
    
    if use_predefined_min_max_param:
        if predefined_min is None or predefined_max is None:
            raise ValueError("Predefined min and max values must be provided when use_predefined_min_max_param is True.")
        min_value = predefined_min
        max_value = predefined_max
    else:
        if arr.ndim == 1:
            min_value = arr.min()
            max_value = arr.max()
        else:
            if use_global_min_max:
                min_value = arr.min()
                max_value = arr.max()
            else:
                min_values = arr.min(axis=1, keepdims=True)
                max_values = arr.max(axis=1, keepdims=True)
                norm_arr = (arr - min_values) / (max_values - min_values) 
                if return_min_max:
                    return norm_arr, (min_values, max_values)
                return norm_arr
    
    norm_arr = (arr - min_value) / (max_value - min_value)
    
    if return_min_max:
        return norm_arr, (min_value, max_value)
    
    return norm_arr

def compute_Tm(initial_T:float, peak_frame_idx:int, rate_per_min:float, exposure_in_sec:float, max:float=90):
    """Function to compute the melting temperature (Tm) based on the peak frame index, heating rate, and exposure time
    
    Args:
    - initial_T (float): The initial temperature
    - peak_frame_idx (int): The index of the peak frame
    - rate_per_min (float): The heating rate per minute
    - exposure_in_sec (float): The exposure time in seconds
    - max (float): The maximum temperature reached in the experiment. Default is 90
    
    Returns:
    - float: The computed melting temperature"""
    tot_t = peak_frame_idx*exposure_in_sec
    tot_t = tot_t/60
    delta_T = rate_per_min*tot_t
    Tm = initial_T+delta_T
    if Tm>=max:
        Tm = max
    return Tm


def compute_frame_idx(target_T, initial_T, rate_per_min, exposure_in_sec):
    """
    Helper function to compute the frame index for a given target temperature.
    """
    delta_T = target_T - initial_T
    if delta_T <= 0:
        return 0
    tot_t_min = delta_T / rate_per_min
    tot_t_sec = tot_t_min * 60
    frame_idx = tot_t_sec / exposure_in_sec
    return int(frame_idx)

def select_by_temp_range(temp_range: tuple, initial_T: float, rate_per_min: float, exposure_in_sec: float, max_temp: float = 90):
    """
    Function to select the index range based on a given temperature range.

    Args:
    - temp_range (tuple): The temperature range as a tuple (min_temp, max_temp).
    - initial_T (float): The initial temperature.
    - rate_per_min (float): The heating rate per minute.
    - exposure_in_sec (float): The exposure time in seconds.
    - max_temp (float): The maximum temperature reached in the experiment. Default is 90.

    Returns:
    - tuple: The calculated index range (min_index, max_index).
    """
    def compute_frame_idx(target_T, initial_T, rate_per_min, exposure_in_sec):
        """
        Helper function to compute the frame index for a given target temperature.
        """
        delta_T = target_T - initial_T
        if delta_T <= 0:
            return 0
        tot_t_min = delta_T / rate_per_min
        tot_t_sec = tot_t_min * 60
        frame_idx = tot_t_sec / exposure_in_sec
        return int(frame_idx)
    
    min_temp, max_temp = temp_range
    min_temp = max(min_temp, initial_T)
    max_temp = min(max_temp, max_temp)
    
    min_index = compute_frame_idx(min_temp, initial_T, rate_per_min, exposure_in_sec)
    max_index = compute_frame_idx(max_temp, initial_T, rate_per_min, exposure_in_sec)
    
    return min_index, max_index



def moving_average(x:np.ndarray, window:int):
    """Fucntion that computes the moving average of a 1D or 2D array

    Args:
    - x (np.array): The input array
    - window (int): The window size for the moving average

    Returns:
    - np.array: The moving average of the input array
    """

    if x.ndim == 1:
        return np.convolve(x, np.ones(window), 'same') / window
    else:
        convolved = np.zeros((x.shape[0], x.shape[1]))
        for i in range(x.shape[0]):
            convolved[i] = np.convolve(x[i], np.ones(window), 'same') / window
        
        return convolved


def snr_moving_avg(array:np.ndarray, window:int, normalize:bool=True, avg:bool=True):
    """Function that estimates the average SNR when the underlying true signal is assuemd to be the moving average
    
    Args:
    - array (np.array): The input data array
    - window (int): The window size for the moving average
    - normalize (bool): Whether to normalize the array. Default is True
    - avg (bool): Whether to return the average SNR. Default is True
    
    Returns:
    - float: The average SNR value when the true signal is assumed to be the
    """
    if normalize == True:
        array = min_max_normalize(array)

    p_signal = moving_average(array, window)
    p_noise = np.sqrt(np.abs(array-p_signal)**2)

    p_noise += 1e-10
    snr = 10 * np.log10(np.abs(p_signal / p_noise))

    if avg == True:
        return np.round(np.mean(snr),4)
    else:
        return np.round(snr,4)





def get_noise_floor(normalized_array: np.ndarray, before_nth_frame: int = 50, after_nth_frame = None, tm_xticks: list = None, tm_temps: list = None, plot: bool = True, compute_SD =False, n_SD = 1):
    """Function that estimates the base noise level for peak finding purposes
    
    Args:
    - normalized_array (np.array): The normalized array
    - before_nth_frame (int): The frame index before which the noise floor is computed. Default is 50
    - tm_xticks (list): The x-ticks for the temperature plot
    - tm_temps (list): The temperature values for the x-ticks
    - plot (bool): Whether to plot the noise floor. Default is True
    - n_SD (int): Number of standard deviations for the upper confidence interval. Default is 1
    
    Returns:
    - Tuple[np.ndarray, np.ndarray, Tuple[float, float]]: The estimated noise floor array, the upper confidence interval, and the slope and intercept of the upper confidence interval line
    """
    # print("got here")
    if after_nth_frame is None:
        after_nth_frame = -before_nth_frame
    # print(after_nth_frame)

    start_points = normalized_array[:, :before_nth_frame].flatten()
    end_points = normalized_array[:, after_nth_frame:].flatten()

    start_points = np.expand_dims(np.median(start_points, axis=0), axis=0)
    end_points = np.expand_dims(np.median(end_points, axis=0), axis=0)

    y_vals = np.concatenate((start_points, end_points))
    x_vals = np.concatenate((np.zeros(start_points.shape), np.ones(end_points.shape) * (normalized_array.shape[1] - 1)))
    
    slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
    noise_floor = slope * np.arange(normalized_array.shape[1]) + intercept
    
    std_dev = np.std(normalized_array - noise_floor, axis=1).mean()

    upper_confidence_interval = noise_floor + n_SD*std_dev

    slope_uc, intercept_uc, r_value_uc, p_value_uc, std_err_uc = linregress(np.arange(normalized_array.shape[1]), noise_floor)

    if plot:
        plt.figure(figsize=(10, 6))
        for signal in normalized_array:
            plt.plot(signal, color='cornflowerblue', alpha=0.1)
        plt.xlabel("Frames")
        plt.ylabel("Normalized -dF/dT")
        plt.title("Noise Floor Visualization")
        plt.plot(noise_floor, color='red', linestyle='dotted', linewidth=3, label='Noise Floor')
        plt.axvline(x=before_nth_frame, color='green', linestyle='dotted', linewidth=2, label='Before nth Frame')
        if after_nth_frame is not None:
            plt.axvline(x=after_nth_frame, color='green', linestyle='dotted', linewidth=2, label='After nth Frame')
        if compute_SD:
            plt.plot(upper_confidence_interval, color='orange', linestyle='dotted', linewidth=3, label= f'Noise floor + ({n_SD} SD)')
        plt.text(0, noise_floor[0] + 0.1, s=f'Start Noise = {noise_floor[0]:.3f}', fontsize=12, color='red', verticalalignment='bottom', horizontalalignment='left')
        plt.text(len(noise_floor) - 1, noise_floor[-1] + 0.1, s=f'End Noise = {noise_floor[-1]:.3f}', fontsize=12, color='red', verticalalignment='bottom', horizontalalignment='right')
        plt.legend()
        if tm_xticks is not None and tm_temps is not None:
            ax = plt.gca()
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(tm_xticks)
            ax2.set_xticklabels(tm_temps, rotation=45)
            ax2.set_xlabel("Temperature (°C)")
        plt.show()
    if not compute_SD:
        upper_confidence_interval = None
        
    return noise_floor, (slope_uc, intercept_uc), upper_confidence_interval[0]




def get_Tm(mcs: np.ndarray, k_peaks: int, first_frame_T: float, heating_rate_per_min: float, img_series_gap_time: float, noise_floor: np.ndarray, noise_floor_params: tuple, 
           tm_temps:np.array, tm_xticks:np.array, max_temp: float = 90.0, widths = 5,
           weight = 0.5, ext_df=None, ext_idx=None, plot=False, return_new_noise_floor=False, height_tolerance=0.15, use_upper_conf=False, ignore_first_nFrames=0):
    """Function that computes the melting temperature (Tm) values from the Global Tm Melting curve data
    
    Args:
    - mcs (np.array): The melting curve data
    - k_peaks (int): The expected number of peaks to find
    - first_frame_T (float): The initial temperature
    - heating_rate_per_min (float): The heating rate per minute
    - img_series_gap_time (float): The exposure time in seconds
    - noise_floor (float): The estimated noise floor value
    - noise_floor_params (tuple): The slope and intercept for the noise floor
    - ext_df (pd.DataFrame): The external dataframe; left for past version compatibility
    - ext_idx (int): The external index; left for past version compatibility
    - plot (bool): Whether to plot the melting curves and noise floor
    
    Returns:
    - np.array: The computed Tm values for valid pairs where the first peak height is greater than the second
    - list: The indices of the curves with only one peak
    - list: The indices of the curves with exactly two peaks and first peak height greater than the second
    - list: The indices of the curves with more than two peaks
    - list: The peak heights for signals with exactly two peaks and first peak height greater than the second
    - np.array: The valid Tm indices
    """

    passed_in_slope, passed_in_intercept = noise_floor_params

    one_peaks = []
    valid_two_peaks = []
    more_than_two_peaks = []
    valid_peak_heights_two_peaks = []
    
    if mcs is None:
        return
    
    if len(mcs.shape) == 1:
        mcs = mcs[np.newaxis, :]
    
    Tm_indices = np.zeros((mcs.shape[0], k_peaks))
    
    for i in range(mcs.shape[0]):
        peaks, properties = find_peaks(mcs[i], width=5, height=noise_floor)
        peak_indices = peaks
        peak_heights = properties["peak_heights"]
        
        if len(peak_indices) < k_peaks:
            if len(peak_indices) == 1:
                one_peaks.append(i)
            if ext_df is not None:
                ext_df.loc[ext_idx, 'Valid'] = 0
            continue
        elif len(peak_indices) == k_peaks:
            if peak_heights[0] > peak_heights[1] - height_tolerance:
                Tm_indices[i, :len(peak_indices)] = peak_indices
                valid_two_peaks.append(i)
                valid_peak_heights_two_peaks.append(peak_heights)
        else:
            more_than_two_peaks.append(i)
            if ext_df is not None:
                ext_df.loc[ext_idx, 'Valid'] = 0
            continue
        
        Tm_indices[i, :len(peak_indices)] = peak_indices
    
    valid_Tms = np.zeros((len(valid_two_peaks), k_peaks))    
    for idx, i in enumerate(valid_two_peaks):
        for j in range(k_peaks):
            valid_Tms[idx, j] = compute_Tm(initial_T=first_frame_T, peak_frame_idx=Tm_indices[i, j],
                                            rate_per_min=heating_rate_per_min, exposure_in_sec=img_series_gap_time, max=max_temp)
    valid_Tm_indices = Tm_indices[valid_two_peaks]

    flattened_valid_Tm_indices = valid_Tm_indices.flatten()
    flattened_valid_peak_heights = np.array(valid_peak_heights_two_peaks).flatten()
    
    # Compute the slope and intercept of the fitted line
    if len(flattened_valid_Tm_indices) > 1:
        fitted_slope, fitted_intercept = np.polyfit(flattened_valid_Tm_indices, flattened_valid_peak_heights, 1)
    else:
        print("pleaes double check the input noise floor")
        fitted_slope = passed_in_slope
        fitted_intercept = passed_in_intercept
    
    if use_upper_conf:
        passed_in_intercept = noise_floor
    


    weight = weight
    new_slope = weight * passed_in_slope + (1 - weight) * fitted_slope
    new_intercept = weight * passed_in_intercept + (1 - weight) * fitted_intercept

    new_noise_floor = new_slope * np.arange(mcs.shape[1]) + new_intercept

    one_peaks = []
    valid_two_peaks = []
    more_than_two_peaks = []
    valid_peak_heights_two_peaks = []
    front_high_back_low = []
    Tm_indices = np.zeros((mcs.shape[0], k_peaks))
    
    # print(mcs.shape[1])
    
    prominence_arr = generate_variable_threshold(signal_length=mcs.shape[1],
                                                           low_threshold_segments=[(ignore_first_nFrames,mcs.shape[1]-1)],
                                                           low_threshold_value=0,
                                                           high_threshold_value=1)

    # print(prominence_arr)
    
    for i in range(mcs.shape[0]):
        peaks, properties = find_peaks(mcs[i], width=widths, height=new_noise_floor, prominence=prominence_arr)
        peak_indices = peaks
        peak_heights = properties["peak_heights"]
        
        if len(peak_indices) < k_peaks:
            if len(peak_indices) == 1:
                one_peaks.append(i)
            if ext_df is not None:
                ext_df.loc[ext_idx, 'Valid'] = 0
            continue
        elif len(peak_indices) == k_peaks:
            if peak_heights[0] >= peak_heights[1]-height_tolerance:
                Tm_indices[i, :len(peak_indices)] = peak_indices
                valid_two_peaks.append(i)
                valid_peak_heights_two_peaks.append(peak_heights)
            else:
                front_high_back_low.append(i)
        else:
            more_than_two_peaks.append(i)
            if ext_df is not None:
                ext_df.loc[ext_idx, 'Valid'] = 0
            continue
        
        Tm_indices[i, :len(peak_indices)] = peak_indices
    
    if plot:
        plt.figure(figsize=(10, 6))
        tick_interval = 25
        ticks = np.arange(0, mcs.shape[1], tick_interval)
        labels = np.round(np.linspace(first_frame_T, first_frame_T + heating_rate_per_min * img_series_gap_time * mcs.shape[1] / 60, mcs.shape[1]), 1)[::tick_interval]
        
        for i in range(mcs.shape[0]):
            plt.plot(np.arange(len(mcs[i])), mcs[i], alpha=0.05)
            # plt.xticks(ticks=ticks, labels=labels)
            plt.xticks(ticks = tm_xticks, labels = tm_temps)

            if i in valid_two_peaks:
                Tm_indices_int = Tm_indices[i].astype(int) 
                plt.plot(Tm_indices_int, mcs[i][Tm_indices_int], "*", markersize=3, color="black")  # Use Tm_indices_int as indices
        
        plt.plot(new_noise_floor, label="New Noise Floor", color="red", linestyle="--")
        plt.xlabel("Temperatures")
        plt.ylabel("Normalized -dF/dT")
        plt.title("Global Tm Melting Curves")
        plt.legend()
        plt.show()


    valid_Tms = np.zeros((len(valid_two_peaks), k_peaks))    
    for idx, i in enumerate(valid_two_peaks):
        for j in range(k_peaks):
            valid_Tms[idx, j] = compute_Tm(initial_T=first_frame_T, peak_frame_idx=Tm_indices[i, j], 
                                           rate_per_min=heating_rate_per_min, exposure_in_sec=img_series_gap_time, max = max_temp)
    valid_Tm_indices = Tm_indices[valid_two_peaks]
    
    if return_new_noise_floor:
        return valid_Tms, one_peaks, valid_two_peaks, more_than_two_peaks, valid_peak_heights_two_peaks, valid_Tm_indices, new_noise_floor,front_high_back_low
    else:
        return valid_Tms, one_peaks, valid_two_peaks, more_than_two_peaks, valid_peak_heights_two_peaks, valid_Tm_indices




def get_Tm_lvl2(global_tm_bkg_subtracted: np.ndarray, one_peak_idxs:list, k_peaks: int, first_frame_T: float, heating_rate_per_min: float, img_series_gap_time: float, 
                 tm_temps:np.array, tm_xticks:np.array, max_temp: float = 90.0, n_SD=1, widths = 5,
           weight = 0.5, ext_df=None, ext_idx=None, plot=False, return_new_noise_floor=False, height_tolerance=0.15, new_window_length = 31, use_upper_conf=True, before_nth_frame=10):

    if len(one_peak_idxs) == 0:
        return np.array([]), [], [], [], [], [], [], []
    
    one_peak_curves = global_tm_bkg_subtracted[one_peak_idxs]
    print(one_peak_curves.shape)
    one_peaks_derivs = min_max_normalize(-1*savgol_filter(one_peak_curves, window_length=new_window_length, 
                                                                     polyorder=2, deriv=1,mode='nearest'), use_global_min_max=True)
        
    global_tm_noise_floor, global_noise_floor_params, upper_confid = get_noise_floor(normalized_array = one_peaks_derivs, before_nth_frame=before_nth_frame, 
                                                plot=True, tm_temps=None, tm_xticks=None, n_SD=n_SD, compute_SD=True)
    
    valid_Tms, one_peaks, valid_two_peaks, more_than_two_peaks, valid_peak_heights_two_peaks, valid_Tm_indices, new_noise_floor,front_high_back_low = get_Tm(mcs = one_peaks_derivs, k_peaks=2, 
                                                        first_frame_T=first_frame_T, heating_rate_per_min = heating_rate_per_min, img_series_gap_time = img_series_gap_time, noise_floor = upper_confid,
                                                        tm_temps=tm_temps, tm_xticks=tm_xticks, max_temp = max_temp, widths = widths,
                                                        noise_floor_params=global_noise_floor_params, return_new_noise_floor=return_new_noise_floor,use_upper_conf=use_upper_conf,
                                                                    plot=plot, weight = weight, height_tolerance=height_tolerance)
    one_peak_idxs = np.array(one_peak_idxs)
    one_peaks = one_peak_idxs[one_peaks].tolist()
    valid_two_peaks = one_peak_idxs[valid_two_peaks].tolist()
    more_than_two_peaks = one_peak_idxs[more_than_two_peaks].tolist()
    front_high_back_low = one_peak_idxs[front_high_back_low].tolist()
    return valid_Tms, one_peaks, valid_two_peaks, more_than_two_peaks, valid_peak_heights_two_peaks, valid_Tm_indices, new_noise_floor,front_high_back_low



def visualize_melt_curve_partitions(global_tm_mcs_deriv:np.array, one_peaks:list, two_peaks:list, more_than_two_peaks:list, global_tm_xticks:np.array, 
                                    global_tm_temps:np.array, fig_width=15, fig_height=10):
    """Function to visualize the melting curve partitions"""
    plt.figure(figsize=(fig_width, fig_height))

    plt.subplot(2, 2, 1) 
    for signal in global_tm_mcs_deriv[one_peaks]:
        plt.plot(signal)
    plt.xticks(ticks=global_tm_xticks, labels=global_tm_temps)
    plt.xlabel("Frames")
    plt.ylabel("-dF/dT")
    plt.title("Melting curves with 1 peak")

    plt.subplot(2, 2, 2)
    for signal in global_tm_mcs_deriv[more_than_two_peaks]:
        plt.plot(signal)
    plt.xticks(ticks=global_tm_xticks, labels=global_tm_temps)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Normalized -dF/dT")
    plt.title("Melting curves with more than 2 peaks")

    plt.subplot(2, 1, 2)
    for signal in global_tm_mcs_deriv[two_peaks]:
        plt.plot(signal)
    plt.xticks(ticks=global_tm_xticks, labels=global_tm_temps)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Normalized -dF/dT")
    plt.title("Melting curves with 2 peaks")

    plt.tight_layout()
    plt.show()






def convert_temperature_delta_to_frame_delta(temperature_delta, rate_per_min, exposure_in_sec):
    """Function to convert temperature delta to frame delta"""
    delta_T = temperature_delta
    if delta_T <= 0:
        return 0
    tot_t_min = delta_T / rate_per_min
    tot_t_sec = tot_t_min * 60
    frame_idx = tot_t_sec / exposure_in_sec
    return int(frame_idx)




def filter_local_tms(all_tms: list, expected_tm_values: list, resolution: float, peak_heights: list, max_n_tms: int):
    """Function that parses and filters the local Tm values based on the expected number of Tms and the resolution

    Args:
    - all_tms (list): The list of all Tm values for each signal ever found
    - expected_tm_values (list): The expected Tm values
    - resolution (float): The resolution of each peak. When filtering, the peak is considered valid if it is within the resolution range of the expected Tm value
    - peak_heights (list): The list of peak heights corresponding to each Tm value
    - max_n_tms (int): The maximum number of Tm values to consider

    Returns:
    - tuple: (output, encoded_output)
      - output (np.array): The parsed and filtered Tm values with validity indicators
      - encoded_output (np.array): The encoded Tm values based on the expected Tm values
    """

    output = np.zeros((len(all_tms), 2 * max_n_tms))
    encoded_output = np.zeros((len(all_tms), 2 * max_n_tms))

    for idx, (tms, heights) in enumerate(zip(all_tms, peak_heights)):
        if len(tms) == 0:
            output[idx, :max_n_tms] = float('nan')
            output[idx, max_n_tms:] = 0
            encoded_output[idx, :max_n_tms] = float('nan')
            encoded_output[idx, max_n_tms:] = 0
            continue

        sorted_indices = np.argsort(heights)[::-1]
        sorted_tms = [tms[i] for i in sorted_indices]
        sorted_heights = [heights[i] for i in sorted_indices]
        
        output[idx, 0] = sorted_tms[0]
        encoded_output[idx, 0] = sorted_tms[0]
        
        for j in range(1, max_n_tms):
            if j < len(sorted_tms):
                output[idx, j] = sorted_tms[j]
                encoded_output[idx, j] = sorted_tms[j]
            else:
                output[idx, j] = float('nan')
                encoded_output[idx, j] = float('nan')
        
        for j in range(max_n_tms):
            if j < len(sorted_tms):
                valid = False
                for N, expected_tm in enumerate(expected_tm_values):
                    if abs(sorted_tms[j] - expected_tm) <= resolution:
                        output[idx, max_n_tms + j] = 1
                        encoded_output[idx, max_n_tms + j] = N + 1
                        valid = True
                        break
                if not valid:
                    output[idx, max_n_tms + j] = 0
                    encoded_output[idx, max_n_tms + j] = 0
            else:
                output[idx, max_n_tms + j] = 0
                encoded_output[idx, max_n_tms + j] = 0

    return output, encoded_output


def visualize_local_tms_distribution(parsed_tms: np.ndarray, expected_tm_values: list, resolution: float, max_n_tms: int, custom_cmap: list = ["red", "green"]):
    """Function that visualizes the distribution of the local Tm values
    
    Args:
    - parsed_tms (np.array): The parsed and filtered Tm values
    - expected_tm_values (list): The expected Tm values
    - resolution (float): The resolution of each peak. Here it is only used for visualization
    - max_n_tms (int): The maximum number of Tm values considered
    - custom_cmap (list): The custom colormap; list of colors
    
    Displays:
    - The distribution of the local Tm values
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = mcolors.ListedColormap(custom_cmap)

    total_count = parsed_tms.shape[0]
    # print(f"Total Count: {total_count}")


    if len(expected_tm_values) == 1:
        for i, expected_tm in enumerate(expected_tm_values):
            valid_tms_group = []
            invalid_tms_group = []
            valid_indices_group = []
            invalid_indices_group = []
            
            for j in range(max_n_tms):
                valid_indices = np.where((parsed_tms[:, max_n_tms + j] == 1) & (~np.isnan(parsed_tms[:, j])))
                invalid_indices = np.where((parsed_tms[:, max_n_tms + j] == 0) & (~np.isnan(parsed_tms[:, j])))

                valid_tms = parsed_tms[valid_indices, j].flatten()
                invalid_tms = parsed_tms[invalid_indices, j].flatten()

                valid_tms_group.extend(valid_tms)
                invalid_tms_group.extend(invalid_tms)
                valid_indices_group.extend(valid_indices[0])
                invalid_indices_group.extend(invalid_indices[0])
            
            # Convert lists to arrays for plotting
            valid_tms_group = np.array(valid_tms_group)
            invalid_tms_group = np.array(invalid_tms_group)
            valid_indices_group = np.array(valid_indices_group)
            invalid_indices_group = np.array(invalid_indices_group)

            ax.scatter(valid_indices_group, valid_tms_group, c='green', label='Valid Tm Values' if i == 0 else "", alpha=0.5, s=15)
            ax.scatter(invalid_indices_group, invalid_tms_group, c='red', label='Non-Specific Tm Values' if i == 0 else "", alpha=0.5, s=15)

            within_resolution = np.ones_like(valid_tms_group).astype(bool)

                                                        

            count_within_resolution = np.sum(within_resolution)
            percentage_within_resolution = (count_within_resolution / total_count) * 100

            within_range_tms = valid_tms_group[within_resolution]
            mean_within_range = np.mean(within_range_tms) if len(within_range_tms) > 0 else 0
            stdev_within_range = np.std(within_range_tms) if len(within_range_tms) > 0 else 0

            ax.axhline(expected_tm, color='navy', linestyle='--', label=f'Expected Tm {i + 1}' if j == 0 else "")
            ax.fill_between(np.arange(total_count), expected_tm - resolution, expected_tm + resolution, color='green', alpha=0.1)

            ax.text(total_count, expected_tm + 3*resolution, 
                    f'Tm {i + 1} Counts: {count_within_resolution}, {percentage_within_resolution:.1f}%, Mean: {mean_within_range:.2f}, Stdev: {stdev_within_range:.2f}',
                    fontsize=10, verticalalignment='center', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            print(f"The mean Tm value for Tm {i + 1} is {mean_within_range:.2f} with a standard deviation of {stdev_within_range:.2f}")
            
        count_invalid = np.sum((parsed_tms[:, max_n_tms:] == 0).all(axis=1))
        percentage_invalid = (count_invalid / total_count) * 100
        
        ax.text(total_count, min(expected_tm_values) - 3 * resolution, 
                f'Non-Specific/No Tm Counts: {count_invalid}, {percentage_invalid:.1f}%',
                fontsize=10, verticalalignment='center', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        valid_tm_values = parsed_tms[:, :max_n_tms].flatten()
        valid_tm_values = valid_tm_values[~np.isnan(valid_tm_values)]
        max_tm_value = np.max(valid_tm_values) if len(valid_tm_values) > 0 else 0
        min_tm_value = np.min(valid_tm_values) if len(valid_tm_values) > 0 else 0
        ax.set_yticks(np.arange(min_tm_value, max_tm_value + 5, 5)) 
        ax.set_xlabel('Index')
        ax.set_ylabel('Tm Values')
        plt.tight_layout()
        plt.legend()
        plt.show()
    else:
        for i, expected_tm in enumerate(expected_tm_values):
            valid_tms_group = []
            invalid_tms_group = []
            valid_indices_group = []
            invalid_indices_group = []
            
            for j in range(max_n_tms):
                valid_indices = np.where((parsed_tms[:, max_n_tms + j] == 1) & (~np.isnan(parsed_tms[:, j])))
                invalid_indices = np.where((parsed_tms[:, max_n_tms + j] == 0) & (~np.isnan(parsed_tms[:, j])))

                valid_tms = parsed_tms[valid_indices, j].flatten()
                invalid_tms = parsed_tms[invalid_indices, j].flatten()

                valid_tms_group.extend(valid_tms)
                invalid_tms_group.extend(invalid_tms)
                valid_indices_group.extend(valid_indices[0])
                invalid_indices_group.extend(invalid_indices[0])
            
            # Convert lists to arrays for plotting
            valid_tms_group = np.array(valid_tms_group)
            invalid_tms_group = np.array(invalid_tms_group)
            valid_indices_group = np.array(valid_indices_group)
            invalid_indices_group = np.array(invalid_indices_group)

            ax.scatter(valid_indices_group, valid_tms_group, c='green', label='Valid Tm Values' if i == 0 else "", alpha=0.5, s=15)
            ax.scatter(invalid_indices_group, invalid_tms_group, c='red', label='Non-Specific Tm Values' if i == 0 else "", alpha=0.5, s=15)

            within_resolution = np.abs(valid_tms_group - expected_tm) <= resolution
            count_within_resolution = np.sum(within_resolution)
            percentage_within_resolution = (count_within_resolution / total_count) * 100

            within_range_tms = valid_tms_group[within_resolution]
            mean_within_range = np.mean(within_range_tms) if len(within_range_tms) > 0 else 0
            stdev_within_range = np.std(within_range_tms) if len(within_range_tms) > 0 else 0

            ax.axhline(expected_tm, color='navy', linestyle='--', label=f'Expected Tm {i + 1}' if j == 0 else "")
            ax.fill_between(np.arange(total_count), expected_tm - resolution, expected_tm + resolution, color='green', alpha=0.1)

            ax.text(total_count, expected_tm + 3*resolution, 
                    f'Tm {i + 1} Counts: {count_within_resolution}, {percentage_within_resolution:.1f}%, Mean: {mean_within_range:.2f}, Stdev: {stdev_within_range:.2f}',
                    fontsize=10, verticalalignment='center', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
        count_invalid = np.sum((parsed_tms[:, max_n_tms:] == 0).all(axis=1))
        percentage_invalid = (count_invalid / total_count) * 100
        ax.text(total_count, min(expected_tm_values) - 3 * resolution, 
                f'Non-Specific/No Tm Counts: {count_invalid}, {percentage_invalid:.1f}%',
                fontsize=10, verticalalignment='center', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        valid_tm_values = parsed_tms[:, :max_n_tms].flatten()
        valid_tm_values = valid_tm_values[~np.isnan(valid_tm_values)]
        max_tm_value = np.max(valid_tm_values) if len(valid_tm_values) > 0 else 0
        min_tm_value = np.min(valid_tm_values) if len(valid_tm_values) > 0 else 0
        ax.set_yticks(np.arange(min_tm_value, max_tm_value + 5, 5)) 
        ax.set_xlabel('Index')
        ax.set_ylabel('Tm Values')
        plt.tight_layout()
        plt.legend()
        plt.show()



def probe_filter_by_shape(data_array: np.ndarray, bkg_signal: np.ndarray, threshold: float, use_shape: str="both"):
    """Subroutine internal function to filter probes based on shape similarity
    
    Args:
    - data_array (np.array): The input data array of melting curves
    - bkg_signal (np.array): The normalized background signal
    - threshold (float): The threshold value for the shape similarity
    - use_shape (str): The shape to use for filtering; choose from 'derivative', 'original', 'both'
    
    Returns:
    - tuple: A tuple containing:
        - np.array: The non-background mask
        - np.array: The background mask"""
    normalized_data = min_max_normalize(data_array, use_global_min_max=False)
    normalized_bkg_signal = bkg_signal

    data_deriv = min_max_normalize(savgol_filter(normalized_data, window_length=55, polyorder=2, deriv=1, mode='nearest'), use_global_min_max=False)
    bkg_deriv = min_max_normalize(savgol_filter(normalized_bkg_signal, window_length=55, polyorder=2, deriv=1, mode='nearest'))

    if use_shape == "derivative":
        combined_bkg = bkg_deriv
        combined_diff = data_deriv
    elif use_shape == "original":
        combined_bkg = normalized_bkg_signal
        combined_diff = normalized_data
    elif use_shape == "both":
        combined_bkg = np.concatenate((bkg_deriv, normalized_bkg_signal))
        combined_diff = np.concatenate([data_deriv, normalized_data], axis=1)
    else:
        raise ValueError("Invalid value for use_shape. Choose from 'derivative', 'original', 'both'.")
    
    # distances = np.array([euclidean(combined_bkg.flatten(), diff.flatten()) for diff in combined_diff])
    
    # Use the correlation coeff as the distance metric
    distances = np.array([1 - np.corrcoef(combined_bkg.flatten(), diff.flatten())[0, 1] for diff in combined_diff])
    normalized_distances = min_max_normalize(distances)

    # for signal in combined_diff:
    #     plt.plot(signal)
    # plt.plot(combined_bkg, color="black")

    non_background_mask = normalized_distances > threshold
    background_mask = ~non_background_mask

    return non_background_mask, background_mask


def visualize_probe_filtering(original_data: np.ndarray, positive_mask: np.ndarray, background_mask: np.ndarray, actual_background: np.ndarray):
    """Function that visualizes the probe filtering process

    Args:
    - original_data (np.array): The original data array
    - positive_mask (np.array): The positive mask
    - background_mask (np.array): The background mask
    - actual_background (np.array): The actual background signal obtained from EvaGreen Negatives

    Displays:
    - The visualization of the probe filtering process
    """
    positive_data = original_data[positive_mask]
    background_data = original_data[background_mask]
    # print(type(positive_data), type(background_data))

    if len(background_data) == 0:
        normalized_background_data = np.zeros((1, original_data.shape[1]))
    else:
        normalized_background_data = min_max_normalize(background_data, use_global_min_max=False)

    normalized_positive_data = min_max_normalize(positive_data, use_global_min_max=False)
    
    fig_width = 15
    fig_height = 5
    plt.figure(figsize=(fig_width, fig_height))

    plt.subplot(1, 2, 1) 
    for signal in normalized_positive_data:
        plt.plot(signal)
    plt.xlabel("Temperature")
    plt.ylabel("Fluorescence Value")
    plt.title("Normalized Local Tm 1 Positive")

    plt.subplot(1, 2, 2)
    for signal in normalized_background_data:
        plt.plot(signal)
    plt.plot(actual_background, color = "black", label="Actual Background", linestyle="dotted", linewidth=4)
    plt.xlabel("Temperature")
    plt.ylabel("Normalized Fluorescence Value")
    plt.title("Normalized Local Tm Negative")
    plt.legend()
    plt.tight_layout()
    plt.show()

def interactive_probe_filtering(data_array, bkg_signal, mode="both", max = 0.005):
    """Function that creates an interactive visualization for probe filtering. When the threshold is adjusted using the slider, the visualization and outputs both updates accordingly

    Args:
    - data_array (np.array): The input data array of melting curves
    - bkg_signal (np.array): The normalized background signal
    - mode (str): The mode to use for filtering; choose from 'derivative', 'original', 'both'

    Returns:
    - dict: A dictionary containing the non-background mask and the background mask
    """
    masks = {'non_background_mask': None, 'background_mask': None}
    output = widgets.Output()
    
    def update_visualization(change):
        threshold = change['new']
        
        if threshold == 0:
            # Include all data points in the non_background_mask if threshold is 0
            non_background_mask = np.ones(data_array.shape[0], dtype=bool)
            background_mask = np.zeros(data_array.shape[0], dtype=bool)
        else:
            non_background_mask, background_mask = probe_filter_by_shape(data_array, bkg_signal, threshold, mode)
        
        masks['non_background_mask'] = non_background_mask
        masks['background_mask'] = background_mask
        
        with output:
            output.clear_output(wait=True)
            visualize_probe_filtering(data_array, non_background_mask, background_mask, bkg_signal)
    
    threshold_slider = widgets.FloatSlider(min=0, max=max, step=0.000001, value=0.1, description='Threshold', layout=widgets.Layout(width='1000px'), readout_format='.5f')
    threshold_slider.observe(update_visualization, names='value')
    
    display(VBox([threshold_slider, output]))
    update_visualization({'new': max})  # Initial plot with the default threshold
    
    return masks




def get_noise_floor_probe(normalized_array: np.ndarray, before_nth_frame: int = 50, last_n_frame=5 , tm_xticks: list = None,
                           tm_temps: list = None, plot: bool = True, 
                           compute_SD=False, n_SD=1, flat_noise=False, normalized_negatives=None, 
                           use_negative_est_SD=False, baseline_offset = False, mannual_offset_fitting_range:tuple = None):
    """Function that estimates the noise floor for the probe channel data
    
    Args:
    - normalized_array (np.array): The normalized array of probe channel data
    - before_nth_frame (int): The number of frames before the nth frame to use for noise floor estimation. Default is 50
    - after_nth_frame (int): The number of frames after the nth frame to use for noise floor estimation. Default is None
    - tm_xticks (list): The xticks for the temperature values
    - tm_temps (list): The temperature values
    - plot (bool): Whether to plot the noise floor visualization
    - compute_SD (bool): Whether to compute the standard deviation
    - n_SD (int): The number of standard deviations to use for the confidence interval
    
    Returns:
    - np.array: The estimated noise floor
    - np.array: The upper confidence interval for the noise floor"""

    # M, N = normalized_array.shape
    # after_nth_frame = N - last_n_frame

    median_vals = np.median(normalized_array, axis=0)
    
    median_of_median = np.median(median_vals)

    # start_points = normalized_array[:, :before_nth_frame].flatten()
    # end_points = normalized_array[:, after_nth_frame:].flatten()


    # y_vals = np.concatenate((start_points, end_points))
    # x_vals = np.concatenate((np.zeros(start_points.shape), np.ones(end_points.shape) * (normalized_array.shape[1] - 1)))
    
    # slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
    # noise_floor = slope * np.arange(normalized_array.shape[1]) + intercept

    M, N = normalized_array.shape
    after_nth_frame = N - last_n_frame

    std_devs = np.std(normalized_array, axis=0)

    least_std_indices = np.argsort(std_devs)[:35]
    # print(least_std_indices)

    y_vals_least_std = normalized_array[:, least_std_indices].flatten()

    # x_vals_least_std = np.repeat(least_std_indices, M)
    x_vals_least_std = np.tile(least_std_indices, M)
    
    slope, intercept, r_value, p_value, std_err = linregress(x_vals_least_std, y_vals_least_std)

    if mannual_offset_fitting_range is not None:
        start_idx, end_idx = mannual_offset_fitting_range
        y_vals = normalized_array[:, start_idx:end_idx].flatten()
        x_vals = np.tile(np.arange(start_idx, end_idx), M)
        slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)


    noise_floor = slope * np.arange(N) + intercept


    filtered_median_vals = median_vals[median_vals > median_of_median]


        # std_dev = np.max(np.std(normalized_negatives, axis=0))

    if len(filtered_median_vals) > 0:
        mad = np.median(np.abs(filtered_median_vals - median_of_median))
        std_dev = mad * 1.4826
    else:
        std_dev = 1.253 * np.std(median_vals) / np.sqrt(M)

    if normalized_negatives is not None:
        weight = 0.5
        negative_x_vals = np.arange(normalized_negatives.shape[1])
        negative_y_vals = np.median(normalized_negatives, axis=0)
        negative_slope, negative_intercept, _, _, _ = linregress(negative_x_vals, negative_y_vals)
        negative_noise_floor = negative_slope * np.arange(normalized_negatives.shape[1]) + negative_intercept

        # noise_floor = weight * noise_floor + (1 - weight) * negative_noise_floor
        noise_floor = weight*slope * np.arange(N) + (1 - weight)*negative_slope * np.arange(N) + intercept
        if use_negative_est_SD:
            std_dev = np.mean(np.std(normalized_negatives, axis=0))
    

    if flat_noise and (mannual_offset_fitting_range is None):
        # print("here 1")
        noise_floor = np.full(normalized_array.shape[1], median_of_median)
    
    elif flat_noise and (mannual_offset_fitting_range is not None):
        # print("here 2")
        normalized_array_with_offset = normalized_array - slope * np.arange(N)
        median_vals_with_offset = np.median(normalized_array_with_offset, axis=0)
        median_of_median_with_offset = np.median(median_vals_with_offset)
        # print(median_of_median_with_offset)
        noise_floor = np.full(normalized_array.shape[1], median_of_median_with_offset)
    else:
        pass
        
    upper_confidence_interval = noise_floor + n_SD * std_dev

    if plot:
        plt.figure(figsize=(10, 6))
        for signal in normalized_array:
            plt.plot(signal, color='cornflowerblue', alpha=0.1)
        plt.xlabel("Frames")
        plt.ylabel("Normalized -dF/dT")
        plt.title("Noise Floor Visualization")
        plt.plot(noise_floor, color='red', linestyle='dotted', linewidth=3, label='Noise Floor')
        plt.axvline(x=before_nth_frame, color='green', linestyle='dotted', linewidth=2, label='Before nth Frame')
        if after_nth_frame is not None:
            plt.axvline(x=after_nth_frame, color='green', linestyle='dotted', linewidth=2, label='After nth Frame')
        if compute_SD:
            plt.plot(upper_confidence_interval, color='orange', linestyle='dotted', linewidth=3, label= f'Noise floor + ({n_SD} SD)')
        plt.text(0, noise_floor[0] + 0.1, s=f'Start Noise = {noise_floor[0]:.3f}', fontsize=12, color='red', verticalalignment='bottom', horizontalalignment='left')
        plt.text(len(noise_floor) - 1, noise_floor[-1] + 0.1, s=f'End Noise = {noise_floor[-1]:.3f}', fontsize=12, color='red', verticalalignment='bottom', horizontalalignment='right')
        plt.legend()
        if tm_xticks is not None and tm_temps is not None:
            ax = plt.gca()
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(tm_xticks)
            ax2.set_xticklabels(tm_temps, rotation=45)
            ax2.set_xlabel("Temperature (°C)")
        plt.show()
    
    if baseline_offset:
        return noise_floor, upper_confidence_interval, std_dev, slope* np.arange(N)
    else:
        return noise_floor, upper_confidence_interval, std_dev
    

def cluster_signals(data_array: np.ndarray, num_clusters: int, Z):
    """Subrountine that uses hierarchical clustering to cluster the signals
    
    Args:
    - data_array (np.array): The input data array of melting curves
    - num_clusters (int): The number of clusters to form
    - Z (np.array): The linkage matrix
    
    Returns:
    - np.array: The cluster labels"""
    labels = fcluster(Z, t=num_clusters, criterion='maxclust')
    return labels - 1






def compute_local_tms(signals: np.ndarray, initial_T: float, final_T: float, heating_rate: float, img_series_gap_time: float, noise_floor, 
                      width: tuple = (15, 60), max_plots: int = 500, plot: bool = True, prominance_array = None, min_temp_between_peaks=10):
    from matplotlib.colors import LogNorm
    min_distance_between_peak = convert_temperature_delta_to_frame_delta(min_temp_between_peaks, 
                                                                       heating_rate, 
                                                                       img_series_gap_time)
    all_tms = []
    all_widths = []
    all_heights = []
    peak_indices = []
    all_prominences = []
    all_plateaus = []
    
    # Store first pass prominences for all signals
    first_pass_prominences = []
    
    if plot:
        fig = plt.figure(figsize=(15, 15))
        gs = GridSpec(4, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, :])
        ax5 = fig.add_subplot(gs[3, :])
        
        tick_interval = 25
        ticks = np.arange(0, signals.shape[1], tick_interval)
        labels = np.round(np.linspace(initial_T, final_T, signals.shape[1]), 1)[::tick_interval]
    
    # First pass to collect all prominences
    for signal in signals:
        min_prominence = 0.01
        initial_peaks, initial_properties = find_peaks(signal, height=noise_floor, width=width, 
                                                     prominence=min_prominence, distance=min_distance_between_peak,
                                                     plateau_size=True)
        if len(initial_peaks) > 0:
            first_pass_prominences.extend(initial_properties['prominences'])
    
    # Calculate global prominence statistics
    if len(first_pass_prominences) > 0:
        max_prominence = np.max(first_pass_prominences)
    else:
        max_prominence = 1.0  # fallback value if no peaks found
    
    # Second pass with region-specific thresholds
    for index, signal in enumerate(signals):
        if prominance_array is not None:
            modified_prominence_array = np.zeros_like(prominance_array, dtype=float)
        
            min_prominence = 0.01
            initial_peaks, initial_properties = find_peaks(signal, height=noise_floor, width=width, 
                                                         prominence=min_prominence, distance=min_distance_between_peak,
                                                         plateau_size=True)
            
            for value in np.unique(prominance_array):
                region_mask = prominance_array == value
                
                if value == 1:  
                    modified_prominence_array[region_mask] = max_prominence * 1.1
                else:
                    percentile = value * 100  # e.g., 0.05 -> 5th percentile
                    threshold = np.percentile(first_pass_prominences, percentile)
                    modified_prominence_array[region_mask] = threshold
            
            # Second pass with modified prominence array
            peaks, properties = find_peaks(signal, height=noise_floor, width=width, 
                                         prominence=modified_prominence_array, distance=min_distance_between_peak,
                                         plateau_size=True)
        else:
            # If no prominence array provided, use first pass results
            min_prominence = 0.01
            peaks, properties = find_peaks(signal, height=noise_floor, width=width, 
                                         prominence=min_prominence, distance=min_distance_between_peak,
                                         plateau_size=True)
        
        tms = [compute_Tm(initial_T=initial_T, peak_frame_idx=peak,
                         rate_per_min=heating_rate,
                         exposure_in_sec=img_series_gap_time,
                         max=final_T) for peak in peaks]
        
        peak_indices.append(peaks)
        all_tms.append(tms)
        all_widths.append(properties['widths'])
        all_heights.append(properties['peak_heights'])
        all_prominences.append(properties['prominences'])
        all_plateaus.append(properties['plateau_sizes'])

        if index < max_plots and plot:
            ax1.plot(np.arange(len(signal)), signal)
            ax1.plot(peaks, signal[peaks], "x", markersize=15)
    
    if plot:
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(labels)
        ax1.set_xlabel("Temperature")
        ax1.set_ylabel("Signal")
        ax1.set_title("Peak Detection Results")
        
        if len(peak_indices) > 0:
            all_peaks_flat = np.concatenate(peak_indices)
            prominences_array = np.concatenate(all_prominences)
            widths_array = np.concatenate(all_widths)
            plateaus_array = np.concatenate(all_plateaus)
            
            hist2d_prom = ax2.hist2d(all_peaks_flat, prominences_array, bins=(50, 50), 
                                    cmap='Blues', norm=LogNorm())
            plt.colorbar(hist2d_prom[3], ax=ax2)
            ax2.set_xlabel("Peak Position (Index)")
            ax2.set_ylabel("Peak Prominence")
            ax2.set_title("Peak Distribution vs Prominence")
            
            ax3.hist(prominences_array, bins=50, color='blue', alpha=0.7)
            ax3.set_xlabel("Peak Prominence")
            ax3.set_ylabel("Count")
            ax3.set_title("Prominence Distribution")
            
            hist2d_width = ax4.hist2d(all_peaks_flat, widths_array, bins=(50, 50),
                                     cmap='Reds', norm=LogNorm())
            plt.colorbar(hist2d_width[3], ax=ax4)
            ax4.set_xlabel("Peak Position (Index)")
            ax4.set_ylabel("Peak Width")
            ax4.set_title("Peak Distribution vs Width")
            
            hist2d_plateau = ax5.hist2d(all_peaks_flat, plateaus_array, bins=(50, 50),
                                       cmap='Greens', norm=LogNorm())
            plt.colorbar(hist2d_plateau[3], ax=ax5)
            ax5.set_xlabel("Peak Position (Index)")
            ax5.set_ylabel("Plateau Size")
            ax5.set_title("Peak Distribution vs Plateau Size")
        
        plt.tight_layout()

    return all_tms, all_widths, all_heights, peak_indices





def visualize_probe_clusters(data_array: np.ndarray, labels: np.ndarray, max_clusters: int, 
                             before_nth_frame: int, n_SD: list, initial_T: float, final_T: float, 
                             heating_rate: float, img_series_gap_time: float, flat_noise: list = None, 
                             normalized_negatives=None, width_range=(15, 100), prominance_array=None, 
                             use_negative_est_SD=False, baseline_offset: list = None, 
                             mannual_offset_fitting_range: tuple = None, min_temp_between_peaks = 10):
    """Internal function to visualize the probe clusters.
    
    Parameters:
        prominance_array: Either a single numpy array applied to all clusters,
                         or a list of arrays with length equal to max_clusters
                         where each array is applied to its respective cluster.
    """
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) if -1 not in unique_labels else len(unique_labels)

    rows = int(np.ceil(max_clusters / 2))
    cols = 2 if max_clusters > 1 else 1

    # Convert single prominence array to list of arrays if needed
    if prominance_array is not None:
        if isinstance(prominance_array, (np.ndarray, list)):
            if not isinstance(prominance_array[0], (np.ndarray, list)):  # Single array case
                prominance_array = [prominance_array] * max_clusters
            else:  # List of arrays case
                if len(prominance_array) != max_clusters:
                    raise ValueError(f"Length of prominance_array list ({len(prominance_array)}) must match max_clusters ({max_clusters})")

    all_tms = [None] * len(data_array)
    all_widths = [None] * len(data_array)
    all_heights = [None] * len(data_array)
    n_sd_away = [None] * len(data_array)

    fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows))
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    tick_interval = 25
    ticks = np.arange(0, data_array.shape[1], tick_interval)
    temp_labels = np.round(np.linspace(initial_T, final_T, data_array.shape[1]), 1)[::tick_interval]

    if flat_noise is None:
        flat_noise = [False] * n_clusters
    elif isinstance(flat_noise, bool):
        flat_noise = [flat_noise] * n_clusters

    if isinstance(n_SD, int):
        n_SD = [n_SD] * max_clusters

    for ax, label in zip(axes, range(max_clusters)):
        if label < n_clusters:
            cluster_flat_noise_status = flat_noise[label]
            cluster_signals = data_array[labels == label]

            if baseline_offset is not None:
                baseline_offset_status = baseline_offset[label]

            # Use the n_SD value corresponding to the current cluster
            n_SD_value = n_SD[label]

            if baseline_offset is not None and baseline_offset_status:
                noise_floor, upper_confidence_interval, estimated_std, baseline = get_noise_floor_probe(
                    cluster_signals, before_nth_frame, plot=False, n_SD=n_SD_value,
                    flat_noise=cluster_flat_noise_status, normalized_negatives=normalized_negatives, 
                    use_negative_est_SD=use_negative_est_SD, baseline_offset=baseline_offset_status, 
                    mannual_offset_fitting_range=mannual_offset_fitting_range)

                cluster_signals = cluster_signals - baseline

            else:
                noise_floor, upper_confidence_interval, estimated_std = get_noise_floor_probe(
                    cluster_signals, before_nth_frame, plot=False, n_SD=n_SD_value,
                    flat_noise=cluster_flat_noise_status, normalized_negatives=normalized_negatives, 
                    use_negative_est_SD=use_negative_est_SD, mannual_offset_fitting_range=mannual_offset_fitting_range)

            for signal in cluster_signals:
                ax.plot(signal, color='cornflowerblue', alpha=0.1)
            ax.plot(noise_floor, color='red', linestyle='dotted', linewidth=2, label='Noise Baseline')
            if upper_confidence_interval is not None:
                ax.plot(upper_confidence_interval, color='orange', linestyle='dotted', linewidth=2, 
                        label=f'Noise Floor = Baseline + {n_SD_value} SD')

            # Get cluster-specific prominence array if available
            current_prominance = None if prominance_array is None else prominance_array[label]
            
            tms, widths, heights, peak_indices = compute_local_tms(
                cluster_signals, initial_T, final_T, heating_rate, img_series_gap_time, 
                upper_confidence_interval, width=width_range, plot=False, prominance_array=current_prominance, min_temp_between_peaks= min_temp_between_peaks)
            # print(widths)

            cluster_indices = np.where(labels == label)[0]
            # print(len(tms), len(widths), len(heights), len(peak_indices),cluster_indices.shape)
            for i, idx in enumerate(cluster_indices):
                all_tms[idx] = tms[i]
                all_widths[idx] = widths[i]
                all_heights[idx] = heights[i]
                n_sd_away[idx] = (heights[i] - noise_floor[peak_indices[i]]) / estimated_std

                ax.plot(peak_indices[i], heights[i], '*', color='black', markersize=5, alpha=0.8)

            ax.set_title(f"Cluster {label + 1}" if label != -1 else "Noise Cluster")
            ax.legend()
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticks)
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(ticks)
            ax2.set_xticklabels(temp_labels, rotation=45)
            ax2.set_xlabel("Temperature (°C)")
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

    all_local_tm_heights_parsed = []
    for arr in n_sd_away:
        if arr.size == 0:
            all_local_tm_heights_parsed.append(0)
        elif arr.size > 1:
            all_local_tm_heights_parsed.append(np.mean(arr))
        else:
            all_local_tm_heights_parsed.append(arr.item())

    all_local_tm_heights_parsed = np.array(all_local_tm_heights_parsed)

    return all_tms, all_local_tm_heights_parsed, all_heights









def interactive_probe_clustering_thresholding(data_array, max_clusters, n_SD, initial_T, final_T, heating_rate, 
                                              img_series_gap_time, before_nth_frame=50, flat_noise:bool=False, normalized_negatives = None,
                                                width_range = (15,100), prominance_array = None, use_negative_est_SD = False,
                                                  baseline_offset:list = None, mannual_offset_fitting_range:tuple = None, return_cluster_indicies=False,
                                                  min_temp_between_peaks=10):
    """Function that creates an interactive visualization for probe clustering. When the number of clusters is adjusted using the slider, the visualization and outputs both updates accordingly
    
    Args:
    - data_array (np.array): The input data array of melting curves
    - max_clusters (int): The maximum number of clusters to form
    - n_SD (int): The number of standard deviations to use for the noise floor
    - initial_T (float): The initial temperature
    - final_T (float): The final temperature
    - heating_rate (float): The heating rate per minute
    - img_series_gap_time (float): The exposure time in seconds
    
    Returns:
    - list: The computed Tm values; each nested list contains the Tm values for each signal
    - list: The widths of the peaks; each nested list contains the widths for each signal
    - list: The heights of the peaks; each nested list contains the heights for each signal"""
    output = widgets.Output()


    data_scaled = min_max_normalize(data_array, use_global_min_max=False)
    pairwise_dist = pdist(data_scaled, metric='euclidean')
    Z = linkage(pairwise_dist, method='ward')
    
    if return_cluster_indicies:
        return all_tms, all_widths, all_heights, Z

    def update_clusters(change):
        num_clusters = change['new']
        labels = cluster_signals(data_array, num_clusters, Z)
        with output:
            output.clear_output(wait=True)
            all_tms, all_widths, all_heights = visualize_probe_clusters(data_array, labels, max_clusters, before_nth_frame, n_SD,
                                                                         initial_T, final_T, heating_rate, img_series_gap_time,
                                                                           flat_noise=flat_noise, normalized_negatives = normalized_negatives, 
                                                                           width_range=width_range, prominance_array=prominance_array, 
                                                                           use_negative_est_SD=use_negative_est_SD, baseline_offset=baseline_offset,
                                                                           mannual_offset_fitting_range = mannual_offset_fitting_range,
                                                                           min_temp_between_peaks= min_temp_between_peaks)
            return all_tms, all_widths, all_heights
    
    cluster_slider = widgets.IntSlider(min=1, max=max_clusters, step=1, value=max_clusters, description='Number of Clusters', layout=widgets.Layout(width='500px'))
    cluster_slider.observe(update_clusters, names='value')
    
    display(VBox([cluster_slider, output]))
    all_tms, all_widths, all_heights = update_clusters({'new': max_clusters})  # Initial plot with the default number of clusters
    
    return all_tms, all_widths, all_heights

# def interactive_probe_clustering_thresholding(data_array, max_clusters, n_SD, initial_T, final_T, heating_rate, 
#                                               img_series_gap_time, before_nth_frame=50, flat_noise:bool=False, normalized_negatives = None,
#                                                 width_range = (15,100), prominance_array = None, use_negative_est_SD = False, baseline_offset:list = None):
#     """Function that creates an interactive visualization for probe clustering. When the number of clusters is adjusted using the slider, the visualization and outputs both updates accordingly
    
#     Args:
#     - data_array (np.array): The input data array of melting curves
#     - max_clusters (int): The maximum number of clusters to form
#     - n_SD (int): The number of standard deviations to use for the noise floor
#     - initial_T (float): The initial temperature
#     - final_T (float): The final temperature
#     - heating_rate (float): The heating rate per minute
#     - img_series_gap_time (float): The exposure time in seconds
    
#     Returns:
#     - list: The computed Tm values; each nested list contains the Tm values for each signal
#     - list: The widths of the peaks; each nested list contains the widths for each signal
#     - list: The heights of the peaks; each nested list contains the heights for each signal"""
#     output = widgets.Output()
    
#     data_scaled = min_max_normalize(data_array, use_global_min_max=False)
#     pairwise_dist = pdist(data_scaled, metric='euclidean')
#     Z = linkage(pairwise_dist, method='ward')
    
#     def update_clusters(change):
#         num_clusters = change['new']
#         labels = cluster_signals(data_array, num_clusters, Z)
#         with output:
#             output.clear_output(wait=True)
#             all_tms, all_widths, all_heights = visualize_probe_clusters(data_array, labels, max_clusters, before_nth_frame, n_SD,
#                                                                          initial_T, final_T, heating_rate, img_series_gap_time,
#                                                                            flat_noise=flat_noise, normalized_negatives = normalized_negatives, 
#                                                                            width_range=width_range, prominance_array=prominance_array, use_negative_est_SD=use_negative_est_SD, baseline_offset=baseline_offset)
#             return all_tms, all_widths, all_heights
    
#     cluster_slider = widgets.IntSlider(min=1, max=max_clusters, step=1, value=max_clusters, description='Number of Clusters', layout=widgets.Layout(width='500px'))
#     cluster_slider.observe(update_clusters, names='value')
    
#     display(VBox([cluster_slider, output]))
#     all_tms, all_widths, all_heights = update_clusters({'new': max_clusters})  # Initial plot with the default number of clusters
    
#     return all_tms, all_widths, all_heights




def list_of_tm_to_index(tm: list, initial_T: float, rate_per_min: float, exposure_in_sec: float, max: float) -> int:
    """Function to convert Tm to index
    
    Args:
    - tm (float): list of list of Tm values
    - initial_T (float): The initial temperature
    - rate_per_min (float): The heating rate per minute
    - exposure_in_sec (float): The exposure time in seconds
    - max (float): The maximum temperature
    
    Returns:
    - int: The index value"""
    idx = np.zeros_like(np.array(tm))
    for i in range(len(tm)):
        if len(tm[i]) != 0:
            # print(tm[i])
            idx[i] = int((tm[i][0] - initial_T) / (rate_per_min * exposure_in_sec / 60))
            # if idx[i] > max:
            #     idx[i] = max
        else:
            idx[i] = np.nan
    return idx

def plot_individual_probe_signal(data_array: np.ndarray,tm_array,
                                initial_T: float, final_T: float, heating_rate: float, img_series_gap_time: float,
                                raw_data_array = None):
    idx_array = list_of_tm_to_index(tm_array, initial_T, heating_rate, img_series_gap_time, final_T)
    # print(idx_array.shape, data_array.shape)
    for i, signal in enumerate(data_array):
        plt.figure(figsize=(7, 3))
        plt.plot(signal)
        plt.xlabel("Frames")
        plt.ylabel("Normalized -dF/dT")
        plt.title(f"Probe signal {i}")
        plt.ylim(0,1)
        # plt.show()

    
        if np.isnan(idx_array[i]):
            continue
        elif type(idx_array[i]) is not int:
            continue
        else:
            # plt.figure(figsize=(7, 3))
            # plt.plot(signal)
            plt.plot(idx_array[i], signal[idx_array[i]], 'x', color='red', markersize=10)
            # plt.xlabel("Frames")
            # plt.ylabel("Normalized -dF/dT")
            # plt.title(f"Probe signal {i}")
            # plt.show()
        plt.show()

        


def plot_individual_probe_signal(data_array: np.ndarray, tm_array,
                                 initial_T: float, final_T: float, heating_rate: float, img_series_gap_time: float,
                                 raw_data_array=None, raw_data_array_tm=None, y_lim=None,
                                   data_array_ticks=None, data_array_labels=None,
                                   raw_data_array_ticks=None, raw_data_array_labels=None):
    idx_array = list_of_tm_to_index(tm_array, initial_T, heating_rate, img_series_gap_time, final_T)

    if raw_data_array is not None:
        raw_data_idx_array = list_of_tm_to_index(raw_data_array_tm, initial_T, heating_rate, img_series_gap_time, final_T)
    
    for i, signal in enumerate(data_array):
        if raw_data_array is not None:
            # If raw_data_array is provided, plot side by side
            raw_signal = raw_data_array[i]
            fig, axs = plt.subplots(1, 2, figsize=(14, 3))
            
            # Plot normalized signal
            axs[0].plot(signal)
            axs[0].set_xlabel("Frames")
            axs[0].set_ylabel("Normalized -dF/dT")
            axs[0].set_title(f"Normalized Probe Signal {i}")
            # axs[0].set_ylim(0, 1)
            if y_lim is not None:
                axs[0].set_ylim(y_lim)
            
            if not (np.isnan(idx_array[i]) or type(idx_array[i]) is not int):
                axs[0].plot(idx_array[i], signal[idx_array[i]], 'x', color='red', markersize=10)
            if data_array_ticks is not None:
                axs[0].set_xticks(data_array_ticks, labels=data_array_labels)
            
            # Plot raw signal
            axs[1].plot(raw_signal)
            axs[1].set_xlabel("Frames")
            axs[1].set_ylabel("Raw Signal")
            axs[1].set_title(f"Raw Probe Signal {i}")
            if y_lim is not None:
                axs[1].set_ylim(y_lim)
            # axs[0].set_ylim(0, 1)

            if raw_data_array_tm is not None:
                if not (np.isnan(raw_data_idx_array[i]) or type(raw_data_idx_array[i]) is not int):
                    axs[1].plot(raw_data_idx_array[i], raw_signal[raw_data_idx_array[i]], 'x', color='red', markersize=10)
            if raw_data_array_ticks is not None:
                axs[1].set_xticks(raw_data_array_ticks, labels=raw_data_array_labels)
            
            # if not (np.isnan(idx_array[i]) or type(idx_array[i]) is not int):
                # axs[1].plot(idx_array[i], raw_signal[idx_array[i]], 'x', color='red', markersize=10)
            
            plt.tight_layout()
            # plt.ylim(0,1)
            plt.show()
        else:
            # Plot only the normalized signal if raw_data_array is not provided
            plt.figure(figsize=(7, 3))
            plt.plot(signal)
            plt.xlabel("Frames")
            plt.ylabel("Normalized -dF/dT")
            plt.title(f"Probe Signal {i}")
            # plt.ylim(0,1)
            if y_lim is not None:
                plt.ylim(y_lim)
            
            if not (np.isnan(idx_array[i]) or type(idx_array[i]) is not int):
                plt.plot(idx_array[i], signal[idx_array[i]], 'x', color='red', markersize=10)
            
            plt.show()





def generate_variable_threshold(signal_length, low_threshold_segments, high_threshold_value, low_threshold_value):
    """
    Generate a variable threshold array for find_peaks.

    Parameters:
    - signal_length (int): Length of the signal.
    - low_threshold_segments (list of tuples): Segments where low threshold should be applied.
        Format: [(start, end), (start,)] where single value tuple means from start to end of signal
    - high_threshold_value (float): Value of the high threshold.
    - low_threshold_value (float): Value of the low threshold.

    Returns:
    - np.ndarray: Array of threshold values.
    """
    # Input validation
    if not isinstance(low_threshold_value, (int, float)):
        raise ValueError(f"low_threshold_value must be numeric, got {type(low_threshold_value)}")
    if not isinstance(high_threshold_value, (int, float)):
        raise ValueError(f"high_threshold_value must be numeric, got {type(high_threshold_value)}")
    
    # Create array with explicit float dtype to prevent integer conversion
    threshold = np.full(signal_length, high_threshold_value, dtype=float)

    for segment in low_threshold_segments:
        if len(segment) == 1:
            # Single value means from that index to the end
            start = segment[0]
            end = signal_length
        elif len(segment) == 2:
            start, end = segment
            start = start if start is not None else 0
            end = end if end is not None else signal_length
        else:
            raise ValueError("Each segment must be a tuple of length 1 or 2")
        
        # Ensure indices are within bounds
        start = max(0, min(start, signal_length))
        end = max(0, min(end, signal_length))
        
        if start < end:
            # Explicitly convert low_threshold_value to float
            threshold[start:end] = float(low_threshold_value)
    
    # Verify the values are set correctly
    if not np.allclose(np.unique(threshold), [low_threshold_value, high_threshold_value]):
        raise ValueError(f"Unexpected values in threshold array. Expected only {low_threshold_value} and {high_threshold_value}, "
                        f"got {np.unique(threshold)}")
    
    return threshold



def join_all_tms(parsed_local_tms_list: List[np.ndarray], global_tms: np.ndarray, max_n_tms_list: List[int], 
                 ordered_channel_names: List[str], expected_tm_vals: List[List[int]], 
                 encoding: bool = False, keep_nonSpecific: bool = False, output_DataFrame: bool = True) -> Union[np.ndarray, pd.DataFrame]:
    """
    Function to join all Tms from local and global sources

    Args:
    - parsed_local_tms_list (list): List of parsed local Tm arrays (each array corresponds to a different channel)
    - global_tms (np.ndarray): Array of global Tm values
    - max_n_tms_list (list): List of max_n_tms for each of the parsed_local_tms
    - ordered_channel_names (list): List of channel names corresponding to each parsed_local_tms array
    - expected_tm_vals (list of lists): List of expected Tm values for each channel
    - encoding (bool): If True, use encoded values; otherwise, use actual Tm values
    - keep_nonSpecific (bool): If False, set non-specific Tm values to NaN
    - output_DataFrame (bool): If True, return a pandas DataFrame

    Returns:
    - np.ndarray or pd.DataFrame: Joined Tm values from local and global sources
    """
    if not all(isinstance(array, np.ndarray) for array in parsed_local_tms_list):
        raise ValueError("All elements in parsed_local_tms_list should be numpy arrays.")
    if not isinstance(global_tms, np.ndarray):
        raise ValueError("global_tms should be a numpy array.")
    if not all(isinstance(n, int) for n in max_n_tms_list):
        raise ValueError("All elements in max_n_tms_list should be integers.")
    if not (len(ordered_channel_names) == len(parsed_local_tms_list) == len(expected_tm_vals)):
        raise ValueError("ordered_channel_names, parsed_local_tms_list, and expected_tm_vals must have the same length.")

    parsed_local_tms_list = [array.copy() for array in parsed_local_tms_list]
    global_tms = global_tms.copy()
    max_n_tms_list = max_n_tms_list.copy()

    if not keep_nonSpecific:
        for array, max_n_tms in zip(parsed_local_tms_list, max_n_tms_list):
            for i in range(array.shape[0]):  # Iterate through rows
                for j in range(max_n_tms):  # Iterate through first max_n_tms columns
                    if array[i, j + max_n_tms] == 0:
                        array[i, j] = np.nan

    all_local_tms = []
    for array, max_n_tms, expected_tms in zip(parsed_local_tms_list, max_n_tms_list, expected_tm_vals):
        if encoding:
            encoded_array = np.zeros_like(array[:, :max_n_tms])
            for i in range(array.shape[0]):  # Iterate through rows
                for j in range(max_n_tms):  # Iterate through first max_n_tms columns
                    tm_indicator = array[i, j + max_n_tms]
                    if tm_indicator == 1:
                        encoded_array[i, j] = expected_tms[0]
                    elif tm_indicator == 2:
                        encoded_array[i, j] = expected_tms[1]
                    else:
                        encoded_array[i, j] = 0
            all_local_tms.append(encoded_array)
        else:
            all_local_tms.append(array[:, :max_n_tms])

    all_local_tms = np.hstack(all_local_tms)
    joined_tms = np.hstack([all_local_tms, global_tms])

    n_global_tms = global_tms.shape[1]
    if output_DataFrame:
        joined_tms = pd.DataFrame(
            joined_tms,
            columns=[f"{channel} Tm {j + 1}" for channel, max_n_tms in zip(ordered_channel_names, max_n_tms_list) for j in range(max_n_tms)] +
                    [f"Global Tm {i + 1}" for i in range(n_global_tms)]
        )

    return joined_tms








def join_meta_data(joined_tms: pd.DataFrame, pos_info: np.ndarray, raw_fluorescence: np.ndarray, confidence: list = None) -> pd.DataFrame:
    """
    Function to join Tm values with positional and raw fluorescence data

    Args:
    - joined_tms (pd.DataFrame): DataFrame containing joined Tm values from local and global sources
    - pos_info (np.ndarray): 2D array containing positional information
    - raw_fluorescence (np.ndarray): 2D array containing raw fluorescence data
    - confidence (list, optional): List of 1D numpy arrays of confidence values

    Returns:
    - pd.DataFrame: Combined DataFrame with Tm values, positional information, and raw fluorescence data
    """
    # Make copies of all the inputs to avoid changing the original data
    pos_info = pos_info.copy()
    raw_fluorescence = raw_fluorescence.copy()


    # Convert numpy arrays to DataFrames
    pos_info_df = pd.DataFrame(pos_info, columns=["Pos X", "Pos Y"])
    raw_fluorescence_df = pd.DataFrame(raw_fluorescence, columns=[f"Intensity {i + 1}" for i in range(raw_fluorescence.shape[1])])
    # print(raw_fluorescence_df.shape, joined_tms.shape, joined_tms.shape)

    if confidence is not None:
        confidence_df = pd.DataFrame({f"Confidence {i + 1}": confidence[i] for i in range(len(confidence))})
        combined_df = pd.concat([pos_info_df, joined_tms, raw_fluorescence_df, confidence_df], axis=1)
    else:
        combined_df = pd.concat([pos_info_df, joined_tms, raw_fluorescence_df], axis=1)
    # print(combined_df.shape)
    return combined_df


def interactive_anomaly_filtering(data_array, mode="iso", max = 0.1):
    """Function that creates an interactive visualization for anomaly filtering using various methods.
    When the contamination is adjusted using the slider, the visualization and outputs both update accordingly.

    Args:
    - data_array (np.array): The input data array
    - mode (str): The method to use for filtering; choose from 'isolation_forest', 'lof', 'one_class_svm', 'elliptic_envelope'

    Returns:
    - dict: A dictionary containing the normal mask and the anomaly mask
    """
    masks = {'normal_mask': None, 'anomaly_mask': None}
    output = widgets.Output()
    
    def update_visualization(change):
        contamination = change['new']
        if contamination == 0:
            normal_mask = np.ones(data_array.shape[0], dtype=bool)
            anomaly_mask = np.zeros(data_array.shape[0], dtype=bool)
        else:
            if mode == "iso":
                normal_mask, anomaly_mask = anomaly_filter_by_isolation_forest(data_array, contamination)
            elif mode == "lof":
                normal_mask, anomaly_mask = anomaly_filter_by_lof(data_array, contamination)
            elif mode == "svm":
                normal_mask, anomaly_mask = anomaly_filter_by_one_class_svm(data_array, contamination)
            elif mode == "autoencoder":
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data_array)
                autoencoder_model = train_autoencoder(scaled_data, epochs=50)
                normal_mask, anomaly_mask = anomaly_filter_by_autoencoder(scaled_data, autoencoder_model, contamination)

        masks['normal_mask'] = normal_mask
        masks['anomaly_mask'] = anomaly_mask
        with output:
            output.clear_output(wait=True)
            visualize_anomaly_filtering(data_array, normal_mask, anomaly_mask)
    
    contamination_slider = widgets.FloatSlider(min=0, max=max, step=0.001, value=0.0, description='Threshold', layout=widgets.Layout(width='1000px'), readout_format='.4f')
    contamination_slider.observe(update_visualization, names='value')
    
    display(VBox([contamination_slider, output]))
    update_visualization({'new': 0.0})
    
    return masks

def anomaly_filter_by_isolation_forest(data, contamination):
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    labels = iso_forest.fit_predict(data)
    normal_mask = labels == 1
    anomaly_mask = labels == -1
    return normal_mask, anomaly_mask

def anomaly_filter_by_lof(data, contamination):
    n_neighbors = int(len(data) * contamination) + 1
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    labels = lof.fit_predict(data)
    normal_mask = labels == 1
    anomaly_mask = labels == -1
    return normal_mask, anomaly_mask

def anomaly_filter_by_one_class_svm(data, contamination):
    one_class_svm = OneClassSVM(nu=contamination, kernel="rbf", gamma='auto')
    labels = one_class_svm.fit_predict(data)
    normal_mask = labels == 1
    anomaly_mask = labels == -1
    return normal_mask, anomaly_mask



def visualize_anomaly_filtering(original_data: np.ndarray, normal_mask: np.ndarray, anomaly_mask: np.ndarray):
    """Function that visualizes the anomaly filtering process

    Args:
    - original_data (np.array): The original data array
    - normal_mask (np.array): The normal mask
    - anomaly_mask (np.array): The anomaly mask

    Displays:
    - The visualization of the anomaly filtering process
    """
    normal_data = original_data[normal_mask]
    anomaly_data = original_data[anomaly_mask]
    
    normalized_normal_data = min_max_normalize(normal_data, use_global_min_max=False)
    normalized_anomaly_data = min_max_normalize(anomaly_data, use_global_min_max=False)

    fig_width = 15
    fig_height = 5
    plt.figure(figsize=(fig_width, fig_height))

    plt.subplot(1, 2, 1) 
    for signal in normalized_normal_data:
        plt.plot(signal, alpha = 0.1, color = "cornflowerblue")
    plt.xlabel("Temperature")
    plt.ylabel("Normalized Value")
    plt.title("Normalized Normal Data")

    plt.subplot(1, 2, 2)
    for signal in normalized_anomaly_data:
        plt.plot(signal, alpha = 0.1, color = "cornflowerblue")
    plt.xlabel("Temperature")
    plt.ylabel("Normalized Value")
    plt.title("Normalized Anomalies")
    plt.tight_layout()
    plt.show()




class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def train_autoencoder(data, epochs = 500, batch_size=64):
    input_dim = data.shape[1]
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    data_tensor = torch.tensor(data, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(data_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for batch_data in dataloader:
            batch_data = batch_data[0]
            output = model(batch_data)
            loss = criterion(output, batch_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

def anomaly_filter_by_autoencoder(data, model, contamination):
    model.eval()
    data_tensor = torch.tensor(data, dtype=torch.float32)
    with torch.no_grad():
        reconstructed = model(data_tensor).cpu().numpy()
    reconstruction_errors = np.mean((data - reconstructed) ** 2, axis=1)
    threshold = np.percentile(reconstruction_errors, 100 * (1 - contamination))
    normal_mask = reconstruction_errors <= threshold
    anomaly_mask = reconstruction_errors > threshold
    return normal_mask, anomaly_mask




def split_dataframe_by_columns(df, columns):
    """
    Splits the DataFrame into a dictionary of subset DataFrames based on unique combinations of the values in the specified columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to split by.

    Returns:
    dict: A dictionary where keys are tuples of unique combinations of the values in the specified columns,
          and values are the subset DataFrames.
    """
    unique_combinations = df[columns].drop_duplicates()
    subset_dfs = {}

    for _, row in unique_combinations.iterrows():
        # Create a mask for the current combination
        mask = (df[columns] == row.values).all(axis=1)
        # Add the subset DataFrame to the dictionary
        subset_dfs[tuple(row.values)] = df[mask]

    return subset_dfs

def plot_subset_scatter(df_dict, col_names, channel_names):
    """
    Creates a 2D scatter plot for each subset DataFrame in the dictionary based on the selected column names.

    Parameters:
    df_dict (dict): Dictionary of subset DataFrames.
    col_names (str): Name of the columns to scatter plot.
    channel_names (list): List of channel names corresponding to the columns in the DataFrame.
    """
    x_col, y_col = col_names
    y_min, x_min = 77, 72
    y_max, x_max = 90.1, 86.5
    
    for key, subset_df in df_dict.items():
        plt.figure(figsize=(15, 13))
        plt.scatter(subset_df[x_col], subset_df[y_col], marker='o', alpha=0.7, s=7)
        
        plt.xlabel(x_col, fontsize=20)
        plt.ylabel(y_col, fontsize=20)
        
        plt.xticks(np.arange(x_min, x_max, 0.5), fontsize=10)
        plt.yticks(np.arange(y_min, y_max, 0.5), fontsize=10)
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        plt.grid(True)
        

        title_parts = []
        for idx, val in enumerate(key):
            col_name = channel_names[idx]
            tm_value = val if val != 0 else "None"
            title_parts.append(f'{col_name}: {tm_value}')
        title = ' | '.join(title_parts)
        
        plt.title(f'Probe Tm: {title}', fontsize=15)
        plt.show()

def plot_subset_scatter(df_dict, col_names, channel_names):
    """
    Creates individual 2D scatter plots for each subset DataFrame and a combined plot with all subsets.

    Parameters:
    df_dict (dict): Dictionary of subset DataFrames.
    col_names (str): Name of the columns to scatter plot.
    channel_names (list): List of channel names corresponding to the columns in the DataFrame.
    """
    x_col, y_col = col_names
    y_min, x_min = 77, 72
    y_max, x_max = 90.1, 86.5
    
    # First create individual plots with original parameters
    for key, subset_df in df_dict.items():
        plt.figure(figsize=(15, 13))
        plt.scatter(subset_df[x_col], subset_df[y_col], marker='o', alpha=0.7, s=7)
        
        plt.xlabel(x_col, fontsize=20)
        plt.ylabel(y_col, fontsize=20)
        
        plt.xticks(np.arange(x_min, x_max, 0.5), fontsize=10)
        plt.yticks(np.arange(y_min, y_max, 0.5), fontsize=10)
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        plt.grid(True)

        title_parts = []
        for idx, val in enumerate(key):
            col_name = channel_names[idx]
            tm_value = val if val != 0 else "None"
            title_parts.append(f'{col_name}: {tm_value}')
        title = ' | '.join(title_parts)
        
        plt.title(f'Probe Tm: {title}', fontsize=15)
        plt.show()
    
    # Create combined plot with same parameters
    plt.figure(figsize=(15, 13))
    
    # Generate a color map with distinct colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(df_dict)))
    
    # Plot each subset with a different color
    for (key, subset_df), color in zip(df_dict.items(), colors):
        plt.scatter(subset_df[x_col], subset_df[y_col], 
                   marker='o', alpha=0.7, s=7, 
                   color=color, 
                   label=' | '.join([f'{channel_names[idx]}: {val if val != 0 else "None"}' 
                                   for idx, val in enumerate(key)]))
    
    plt.xlabel(x_col, fontsize=20)
    plt.ylabel(y_col, fontsize=20)
    
    plt.xticks(np.arange(x_min, x_max, 0.5), fontsize=10)
    plt.yticks(np.arange(y_min, y_max, 0.5), fontsize=10)
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    plt.grid(True)
    plt.title('Probe Tm: Combined Plot', fontsize=15)
    
    # Add legend with smaller font
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Adjust layout to prevent legend from being cut off
    plt.tight_layout()
    plt.show()


def shift_nonzero_to_first(df, column_pairs):
    """
    Shifts non-zero values to the first column of each pair if the first column is 0.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_pairs (list of tuples): List of column pairs to check and shift values.

    Returns:
    pd.DataFrame: The modified DataFrame.
    """
    for col1, col2 in column_pairs:
        mask = (df[col1] == 0) & (df[col2] != 0)
        df.loc[mask, col1] = df.loc[mask, col2]
        df.loc[mask, col2] = 0
    return df



def save_subset_dfs(df_dict):
    """
    Saves the subset DataFrames in the dictionary to individual CSV files.

    Parameters:
    df_dict (dict): Dictionary of subset DataFrames.
    """

    cwd = os.getcwd()
    folder_name = os.path.basename(cwd)

    for key, subset_df in df_dict.items():
        file_name = f"{folder_name}_{key}.csv"
        subset_df.to_csv(file_name, index=True)
        print(f"{key} Subset is saved as '{file_name}'.")
    
    return










def interactive_visual_QC(plot_dfs, external_arr_list1, selected_col_names,
                          external_arr_list2, 
                          external_arr_list3=None, 
                          external_arr_list4=None, 
                          plot_12_ticks=None, plot_34_ticks=None, local_tm_range=None):
    """
    Function to create an interactive visualization for quality control of data. When hovering over a point in the scatter plot, the corresponding external plots are updated.
    When clicking on a point in the scatter plot, the validity status of the point is toggled.
    
    Args:
    - plot_dfs (list): List of DataFrames to plot
    - external_arr_list1 (list): List of arrays for external plot 1
    - selected_col_names (list): List of column names to plot
    - external_arr_list2 (list): List of arrays for external plot 2
    - external_arr_list3 (list, optional): List of arrays for external plot 3
    - external_arr_list4 (list, optional): List of arrays for external plot 4
    - plot_12_ticks (list, optional): List of ticks for external plots 1 and 2
    - plot_34_ticks (list, optional): List of ticks for external plots 3 and 4
    
    Returns:
    - Updated plot_dfs"""
    assert len(plot_dfs) == len(external_arr_list1) == len(external_arr_list2), "Input lists must have the same length"
    if external_arr_list3 is not None:
        assert len(plot_dfs) == len(external_arr_list3)
    if external_arr_list4 is not None:
        assert len(plot_dfs) == len(external_arr_list4)

    plot_dfs_copy = [df.copy(deep=True) for df in plot_dfs]

    if local_tm_range is not None:
        for i, external_arr in enumerate(external_arr_list3):
            external_arr_list3[i] = external_arr[:, local_tm_range[0]:local_tm_range[1]]
        for i, external_arr in enumerate(external_arr_list4):
            external_arr_list4[i] = external_arr[:, local_tm_range[0]:local_tm_range[1]]


    for df in plot_dfs_copy:
        # print("got here")
        df['validity_status'] = 1

    app = dash.Dash(__name__)

    combined_df = pd.concat(plot_dfs_copy, keys=range(len(plot_dfs_copy)), names=['df_index'])
    combined_df.reset_index(level='df_index', inplace=True)
    combined_df['external_index'] = combined_df.iloc[:, 1]
    # print(combined_df["external_index"])
    # print(combined_df)

    scatter_fig = px.scatter(
        combined_df, 
        x=selected_col_names[0], 
        y=selected_col_names[1], 
        color=combined_df['df_index'].astype(str), 
        labels={'df_index': 'DataFrame'},
        custom_data=['df_index', 'external_index', 'validity_status'],
        opacity=[1 if status == 1 else 0.3 for status in combined_df['validity_status']]
    )

    scatter_width = 800
    scatter_height = 800

    scatter_fig.update_layout(
        xaxis=dict(range=[None, None]),
        yaxis=dict(range=[None, None]),
        height=scatter_height,
        width=scatter_width
    )
    app.layout = html.Div([
        html.Div([
            html.P("Click on points to toggle their validity status. Gray points are marked as invalid."),
            dcc.Graph(id='main-scatter', figure=scatter_fig, style={'height': scatter_height, 'width': scatter_width})
        ], style={'grid-column': '2 / 3', 'grid-row': '1 / 5'}),
        
        dcc.Graph(id='external-plot1', style={'grid-column': '1 / 2', 'grid-row': '1 / 2', 'height': '35vh'}),
        dcc.Graph(id='external-plot2', style={'grid-column': '1 / 2', 'grid-row': '2 / 3', 'height': '35vh'}),
        dcc.Graph(id='external-plot3', style={'grid-column': '1 / 2', 'grid-row': '3 / 4', 'height': '40vh'}),
        dcc.Graph(id='external-plot4', style={'grid-column': '1 / 2', 'grid-row': '4 / 5', 'height': '40vh'}),
    ], style={
        'display': 'grid',
        'grid-template-columns': '50%, 50%',
        'grid-template-rows': 'repeat(4, 40vh)',
        'height': '100vh',
        'width': '105vw'
    })
    @app.callback(
        [Output('external-plot1', 'figure'),
        Output('external-plot2', 'figure'),
        Output('external-plot3', 'figure'),
        Output('external-plot4', 'figure')],
        [Input('main-scatter', 'hoverData')]
    )
    def update_side_plots(hoverData):
        if hoverData is None:
            return [go.Figure(), go.Figure(), go.Figure(), go.Figure()]

        point = hoverData['points'][0]
        df_index = int(point['customdata'][0])
        external_index = int(point['customdata'][1])


        def create_figure(data, title, ticks):
            fig = go.Figure(data=go.Scatter(y=data, x=list(range(len(data)))))
            fig.update_layout(title=title)
            if ticks is not None:
                tick_interval = len(data) / (len(ticks) - 1)
                fig.update_xaxes(
                    tickmode='array',
                    tickvals=[i * tick_interval for i in range(len(ticks))],
                    ticktext=ticks,
                    # tilt the ticks
                    tickangle=45
                )
            return fig
        
        # print(f"DataFrame {df_index}, External Index {external_index}")

        fig1 = create_figure(external_arr_list1[df_index][external_index], 'EvaGreen Raw', plot_12_ticks)
        fig2 = create_figure(external_arr_list2[df_index][external_index], 'EvaGreen Derivative', plot_12_ticks)
        # print(external_arr_list1[df_index].shape, external_arr_list2[df_index].shape)


        if external_arr_list3 is not None:
            fig3 = create_figure(external_arr_list3[df_index][external_index], 'Hex Derivative', plot_34_ticks)
        else:
            fig3 = go.Figure()
            fig3.update_layout(title='Hex Derivative (Not Available)')

        if external_arr_list4 is not None:
            fig4 = create_figure(external_arr_list4[df_index][external_index], 'CY5 Derivative', plot_34_ticks)
        else:
            fig4 = go.Figure()
            fig4.update_layout(title='CY5 Derivative (Not Available)')

        return [fig1, fig2, fig3, fig4]
    
    @app.callback(
        Output('main-scatter', 'figure'),
        Input('main-scatter', 'clickData'),
        State('main-scatter', 'figure')
    )
    def update_point_validity(clickData, current_figure):
        if clickData is None:
            return current_figure

        point = clickData['points'][0]
        df_index = int(point['customdata'][0])
        external_index = int(point['customdata'][1])
        clicked_x = point['x']
        clicked_y = point['y']

        # print(f"DataFrame {df_index}, External Index {external_index}, x={clicked_x}, y={clicked_y}")

        df = plot_dfs_copy[df_index]

        # print(f"plot_df_shape: {df.shape}")

        matching_rows = df.index[(df[selected_col_names[0]] == clicked_x) & (df[selected_col_names[1]] == clicked_y)]

        if len(matching_rows) == 0:
            print(f"Warning: No matching rows found for x={clicked_x}, y={clicked_y} in DataFrame {df_index}")
            return current_figure

        for row_index in matching_rows:
            current_validity = df.loc[row_index, 'validity_status']
            new_validity = 1 - current_validity
            df.loc[row_index, 'validity_status'] = new_validity

            for trace in current_figure['data']:
                for i, (custom_df_index, custom_external_index, _) in enumerate(trace['customdata']):
                    if trace['x'][i] == clicked_x and trace['y'][i] == clicked_y:
                        trace['marker']['opacity'][i] = 1 if new_validity == 1 else 0.3

        return current_figure
    
    free_port = find_free_port()

    app.run_server(debug=True, port=free_port)
    return plot_dfs_copy






def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port



def scatter_plot_dfs(dfs, x_col, y_col):
    """
    Scatter plots selected columns of all dataframes on the same 2D plot.

    Parameters:
    dfs (list of pd.DataFrame): List of dataframes to plot
    x_col (str): Column name to plot on the x-axis
    y_col (str): Column name to plot on the y-axis
    """
    plt.figure(figsize=(15, 13))

    y_min, x_min = 77, 72
    y_max, x_max = 90.1, 86.5

    colors = plt.cm.get_cmap('tab10', len(dfs))  # Get a colormap with distinct colors

    for i, df in enumerate(dfs):
        if i == 0:
            plt.scatter(df[x_col], df[y_col], color="black", label=f'DF {i+1}', alpha=1, s=60)
        else:
            plt.scatter(df[x_col], df[y_col], color=colors(i), label=f'DF {i+1}', alpha=0.6, s=10)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('Scatter Plot of DataFrames')
    plt.legend()
    
    plt.grid(True)
    plt.show()



def greedy_surjective_constrained_matching(
    set_A,
    set_B,
    cost_matrix: np.ndarray,
    max_cost_threshold: float
) -> pd.DataFrame:
    """
    Perform a surjective constrained matching from set A to set B.
    
    Args:
    set_A (List[Any]): The larger set with size 'a'.
    set_B (List[Any]): The smaller set with size 'b'.
    cost_matrix (np.ndarray): A 2D array of shape (a, b) containing the cost of each possible pairing.
    max_cost_threshold (float): The maximum allowable cost for any single mapping.
    
    Returns:
    pd.DataFrame: A DataFrame where each column represents an element in set B,
                  and each cell contains the ID of an element from set A or np.nan.
    """
    a, b = len(set_A), len(set_B)
    # assert a > b, "Set A must be larger than set B"
    # assert cost_matrix.shape == (a, b), f"Cost matrix shape {cost_matrix.shape} doesn't match set sizes ({a}, {b})"
    
    valid_mask = cost_matrix <= max_cost_threshold
    

    matching = {elem_B: [] for elem_B in set_B}
    
    for i, elem_A in enumerate(set_A):
        valid_options = np.where(valid_mask[i])[0]
        if len(valid_options) > 0:
            best_match = valid_options[np.argmin(cost_matrix[i][valid_options])]
            matching[set_B[best_match]].append(elem_A)
    
    for i, elem_A in enumerate(set_A):
        if all(elem_A not in matched for matched in matching.values()):
            valid_options = np.where(valid_mask[i])[0]
            for j in valid_options:
                if len(matching[set_B[j]]) < len(set_A): 
                    matching[set_B[j]].append(elem_A)
                    break
    
    max_matches = max(len(v) for v in matching.values())
    df = pd.DataFrame({k: v + [np.nan] * (max_matches - len(v)) for k, v in matching.items()})
    
    # Check if the matching is surjective
    total_matched = df.notna().sum().sum()
    if total_matched < a:
        print(f"Warning: Not all elements in A could be matched. {total_matched}/{a} elements matched.")
    
    return df





def plot_matching_interactive(matching_df, external_df, selected_column_names, std_x=0.25, std_y=0.25, off_diag=0, default_n_SD=1.5):
    result = {'updated_matching_df': pd.DataFrame(), 'stats_df': pd.DataFrame()}
    output = widgets.Output()
    final_fig = None

    updated_matching_df = pd.DataFrame()
    stats_df = pd.DataFrame()

    save_button = widgets.Button(description='Save Results', button_style='success')

    def calculate_confidence_scores(x, y, center_x, center_y, std_x, std_y):
        # Calculate normalized distances
        x_dist = (x - center_x) / std_x
        y_dist = (y - center_y) / std_y
        
        distances = np.sqrt(x_dist**2 + y_dist**2)
        
        # Convert distances to confidence scores using Gaussian function
        confidence_scores = np.exp(-distances**2 / (2 * 2**2))
        return confidence_scores

    def update_plot(change):
        nonlocal final_fig, updated_matching_df, stats_df
        n_SD = n_SD_slider.value

        with output:
            clear_output(wait=True)
            
            fig, ax = plt.subplots(figsize=(13, 16))
            y_min, x_min = external_df[selected_column_names[1]].min(), external_df[selected_column_names[0]].min()
            y_max, x_max = external_df[selected_column_names[1]].max(), external_df[selected_column_names[0]].max()
            is_multi_index = isinstance(external_df.index, pd.MultiIndex)
            matched_indices = set()
            unmatched_indices = set()

            updated_matching_df = pd.DataFrame(index=matching_df.index, columns=matching_df.columns)

            for col in matching_df.columns:
                if not is_multi_index:
                    indices = matching_df[col].dropna().astype(int).tolist()
                else:
                    indices = matching_df[col].dropna().tolist()

                matched_indices.update(indices)

                if is_multi_index:
                    multi_index_tuples = [(i,) if not isinstance(i, tuple) else i for i in indices]
                    cluster_data = external_df.loc[multi_index_tuples]
                else:
                    cluster_data = external_df.iloc[indices]
                x = cluster_data[selected_column_names[0]]
                y = cluster_data[selected_column_names[1]]

                ax.scatter(x, y, label=col, s=10)
                if len(x) > 0 and len(y) > 0:
                    center_x = np.median(x)
                    center_y = np.median(y)

                    x_dist = (x - center_x) / std_x
                    y_dist = (y - center_y) / std_y

                    inside_ellipse = (x_dist**2 + y_dist**2) <= n_SD**2

                    cov_xy = off_diag
                    cov_matrix = np.array([[std_x**2, cov_xy],
                                        [cov_xy, std_y**2]])
                    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
                    eigvals = np.maximum(eigvals, 0)

                    largest_eigvec = eigvecs[:, 1]
                    angle_of_tilt = np.degrees(np.arctan2(largest_eigvec[1], largest_eigvec[0])) 
                    a = n_SD * std_x
                    b = n_SD * std_y
                    
                    inside_ellipse = (((x - center_x) * np.cos(np.radians(angle_of_tilt)) + (y - center_y) * np.sin(np.radians(angle_of_tilt))) ** 2 / a ** 2 + 
                                    ((x - center_x) * np.sin(np.radians(angle_of_tilt)) - (y - center_y) * np.cos(np.radians(angle_of_tilt))) ** 2 / b ** 2) <= 1

                    x_inside = x[inside_ellipse]
                    y_inside = y[inside_ellipse]

                    if len(x_inside) > 0 and len(y_inside) > 0:
                        center_x = np.median(x_inside)
                        center_y = np.median(y_inside)

                        cov_xy = off_diag
                        cov_matrix = np.array([[std_x**2, cov_xy],
                                            [cov_xy, std_y**2]])
                        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
                        eigvals = np.maximum(eigvals, 0)

                        largest_eigvec = eigvecs[:, 1]
                        angle_of_tilt = np.degrees(np.arctan2(largest_eigvec[1], largest_eigvec[0])) 

                        a = n_SD * std_x
                        b = n_SD * std_y
                        
                        inside_ellipse = (((x - center_x) * np.cos(np.radians(angle_of_tilt)) + (y - center_y) * np.sin(np.radians(angle_of_tilt))) ** 2 / a ** 2 + 
                                        ((x - center_x) * np.sin(np.radians(angle_of_tilt)) - (y - center_y) * np.cos(np.radians(angle_of_tilt))) ** 2 / b ** 2) <= 1

                        ellipse = patches.Ellipse((center_x, center_y), width=2*a, height=2*b,
                                                angle=angle_of_tilt, edgecolor='cornflowerblue', fc='None', lw=1, alpha=0.5)

                        ax.add_patch(ellipse)

                        ax.plot(center_x, center_y, marker='+', color='cornflowerblue', markersize=5)

                        matched_within_ellipse = len(x_inside)
                        ax.annotate(f'{col}: {matched_within_ellipse}', (center_x, center_y), fontsize=12, ha='right')

                        if is_multi_index:
                            updated_matching_df[col] = pd.Series([(i,) if not isinstance(i, tuple) else i for i in cluster_data.index[inside_ellipse]])
                        else:
                            updated_matching_df[col] = pd.Series(cluster_data.index[inside_ellipse])

                    unmatched_indices.update(cluster_data.index[~inside_ellipse])

            if is_multi_index:
                all_indices = set(external_df.index)
            else:
                all_indices = set(external_df.index.tolist())

            unmatched_indices.update(all_indices - matched_indices)
            if is_multi_index:
                unmatched_data = external_df.loc[list(unmatched_indices)]
            else:
                unmatched_data = external_df.iloc[list(unmatched_indices)]

            x_unmatched = unmatched_data[selected_column_names[0]]
            y_unmatched = unmatched_data[selected_column_names[1]]

            ax.scatter(x_unmatched, y_unmatched, color='black', label='Unmatched', s=10)

            num_clusters = sum(matching_df[col].notna().any() for col in matching_df.columns)
            ax.set_title(f'Number of Clusters Discovered: {num_clusters}')
            ax.set_xlabel(selected_column_names[0])
            ax.set_ylabel(selected_column_names[1])
            ax.grid(True)
            plt.legend()
            plt.show()

            stats = []
            for col in updated_matching_df.columns:
                matched_indices = updated_matching_df[col].dropna().tolist()
                if matched_indices:
                    if is_multi_index:
                        cluster_data = external_df.loc[matched_indices]
                    else:
                        cluster_data = external_df.iloc[matched_indices]

                    count = len(cluster_data)
                    median_low_tm = cluster_data['LowTm'].median()
                    median_high_tm = cluster_data['HighTm'].median()

                    # Calculate confidence scores for the cluster
                    x = cluster_data[selected_column_names[0]]
                    y = cluster_data[selected_column_names[1]]
                    center_x = np.median(x)
                    center_y = np.median(y)
                    
                    confidence_scores = calculate_confidence_scores(x, y, center_x, center_y, std_x, std_y)
                    counts_by_confidence = np.sum(confidence_scores)

                    median_low_tm = round(median_low_tm, 2)
                    median_high_tm = round(median_high_tm, 2)
                    counts_by_confidence = round(counts_by_confidence, 2)

                    stats.append({
                        'cluster_name': col,
                        'counts': count,
                        'counts_by_confidence': counts_by_confidence,
                        'LowTm': median_low_tm,
                        'HighTm': median_high_tm
                    })

            stats_df = pd.DataFrame(stats)
            final_fig = fig

    def save_results(b):
        result['updated_matching_df'] = updated_matching_df
        result['stats_df'] = stats_df
        print("Results have been saved to the 'result' dictionary.")

    n_SD_slider = widgets.FloatSlider(value=default_n_SD, min=0.1, max=3.0, step=0.05, description='n_SD:', layout=widgets.Layout(width='600px'), readout_format='.1f')
    n_SD_slider.observe(update_plot, names='value')

    save_button.on_click(save_results)

    display(VBox([HBox([n_SD_slider, save_button]), output]))
    update_plot(None)

    return result



def plot_matching(matching_df, external_df, selected_column_names):
    cluster_stats = {}

    fig, ax = plt.subplots(figsize=(13, 13))
    y_min, x_min = -0.1, -0.2
    y_max, x_max = 1.1, 1.1
    is_multi_index = isinstance(external_df.index, pd.MultiIndex)

    matched_indices = set()

    for col in matching_df.columns:
        if not is_multi_index:
            indices = matching_df[col].dropna().astype(int).tolist()
        else:
            indices = matching_df[col].dropna().tolist()

        matched_indices.update(indices)

        if is_multi_index:
            multi_index_tuples = [(i,) if not isinstance(i, tuple) else i for i in indices]
            cluster_data = external_df.loc[multi_index_tuples]
        else:
            cluster_data = external_df.iloc[indices]
        x = cluster_data[selected_column_names[0]]
        y = cluster_data[selected_column_names[1]]

        ax.scatter(x, y, label=col, s=10)
        cluster_stats[col] = {'Count': len(indices), 'ID': indices}
        if len(x) > 0 and len(y) > 0:
            center_x = np.median(x)
            center_y = np.median(y)
            ax.annotate(f'{col}:{cluster_stats[col]["Count"]}', (center_x, center_y), fontsize=12, ha='right')

        cluster_stats[col] = {'Count': len(indices), 'ID': indices}

    if is_multi_index:
        all_indices = set(external_df.index)
    else:
        all_indices = set(external_df.index.tolist())

    unmatched_indices = all_indices - matched_indices
    if is_multi_index:
        unmatched_data = external_df.loc[list(unmatched_indices)]
    else:
        unmatched_data = external_df.iloc[list(unmatched_indices)]

    x_unmatched = unmatched_data[selected_column_names[0]]
    y_unmatched = unmatched_data[selected_column_names[1]]

    ax.scatter(x_unmatched, y_unmatched, color='black', label='Unmatched', s=10)

    num_clusters = sum(matching_df[col].notna().any() for col in matching_df.columns)
    ax.set_title(f'Number of Clusters Discovered: {num_clusters}')
    ax.set_xlabel(selected_column_names[0])
    ax.set_ylabel(selected_column_names[1])
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    ax.grid(True)
    # ax.set_xticks(np.arange(x_min, x_max, 0.1))
    # ax.set_yticks(np.arange(y_min, y_max, 0.1))
    plt.legend()
    plt.show()



def process_df_list(df_list):
    c2_with_ids = []
    for i, df in enumerate(df_list):
        df['area_id'] = i 
        c2_with_ids.append(df)
    concatenated_c2 = pd.concat(c2_with_ids)

    if "idx" in concatenated_c2.columns:
        concatenated_c2.set_index(['area_id', 'idx'], inplace=True, drop=False)
    else:
        concatenated_c2.set_index(['area_id', 'Unnamed: 0'], inplace=True, drop=False)

    # if 'Unnamed: 0' in concatenated_c2.columns:
    #     concatenated_c2.set_index(['area_id', "Unnamed: 0"], inplace=True, drop=False)
    # else:
    #     concatenated_c2.set_index(['area_id', 'idx'], inplace=True, drop=False)
    concatenated_c2.index.names = ['Area_ID', 'Row_ID']
    return concatenated_c2





def refine_clusters(matching_df, external_df, selected_column_names, prior=None, reg_covar=None, model = "Both"):

    if reg_covar is None:
        reg_covar = 0.0001
    refined_df = matching_df.copy()
    
    n_components = sum(~matching_df.isna().all())

    data = []
    means_init = []
    all_indices = []
    
    is_multi_index = isinstance(external_df.index, pd.MultiIndex)

    for col in matching_df.columns:
        non_nan_indices = matching_df[col].dropna()
        if len(non_nan_indices) > 0:
            if is_multi_index:
                multi_index_tuples = [(i,) if not isinstance(i, tuple) else i for i in non_nan_indices]
                col_data = external_df.loc[multi_index_tuples, selected_column_names].values
            else:
                non_nan_indices = non_nan_indices.astype(int)
                col_data = external_df.loc[non_nan_indices, selected_column_names].values
                
            data.extend(col_data)
            means_init.append(np.mean(col_data, axis=0))
            all_indices.extend(non_nan_indices)

    data = np.array(data)
    means_init = np.array(means_init)

    if prior is not None:
        means_init = prior
    
    if model == "GMM":
        gmm = GaussianMixture(n_components=n_components, 
                            means_init=means_init,
                            max_iter=700, covariance_type="tied", reg_covar=reg_covar, n_init=20)
        gmm.fit(data)
        avg_cov_matrix = gmm.covariances_
        predictions = gmm.predict(data)
    elif model == "KMeans":
        gmm = KMeans(n_clusters = n_components, init=means_init, tol=reg_covar)
        gmm.fit(data)
        predictions = gmm.predict(data)
        avg_cov_matrix = None

    else:
        gmm = GaussianMixture(n_components=n_components,
                            means_init=means_init,
                            max_iter=700, covariance_type="tied", reg_covar=reg_covar, n_init=20)
        gmm.fit(data)
        avg_cov_matrix = gmm.covariances_

        gmm = KMeans(n_clusters = n_components, init=means_init, tol=reg_covar)
        gmm.fit(data)
        predictions = gmm.predict(data)

    non_empty_cols = [col for col in matching_df.columns if not matching_df[col].isna().all()]
    prediction_dict = dict(zip(all_indices, predictions))
    
    for i, col in enumerate(non_empty_cols):
        cluster_indices = [idx for idx, pred in prediction_dict.items() if pred == i]
        refined_df[col] = pd.Series(cluster_indices, index=range(len(cluster_indices)))
    
    return refined_df, avg_cov_matrix







def plot_tm_levels(df_list, name_list, color_list, plot_size=(10, 8), size_scale=10, z_spacing=5):
    assert len(df_list) == len(name_list) == len(color_list), "All input lists must have the same length"
    
    fig = plt.figure(figsize=plot_size)
    ax = fig.add_subplot(111, projection='3d')
    
    total_points = 0  # Initialize a counter for the total number of points
    max_x, max_y, max_z = 0, 0, 0
    for idx, df in enumerate(df_list):
        x = df['LowTm']
        y = df['HighTm']
        z = np.full(len(df), idx * z_spacing)
        sizes = df['counts'] * size_scale
        color = color_list[idx]
        
        ax.scatter(x, y, z, s=sizes, c=color, label=name_list[idx], alpha=0.8, edgecolors="grey", linewidth=0.8)
        
        # Count the number of points scattered in this loop
        total_points += len(x)
        
        max_x = max(max_x, x.max())
        max_y = max(max_y, y.max())
        max_z = max(max_z, z.max())
    
    ax.set_xlabel('LowTm', fontsize=12, fontweight='bold')
    ax.set_ylabel('HighTm', fontsize=12, fontweight='bold')

    ax.set_zticks([i * z_spacing for i in range(len(name_list))])
    ax.set_zticklabels(name_list, fontsize=13)

    ax.set_zlim(0, max_z + 10)
    
    ax.set_box_aspect((max_x, max_y, max_z + z_spacing))

    ax.tick_params(axis='z', pad=40) 

    zlim = ax.get_zlim() 
    ax.text(max_x+3.5, max_y, zlim[1] + 6, 'Local Tms ($^\circ$C) with Color', fontsize=14, fontweight='bold', ha='center')

    for size in [10]:
        ax.scatter([], [], [], s=size * size_scale, c='gray', alpha=0.6, edgecolors='w', linewidth=0.5,
                   label=f'{size} counts')
    
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=15, borderpad=1)

    plt.show()

    print(f'Total number of distinct molecules resolved: {total_points}')




from scipy.interpolate import interp1d

def align_datasets(data1:pd.DataFrame, data2:pd.DataFrame, size=32, max_shift_x=0.5, max_shift_y= 0.5, cols_to_align = ['LowTm', 'HighTm']):
    data1_raw = data1.copy()
    data2_raw = data2.copy()

    data1 = data1_raw[cols_to_align].values
    data2 = data2_raw[cols_to_align].values


    def create_image(data, size):
        img = np.zeros((size, size))
        for point in data:
            x, y = point
            img[int(y * size) % size, int(x * size) % size] += 1

        img = img / np.max(img)
        return img

    def convolution_alignment(data1, data2, size, max_shift_x, max_shift_y):
        img1 = create_image(data1, size)
        img2 = create_image(data2, size)

        # max_shift_pixels = int(max_shift * size)
        max_shift_x_pixels = int(max_shift_x * size)
        max_shift_y_pixels = int(max_shift_y * size)        
        max_value = -np.inf
        best_shift = (0, 0)
        
        for x_shift in range(-max_shift_x_pixels, max_shift_x_pixels + 1):
            for y_shift in range(-max_shift_y_pixels, max_shift_y_pixels + 1):
                shifted_img2 = np.roll(np.roll(img2, y_shift, axis=0), x_shift, axis=1)
                conv_value = np.sum(img1 * shifted_img2)
                if conv_value > max_value:
                    max_value = conv_value
                    best_shift = (x_shift, y_shift)
        print(np.array(best_shift) / size)

        return np.array(best_shift) / size

    def apply_shift(data, shift):
        shifted_data = data.copy()
        shifted_data[:, 0] += shift[0]  # Apply x shift
        shifted_data[:, 1] += shift[1]  # Apply y shift
        return shifted_data

    shift = convolution_alignment(data1, data2, size, max_shift_x, max_shift_y)

    aligned_data2 = apply_shift(data2, shift)

    plt.figure(figsize=(18, 8))
    y_min, x_min = 77, 72
    y_max, x_max = 90.1, 86.5

    plt.subplot(1, 2, 1)
    plt.scatter(data1[:, 0], data1[:, 1], label='Data1', alpha=0.5)
    plt.scatter(data2[:, 0], data2[:, 1], label='Data2', alpha=0.5)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('Before Alignment')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(data1[:, 0], data1[:, 1], label='Data1', alpha=0.5)
    plt.scatter(aligned_data2[:, 0], aligned_data2[:, 1], label='Aligned Data2', alpha=0.5)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('After Alignment')
    plt.legend()

    plt.show()

    data2_aligned = data2_raw.copy()
    data2_aligned[cols_to_align] = aligned_data2

    return data2_aligned, shift






def align_datasets(data1: pd.DataFrame, data2: pd.DataFrame, size=32, max_shift_x=0.5, max_shift_y=0.5, cols_to_align=['LowTm', 'HighTm'], manual: bool = False):
    data1_raw = data1.copy()
    data2_raw = data2.copy()

    data1 = data1_raw[cols_to_align].values
    data2 = data2_raw[cols_to_align].values

    def create_image(data, size):
        img = np.zeros((size, size))
        for point in data:
            x, y = point
            img[int(y * size) % size, int(x * size) % size] += 1
        img = img / np.max(img)
        return img

    def convolution_alignment(data1, data2, size, max_shift_x, max_shift_y):
        img1 = create_image(data1, size)
        img2 = create_image(data2, size)

        max_shift_x_pixels = int(max_shift_x * size)
        max_shift_y_pixels = int(max_shift_y * size)
        max_value = -np.inf
        best_shift = (0, 0)

        for x_shift in range(-max_shift_x_pixels, max_shift_x_pixels + 1):
            for y_shift in range(-max_shift_y_pixels, max_shift_y_pixels + 1):
                shifted_img2 = np.roll(np.roll(img2, y_shift, axis=0), x_shift, axis=1)
                conv_value = np.sum(img1 * shifted_img2)
                if conv_value > max_value:
                    max_value = conv_value
                    best_shift = (x_shift, y_shift)
        return np.array(best_shift) / size

    def apply_shift(data, shift):
        shifted_data = data.copy()
        shifted_data[:, 0] += shift[0]  # Apply x shift
        shifted_data[:, 1] += shift[1]  # Apply y shift
        return shifted_data

    if manual:
        shift_x = widgets.FloatSlider(min=-max_shift_x, max=max_shift_x, step=0.005, value=0, description="Shift X", layout=widgets.Layout(width='1000px'), readout_format='.5f')
        shift_y = widgets.FloatSlider(min=-max_shift_y, max=max_shift_y, step=0.005, value=0, description="Shift Y", layout=widgets.Layout(width='1000px'), readout_format='.5f')

        manual_shift = [0, 0]

        def update_plot(shift_x_val, shift_y_val):
            nonlocal manual_shift
            manual_shift[0] = shift_x_val
            manual_shift[1] = shift_y_val
            
            aligned_data2 = apply_shift(data2, manual_shift)
            
            plt.figure(figsize=(18, 8))
            y_min, x_min = 77, 72
            y_max, x_max = 90.1, 86.5

            plt.subplot(1, 2, 1)
            plt.scatter(data1[:, 0], data1[:, 1], label='Data1', alpha=0.5)
            plt.scatter(data2[:, 0], data2[:, 1], label='Data2', alpha=0.5)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.title('Before Alignment')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.scatter(data1[:, 0], data1[:, 1], label='Data1', alpha=0.5)
            plt.scatter(aligned_data2[:, 0], aligned_data2[:, 1], label='Manually Aligned Data2', alpha=0.5)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.title(f'After Manual Alignment (Shift X: {shift_x_val:.2f}, Shift Y: {shift_y_val:.2f})')
            plt.legend()

            plt.show()
            print(f"Shift X: {shift_x_val:.2f}, Shift Y: {shift_y_val:.2f}")

        interactive_plot = widgets.interactive(update_plot, shift_x_val=shift_x, shift_y_val=shift_y)
        display(interactive_plot)

        aligned_data2 = apply_shift(data2, manual_shift)
        data2_aligned = data2_raw.copy()
        data2_aligned[cols_to_align] = aligned_data2

        return data2_aligned, manual_shift
    
    else:
        # Perform automatic convolution alignment
        shift = convolution_alignment(data1, data2, size, max_shift_x, max_shift_y)
        aligned_data2 = apply_shift(data2, shift)

        plt.figure(figsize=(18, 8))
        y_min, x_min = 77, 72
        y_max, x_max = 90.1, 86.5

        plt.subplot(1, 2, 1)
        plt.scatter(data1[:, 0], data1[:, 1], label='Data1', alpha=0.5)
        plt.scatter(data2[:, 0], data2[:, 1], label='Data2', alpha=0.5)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title('Before Alignment')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(data1[:, 0], data1[:, 1], label='Data1', alpha=0.5)
        plt.scatter(aligned_data2[:, 0], aligned_data2[:, 1], label='Aligned Data2', alpha=0.5)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(f'After Alignment (Shift X: {shift[0]:.2f}, Shift Y: {shift[1]:.2f})')
        plt.legend()

        plt.show()

        data2_aligned = data2_raw.copy()
        data2_aligned[cols_to_align] = aligned_data2
        

        return data2_aligned, shift



def apply_global_shift(data:pd.DataFrame, shift:np.ndarray, cols_to_shift=['LowTm', 'HighTm'], plot=False):
    data_raw = data.copy()
    data = data_raw[cols_to_shift].values

    shifted_data = data.copy()
    shifted_data[:, 0] += shift[0]  # Apply x shift
    shifted_data[:, 1] += shift[1]  # Apply y shift

    data_aligned = data_raw.copy()
    data_aligned[cols_to_shift] = shifted_data
    if plot:

        plt.figure(figsize=(8, 8))
        y_min, x_min = 77, 72
        y_max, x_max = 90.1, 86.5

        plt.scatter(data[:, 0], data[:, 1], label='Data', alpha=0.5)
        plt.scatter(shifted_data[:, 0], shifted_data[:, 1], label='Shifted Data', alpha=0.5)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title('Before and After Global Shift')
        plt.legend()

        plt.show()
    
    # data_name = retrieve_name(data)[0]
    print(f"The amount of shift applied to the data is: {np.array(np.round(shift, 2))}")
    
    return data_aligned


def create_grid(points, fixed_n_line_x, fixed_n_line_y, x_init=None, y_init=None, x_range=None, y_range=None, gmm=False, manual=False):

    if not isinstance(points, np.ndarray):
        points = np.array(points)

    #filter points based on range
    if x_range is not None:
        x_min, x_max = x_range
        points = points[(points[:, 0] >= x_min) & (points[:, 0] <= x_max)]
    if y_range is not None:
        y_min, y_max = y_range
        points = points[(points[:, 1] >= y_min) & (points[:, 1] <= y_max)]

    if manual:
        selected_x = np.array(x_init).flatten()
        selected_y = np.array(y_init).flatten()
        plt.figure(figsize=(5, 5))
        plt.scatter(points[:, 0], points[:, 1], s=7)

        for x in selected_x:
            plt.axvline(x=x, color='r', linestyle='--', linewidth=0.5)
        for y in selected_y:
            plt.axhline(y=y, color='r', linestyle='--', linewidth=0.5)
        plt.xlabel('LowTm')
        plt.ylabel('HighTm')

        plt.show()

        return selected_x, selected_y
    if x_init is not None:
        x_init = np.array(x_init).reshape(-1, 1)
        kmeans_x = KMeans(n_clusters=fixed_n_line_x, max_iter=100000, tol=1e-15, n_init=10, init=x_init).fit(points[:, 0].reshape(-1, 1))
        if gmm:
            kmeans_x = GaussianMixture(n_components=fixed_n_line_x, means_init=x_init, max_iter=100000, tol=1e-5, n_init=10).fit(points[:, 0].reshape(-1, 1))
        # kmeans_x = GaussianMixture(n_components=fixed_n_line_x, means_init=x_init, max_iter=100000, tol=1e-15, n_init=10).fit(points[:, 0].reshape(-1, 1))
    else:
        kmeans_x = KMeans(n_clusters=fixed_n_line_x, max_iter=100000, tol=1e-15, n_init=10).fit(points[:, 0].reshape(-1, 1))
        if gmm:
            kmeans_x = GaussianMixture(n_components=fixed_n_line_x, max_iter=100000, tol=1e-5, n_init=10).fit(points[:, 0].reshape(-1, 1))
    if y_init is not None:
        y_init = np.array(y_init).reshape(-1, 1)
        kmeans_y = KMeans(n_clusters=fixed_n_line_y, max_iter=100000, tol=1e-15, n_init=10, init=y_init).fit(points[:, 1].reshape(-1, 1))
        if gmm:
            kmeans_y = GaussianMixture(n_components=fixed_n_line_y, means_init=y_init, max_iter=100000, tol=1e-5, n_init=10).fit(points[:, 1].reshape(-1, 1))
    else:
        kmeans_y = KMeans(n_clusters=fixed_n_line_y, max_iter=100000, tol=1e-15, n_init=10).fit(points[:, 1].reshape(-1, 1))
        if gmm:
            kmeans_y = GaussianMixture(n_components=fixed_n_line_y, max_iter=100000, tol=1e-5, n_init=10).fit(points[:, 1].reshape(-1, 1))
    

    
    # kmeans_x = KMeans(n_clusters=fixed_n_line_x, max_iter=100000, tol=1e-15, n_init=10).fit(points[:, 0].reshape(-1, 1))
    if gmm:
        selected_x = sorted(kmeans_x.means_.flatten())
        selected_y = sorted(kmeans_y.means_.flatten())
    else:
        selected_x = sorted(kmeans_x.cluster_centers_.flatten())
    # kmeans_y = KMeans(n_clusters=fixed_n_line_y, max_iter=100000, tol=1e-15, n_init=10).fit(points[:, 1].reshape(-1, 1))
        selected_y = sorted(kmeans_y.cluster_centers_.flatten())



    plt.figure(figsize=(5, 5))
    plt.scatter(points[:, 0], points[:, 1], s=7)

    for x in selected_x:
        plt.axvline(x=x, color='r', linestyle='--', linewidth=0.5)
    for y in selected_y:
        plt.axhline(y=y, color='r', linestyle='--', linewidth=0.5)
    plt.xlabel('LowTm')
    plt.ylabel('HighTm')
    plt.grid(False)

    plt.show()

    return selected_x, selected_y




def grid_transform(domain_grid, target_grid, provided_transform=None, data_to_transform=None, cols_to_transform=['LowTm', 'HighTm'], save_name = None):
    X,Y = domain_grid
    U,V = target_grid

    f_u = interp1d(X, U, kind='linear', fill_value='extrapolate')
    f_v = interp1d(Y, V, kind='linear', fill_value='extrapolate')

    if provided_transform is not None:
        f_u = provided_transform[0]
        f_v = provided_transform[1]

    if save_name is not None:
        with open(f'{save_name}_f_u.pkl', 'wb') as f:
            pickle.dump(f_u, f)
        with open(f'{save_name}_f_v.pkl', 'wb') as f:
            pickle.dump(f_v, f)

    U_transformed = f_u(X)
    V_transformed = f_v(Y)

    transformed_data = None
    if data_to_transform is not None:
        transformed_data = data_to_transform.copy()
        transformed_data[cols_to_transform[0]] = f_u(data_to_transform[cols_to_transform[0]])
        transformed_data[cols_to_transform[1]] = f_v(data_to_transform[cols_to_transform[1]])

    # X_grid, Y_grid = np.meshgrid(X, Y)
    # U_grid, V_grid = np.meshgrid(U_transformed, V_transformed)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for x in X:
        ax[0].plot([x] * len(Y), Y, 'b-', alpha=0.5)
    for y in Y:
        ax[0].plot(X, [y] * len(X), 'b-', alpha=0.5)
    for u in U:
        ax[0].plot([u] * len(V), V, 'r-', alpha=0.5)
    for v in V:
        ax[0].plot(U, [v] * len(U), 'r-', alpha=0.5)

    ax[0].set_title('Pred Grid overlaid with Data Grid')
    ax[0].set_xlabel('X-axis / U-axis')
    ax[0].set_ylabel('Y-axis / V-axis')

    for u in U_transformed:
        ax[1].plot([u] * len(V), V, alpha=0.5, color="blue")
    for v in V_transformed:
        ax[1].plot(U_transformed, [v] * len(U_transformed), alpha=0.5, color="blue")
    for u in U:
        ax[1].plot([u] * len(V), V, 'r-', alpha=0.5)
    for v in V:
        ax[1].plot(U, [v] * len(U), 'r-', alpha=0.5)

    ax[1].set_title('Transformed Pred overlaid with Data Grid')
    ax[1].set_xlabel('X-axis / U-axis')
    ax[1].set_ylabel('Y-axis / V-axis')

    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    ax[0].plot(X, U_transformed, 'go-', label='Transformed U')
    ax[0].plot(X, U[:len(X)], 'bo-', label='Target U')
    ax[0].set_title('Pred to Data Transformation')
    ax[0].set_xlabel('Predicted X-axis')
    ax[0].set_ylabel('Transformed / Data X-axis')
    ax[0].legend()

    ax[1].plot(Y, V_transformed, 'go-', label='Transformed V')
    ax[1].plot(Y, V[:len(Y)], 'bo-', label='Target V')
    ax[1].set_title('Pred to Data Transformation')
    ax[1].set_xlabel('Predicted Y-axis')
    ax[1].set_ylabel('Transformed / Data Y-axis')
    ax[1].legend()

    plt.show()

    return [f_u, f_v], transformed_data




def load_directory(parent_dir, extension, file_path_variable_list, local_scope, font_size='14px'):
    if not parent_dir.endswith('/'):
        parent_dir += '/'
    
    search_pattern = os.path.join(parent_dir, f"*.{extension}")
    available_files = glob.glob(search_pattern, recursive=True)

    if not available_files:
        print(f"No files with .{extension} found in {parent_dir}")
        return
    
    file_names = [os.path.basename(file) for file in available_files]
    file_names.sort() 
    file_names_with_none = ['None'] + file_names

    selected_files = {} 
    
    def load_files(button):
        for var_name, dropdown in dropdowns.items():
            selected_value = dropdown.value
            if selected_value == 'None':
                selected_files[var_name] = None
            else:
                selected_files[var_name] = os.path.join(parent_dir, selected_value)
    
        local_scope.update(selected_files)
        
        for var_name, file_path in selected_files.items():
            print(f"{var_name} = {file_path}")

    dropdowns = {}
    for var_name in file_path_variable_list:
        dropdown = widgets.Dropdown(
            options=file_names_with_none,
            description=f'{var_name}:',
            style={'description_width': 'initial', 'font_size': font_size},  
            layout=widgets.Layout(width='80%', font_size=font_size),
            font_size=25  
        )
        dropdowns[var_name] = dropdown
        display(dropdown)
    
    button = widgets.Button(description="Load Files")
    button.on_click(load_files)
    
    display(button)




def update_cluster_assignment(matching_df, df_list, nth_col_insertion=3):
    for i in range(len(df_list)):
        df_list[i]['cluster_name'] = np.nan

        cols = [col for col in df_list[i].columns if col != 'cluster_name']
        df_list[i] = df_list[i][cols[:nth_col_insertion] + ['cluster_name'] + cols[nth_col_insertion:]]

    k = 0
    for col in matching_df.columns:
        for i, val in matching_df[col].items():
            # print(val)
            if isinstance(val, tuple):
                df_idx, row_idx = val
                # print(df_idx, row_idx)
                if 0 <= df_idx < len(df_list):
                    df = df_list[df_idx]
                    # print(row_idx)
                    if row_idx in df.index:
                        # print(k)
                        # k += 1
                        df.at[row_idx, 'cluster_name'] = col

    return df_list




from scipy.ndimage import gaussian_filter1d



def subtract_background(local_tm1_positive_df, local_tm1_negative_df, neighborhood_size):
    """
    Perform background subtraction on positive data points using nearby negative points.
    
    Parameters:
    -----------
    local_tm1_positive_df : pandas.DataFrame
        DataFrame containing positive data points with Pos column and T1-TN data columns
    local_tm1_negative_df : pandas.DataFrame
        DataFrame containing negative data points with same structure
    neighborhood_size : float
        Maximum distance to consider for finding neighboring negative points
        
    Returns:
    --------
    tuple:
        - pandas.DataFrame: New DataFrame with background-subtracted positive data points
        - pandas.DataFrame: QC information containing:
            * positive_position: Position of the positive point
            * num_neighbors: Number of negative neighbors found
            * neighbor_positions: List of positions of neighboring negative points
            * neighbor_distances: List of distances to neighboring negative points
    """
    
    def extract_coordinates(pos_str):
        try:
            return literal_eval(pos_str)
        except:
            return pos_str
    
    def calculate_distance(pos1, pos2):
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
    
    data_columns = [col for col in local_tm1_positive_df.columns if col.startswith('T')]
    
    result_df = local_tm1_positive_df.copy()
    
    qc_info = []
    
    for idx, pos_row in tqdm(local_tm1_positive_df.iterrows(), 
                            total=len(local_tm1_positive_df),
                            desc="Processing positive data points",
                            unit="point"):
        pos_coords = extract_coordinates(pos_row['Pos'])
        
        nearby_negatives = []
        neighbor_info = {
            'positive_position': pos_row['Pos'],
            'neighbor_positions': [],
            'neighbor_distances': []
        }
        
        for neg_idx, neg_row in local_tm1_negative_df.iterrows():
            neg_coords = extract_coordinates(neg_row['Pos'])
            distance = calculate_distance(pos_coords, neg_coords)
            
            if distance <= neighborhood_size:
                nearby_negatives.append(neg_row[data_columns].values)
                neighbor_info['neighbor_positions'].append(neg_row['Pos'])
                neighbor_info['neighbor_distances'].append(float(distance)) 
        
        neighbor_info['num_neighbors'] = len(nearby_negatives)
        qc_info.append(neighbor_info)
        
        if nearby_negatives:
            negative_median = np.median(nearby_negatives, axis=0)
            # print(type(negative_median))
            negative_median = np.array(negative_median, dtype=float)
            negative_median = gaussian_filter1d(negative_median, sigma=10, radius=35)
            
            result_df.loc[idx, data_columns] = pos_row[data_columns] - negative_median
        else:
            result_df.loc[idx, data_columns] = pos_row[data_columns]
    
    qc_df = pd.DataFrame(qc_info)
    
    return result_df, qc_df





def visualize_background_subtraction_qc(qc_df, local_tm1_positive_df, local_tm1_negative_df, result_df,
                                      local_tm1_mcs_deriv=None, additional_array=None, image_size=(1200, 1200), 
                                      corner_threshold=150, n_corner_points_to_sample=4, n_central_points_to_sample=2):
    """
    Create QC plots showing original signals, background-subtracted results, derivatives, and optional additional data.
    
    Parameters:
    -----------
    qc_df : pandas.DataFrame
        QC information DataFrame from subtract_background function
    local_tm1_positive_df : pandas.DataFrame
        Original positive data points DataFrame
    local_tm1_negative_df : pandas.DataFrame
        Original negative data points DataFrame
    result_df : pandas.DataFrame
        Background-subtracted results DataFrame
    local_tm1_mcs_deriv : numpy.ndarray, optional
        Derivative signals corresponding to result_df positions
    additional_array : numpy.ndarray, optional
        Additional time series data to plot in a fourth column
    image_size : tuple
        Size of the image in pixels (width, height)
    corner_threshold : int
        Distance from corners to consider as "corner region"
    n_corner_points_to_sample : int
        Number of corner points to sample
    n_central_points_to_sample : int
        Number of central points to sample
    """
    
    def extract_coordinates(pos_str):
        try:
            return literal_eval(pos_str)
        except:
            return pos_str
    
    def get_distance_to_corner(coords, corner_coords):
        return np.sqrt((coords[0] - corner_coords[0])**2 + 
                      (coords[1] - corner_coords[1])**2)
    
    def is_central_position(coords, image_size):
        x, y = coords
        center_x, center_y = image_size[0]/2, image_size[1]/2
        return (abs(x - center_x) <= image_size[0]/4 and 
                abs(y - center_y) <= image_size[1]/4)
    
    def find_qc_row(pos_value, qc_df):
        """Helper function to find matching QC row with error handling"""
        matching_rows = qc_df[qc_df['positive_position'] == pos_value]
        if len(matching_rows) == 0:
            return None
        return matching_rows.iloc[0]
    
    time_columns = [col for col in local_tm1_positive_df.columns if col.startswith('T')]
    time_points = range(len(time_columns))
    
    corners = [
        (0, 0),                    # Top-left
        (0, image_size[1]),        # Bottom-left
        (image_size[0], 0),        # Top-right
        (image_size[0], image_size[1])  # Bottom-right
    ]
    
    all_corner_candidates = []
    all_central_candidates = []
    
    for idx, row in local_tm1_positive_df.iterrows():
        if find_qc_row(row['Pos'], qc_df) is None:
            continue
            
        coords = extract_coordinates(row['Pos'])
        
        if is_central_position(coords, image_size):
            all_central_candidates.append((idx, coords))
            continue
        
        min_distance = float('inf')
        nearest_corner = None
        
        for corner in corners:
            distance = get_distance_to_corner(coords, corner)
            if distance <= corner_threshold and distance < min_distance:
                min_distance = distance
                nearest_corner = corner
        
        if nearest_corner is not None:
            all_corner_candidates.append((idx, coords, nearest_corner, min_distance))
    
    selected_corners = []
    if all_corner_candidates:
        sorted_corner_candidates = sorted(all_corner_candidates, key=lambda x: x[3])
        selected_corners = [(pos[0], pos[1]) for pos in sorted_corner_candidates[:n_corner_points_to_sample]]
    
    selected_central = []
    if all_central_candidates:
        if len(all_central_candidates) <= n_central_points_to_sample:
            selected_central = all_central_candidates
        else:
            selected_indices = np.random.choice(
                len(all_central_candidates), 
                size=n_central_points_to_sample, 
                replace=False
            )
            selected_central = [all_central_candidates[i] for i in selected_indices]
    
    all_selected = selected_corners + selected_central
    
    print(f"Selected {len(selected_corners)} corner points and {len(selected_central)} central points")
    
    if not all_selected:
        raise ValueError("No positions were selected. Please check your threshold and sampling parameters.")
    

    n_cols = 4 if additional_array is not None else 3
    
    fig = plt.figure(figsize=(10*n_cols, 6*len(all_selected))) 
    gs = GridSpec(len(all_selected), n_cols, figure=fig)
    # fig.suptitle('Background Subtraction QC: Original Signals, Subtracted Results, and Derivatives', fontsize=16)
    
    for i, (idx, coords) in enumerate(all_selected):

        ax_left = fig.add_subplot(gs[i, 0])
        
        positive_signal = local_tm1_positive_df.loc[idx, time_columns].values
        ax_left.plot(time_points, positive_signal, 'b-', label='Positive Signal', linewidth=2)
        
        qc_row = find_qc_row(local_tm1_positive_df.loc[idx, 'Pos'], qc_df)
        if qc_row is not None:
            negative_signals = []
            
            for neg_pos in qc_row['neighbor_positions']:
                try:
                    neg_idx = local_tm1_negative_df[local_tm1_negative_df['Pos'] == neg_pos].index[0]
                    neg_signal = local_tm1_negative_df.loc[neg_idx, time_columns].values
                    negative_signals.append(neg_signal)
                    ax_left.plot(time_points, neg_signal, 'r-', alpha=0.3, linewidth=1)
                except IndexError:
                    continue
            
            if negative_signals:
                median_signal = np.median(negative_signals, axis=0)
                ax_left.plot(time_points, median_signal, 'k--', label='Negative Median', linewidth=2)
        
        ax_left.set_xlabel('Time Point')
        ax_left.set_ylabel('Signal Intensity')
        ax_left.set_title(f'Original Signals at Position {coords}\n({len(negative_signals) if "negative_signals" in locals() else 0} negative neighbors)')
        if i == 0: ax_left.legend()
        
        ax_middle = fig.add_subplot(gs[i, 1])
        subtracted_signal = result_df.loc[idx, time_columns].values
        ax_middle.plot(time_points, subtracted_signal, 'g-', label='Subtracted Signal', linewidth=2)
        ax_middle.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        ax_middle.set_xlabel('Time Point')
        ax_middle.set_ylabel('Background-Subtracted Intensity')
        ax_middle.set_title(f'Background-Subtracted Signal\nat Position {coords}')
        if i == 0: ax_middle.legend()
        
        if local_tm1_mcs_deriv is not None:
            ax_right = fig.add_subplot(gs[i, 2])
            derivative_signal = local_tm1_mcs_deriv[idx]
            ax_right.plot(time_points, derivative_signal, 'm-', label='Derivative', linewidth=2)
            ax_right.axhline(y=0, color='k', linestyle=':', alpha=0.5)
            ax_right.set_xlabel('Time Point')
            ax_right.set_ylabel('Signal Derivative')
            ax_right.set_title(f'Signal Derivative\nat Position {coords}')
            if i == 0: ax_right.legend()
        
        if additional_array is not None:
            ax_additional = fig.add_subplot(gs[i, 3])
            additional_signal = additional_array[idx]
            ax_additional.plot(time_points, additional_signal, 'c-', label='Additional Signal', linewidth=2)
            ax_additional.axhline(y=0, color='k', linestyle=':', alpha=0.5)
            ax_additional.set_xlabel('Time Point')
            ax_additional.set_ylabel('Additional Signal')
            ax_additional.set_title(f'Additional Signal\nat Position {coords}')
            if i == 0: ax_additional.legend()
        
        xlim = ax_left.get_xlim()
        ax_middle.set_xlim(xlim)
        if local_tm1_mcs_deriv is not None:
            ax_right.set_xlim(xlim)
        if additional_array is not None:
            ax_additional.set_xlim(xlim)
    
    plt.tight_layout()



def gaussian_smooth(signals, sigma=2.0, truncate=4.0, mode='nearest'):
    """
    Apply Gaussian smoothing to multiple time series signals.
    
    Parameters:
    -----------
    signals : ndarray
        2D array of shape (M, N) containing M signals of length N
    sigma : float
        Standard deviation for Gaussian kernel
    truncate : float
        Number of standard deviations to truncate the Gaussian kernel
    mode : str
        How to handle boundaries. Options: 'reflect', 'constant', 'nearest', 'mirror', 'wrap'
        
    """
    
    smoothed = np.array([
        gaussian_filter1d(signal, sigma=sigma, truncate=truncate, mode=mode)
        for signal in signals
    ])
    
    return smoothed



def wittwer_background_subtract(signals, TL_idx, TR_idx, eps=2, plot = False):
    M, N = signals.shape
    subtracted_signals = np.zeros_like(signals)
    x = np.arange(N)
    
    for i in range(M):
        try:
            signal = signals[i]
            derivatives = savgol_filter(signal, 45, 1, deriv=1, mode="nearest")
            
            TL_start = max(0, TL_idx - eps)
            TL_end = min(N, TL_idx + eps + 1)
            TR_start = max(0, TR_idx - eps)
            TR_end = min(N, TR_idx + eps + 1)
            
            B_prime_TL = np.median(derivatives[TL_start:TL_end])
            B_prime_TR = np.median(derivatives[TR_start:TR_end])
            # print(B_prime_TR, B_prime_TL)
            if (B_prime_TR / B_prime_TL) <= 0:
                # print(B_prime_TR, B_prime_TL)
                B_prime_TR = -1* B_prime_TR
                # print(f"Skipping signal {i}: negative derivative ratio")
                # continue
                
            a = np.log(B_prime_TR / B_prime_TL) / (TR_idx - TL_idx)
            C = B_prime_TL / a

            # print(a, C)
            
            x_shifted = x - TL_idx
            background = C * np.exp(a * x_shifted)
            subtracted_signals[i] = signal - background

            if plot:
                plt.figure(figsize=(18, 4))
                plt.subplot(121)
                plt.plot(x, signal, label='Original')
                plt.plot(x, background, '--', label='Background')
                plt.axvline(TL_idx, color='g', linestyle=':', label='TL')
                plt.axvline(TR_idx, color='r', linestyle=':', label='TR')
                plt.title(f'Melt Curvve {i+1}')
                plt.legend()
                # plt.subplot(132)
                # plt.plot(x, background, label='Background')
                # plt.title('Background Model')
                # plt.legend()
                plt.subplot(122)
                plt.plot(x, subtracted_signals[i], label='Background Subtracted')
                plt.title('After Bkg Subtraction')
                plt.legend()
                
                plt.tight_layout()
                plt.show()
            
        except Exception as e:
            print(f"Failed to process signal {i}: {str(e)}")
            continue
    
    return subtracted_signals








from typing import Dict, List, Tuple, Any
def save_data(data_dict: Dict[str, Any], folder_path: str = "raw_data") -> None:
    """
    Save variables to disk in the specified folder.
    
    Args:
        data_dict: Dictionary of variable names and their values to save
        folder_path: Path to folder where data will be saved
    """
    os.makedirs(folder_path, exist_ok=True)
    
    with open(os.path.join(folder_path, "data.pkl"), "wb") as f:
        pickle.dump(data_dict, f)
    
    print(f"Data saved to {folder_path}/data.pkl")

def load_data(folder_path: str = "raw_data") -> Dict[str, Any]:
    """
    Load variables from disk.
    
    Args:
        folder_path: Path to folder where data is stored
        
    Returns:
        Dictionary containing all saved variables
    """
    file_path = os.path.join(folder_path, "data.pkl")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)
    
    print(f"Data loaded from {file_path}")
    return data_dict




def print_version():
    print("Currently running 01312025 version")



def plot_subset_scatter(df_dict, col_names, channel_names, key_to_tm_mapping):
    """
    Creates individual 2D scatter plots for each subset DataFrame and a combined plot with all subsets.

    Parameters:
    df_dict (dict): Dictionary of subset DataFrames. Keys should be descriptive strings like "B1R1".
    col_names (tuple): Tuple of column names to scatter plot (x_col, y_col).
    channel_names (list): List of channel names (e.g., ["B1", "R1", "G1", "B2"]).
    key_to_tm_mapping (dict): Mapping from df_dict keys (str) to the original Tm tuple (e.g., (78.3, 0.0, 82.5, 0.0)).
    """
    x_col, y_col = col_names
    y_min, x_min = 77, 72
    y_max, x_max = 90.1, 86.5

    for key, subset_df in df_dict.items():
        plt.figure(figsize=(15, 13))
        plt.scatter(subset_df[x_col], subset_df[y_col], marker='o', alpha=0.7, s=7)

        plt.xlabel(x_col, fontsize=20)
        plt.ylabel(y_col, fontsize=20)

        plt.xticks(np.arange(x_min, x_max, 0.5), fontsize=10)
        plt.yticks(np.arange(y_min, y_max, 0.5), fontsize=10)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.grid(True)
        tm_tuple = key_to_tm_mapping.get(key, None)

        if tm_tuple:
            title_parts = [f'{channel_names[i]}: {tm_tuple[i] if tm_tuple[i] != 0 else "None"}' for i in range(len(channel_names))]
            title_label = ' | '.join(title_parts)
            title = f'{key} | {title_label}'
        else:
            title = key

        plt.title(f'Probe Channel: {title}', fontsize=15)
        plt.tight_layout()
        plt.show()

    # Combined plot
    plt.figure(figsize=(15, 13))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(df_dict)))

    for (key, subset_df), color in zip(df_dict.items(), colors):
        tm_tuple = key_to_tm_mapping.get(key, None)
        if tm_tuple:
            label_parts = [f'{channel_names[i]}: {tm_tuple[i] if tm_tuple[i] != 0 else "None"}' for i in range(len(channel_names))]
            label = f'{key} | ' + ' | '.join(label_parts)
        else:
            label = key

        plt.scatter(subset_df[x_col], subset_df[y_col],
                    marker='o', alpha=0.7, s=7,
                    color=color, label=label)

    plt.xlabel(x_col, fontsize=20)
    plt.ylabel(y_col, fontsize=20)

    plt.xticks(np.arange(x_min, x_max, 0.5), fontsize=10)
    plt.yticks(np.arange(y_min, y_max, 0.5), fontsize=10)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.grid(True)
    plt.title('Probe Tm: Combined Plot', fontsize=15)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()



def save_subset_dfs(df_dict):
    """
    Saves the subset DataFrames in the dictionary to individual CSV files.

    Parameters:
    df_dict (dict): Dictionary of subset DataFrames.
    """

    cwd = os.getcwd()
    folder_name = os.path.basename(cwd)

    for key, subset_df in df_dict.items():
        file_name = f"{folder_name}_{key}.csv"
        subset_df.to_csv(file_name, index=True)
        print(f"{key} Subset is saved as '{file_name}'.")
    
    return


