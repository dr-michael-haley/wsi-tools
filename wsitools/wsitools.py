# --- Standard Library ---
import json
import logging
import os
import random
import re
import shutil
import subprocess as sp
from collections import defaultdict
from glob import glob
from pathlib import Path
from IPython.core.display import HTML

# --- Third-party Packages ---
import dask.array as da
from dask.array import Array as DaskArray
from dask.distributed import Client, LocalCluster
from dask_image.ndfilters import gaussian_filter, median_filter

from ome_types import from_xml, to_xml

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import napari
from napari.utils.notifications import show_info, show_warning, show_error

import numpy as np
import pandas as pd

from magicgui import magicgui, widgets

from ome_types import from_xml
from ome_types.model import OME, Image, Pixels, Channel, TiffData

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QMessageBox

from scipy import ndimage
from scipy.ndimage import grey_opening

from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.io import imsave
from skimage.morphology import opening as grey_opening
from skimage.transform import downscale_local_mean

from tifffile import TiffFile

import zarr


def add_java_paths(java_paths=None):
    """Add Java-related tool paths to environment and report status."""

    # Increase Java heap memory
    os.environ['JAVA_TOOL_OPTIONS'] = '-Xmx128G'

    # Default paths
    if not java_paths:
        java_paths = [
            r"C:/java_packages/bftools",
            r"C:/java_packages/bioformats2raw-0.10.0-rc2/bin",
            r"C:/java_packages/raw2ometiff-0.8.0-rc1/bin",
            r"C:/java_packages/maven-mvnd-1.0.2-windows-amd64/bin",
            r"C:/java_packages/vips-dev-8.16/bin"
        ]

    print("🔧 Adding Java tool paths to environment:")

    for path in java_paths:
        path = Path(path)
        norm_path = str(path)

        # Check existence and warn if missing
        if not path.exists():
            print(f"  ⚠️ Warning: path does not exist — {norm_path}")

        # Add to PATH if not already included
        if norm_path not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + norm_path

        # Extract readable tool name
        base_tool = path.parent.name if path.name.lower() == "bin" else path.name
        print(f"  ✅ Imported: {base_tool}")

    print("✅ Java environment updated.")



def start_dask_cluster(n_workers=12, threads_per_worker=2, memory_limit='32GB'):
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)
    client = Client(cluster)
    return client

def run_commandline(cmd: str, verbose: int = 0, return_readout = False, print_command=False):
    '''
    Wrapper to run command line (ie. java tools
    '''
    
    if print_command: print(cmd)
    
    env = os.environ.copy()
    
    if return_readout: readout = []

    if verbose == 0:
        # capture the output silently when verbose is disabled
        result = sp.run(cmd, shell=True, env=env, capture_output=True, text=True)

        # check if the process failed
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with error: {result.stderr}")

    else:
        # stream output in real-time if verbose is enabled
        process = sp.Popen(cmd, shell=True, env=env, stdout=sp.PIPE, stderr=sp.PIPE, text=True)

        # stream stdout in real-time
        for stdout_line in iter(process.stdout.readline, ""):
            logging.info(stdout_line.strip())  # Log each line of output
            
            if return_readout:
                readout.append(stdout_line.strip())
            else:
                print(stdout_line.strip())
        
        process.stdout.close()

        # wait for process to finish
        process.wait()

        # check if the process failed
        if process.returncode != 0:
            stderr_output = process.stderr.read().strip()
            process.stderr.close()
            raise RuntimeError(f"Command failed with error: {stderr_output}")

        process.stderr.close()
    
    if return_readout:
        return readout


def parse_showinf_series_metadata(log_lines, series_number):
    series_data = {}
    current_series = None
    series_block = []

    # Preprocess and segment the input into series blocks
    for line in log_lines:
        match = re.match(r"Series #(\d+) :", line)
        if match:
            if current_series is not None:
                series_data[current_series] = series_block
            current_series = int(match.group(1))
            series_block = []
        elif current_series is not None:
            series_block.append(line)

    if current_series is not None:
        series_data[current_series] = series_block  # Save the last series block

    # Check if the requested series exists
    if series_number not in series_data:
        raise ValueError(f"Series #{series_number} not found.")

    # Parse the requested series block into a dictionary
    series_info = {}
    for line in series_data[series_number]:
        if line.strip() == '' or line.strip() == '-----':
            continue

        key_value = re.match(r"(.+?) = (.+)", line)
        if key_value:
            key, value = key_value.groups()
            key = key.strip()
            value = value.strip()

            # Attempt to convert value to appropriate type
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif re.match(r'^-?\d+$', value):
                value = int(value)
            elif re.match(r'^-?\d+\.\d+$', value):
                value = float(value)

            series_info[key] = value

    return series_info
    
def vsi_folder_to_zarr_UNUSED(raw_folder = "Images_raw",
                                tile_size = 512,
                                dtype = np.uint16,
                                series = 1,
                                zarr_level = 0,
                                pyramid_level = 0):
    '''
    This function takes as input the output of the bioformats2raw conversion of a vsi file (https://github.com/glencoesoftware/bioformats2raw),
    which creates a folder in a 'zarr' compatible format
    '''
    
    # Open raw folder as Zarr
    store = zarr.open(raw_folder, mode='r+')[str(zarr_level)]

    # Access specific level of pyramid (usually top, 0)
    raw_zarr = store[str(pyramid_level)]
    print(f"Level 0 shape: {raw_zarr.shape}, chunks: {raw_zarr.chunks}")

    # Convert to Dask    
    raw_dask = da.from_zarr(raw_zarr)
    
    # Open and clean metadata
    with open(os.path.join(raw_folder, "OME", "METADATA.ome.xml"), encoding='utf-8') as f:
        xml_text = f.read()

    xml_text = xml_text.replace("Âµm", "µm")

    # Now parse with ome-types
    ome = from_xml(xml_text)
    
    pixels = ome.images[0].pixels
    size_z, size_c, size_y, size_x = pixels.size_z, pixels.size_c, pixels.size_y, pixels.size_x


def subtract_background(array, methods, axis_order='CYX', verbose=True):
    """
    Apply background subtraction to specified channels of a (possibly Dask) array.
    Parameters:
      - array: Dask or NumPy array with shape corresponding to axis_order.
      - methods: List of dicts specifying:
          - method: one of 'rolling_ball', 'median', etc.
          - channels: list of channel indices (relative to C axis)
          - other method-specific parameters
      - axis_order: string like 'CYX' or 'TCZYX' describing array shape.
    """

    is_dask = isinstance(array, da.Array)
    full_ndim = array.ndim
    channel_axis = axis_order.index('C')
    result = array.copy()

    def _slice_channel(arr, c):
        slicer = [slice(None)] * arr.ndim
        slicer[channel_axis] = c
        return arr[tuple(slicer)]

    def _set_channel(arr, c, new_data):
        slicer = [slice(None)] * arr.ndim
        slicer[channel_axis] = c
        arr[tuple(slicer)] = new_data
        return arr

    for step in methods:
        m = step["method"]
        ch_idxs = step.get("channels", list(range(array.shape[channel_axis])))

        if verbose:
            print(f"🔧 {m} on channels {ch_idxs} with params={step}")

        for c in ch_idxs:
            channel = _slice_channel(result, c)
            ch_shape = channel.shape
            ch_ndim = channel.ndim
            ch_axis_order = [ax for i, ax in enumerate(axis_order) if i != channel_axis]
            spatial_axes = [i for i, ax in enumerate(ch_axis_order) if ax in ('Y', 'X')]

            if m == "rolling_ball":
                σ = float(step.get("sigma_px", 50))
                sigma = tuple(σ if i in spatial_axes else 0 for i in range(ch_ndim))
                fn = gaussian_filter if is_dask else lambda x: ndimage.gaussian_filter(x, sigma=sigma)
                bg = fn(channel, sigma=sigma) if is_dask else fn(channel)
                sub = (channel.astype('int32') - bg.astype('int32')).clip(0).astype('uint16')
                if step.get("rescale", False):
                    fn2 = lambda b: rescale_intensity(b, in_range='image', out_range='uint16').astype('uint16')
                    sub = sub.map_blocks(fn2, dtype='uint16') if is_dask else fn2(sub)

            elif m == "median":
                s = int(step.get("size_px", 50))
                size = tuple(s if i in spatial_axes else 1 for i in range(ch_ndim))
                fn = median_filter if is_dask else lambda x: ndimage.median_filter(x, size=size)
                bg = fn(channel)
                sub = (channel.astype('int32') - bg.astype('int32')).clip(0).astype('uint16')
                if step.get("rescale", False):
                    fn2 = lambda b: rescale_intensity(b, in_range='image', out_range='uint16').astype('uint16')
                    sub = sub.map_blocks(fn2, dtype='uint16') if is_dask else fn2(sub)

            elif m == "morph_opening":
                s = int(step.get("size_px", 50))
                footprint = np.ones(tuple(s if i in spatial_axes else 1 for i in range(ch_ndim)))
                sub = (
                    channel.map_blocks(lambda b: grey_opening(b, footprint=footprint), dtype=channel.dtype)
                    if is_dask else
                    grey_opening(channel, footprint=footprint)
                )

            elif m == "clahe":
                fn = lambda b: equalize_adapthist(b.astype('float32')).astype('float32')
                sub = channel.map_blocks(fn, dtype='float32') if is_dask else fn(channel)

            elif m == "rescale":
                ir = step.get("in_range", (0, 65535))
                orng = step.get("out_range", 'uint16')
                fn = lambda b: rescale_intensity(b, in_range=ir, out_range=orng).astype(orng)
                sub = channel.map_blocks(fn, dtype=orng) if is_dask else fn(channel)

            elif m == "threshold":
                t = step.get("value", 10)
                fn = lambda b: np.where(b < t, 0, b).astype(b.dtype)
                sub = channel.map_blocks(fn, dtype=channel.dtype) if is_dask else fn(channel)

            else:
                raise ValueError(f"Unknown method {m}")

            result = _set_channel(result, c, sub)

    return result


# Helper: map emission wavelength to a rough color name
def wavelength_to_color(wavelength_nm):
    if wavelength_nm < 500:
        return 'blue'
    elif 500 <= wavelength_nm < 550:
        return 'green'
    elif 550 <= wavelength_nm < 590:
        return 'orange'
    elif 590 <= wavelength_nm < 750:
        return 'red'
    else:
        return 'gray'  # fallback for IR/UV or unknown

def wavelength_to_color2(wavelength_nm):
    """
    Convert wavelength (nm) to an approximate RGB color.
    Based on: https://stackoverflow.com/a/16854885
    """


    # Use matplotlib's colormap as fallback
    try:
        cmap = plt.get_cmap('nipy_spectral')
        norm = plt.Normalize(400, 700)
        return cmap(norm(wavelength_nm))[:3]  # return RGB tuple
    except Exception:
        return (1.0, 1.0, 1.0)  # white fallback


def napari_tile_inspector(zarr_path,
                          chunk_size=(1024,1024),
                          channel_order = 'TCZYX',
                          channel_names=None,
                          channel_colors=None,
                          zarr_level="0",
                          display_level="0"):
    """
    Napari tile inspector with:
      - Random tile loading
      - Background‑sub GUI with parameter widgets
      - Sync contrast button (union/intersection)
      - Clear adjusted toggle
      - Channel color defaults
    """
    
    # Load to dask from on disk zarr
    store = zarr.open(zarr_path, mode='r')[zarr_level]
    original_dask = da.from_zarr(store[display_level])

    # Extract indices of different components of data (time, channels, z, y, x)
    axis_indices = {dim: channel_order.index(dim) for dim in "TCZYX"}
    shape = original_dask.shape

    # Get the actual dimensions from shape
    Y = shape[axis_indices["Y"]]
    X = shape[axis_indices["X"]]
    C = shape[axis_indices["C"]]

    # Read in metadata
    with open(os.path.join(zarr_path, "OME", "METADATA.ome.xml"), encoding='utf-8') as f:
        xml_text = f.read()

    # Fix bad characters
    xml_text = xml_text.replace("Âµm", "µm")

    # Parse with ome-types
    ome = from_xml(xml_text)

    # Extract channels
    channels = ome.images[0].pixels.channels

    # List of channel names
    if not channel_names:
        channel_names = [ch.name for ch in channels]

    # List of colors for napari visualization
    if not channel_colors:
        channel_colors = [wavelength_to_color(ch.emission_wavelength) for ch in channels]

    viewer = napari.Viewer()
    raw_layers = []
    adjusted_names = []
    contrast_limits = []

    # Add raw placeholders
    for c,name in enumerate(channel_names):
        layer = viewer.add_image(
            np.zeros(chunk_size),
            name=name,
            colormap=channel_colors[c],
            blending='additive'
        )
        raw_layers.append(layer)

    def load_random(clear_adjusted=False):
        # optional clearing
        if clear_adjusted:
            for nm in adjusted_names:
                if nm in viewer.layers:
                    viewer.layers.remove(nm)
            adjusted_names.clear()

        yidx = random.randrange(Y // chunk_size[0])
        xidx = random.randrange(X // chunk_size[1])
        ys, xs = yidx * chunk_size[0], xidx * chunk_size[1]

        for c in range(C):
            tile = original_dask[0, c, 0, ys:ys + chunk_size[0], xs:xs + chunk_size[1]].compute()
            tile = tile.astype(np.uint16)
            raw_layers[c].data = tile

            # auto contrast
            if tile.max() > 0:
                vmin = np.percentile(tile, 1)
                vmax = np.percentile(tile, 99.5)
            else:
                vmin, vmax = 0, 1
            raw_layers[c].contrast_limits = (vmin, vmax)
            if len(contrast_limits) <= c:
                contrast_limits.append((vmin, vmax))
            else:
                contrast_limits[c] = (vmin, vmax)

    # Random + clear toggle
    @magicgui(call_button="Random Tile", clear_adjusted={"label": "Clear adjusted"})
    def btn_random(clear_adjusted: bool = True):
        load_random(clear_adjusted)
    viewer.window.add_dock_widget(btn_random, area='left', name="Tile Controls")

    # Sync contrast helper
    def sync_contrast(strategy: str = "union"):
        """
        Sync contrast limits of all layers matching each original channel name.
        """

        groups = defaultdict(list)

        for base_name in channel_names:
            for layer in viewer.layers:
                if base_name in layer.name:
                    groups[base_name].append(layer)

        for base_name, layers in groups.items():
            vmins = [l.contrast_limits[0] for l in layers]
            vmaxs = [l.contrast_limits[1] for l in layers]
            if strategy == "union":
                cl = (min(vmins), max(vmaxs))
            elif strategy == "intersection":
                cl = (max(vmins), min(vmaxs))
            else:
                continue

            for l in layers:
                l.contrast_limits = cl

    @magicgui(strategy={"choices":["union","intersection"]}, call_button="Sync Contrast")
    def btn_sync(strategy: str):
        sync_contrast(strategy)
    viewer.window.add_dock_widget(btn_sync, area='right', name="Contrast Sync")

    # Background GUI
    @magicgui(
        method={"choices": ["rolling_ball", "median", "morph_opening", "clahe", "rescale", "threshold"]},
        layer_select={
            "label": "Target Layers",
            "choices": lambda w: [layer.name for layer in viewer.layers if isinstance(layer.data, np.ndarray)],
            "allow_multiple": True
        },
        sigma_px={"label": "Sigma (px)", "min": 1, "max": 500, "step": 1, "value": 50},
        size_px={"label": "Size (px)", "min": 1, "max": 500, "step": 1, "value": 50},
        rescale={"label": "Rescale", "value": False},
        autoscale_contrast={"label": "Autoscale contrast", "value": True},
        threshold={"label": "Thresh", "min": 0, "max": 1000, "step": 1, "value": 10},
        call_button="Apply"
    )
    def apply_bg(
            method: str,
            sigma_px: int,
            size_px: int,
            rescale: bool,
            threshold: int,
            autoscale_contrast: bool,
            layer_select: list = []
    ):
        try:
            params = {"method": method}
            if method == "rolling_ball":
                params["sigma_px"] = sigma_px
                if rescale:
                    params["rescale"] = True
            elif method == "median":
                params["size_px"] = size_px
                if rescale:
                    params["rescale"] = True
            elif method == "threshold":
                params["value"] = threshold
            elif method == "rescale":
                params["in_range"] = (0, 65535)
                params["out_range"] = "uint16"

            # Apply to selected layers, or fallback to raw tile
            targets = layer_select or [layer.name for layer in viewer.layers if layer.name in channel_names]



            if not targets:
                show_error("No target layers found to apply background.")
                return

            for layer_name in targets:
                layer = viewer.layers[layer_name]
                data = layer.data

                # Handle 2D vs 3D
                if data.ndim == 2:
                    arr = data[None, ...]
                    axis_order = "CYX"
                    channel_idxs = [0]
                elif data.ndim == 3:
                    arr = data
                    axis_order = "CYX"
                    channel_idxs = list(range(arr.shape[0]))
                else:
                    show_error(f"Unsupported array shape {data.shape} in layer '{layer_name}'")
                    continue

                # Apply background subtraction
                methods = [dict(params, channels=channel_idxs)]
                show_info(f"Applying: {str(methods)}. Axis order: {str(axis_order)}.")
                corrected = subtract_background(arr, methods, axis_order=axis_order,
                                                verbose=False)
                if isinstance(corrected, da.Array):
                    corrected = corrected.compute()

                for c in channel_idxs:
                    name_suffix = f"{method} - {layer_name} [C{c}]"

                    if autoscale_contrast:
                        cl = (np.percentile(corrected[c], 1), np.percentile(corrected[c], 99.5))
                    else:
                        cl = layer.contrast_limits

                    cmap = layer.colormap.name if hasattr(layer, "colormap") else "gray"
                    viewer.add_image(corrected[c], name=name_suffix, blending='additive', colormap=cmap,
                                     contrast_limits=cl)
                    adjusted_names.append(name_suffix)

            show_info(f"Applied {method} to {targets}")
        except Exception as e:
            show_error(str(e))

    viewer.window.add_dock_widget(apply_bg, area='right', name="Bg Subtraction")

    @magicgui(call_button="Save Snapshot")
    def save_snap():
        try:
            combo = [raw_layers[c].data for c in range(C)]
            arr = np.stack(combo,axis=0)
            fn=f"tile_{random.randint(0,9999)}.tif"
            imsave(fn,arr.astype('uint16'))
            show_info("Saved "+fn)
        except Exception as e:
            show_error(str(e))
    viewer.window.add_dock_widget(save_snap, area='right', name="Export")

    load_random(clear_adjusted=True)
    return viewer

def build_pyramid_numpy(base, num_levels, axis_order='TCZYX'):
    """Build a pyramid using skimage downscale on NumPy array."""
    pyramid = [base]
    shape = base.shape
    spatial_axes = [i for i, ax in enumerate(axis_order) if ax in ('Y', 'X')]

    for _ in range(1, num_levels):
        downscale = tuple(2 if i in spatial_axes else 1 for i in range(len(shape)))
        base = downscale_local_mean(base, downscale).astype(base.dtype)
        pyramid.append(base)

    return pyramid


def save_modified_zarr_pyramid(zarr_path, zarr_level="0", methods=None, load_into_memory=False, axis_order="TCZYX"):

    if methods is None:
        raise ValueError("Must provide background subtraction methods.")

    zgroup = zarr.open(zarr_path, mode="r+")[zarr_level]
    level_keys = sorted(k for k in zgroup.array_keys() if k.isdigit())
    num_levels = len(level_keys)
    print(f"📂 Found {num_levels} pyramid arrays in: {zarr_path}/{zarr_level}/")

    level0_key = level_keys[0]
    level0_dask = da.from_zarr(zgroup[level0_key])
    original_chunks = zgroup[level0_key].chunks
    print(f"📏 Level 0 shape: {level0_dask.shape}, chunks: {original_chunks}, dtype: {level0_dask.dtype}")

    spatial_axes = [i for i, ax in enumerate(axis_order) if ax in ("Y", "X")]
    coarsen_factors = {i: 2 for i in spatial_axes}

    if load_into_memory:
        print("🧠 Loading into memory...")
        level0_np = level0_dask.compute()
        corrected = subtract_background(level0_np, methods=methods, axis_order=axis_order)
        pyramid = build_pyramid_numpy(corrected, num_levels, axis_order=axis_order)

        for i, level in enumerate(pyramid):
            print(f"💾 Writing NumPy level {i}")
            da.from_array(level, chunks=original_chunks).to_zarr(zgroup[str(i)], overwrite=True)
    else:
        print("🐢 Using Dask lazy eval with compute/persist...")
        corrected = subtract_background(level0_dask, methods=methods, axis_order=axis_order).persist()
        base = corrected

        for i in range(num_levels):
            print(f"💾 Writing Dask level {i}")
            base.to_zarr(zgroup[str(i)], overwrite=True)
            base = da.coarsen(np.mean, base, coarsen_factors, trim_excess=True)

    print("✅ Zarr pyramid update complete.")


    
def vsi_background_subtract(vsi_path, 
                            raw_folder,
                            output_ometiff='pyramid_output.ome.tiff',
                            methods=[{"method": "rolling_ball", "sigma_px": 50}],
                            vsi_series=2,
                            zarr_level="0",
                            max_workers=8,
                            load_into_memory=False,
                            overwrite=True,
                            patch_size=512,
                            axis_order="TCZYX"):
    print(f"Using showinf to read metadata for \"{vsi_path}\", series# {vsi_series}...\n")
    showinf_readout = run_commandline(f"showinf -nopix -noflat -series {vsi_series} \"{vsi_path}\"", verbose=1, return_readout=True)
    series_metadata = parse_showinf_series_metadata(showinf_readout, series_number=vsi_series)

    print(f'Metadata for \"{vsi_path}\", series# {vsi_series}:')
    for i, v in series_metadata.items():
        print(f"{i} = {v}")

    if not os.path.isdir(raw_folder) or overwrite:
        print('\nConverting .vsi whole slide image files into zarr folder...')
        run_commandline(f"bioformats2raw --overwrite --resolutions {series_metadata['Resolutions']} --tile-width {patch_size} --max-workers {max_workers} --series {vsi_series} \"{vsi_path}\" {raw_folder}", verbose=1, print_command=True)
    else:
        print(f'\nExisting raw folder found at {raw_folder}, skipping extraction (select overwrite=True to overwrite existing folder)')

    save_modified_zarr_pyramid(raw_folder, zarr_level=zarr_level, methods=methods, load_into_memory=load_into_memory, axis_order=axis_order)

    run_commandline(f"raw2ometiff {raw_folder} {output_ometiff}", verbose=1, print_command=True)


def generate_vsi_path_list(vsi_paths, csv_path='vsi_list.csv', append=True, display_table=True):
    # Ensure vsi_paths is a list
    if not isinstance(vsi_paths, list):
        vsi_paths = [vsi_paths]

    # Collect all .vsi files from the list of directories
    vsi_path_list = []
    for path in vsi_paths:
        vsi_path_list.extend(glob(os.path.join(path, "*.vsi")))

    # Extract filenames without extension
    vsi_files = [Path(x).stem for x in vsi_path_list]

    # Create dataframe
    df = pd.DataFrame({
        'Filename': vsi_files,
        'Path': vsi_path_list,
        'Include': [True] * len(vsi_files),
        'Rename': [""] * len(vsi_files)
    }).set_index('Filename')

    # If CSV doesn't exist, save new
    if not os.path.exists(csv_path):
        df.to_csv(csv_path)
        print(f"Created new CSV at {csv_path}")

    # If CSV exists and appending
    elif append:
        df_exist = pd.read_csv(csv_path, index_col=0)
        df_merged = pd.concat([df, df_exist])

        # Remove duplicates, keeping the first occurrence
        df_merged = df_merged[~df_merged.index.duplicated(keep='first')]

        df_merged.to_csv(csv_path)
        print(f"Appended new entries to {csv_path}, duplicates removed if any found")
        df = df_merged

    else:
        print(f"Existing file found: {csv_path}. No changes made.")
        df = pd.read_csv(csv_path, index_col=0)

    if display_table:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            display(HTML(df.to_html()))

    return


def is_valid_folder_name(name):
    # Check for common invalid folder name characters across major OS
    return bool(re.match(r'^[^<>:"/\\|?*]+$', name)) and name.strip() != ""

def vsi_to_zarr_batch(output_path,
                      vsi_list="vsi_list.csv",
                      vsi_files_path=None,
                      vsi_series=2,
                      patch_size=1024,
                      max_workers=8,
                      overwrite=True,
                      rename=True):
    """
    Converts all .vsi files in a folder to Zarr using bioformats2raw.
    Each .vsi file gets its own subfolder inside output_path.
    """

    if vsi_files_path:
        vsi_files = glob(os.path.join(vsi_files_path, "*.vsi"))
        if not vsi_files:
            print(f"No .vsi files found in {vsi_files_path}")
            return

    vsi_df = None
    if vsi_list and os.path.exists(vsi_list):
        vsi_df = pd.read_csv(vsi_list, index_col=0)

        if 'Include' not in vsi_df.columns:
            print("Warning: 'Include' column missing from VSI list. Assuming all True.")
            vsi_df['Include'] = True

        if 'Path' not in vsi_df.columns:
            raise ValueError("CSV must contain a 'Path' column.")

        if rename:
            if 'Rename' not in vsi_df.columns:
                raise ValueError("Rename column is required but missing in the VSI list.")

            missing = vsi_df[vsi_df.Include & vsi_df.Rename.isnull()]
            if not missing.empty:
                raise ValueError(f"The following included files are missing Rename values:\n{missing.index.tolist()}")

            renames = vsi_df.loc[vsi_df.Include, 'Rename']
            if renames.duplicated().any():
                raise ValueError("Duplicate entries found in Rename column.")

            invalid_names = [r for r in renames if not is_valid_folder_name(r)]
            if invalid_names:
                raise ValueError(f"Invalid folder names found in Rename column: {invalid_names}")

        num_inc = vsi_df.Include.sum()
        num_exc = (~vsi_df.Include).sum()
        print(f'Using stored VSI include list. # included: {num_inc}, # excluded: {num_exc}')

        # Filter only included files using absolute paths
        vsi_include_paths = vsi_df.loc[vsi_df.Include, 'Path'].tolist()
        vsi_files = vsi_include_paths #[x for x in vsi_files if x in vsi_include_paths]

    else:
        if vsi_list:
            print(f"Warning: VSI list {vsi_list} not found. Proceeding with all files.")

    print(f'{len(vsi_files)} .vsi files to process from {vsi_files_path}')
    os.makedirs(output_path, exist_ok=True)

    for vsi_path in vsi_files:
        base_name = os.path.splitext(os.path.basename(vsi_path))[0]

        if rename and vsi_df is not None:
            if base_name in vsi_df.index:
                new_name = vsi_df.loc[base_name, 'Rename']
                if pd.notnull(new_name):
                    base_name = new_name

        raw_folder = os.path.join(output_path, base_name)
        print(f"\nProcessing \"{vsi_path}\" → {raw_folder}")

        # Get metadata
        print(f"Reading metadata for series {vsi_series} using showinf...")
        showinf_readout = run_commandline(
            f"showinf -nopix -noflat -series {vsi_series} \"{vsi_path}\"",
            verbose=1,
            return_readout=True
        )
        series_metadata = parse_showinf_series_metadata(showinf_readout, series_number=vsi_series)

        # Run bioformats2raw conversion
        if not os.path.isdir(raw_folder) or overwrite:
            print(f"\nConverting \"{vsi_path}\" to Zarr folder...")
            run_commandline(
                f"bioformats2raw --overwrite --resolutions {series_metadata['Resolutions']} "
                f"--tile-width {patch_size} --max-workers {max_workers} "
                f"--series {vsi_series} \"{vsi_path}\" {raw_folder}",
                verbose=1,
                print_command=True
            )
        else:
            print(f"Zarr folder already exists at {raw_folder}; skipping (set overwrite=True to force)")


# Generate the complete region crop tool for WSI using napari + zarr + dask
# This includes:
# 1. Napari viewer for annotation on low-res pyramid
# 2. Cropper that slices the full-res pyramid based on user-drawn regions
# 3. Writes full Zarr pyramids with metadata for raw2ometiff/QuPath compatibility


# ------------------------------
# Utility: Update .zattrs metadata
# ------------------------------
def update_multiscale_metadata(zarr_group, num_levels, axes=None):
    """Attach OME-NGFF multiscales metadata to the Zarr group."""
    datasets = [{"path": str(i)} for i in range(num_levels)]
    axes = axes or [
        {"name": "t", "type": "time"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"}
    ]
    zarr_group.attrs["multiscales"] = [{
        "version": "0.4",
        "datasets": datasets,
        "axes": axes
    }]

# ------------------------------
# Function 1: Launch Viewer
# ------------------------------

def list_zarr_folders(zarr_path):
    return sorted([
        name for name in os.listdir(zarr_path)
        if os.path.isdir(os.path.join(zarr_path, name))
    ])

def load_saved_shapes(json_folder, scale):
    shapes = []
    for fname in sorted(os.listdir(json_folder)):
        if fname.endswith('.json'):
            with open(os.path.join(json_folder, fname), 'r') as f:
                region = json.load(f)
                ymin = region['ymin']
                ymax = region['ymax']
                xmin = region['xmin']
                xmax = region['xmax']
                shape = np.array([
                    [ymin, xmin],
                    [ymin, xmax],
                    [ymax, xmax],
                    [ymax, xmin]
                ]) / scale  # ✅ Downscale for display
                shapes.append(shape)
    return np.array(shapes)

def launch_annotation_viewer(zarr_path_root, output_root, zarr_level="0", axis_order = 'TCZYX',):
    zarr_folders = list_zarr_folders(zarr_path_root)
    viewer = napari.Viewer()

    def load_zarr_folder(zarr_folder_name, load_regions=True, normalize_contrast=True, normalization_percentile=0.99, display_level=6):

        viewer.layers.clear()
        full_zarr_path = os.path.join(zarr_path_root, zarr_folder_name)
        store = zarr.open(full_zarr_path, mode='r')[zarr_level]
        lowres = da.from_zarr(store[str(display_level)])

        scale = 2 ** display_level
        output_path = os.path.join(output_root, zarr_folder_name)

        # Extract channel names and colors from OME
        try:
            with open(os.path.join(full_zarr_path, "OME", "METADATA.ome.xml"), encoding='utf-8') as f:
                xml_text = f.read()
            xml_text = xml_text.replace("Âµm", "µm")

            from ome_types import from_xml
            ome = from_xml(xml_text)
            channels = ome.images[0].pixels.channels

            channel_names = [ch.name for ch in channels]
            channel_colors = [wavelength_to_color(ch.emission_wavelength) if ch.emission_wavelength else (1.0, 1.0, 1.0)
                              for ch in channels]

        except Exception as e:
            print(f"⚠️ Failed to read OME metadata: {e}")
            try:
                channel_axis = axis_order.index('C')
                num_channels = lowres.shape[channel_axis]
            except ValueError:
                print("❌ 'C' not found in axis_order.")
                num_channels = 1

            channel_names = [f"Channel {i}" for i in range(num_channels)]
            channel_colors = [(1.0, 1.0, 1.0)] * num_channels

        # Determine channel axis and move to front
        try:
            channel_axis = axis_order.index('C')
            num_channels = lowres.shape[channel_axis]
            lowres_channels_first = da.moveaxis(lowres, channel_axis, 0)

        except Exception as e:
            print(f"❌ Error moving channel axis: {e}")
            viewer.add_image(
                lowres,
                name="Image",
                colormap='gray',
                scale=(scale, scale)
            )
            num_channels = 1
            lowres_channels_first = [lowres]

        # Add each channel as its own image layer with normalization
        for i in range(num_channels):
            # 1. Squeeze and prepare channel image
            channel_img = da.squeeze(lowres_channels_first[i])

            # 2. Add image layer first
            image_layer = viewer.add_image(
                channel_img,
                name=channel_names[i] if i < len(channel_names) else f"Channel {i}",
                colormap=channel_colors[i],
                scale=(scale, scale),
                blending="additive"
            )

            # 3. Optionally normalize contrast
            if normalize_contrast:
                try:
                    if isinstance(image_layer.data, DaskArray):
                        data = image_layer.data.compute()
                    else:
                        data = np.asarray(image_layer.data)

                    vmin = float(np.percentile(data, 0))
                    vmax = float(np.percentile(data, normalization_percentile * 100))

                    image_layer.contrast_limits = (vmin, vmax)
                except Exception as e:
                    print(f"⚠️ Could not compute contrast for channel {i}: {e}")

        # Load and add shapes
        if load_regions and os.path.isdir(output_path):
            shapes = load_saved_shapes(output_path, scale)
        else:
            shapes = []

        if "Regions" in viewer.layers:
            viewer.layers.remove("Regions")

        new_shapes_layer = viewer.add_shapes(
            data=shapes,
            name="Regions",
            shape_type="rectangle",
            edge_color="red",
            edge_width=10,
            face_color="transparent",
            scale=(scale, scale)
        )

        save_shapes.current_layer = new_shapes_layer
        save_shapes.zarr_folder_name = zarr_folder_name
        save_shapes.scale = scale

        print(f"✅ Loaded Zarr: {zarr_folder_name} with {'regions' if load_regions else 'no regions'}")

    @magicgui(
        zarr_folder={"choices": zarr_folders},
        load_saved_regions={"label": "Load saved regions"},
        normalize_contrast={"label": "Normalize contrast"},
        normalization_percentile={"widget_type": "FloatSpinBox", "min": 0.0, "max": 1.0, "step": 0.01,
                                  "label": "Normalize to percentile"},
        display_level={"widget_type": "SpinBox", "min": 0, "max": 10, "step": 1, "label": "Display level"},
        call_button="Load Zarr"
    )
    def select_zarr(zarr_folder, load_saved_regions=True, normalize_contrast=True, normalization_percentile=0.99,
                    display_level=6):
        load_zarr_folder(
            zarr_folder,
            load_regions=load_saved_regions,
            normalize_contrast=normalize_contrast,
            normalization_percentile=normalization_percentile,
            display_level=display_level
        )

    @magicgui(
        overwrite={"label": "Overwrite saved regions"},
        call_button="Save Region Shapes"
    )
    def save_shapes(overwrite: bool = True):
        output_path = os.path.join(output_root, save_shapes.zarr_folder_name)
        display_level = select_zarr.display_level.value
        save_annotation_shapes(
            save_shapes.current_layer.data,
            output_root=output_path,
            display_level=display_level,
            overwrite=overwrite
        )
    @magicgui(call_button="Clear Regions")
    def clear_shapes():
        if hasattr(save_shapes, "current_layer") and save_shapes.current_layer in viewer.layers:
            save_shapes.current_layer.data = []
            print("🧹 Cleared all shapes from the viewer.")
        else:
            print("⚠️ No shapes layer found.")

    viewer.window.add_dock_widget(select_zarr, area='right')
    viewer.window.add_dock_widget(save_shapes, area='right')
    viewer.window.add_dock_widget(clear_shapes, area='right')

    if zarr_folders:
        load_zarr_folder(zarr_folders[0], load_regions=True)

    @magicgui(call_button="Clear Regions")
    def clear_shapes():
        if hasattr(save_shapes, "current_layer") and save_shapes.current_layer in viewer.layers:
            save_shapes.current_layer.data = []
            print("🧹 Cleared all shapes from the viewer.")
        else:
            print("⚠️ No shapes layer found.")



    print(f"✅ Annotation viewer initialized for root: {zarr_path_root}")
    return viewer



def save_annotation_shapes(shapes_data, output_root, display_level=3, overwrite=True):
    os.makedirs(output_root, exist_ok=True)
    scale = 2 ** display_level

    current_files = []
    for i, shape in enumerate(shapes_data):
        coords = (np.array(shape) * scale).astype(int)
        ymin, xmin = np.min(coords, axis=0).tolist()
        ymax, xmax = np.max(coords, axis=0).tolist()

        region_data = {
            "index": i + 1,
            "ymin": ymin,
            "ymax": ymax,
            "xmin": xmin,
            "xmax": xmax
        }

        json_filename = f"region_{i+1:03d}.json"
        json_path = os.path.join(output_root, json_filename)
        current_files.append(json_filename)

        if not overwrite and os.path.exists(json_path):
            print(f"⚠️  Skipping existing file: {json_path}")
            continue

        with open(json_path, "w") as f:
            json.dump(region_data, f)

    # Delete any old region JSONs not in the current set
    existing_files = [f for f in os.listdir(output_root) if f.endswith(".json")]
    stale_files = set(existing_files) - set(current_files)

    for stale in stale_files:
        stale_path = os.path.join(output_root, stale)
        os.remove(stale_path)
        print(f"🗑️ Removed stale region: {stale_path}")

    print(f"✅ Saved {len(shapes_data)} region JSONs to: {output_root}")




def build_pyramid_numpy(base, num_levels):
    pyramid = [base]
    for _ in range(1, num_levels):
        base = downscale_local_mean(base, (1, 1, 1, 2, 2)).astype(base.dtype)
        pyramid.append(base)
    return pyramid



def save_region_pyramid(pyramid, out_path, zarr_level="0", chunks=None, dtype="uint16"):
    """
    Save a pyramid (list of NumPy arrays) to a new Zarr structure with nested chunk layout.

    Parameters:
    - pyramid: list of NumPy arrays for each resolution level
    - out_path: destination folder for the region
    - zarr_level: name of the subfolder (e.g., '0') to store pyramid arrays
    - chunks: chunk size to use (e.g., from original Zarr)
    - dtype: datatype to enforce in output
    """
    pyramid_path = os.path.join(out_path, zarr_level)
    os.makedirs(pyramid_path, exist_ok=True)

    for i, level in enumerate(pyramid):
        # Wrap each level as a Dask array and save to its own subfolder
        arr = da.from_array(level.astype(dtype), chunks=chunks)
        arr.to_zarr(
            os.path.join(pyramid_path, str(i)),
            overwrite=True,
            dimension_separator="/"  # Use nested folder format
        )

def copy_metadata_structure(zarr_path, region_out_path, zarr_level="0"):
    """
    Copy required metadata files and OME folder for OME-Zarr compatibility.
    Includes:
    - .zattrs and .zgroup from both root and pyramid folder
    - OME folder if present
    """

    os.makedirs(region_out_path, exist_ok=True)

    # 1. Copy OME folder (image/channel metadata)
    ome_src = os.path.join(zarr_path, "OME")
    ome_dst = os.path.join(region_out_path, "OME")
    if os.path.isdir(ome_src):
        shutil.copytree(ome_src, ome_dst, dirs_exist_ok=True)
        print("📁 Copied OME/ metadata folder")

    # 2. Copy root-level Zarr metadata (.zattrs, .zgroup)
    for fname in [".zattrs", ".zgroup"]:
        src_file = os.path.join(zarr_path, fname)
        dst_file = os.path.join(region_out_path, fname)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"📄 Copied root-level {fname}")

    # 3. Copy Zarr metadata from pyramid folder (e.g. zarr_path/0 → region_out_path/0)
    pyramid_src = os.path.join(zarr_path, zarr_level)
    pyramid_dst = os.path.join(region_out_path, zarr_level)
    os.makedirs(pyramid_dst, exist_ok=True)
    for fname in [".zattrs", ".zgroup"]:
        src_file = os.path.join(pyramid_src, fname)
        dst_file = os.path.join(pyramid_dst, fname)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"📄 Copied {zarr_level}/ level {fname}")



def update_ome_xml_dimensions(region_path, shape, dtype="uint16"):
    """
    Patch the OME-XML file in a region folder to match the cropped image dimensions.

    Parameters:
    - region_path: Path to region folder containing `OME/METADATA.ome.xml`
    - shape: Tuple of (T, C, Z, Y, X) from the cropped data
    - dtype: Output data type (default: 'uint16')
    """
    xml_path = os.path.join(region_path, "OME", "METADATA.ome.xml")

    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"No OME-XML found at: {xml_path}")

    with open(xml_path, encoding='utf-8') as f:
        xml_text = f.read().replace("Âµm", "µm")

    ome = from_xml(xml_text)

    T, C, Z, Y, X = shape
    pixels = ome.images[0].pixels
    pixels.size_t = T
    pixels.size_c = C
    pixels.size_z = Z
    pixels.size_y = Y
    pixels.size_x = X
    pixels.type = dtype
    pixels.interleaved = False  # Conservative default

    # You might also clear or fix planes if needed:
    pixels.planes = []

    # Save updated XML
    new_xml = to_xml(ome)
    with open(xml_path, "w", encoding='utf-8') as f:
        f.write(new_xml)

    print(f"📄 Updated OME-XML dimensions in: {xml_path}")


def process_all_zarrs_and_regions(
    zarr_path_root,
    regions_root,
    zarr_crops_path="zarr_crops",
    ometiffs_path="ometiffs",
    convert_to_ome_tiff=True,
    flatten_ometiff_folder=True,
    delete_zarr_crops_after=True,
    zarr_level="0",
    dtype='uint16',
    pyramid_levels=None,
    rename_files=False,
    vsi_csv = 'vsi_include_list.csv'
):
    os.makedirs(zarr_crops_path, exist_ok=True)
    if convert_to_ome_tiff:
        os.makedirs(ometiffs_path, exist_ok=True)

    zarr_folders = sorted([
        f for f in os.listdir(zarr_path_root)
        if os.path.isdir(os.path.join(zarr_path_root, f))
    ])

    # Load or generate rename map
    rename_map = {}
    if rename_files:
        if not os.path.exists(vsi_csv):
            df = pd.DataFrame(index=zarr_folders).rename_axis('Filename')
            df["Rename"] = ""
            df.to_csv(vsi_csv)
            print(f"⚠️ Created template {vsi_csv}. Please fill it and re-run.")
            return

        df = pd.read_csv(vsi_csv, index_col=0)

        if 'Include' in df.columns:
            df = df.loc[df['Include'] == True, :]

        missing = [zf for zf in zarr_folders if zf not in df.index or not isinstance(df.loc[zf, 'Rename'], str) or df.loc[zf, 'Rename'].strip() == ""]
        if missing:
            print(f"⚠️ Rename entries missing for: {missing}")
            print(f"Please update {vsi_csv} and re-run.")
            return

        rename_map = df["Rename"].to_dict()

    for zarr_name in zarr_folders:
        zarr_path = os.path.join(zarr_path_root, zarr_name)
        region_dir = os.path.join(regions_root, zarr_name)
        if not os.path.isdir(region_dir):
            print(f"⚠️ No region folder for {zarr_name}, skipping.")
            continue

        print(f"\n🚀 Processing Zarr: {zarr_name}")
        region_files = sorted(glob(os.path.join(region_dir, "*.json")))
        if not region_files:
            print("⚠️ No regions found.")
            continue

        # Load Zarr
        zarr_store = zarr.open(zarr_path, mode='r')
        level0 = zarr_store[zarr_level]["0"]
        fullres = da.from_zarr(level0)
        original_chunks = level0.chunks
        if not dtype: dtype = fullres.dtype

        if pyramid_levels is None:
            pyramid_levels = len(list(zarr_store[zarr_level].array_keys())) + 1

        base_name = rename_map.get(zarr_name, zarr_name)

        for region_file in region_files:
            region_id = Path(region_file).stem
            with open(region_file, "r") as f:
                region_info = json.load(f)

            ymin = max(0, region_info["ymin"])
            ymax = min(fullres.shape[-2], region_info["ymax"])
            xmin = max(0, region_info["xmin"])
            xmax = min(fullres.shape[-1], region_info["xmax"])

            if ymin >= ymax or xmin >= xmax:
                print(f"⚠️ Skipping region {region_id} — invalid crop area after bounding.")
                continue

            print(f"✂️ Cropping {zarr_name} — {region_id} → Y: {ymin}-{ymax}, X: {xmin}-{xmax}")

            cropped = fullres[:, :, :, ymin:ymax, xmin:xmax].compute()

            pyramid = build_pyramid_numpy(cropped, num_levels=pyramid_levels)

            out_crop_name = f"{base_name}_{region_id}"
            out_crop_path = os.path.join(zarr_crops_path, out_crop_name)

            save_region_pyramid(
                pyramid,
                out_crop_path,
                zarr_level=zarr_level,
                chunks=original_chunks,
                dtype=dtype
            )

            copy_metadata_structure(zarr_path, out_crop_path, zarr_level=zarr_level)
            update_ome_xml_dimensions(out_crop_path, shape=cropped.shape, dtype=str(dtype))

            print(f"✅ Saved crop: {out_crop_path}")

            # Convert to OME-TIFF
            if convert_to_ome_tiff:
                if flatten_ometiff_folder:
                    tiff_path = os.path.join(ometiffs_path, f"{out_crop_name}.ome.tiff")
                else:
                    subfolder = os.path.join(ometiffs_path, base_name)
                    os.makedirs(subfolder, exist_ok=True)
                    tiff_path = os.path.join(subfolder, f"{region_id}.ome.tiff")

                command = f"raw2ometiff {out_crop_path} {tiff_path}"
                run_commandline(command, verbose=0, print_command=True)

        print(f"✅ Finished processing: {zarr_name}")

    if convert_to_ome_tiff and delete_zarr_crops_after:
        print(f"🧹 Deleting temporary crops in: {zarr_crops_path}")
        shutil.rmtree(zarr_crops_path)


def generate_summary_pngs(
    input_dir,
    output_dir="summary_pngs",
    display_level=8,
    quantile=0.95,
    squash_channels=False,
    max_images=None,
    zarr_level="0"
):
    """
    Generate summary PNGs from a folder of OME-TIFFs or Zarr folders.

    Parameters:
    - input_dir: folder containing either .ome.tiff files or Zarr folders
    - output_dir: where to save PNGs
    - display_level: pyramid level to use (higher = more downsampled)
    - quantile: quantile to use for contrast stretching (0.95 = 95th percentile)
    - squash_channels: sum all channels into grayscale
    - max_images: optional limit on number of images
    - zarr_level: subfolder level inside Zarr (usually "0")
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # --- Auto-detect format ---
    entries = sorted(list(input_dir.glob("*.ome.tif*")))
    from_format = "ometiff" if entries else "zarr"
    if not entries:
        entries = sorted([p for p in input_dir.iterdir() if (p / zarr_level).exists()])

    if not entries:
        print("❌ No OME-TIFFs or Zarr folders found.")
        return

    if max_images:
        entries = entries[:max_images]

    print(f"📂 Detected format: {from_format} — processing {len(entries)} items")

    for entry in entries:
        try:
            name = entry.stem
            if from_format == "zarr":
                arr_path = entry / zarr_level / str(display_level)
                arr = da.from_zarr(str(arr_path))

                if arr.ndim == 5:
                    arr = arr[0, :, 0, :, :]  # (C, Y, X)
                elif arr.ndim == 4:
                    arr = arr[0, :, :, :]     # (C, Y, X)

                data = arr.compute().astype(np.float32)

            else:  # OME-TIFF
                with TiffFile(str(entry)) as tif:
                    series = tif.series[0]
                    levels = series.levels
                    level = levels[min(display_level, len(levels) - 1)]
                    data = level.asarray().astype(np.float32)

                    if data.ndim == 2:
                        data = data[np.newaxis, ...]
                    elif data.ndim == 3 and data.shape[-1] in (1, 3, 4):
                        data = data.transpose(2, 0, 1)  # (C, Y, X)

            # --- Normalize and optionally squash ---
            if squash_channels:
                combined = np.sum(data, axis=0)
                vmin, vmax = np.percentile(combined, [0, quantile * 100])
                img = np.clip((combined - vmin) / (vmax - vmin), 0, 1)
                img_uint8 = (img * 255).astype(np.uint8)
                rgb = np.stack([img_uint8] * 3, axis=-1)

            elif data.shape[0] == 1:
                ch = data[0]
                vmin, vmax = np.percentile(ch, [0, quantile * 100])
                norm = np.clip((ch - vmin) / (vmax - vmin), 0, 1)
                rgb = np.stack([(norm * 255).astype(np.uint8)] * 3, axis=-1)

            else:
                channels = []
                for ch in data[:3]:
                    vmin, vmax = np.percentile(ch, [0, quantile * 100])
                    ch_norm = np.clip((ch - vmin) / (vmax - vmin), 0, 1)
                    channels.append((ch_norm * 255).astype(np.uint8))
                while len(channels) < 3:
                    channels.append(np.zeros_like(channels[0]))
                rgb = np.stack(channels[:3], axis=-1)

            save_path = output_dir / f"{name}.png"
            imsave(str(save_path), rgb)
            print(f"✅ Saved summary: {save_path.name}")

        except Exception as e:
            print(f"❌ Failed for {entry.name}: {e}")







def organize_ometiffs(ometiffs_path="ometiffs", reverse=False):
    """
    Organizes or reverses organization of OME-TIFFs in a given folder using a CSV.

    Parameters:
    - ometiffs_path (str): Root directory containing .ome.tiff files or subfolders.
    - reverse (bool): If True, flattens any subfolders and moves files back to root.
    """

    ometiffs_path = Path(ometiffs_path)

    if reverse:
        print("🔁 Reversing subfolder organization...")

        ome_files = list(ometiffs_path.glob("**/*.ome.tiff"))
        ome_files = [f for f in ome_files if f.parent != ometiffs_path]

        if not ome_files:
            print("⚠️ No .ome.tiff files found in subfolders to reverse.")
            return

        for f in ome_files:
            target = ometiffs_path / f.name
            if target.exists():
                print(f"⚠️ File already exists at root: {target.name}, skipping.")
                continue
            shutil.move(str(f), str(target))

        # Clean up empty subfolders
        for subdir in ometiffs_path.glob("*/"):
            if subdir.is_dir() and not any(subdir.iterdir()):
                subdir.rmdir()
                print(f"🧹 Removed empty folder: {subdir}")

        print("✅ All .ome.tiff files moved back to root.")
        return

    # --- Normal mode (organize into subfolders) ---
    # Check if any ome.tiffs are already inside subfolders
    already_nested = any(f.is_file() and f.parent != ometiffs_path for f in ometiffs_path.glob("**/*.ome.tiff"))
    if already_nested:
        print("⚠️ Some .ome.tiff files already exist in subfolders. Use reverse=True to undo.")
        return

    csv_path = Path("organise_ometiffs.csv")
    ome_files = sorted([f.name for f in ometiffs_path.glob("*.ome.tiff")])

    if not csv_path.exists():
        df = pd.DataFrame(index=ome_files)
        df["subfolder"] = ""
        df.to_csv(csv_path)
        print(f"⚠️ Created template CSV: {csv_path}. Please complete and re-run.")
        return

    df = pd.read_csv(csv_path, index_col=0)

    missing = [f for f in ome_files if f not in df.index or not isinstance(df.loc[f, 'subfolder'], str) or df.loc[f, 'subfolder'].strip() == ""]
    if missing:
        print(f"⚠️ Missing or incomplete subfolder assignments for: {missing}")
        print(f"Please update {csv_path} and re-run.")
        return

    # Perform the move operations
    for file_name, subfolder in df["subfolder"].items():
        src = ometiffs_path / file_name
        dst_folder = ometiffs_path / subfolder
        dst_folder.mkdir(parents=True, exist_ok=True)
        dst = dst_folder / file_name

        if not src.exists():
            print(f"⚠️ Source file not found: {file_name}")
            continue

        shutil.move(str(src), str(dst))
        print(f"📁 Moved {file_name} → {subfolder}/")

    print("✅ OME-TIFFs organized into subfolders.")


def vsi_to_slidelabels(temp_zarr_path='temp',
                       image_output_path='slidelabels',
                      vsi_files_path=None,
                      vsi_series=0,
                      patch_size=1024,
                      max_workers=8):
    """
    Converts all .vsi files in a folder to Zarr using bioformats2raw.
    Each .vsi file gets its own subfolder inside output_path.
    """

    if vsi_files_path:
        vsi_files = glob(os.path.join(vsi_files_path, "*.vsi"))
        if not vsi_files:
            print(f"No .vsi files found in {vsi_files_path}")
            return

    print(f'{len(vsi_files)} .vsi files to process from {vsi_files_path}')
    os.makedirs(temp_zarr_path, exist_ok=True)

    for vsi_path in vsi_files:
        base_name = os.path.splitext(os.path.basename(vsi_path))[0]

        temp_zarr_folder = os.path.join(temp_zarr_path, base_name)
        print(f"\nProcessing \"{vsi_path}\" → {temp_zarr_folder}")

        # Run bioformats2raw conversion
        print(f"\nConverting \"{vsi_path}\" to Zarr folder...")
        run_commandline(
            f"bioformats2raw --overwrite "
            f"--tile-width {patch_size} --max-workers {max_workers} "
            f"--series {vsi_series} \"{vsi_path}\" {temp_zarr_folder}",
            verbose=1,
            print_command=True
        )

        tiff_path = os.path.join(image_output_path, (base_name + ".ome.tiff"))

        command = f"raw2ometiff {temp_zarr_folder} {tiff_path}"
        run_commandline(command, verbose=0, print_command=True)
