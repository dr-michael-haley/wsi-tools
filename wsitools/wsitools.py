import logging
import os
import re
from glob import glob
import subprocess as sp
import numpy as np
import dask.array as da
from skimage.transform import downscale_local_mean
from ome_types import from_xml
import subprocess
import zarr
from dask_image.ndfilters import gaussian_filter, median_filter
from scipy.ndimage import grey_opening
from scipy import ndimage
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.transform import downscale_local_mean
import numpy as np

import napari
import numpy as np
import dask.array as da
import random
from magicgui import magicgui, widgets
from skimage.io import imsave
from napari.utils.notifications import show_info, show_warning, show_error
from qtpy.QtCore import QTimer
from collections import defaultdict

from dask.distributed import Client, LocalCluster

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
    from ome_types import from_xml
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
    import numpy as np
    import dask.array as da
    from dask_image.ndfilters import gaussian_filter, median_filter
    from skimage.exposure import rescale_intensity, equalize_adapthist
    from skimage.morphology import opening as grey_opening
    from scipy import ndimage

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
        from collections import defaultdict

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
    import dask.array as da
    import zarr
    import numpy as np

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
    print(f"Using showinf to read metadata for {vsi_path}, series# {vsi_series}...\n")
    showinf_readout = run_commandline(f"showinf -nopix -noflat -series {vsi_series} {vsi_path}", verbose=1, return_readout=True)
    series_metadata = parse_showinf_series_metadata(showinf_readout, series_number=vsi_series)

    print(f'Metadata for {vsi_path}, series# {vsi_series}:')
    for i, v in series_metadata.items():
        print(f"{i} = {v}")

    if not os.path.isdir(raw_folder) or overwrite:
        print('\nConverting .vsi whole slide image files into zarr folder...')
        run_commandline(f"bioformats2raw --overwrite --resolutions {series_metadata['Resolutions']} --tile-width {patch_size} --max-workers {max_workers} --series {vsi_series} {vsi_path} {raw_folder}", verbose=1, print_command=True)
    else:
        print(f'\nExisting raw folder found at {raw_folder}, skipping extraction (select overwrite=True to overwrite existing folder)')

    save_modified_zarr_pyramid(raw_folder, zarr_level=zarr_level, methods=methods, load_into_memory=load_into_memory, axis_order=axis_order)

    run_commandline(f"raw2ometiff {raw_folder} {output_ometiff}", verbose=1, print_command=True)


def vsi_to_zarr_batch(vsi_files_path,
                      output_path,
                      vsi_series=2,
                      patch_size=None,
                      max_workers=8,
                      overwrite=True):
    """
    Converts all .vsi files in a folder to zarr using bioformats2raw.
    Each .vsi file gets its own subfolder inside output_path.
    """
    vsi_files = glob(os.path.join(vsi_files_path, "*.vsi"))
    if not vsi_files:
        print(f"No .vsi files found in {vsi_files_path}")
        return

    print(f'{len(vsi_files)} .vsi files found in {str(vsi_files_path)}')

    os.makedirs(output_path, exist_ok=True)

    for vsi_path in vsi_files:
        base_name = os.path.splitext(os.path.basename(vsi_path))[0]
        raw_folder = os.path.join(output_path, base_name)

        print(f"\nProcessing {vsi_path} → {raw_folder}")

        # Get metadata
        print(f"Reading metadata for series {vsi_series} using showinf...")
        showinf_readout = run_commandline(
            f"showinf -nopix -noflat -series {vsi_series} {vsi_path}",
            verbose=1,
            return_readout=True
        )
        series_metadata = parse_showinf_series_metadata(showinf_readout, series_number=vsi_series)

        if not os.path.isdir(raw_folder) or overwrite:
            print(f"\nConverting {vsi_path} to Zarr folder...")
            run_commandline(
                f"bioformats2raw --overwrite --resolutions {series_metadata['Resolutions']} "
                f"--tile-width {patch_size} --max-workers {max_workers} "
                f"--series {vsi_series} {vsi_path} {raw_folder}",
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

import numpy as np
import dask.array as da
import zarr
import os
from magicgui import magicgui
from skimage.transform import downscale_local_mean
import napari
from qtpy.QtWidgets import QMessageBox
from ome_types.model import OME, Image, Pixels, Channel, TiffData

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

def launch_annotation_viewer(zarr_path, output_root, zarr_level="0", display_level=6):
    """
    Launch Napari for region annotation on a WSI and crop regions into new Zarr pyramids.

    Parameters:
    - zarr_path (str): path to top-level Zarr (e.g. "Images_raw")
    - output_root (str): folder to save cropped region Zarrs into
    - zarr_level (str): subfolder of Zarr with pyramid arrays (default: "0")
    - display_level (int): which downsample level to display for annotation (default: 3)
    - pyramid_levels (int): how many pyramid levels to build per cropped region
    """
    store = zarr.open(zarr_path, mode='r')[zarr_level]
    lowres = da.from_zarr(store[str(display_level)])

    viewer = napari.Viewer()
    viewer.add_image(lowres, name="WSI LowRes", colormap='gray')
    shapes = viewer.add_shapes(name="Regions", shape_type='rectangle', edge_color='red', face_color='transparent')

    @magicgui(call_button="Save Region Shapes")
    def save_shapes():
        save_annotation_shapes(shapes.data, output_root=output_root, display_level=display_level)

    viewer.window.add_dock_widget(save_shapes, area='right')
    print(f"✅ Annotation viewer launched for {zarr_path}, saving to {output_root}")
    return viewer

def save_annotation_shapes(shapes_data, output_root, display_level=3):
    """
    Save napari shape rectangles to JSON files scaled to level 0.

    Parameters:
    - shapes_data: List of (N, 4, 2) arrays from napari shapes
    - output_root: Folder to write region_###.json files into
    - display_level: downsample level (e.g. 3 → scale=8)
    """
    os.makedirs(output_root, exist_ok=True)
    scale = 2 ** display_level

    for i, shape in enumerate(shapes_data):
        coords = (np.array(shape) * scale).astype(int)
        ymin, xmin = np.min(coords, axis=0).tolist()
        ymax, xmax = np.max(coords, axis=0).tolist()

        region_data = {"index": i+1, "ymin": ymin, "ymax": ymax, "xmin": xmin, "xmax": xmax}
        with open(os.path.join(output_root, f"region_{i+1:03d}.json"), "w") as f:
            json.dump(region_data, f)

    print(f"✅ Saved {len(shapes_data)} region JSONs to: {output_root}")

def build_pyramid_numpy(base, num_levels):
    pyramid = [base]
    for _ in range(1, num_levels):
        base = downscale_local_mean(base, (1, 1, 1, 2, 2)).astype(base.dtype)
        pyramid.append(base)
    return pyramid


import os
import json
import shutil
import zarr
import dask.array as da
import numpy as np
from skimage.transform import downscale_local_mean
from glob import glob

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

from ome_types import from_xml, to_xml
import os

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


def process_saved_regions(zarr_path, region_dir, zarr_level="0", pyramid_levels=None):
    """
    Process all saved region definitions and save as separate Zarr pyramids.

    Parameters:
    - zarr_path: original BioFormats2Raw-style Zarr folder.
    - region_dir: folder with JSON files for region definitions.
    - zarr_level: typically "0" — the main group holding pyramids.
    - pyramid_levels: how many pyramid levels to generate (match original if None).
    """

    # Load full-resolution Zarr array
    zarr_store = zarr.open(zarr_path, mode='r')
    level0 = zarr_store[zarr_level]["0"]
    fullres = da.from_zarr(level0)
    print(fullres)
    original_chunks = level0.chunks
    dtype = fullres.dtype

    if pyramid_levels is None:
        pyramid_levels = len(list(zarr_store[zarr_level].array_keys())) + 1

    print(f"🔍 Found {pyramid_levels} pyramid levels with chunk size {original_chunks}")

    # Find region definition files
    region_files = sorted(glob(os.path.join(region_dir, "*.json")))
    print(f"🔍 Found {len(region_files)} region files.")

    for region_file in region_files:
        region_id = os.path.splitext(os.path.basename(region_file))[0]
        with open(region_file, "r") as f:
            region_info = json.load(f)

        ymin = region_info["ymin"]
        ymax = region_info["ymax"]
        xmin = region_info["xmin"]
        xmax = region_info["xmax"]

        print(f"✂️ Cropping region: {region_id} → Y: {ymin}-{ymax}, X: {xmin}-{xmax}")

        cropped = fullres[:, :, :, ymin:ymax, xmin:xmax].compute()
        pyramid = build_pyramid_numpy(cropped, num_levels=pyramid_levels)

        out_path = os.path.join(region_dir, region_id)
        save_region_pyramid(
            pyramid,
            out_path,
            zarr_level=zarr_level,
            chunks=original_chunks,
            dtype=dtype
        )

        copy_metadata_structure(zarr_path, out_path, zarr_level=zarr_level)
        update_ome_xml_dimensions(region_path=out_path, shape=cropped.shape)

        print(f"✅ Saved region to: {out_path}")

