import os.path

import matplotlib.axes
import rasterio
from rasterio import plot as rasterioplt
import rasterio.windows
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Union, Optional, List, Tuple
from ml4floods.data.worldfloods.configs import BANDS_S2, BANDS_L8, CHANNELS_CONFIGURATIONS#plot_s2_rbg_image
from ml4floods.models.model_setup import get_channel_configuration_bands
from ml4floods.data.worldfloods import configs
from ml4floods.data import utils
from matplotlib.patches import Patch
# from ml4floods.data.worldfloods.configs import BANDS_S2, CHANNELS_CONFIGURATIONS 
import geopandas as gpd

def download_tiff(local_folder: str, tiff_input: str, folder_ground_truth: str,
                  folder_permanent_water: Optional[str] = None, requester_pays:bool=True) -> str:
    """
    Download a set of tiffs from the google bucket to a local folder

    Args:
        local_folder: local folder to download
        tiff_input: input tiff file
        folder_ground_truth: folder with ground truth images
        folder_permanent_water: folder with permanent water images
        requester_pays: Requester pays option of the bucket

    Returns:
        location of tiff_input in the local file system

    """
    import fsspec
    fs = fsspec.filesystem("gs", requester_pays=requester_pays)

    folders = ["/S2/", folder_ground_truth]
    if folder_permanent_water is not None:
        folders.append(folder_permanent_water)

    for folder in folders:
        file_to_download = tiff_input.replace("/S2/", folder)
        if folder.startswith("/"):
            folder = folder[1:]
        folder_iter = os.path.join(local_folder, folder)  # remove /
        file_local = os.path.join(folder_iter, os.path.basename(file_to_download))
        if folder == "S2/":
            return_folder = file_local
        if os.path.exists(file_local):
            continue
        if not fs.exists(file_to_download):
            print(f"WARNING!! file {file_to_download} does not exists")
            continue

        os.makedirs(folder_iter, exist_ok=True)
        fs.get_file(file_to_download, file_local)
        print(f"Downloaded file {file_local}")

    return return_folder

def plot_s2_rbg_image(input: Union[str, np.ndarray], transform:Optional[rasterio.Affine]=None,
                      window:Optional[rasterio.windows.Window]=None,
                      max_clip_val:Optional[float]=3000.,
                      min_clip_val:Optional[float]=0.,
                      channel_configuration:str="all",
                      size_read:Optional[int]=None,
                      **kwargs):
    """
    Plot bands B4, B3, B2 of a Sentinel-2 image. Input could be an array or a str. Values are clipped to 3000
    (it assumes the image has values in [0, 10_000] -> https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2 )

    Tip: Use `size_read` to read from the image pyramid if input is a COG GeoTIFF to speed up the reading.

    Args:
        input: str of array with (C, H, W) configuration.
        transform: geospatial transform if input is an array
        window: window to read from the input
        max_clip_val: value to clip the the input for visualization
        min_clip_val: value to clip the the input for visualization
        channel_configuration: Expected bands of the inputs
        size_read: max size to read. Use this to read from the overviews of the image.
        **kwargs: extra args for rasterio.plot.show

    Returns:
        ax : matplotlib Axes
            Axes with plot.

    """

    BANDS_S2 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]

    # 0 based channels based on BANDS_S2
    CHANNELS_CONFIGURATIONS = {
        "all": list(range(0,len(BANDS_S2))),
        "rgb": [3, 2, 1],
        "swirnirred": [11, 7, 3],
        "bgr": [1, 2, 3],
        "bgri": [1, 2, 3, 7],
        "riswir" : [3, 7, 11],
        "bgriswir" : [1, 2, 3, 7, 11],
        "bgriswirs" : [1, 2, 3, 7, 11, 12],
        "l89s2": [0, 1, 2, 3, 7, 10, 11, 12], # Same bands as Landsat-7 and Landsat-8
        "sub_20": [1, 2, 3, 4, 5, 6, 7, 8, 11, 12],
        "hyperscout2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    band_names_current_image = [BANDS_S2[iband] for iband in CHANNELS_CONFIGURATIONS[channel_configuration]]
    bands = [band_names_current_image.index(b) for b in ["B4", "B3", "B2"]]
    image, transform = get_image_transform(input, transform=transform, bands=bands, window=window,
                                           size_read=size_read)

    if max_clip_val is not None:
        min_clip_val = 0 if min_clip_val is None else min_clip_val
        image = np.clip((image-min_clip_val)/(max_clip_val - min_clip_val), 0, 1)

    return rasterioplt.show(image, transform=transform, **kwargs)
