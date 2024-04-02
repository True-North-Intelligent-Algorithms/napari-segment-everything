from segment_anything import sam_model_registry, SamPredictor 
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
import urllib.request
import warnings
from pathlib import Path
from typing import Optional
from skimage.measure import label                    
from skimage.measure import regionprops
from skimage import color

import toolz as tz
from napari.utils import progress

# Some code in this file copied from https://github.com/royerlab/napari-segment-anything/blob/main/src/napari_segment_anything/utils.py

SAM_WEIGHTS_URL = {
    "default": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


@tz.curry
def _report_hook(
    block_num: int,
    block_size: int,
    total_size: int,
    pbr: "progress" = None,
) -> None:
    downloaded = block_num * block_size
    percent = downloaded * 100 / total_size
    downloaded_mb = downloaded / 1024 / 1024
    total_size_mb = total_size / 1024 / 1024
    increment = int(percent) - pbr.n
    if increment > 1:  # faster than increment at every iteration
        pbr.update(increment)
    print(
        f"Download progress: {percent:.1f}% ({downloaded_mb:.1f}/{total_size_mb:.1f} MB)",
        end="\r",
    )


def download_weights(weight_url: str, weight_path: "Path"):
    print(f"Downloading {weight_url} to {weight_path} ...")
    pbr = progress(total=100)
    try:
        urllib.request.urlretrieve(
            weight_url, weight_path, reporthook=_report_hook(pbr=pbr)
        )
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        urllib.error.ContentTooShortError,
    ) as e:
        warnings.warn(f"Error downloading {weight_url}: {e}")
        return None
    else:
        print("\rDownload complete.                            ")
    pbr.close()


def get_weights_path(model_type: str) -> Optional[Path]:
    """Returns the path to the weight of a given model architecture."""
    weight_url = SAM_WEIGHTS_URL[model_type]

    cache_dir = Path.home() / ".cache/tnia-sam"
    cache_dir.mkdir(parents=True, exist_ok=True)

    weight_path = cache_dir / weight_url.split("/")[-1]

    # Download the weights if they don't exist
    if not weight_path.exists():
        download_weights(weight_url, weight_path)

    return weight_path


def get_sam(model_type: str):
    sam = sam_model_registry[model_type](get_weights_path(model_type))
    #sam.to(self._device)
    return sam

def get_sam_predictor(model_type: str):
    sam = sam_model_registry[model_type](get_weights_path(model_type))
    #sam.to(self._device)
    return SamPredictor(sam)

def get_sam_automatic_mask_generator(model_type: str, points_per_side=32, pred_iou_thresh=0.1, stability_score_thresh=0.1, box_nms_thresh=0.5):
    sam = sam_model_registry[model_type](get_weights_path(model_type))

    sam_anything_predictor = SamAutomaticMaskGenerator(sam,
        points_per_side=int(points_per_side),
        #points_per_batch=int(self.le_points_per_batch.text()),
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        #stability_score_offset=float(self.le_stability_score_offset.text()),
        box_nms_thresh=box_nms_thresh,
        #crop_n_layers=int(self.le_crop_n_layers.text()),
        #crop_nms_thresh=float(self.le_crop_nms_thresh.text()),
        #crop_overlap_ratio=float(self.le_crop_overlap_ratio.text()),
        #crop_n_points_downscale_factor=int(self.le_crop_n_points_downscale_factor.text()),
        #min_mask_region_area=int(self.le_min_mask_region_area.text()),
        )
    #sam.to(self._device)
    return sam_anything_predictor

import numpy as np

def make_label_image_3d(masks):
    '''
    Creates a label image by adding one mask at a time onto an empty image
    '''
    num_masks = len(masks)
    label_image = np.zeros( (num_masks, masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1]), dtype='uint16')

    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    for enum, mask in enumerate(sorted_masks):
        mnarray = mask['segmentation']
        label_image[enum,:,:] = mnarray.astype('uint16') * (enum+1)

    return label_image

def filter_labels_3d_multi(label_image, sorted_results, stats, mins, maxes, napari_label=None): 
    for enum, result in enumerate(sorted_results):
        keep = all(min <= result[stat] <= max for stat, min, max in zip(stats, mins, maxes))
        if keep:
            if result['keep']==True:
                continue
            result['keep'] = True
            coords = np.where(result['segmentation'])
            if napari_label is None:
                temp = label_image[enum, :,:]
                temp[coords] = enum+1
            else:
                z = np.full(coords[0].shape, enum)
                coords = (z, coords[0], coords[1])
                if enum < label_image.shape[0]-1:
                    napari_label.data_setitem(coords, enum+1, True)
                else:
                    napari_label.data_setitem(coords, enum+1, True)
        else:
            if result['keep']==False:
                continue
            result['keep'] = False
            coords = np.where(result['segmentation'])
            if napari_label is None:
                temp = label_image[enum, :,:]
                temp[coords] = 0
            else:
                z = np.full(coords[0].shape, enum)
                coords = (z, coords[0], coords[1])
                if enum < label_image.shape[0]-1:
                    napari_label.data_setitem(coords, 0, True)
                else:
                    napari_label.data_setitem(coords, 0, True)

def filter_labels_3d(label_image, sorted_results, stat, min=0, max=10000000000, napari_label=None): 

    for enum, result in enumerate(sorted_results):
        if result[stat] < min or result[stat] > max:
            if result['keep']==False:
                continue
            result['keep'] = False
            coords = np.where(result['segmentation'])
            if napari_label is None:
                temp = label_image[enum, :,:]
                temp[coords] = 0
            else:
                z = np.full(coords[0].shape, enum)
                coords = (z, coords[0], coords[1])
                if enum < label_image.shape[0]-1:
                    napari_label.data_setitem(coords, 0, True)
                else:
                    napari_label.data_setitem(coords, 0, True)
        else:
            if result['keep']==True:
                continue
            result['keep'] = True
            coords = np.where(result['segmentation'])
            if napari_label is None:
                temp = label_image[enum, :,:]
                temp[coords] = enum+1
            else:
                z = np.full(coords[0].shape, enum)
                coords = (z, coords[0], coords[1])
                if enum < label_image.shape[0]-1:
                    napari_label.data_setitem(coords, enum+1, True)
                else:
                    napari_label.data_setitem(coords, enum+1, True)


def add_properties_to_label_image(orig_image, sorted_results):

    hsv_image = color.rgb2hsv(orig_image)

    hue = 255*hsv_image[:,:,0]
    saturation = 255*hsv_image[:,:,1]
    intensity = 255*hsv_image[:,:,2]

    for enum, result in enumerate(sorted_results):
        segmentation = result['segmentation']
        coords = np.where(segmentation)
        regions = regionprops(segmentation.astype('uint8'))

        # calculate circularity
        result['circularity'] = 4*np.pi*regions[0].area / (regions[0].perimeter**2)
        result['solidity'] = regions[0].solidity
        intensity_pixels = intensity[coords]
        result['mean_intensity'] = np.mean(intensity_pixels)
        result['10th_percentile_intensity'] = np.percentile(intensity_pixels, 10)
        hue_pixels = hue[coords]
        result['mean_hue'] = np.mean(hue_pixels)
        saturation_pixels = saturation[coords]
        result['mean_saturation'] = np.mean(saturation_pixels)


def make_label_image(label_image, masks, orig_img, min_size, max_size, min_intensity=10):
    '''
    Creates a label image by adding one mask at a time onto an empty image, given the masks from an ultralytics prediction

    Inputs:
    - A label image of zeros, in the same size and shape of your original image.
    - A list of masks from an ultralytics segmentation
    Outputs:
    - A label-image of all masks
    '''

    # sort masks by size.  This is done so that small masked are rendered last, and are not overwritten by larger masks.
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    for enum, mask in enumerate(sorted_masks):
        mnarray = mask['segmentation']
        min_col, min_row, width, height = mask['bbox']

        width = width+1
        height = height+1

        # Crop the image
        mnarray = mnarray[min_row:min_row+height, min_col:min_col+width]
        orig_img_cropped = orig_img[min_row:min_row+height, min_col:min_col+width]
        
        # remove low intensities from mask
        mnarray[orig_img_cropped<min_intensity] = 0
        pixels = orig_img_cropped[mnarray]

        if pixels.size == 0:
            continue

        #mnarray = morphology.opening(mnarray, selem)
        if pixels.size > min_size and pixels.size < max_size:
            #label_image[mnarray] = enum + 1 # set each mask to a unique ID (enum)
            add_small_to_large_2d(label_image, ((enum+1)*mnarray).astype('uint16'), min_col, min_row, mode = 'replace_non_zero', center=False)


def add_small_to_large_2d(large_arr, small_arr, x, y, check_empty=False, mode='add', center = True):
    """ Adds a small array 2d to a larger array.

    Notes:  This function is essentially the same as add_small_to_large but for 2D.  The code could be made more 'DRY' in the future.

    Args:
        large_arr (numpy.ndarray): The larger array.
        small_arr (numpy.ndarray): The smaller array to add to the larger array.
        x (int): The x coordinate of the center of the smaller array.
        y (int): The y coordinate of the center of the smaller array.
        check_empty (bool, optional): If True, the function will check if the overlapping slice is empty before adding the small array. Default is False.
    Returns:
        bool: True if the small array was added to the larger array, False otherwise.
    """
    # Get the dimensions of the arrays
    h, w = large_arr.shape
    sh, sw = small_arr.shape

    if center:
        # Calculate the start and end indices for the small array in each dimension
        start_x = x - sw//2
        end_x = start_x + sw
        start_y = y - sh//2
        end_y = start_y + sh
    else:
        start_x = x
        end_x = x + sw
        start_y = y
        end_y = y + sh

    # Calculate the overlapping slice of the small array
    slice_y = slice(max(start_y, 0), min(end_y, h))
    slice_x = slice(max(start_x, 0), min(end_x, w))
    small_slice_y = slice(max(-start_y, 0), min(sh - (end_y - h), sh))
    small_slice_x = slice(max(-start_x, 0), min(sw - (end_x - w), sw))
    small_slice = small_arr[small_slice_y, small_slice_x]

    if check_empty:
        # Check if the slice is empty
        if np.any(np.logical_and(large_arr[slice_y, slice_x]>0,small_slice>0)):
            return False

    # Add the small slice to the large array
    if mode=='add':
        large_arr[slice_y, slice_x] += small_slice
    elif mode=='replace':
        large_arr[slice_y, slice_x] = small_slice
    elif mode=='replace_non_zero':
        indices = np.where(small_slice>0)
        large_arr[slice_y, slice_x][indices] = small_slice[indices]
    return True
