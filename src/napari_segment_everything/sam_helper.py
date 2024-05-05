import gc
import urllib.request
import warnings
from pathlib import Path
from typing import Optional

import cv2
import gdown
import numpy as np
import toolz as tz
import torch
from napari.utils import progress
from segment_anything import SamPredictor
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from skimage import color
from skimage.measure import regionprops

from napari_segment_everything.minimal_detection.detect_and_segment import (
    create_MS_model,
    segment_from_bbox,
)
from napari_segment_everything.minimal_detection.mobilesamv2 import (
    SamPredictor as SamPredictorV2,
)
from napari_segment_everything.minimal_detection.mobilesamv2 import (
    sam_model_registry,
)
from napari_segment_everything.minimal_detection.prompt_generator import (
    RcnnDetector,
    YoloDetector,
)

# Some code in this file copied from https://github.com/royerlab/napari-segment-anything/blob/main/src/napari_segment_anything/utils.py

SAM_WEIGHTS_URL = {
    "default": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "ObjectAwareModel": "https://drive.google.com/uc?id=1_vb_0SHBUnQhtg5SEE24kOog9_5Qpk5Z/ObjectAwareModel.pt",
    "ObjectAwareModel_Cell_FT": "https://drive.google.com/uc?id=1efZ40ti87O346dJW5lp7inCZ84N_nugS/ObjectAwareModel_Cell_FT.pt",
    "efficientvit_l2": "https://drive.google.com/uc?id=10Emd1k9obcXZZALiqlW8FLIYZTfLs-xu/l2.pt",
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
        if weight_url.startswith("https://drive.google.com/"):
            google_weight_url = "/".join(weight_url.split("/")[0:-1])

            gdown.download(google_weight_url, str(weight_path))
        else:
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


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_sam(model_type: str):
    sam = sam_model_registry[model_type](get_weights_path(model_type))
    # sam.to(self._device)
    return sam


def get_sam_predictor(model_type: str):
    sam = sam_model_registry[model_type](get_weights_path(model_type))
    # sam.to(self._device)
    return SamPredictor(sam)


def get_sam_automatic_mask_generator(
    model_type: str,
    points_per_side=32,
    pred_iou_thresh=0.1,
    stability_score_thresh=0.1,
    box_nms_thresh=0.5,
    crop_n_layers=1,
):

    device = get_device()
    if device == "mps":
        device = "cpu"
    sam = sam_model_registry[model_type](get_weights_path(model_type))
    sam.to()
    sam_anything_predictor = SamAutomaticMaskGenerator(
        sam,
        points_per_side=int(points_per_side),
        # points_per_batch=64,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        # stability_score_offset=1.0
        box_nms_thresh=box_nms_thresh,
        crop_n_layers=crop_n_layers,
        # crop_nms_thresh=0.7,
        # crop_overlap_ratio: float = 512 / 1500,
        # crop_n_points_downscale_factor=1,
        # in_mask_region_area=0,
        # point_grids: Optional[List[np.ndarray]] = None,
    )
    # sam.to(self._device)
    return sam_anything_predictor


def get_bounding_boxes(
    image,
    detector_model,
    device="cpu",
    conf=0.4,
    iou=0.5,
    imgsz=1024,
    max_det=10000,
):
    if detector_model == "YOLOv8":
        model = YoloDetector(
            str(get_weights_path("ObjectAwareModel")), device=device
        )
        bounding_boxes = model.get_bounding_boxes(
            image, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det
        )
    elif detector_model == "Finetuned":
        model = RcnnDetector(
            str(get_weights_path("ObjectAwareModel_Cell_FT")), device=device
        )
        bounding_boxes = model.get_bounding_boxes(image, conf=conf, iou=iou)
    #    print(bounding_boxes)
    return bounding_boxes


def get_mobileSAMv2(image=None, bounding_boxes=None, device=get_device()):
    """
    Uses a SAM model to make predictions from bounding boxes.

    Parameters
    ----------
    image : numpy.ndarray, optional
        A 2D-image in grayscale or RGB. The default is None.
    bounding_boxes : numpy.ndarray, optional
        An array of boxes in xyxy-coordinates. The default is None.

    Returns
    -------
    sam_masks : LIST
        A list of results dictionaries, one for each segmentation mask.
        Each sam_mask has keys for segmentation, area, predicted_iou, and stability_score.

    """
    if image is None:
        print("Upload an image first")
        return
    if image.ndim < 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # device = "cpu"
    weights_path_VIT = get_weights_path("efficientvit_l2")
    samV2 = create_MS_model()

    samV2.image_encoder = sam_model_registry["efficientvit_l2"](
        weights_path_VIT
    )
    if device == "mps":
        device="cpu"
    samV2.to(device=device)
    samV2.eval()
    predictor = SamPredictorV2(samV2)
    predictor.set_image(image)
    sam_masks = segment_from_bbox(
        bounding_boxes, predictor, samV2, device=device
    )
    del bounding_boxes

    gc.collect()
    torch.cuda.empty_cache()

    return sam_masks


def make_label_image_3d(masks):
    """
    Creates a label image by adding one mask at a time onto an empty image
    """
    num_masks = len(masks)
    label_image = np.zeros(
        (
            num_masks,
            masks[0]["segmentation"].shape[0],
            masks[0]["segmentation"].shape[1],
        ),
        dtype="uint16",
    )

    for enum, mask in enumerate(masks):
        mnarray = mask["segmentation"]
        label_image[enum, :, :] = mnarray.astype("uint16") * (enum + 1)

    return label_image


def filter_labels_3d_multi(
    label_image, sorted_results, stats, mins, maxes, napari_label=None
):
    for enum, result in enumerate(sorted_results):
        keep = all(
            min <= result[stat] <= max
            for stat, min, max in zip(stats, mins, maxes)
        )
        if keep:
            if result["keep"] == True:
                continue
            result["keep"] = True
            coords = np.where(result["segmentation"])
            if napari_label is None:
                temp = label_image[enum, :, :]
                temp[coords] = enum + 1
            else:
                z = np.full(coords[0].shape, enum)
                coords = (z, coords[0], coords[1])
                if enum < label_image.shape[0] - 1:
                    napari_label.data_setitem(coords, enum + 1, True)
                else:
                    napari_label.data_setitem(coords, enum + 1, True)
        else:
            if result["keep"] == False:
                continue
            result["keep"] = False
            coords = np.where(result["segmentation"])
            if napari_label is None:
                temp = label_image[enum, :, :]
                temp[coords] = 0
            else:
                z = np.full(coords[0].shape, enum)
                coords = (z, coords[0], coords[1])
                if enum < label_image.shape[0] - 1:
                    napari_label.data_setitem(coords, 0, True)
                else:
                    napari_label.data_setitem(coords, 0, True)


def add_properties_to_label_image(orig_image, sorted_results):

    hsv_image = color.rgb2hsv(orig_image)
    # switch to this? https://forum.image.sc/t/looking-for-a-faster-version-of-rgb2hsv/95214/12

    hue = 255 * hsv_image[:, :, 0]
    saturation = 255 * hsv_image[:, :, 1]
    intensity = 255 * hsv_image[:, :, 2]

    for enum, result in enumerate(sorted_results):
        segmentation = result["segmentation"]
        coords = np.where(segmentation)
        regions = regionprops(segmentation.astype("uint8"))

        # calculate circularity
        result["circularity"] = (
            4 * np.pi * regions[0].area / (regions[0].perimeter ** 2)
        )
        # for small pixelated objects, circularity can be > 1 so we cap it
        if result["circularity"] > 1:
            result["circularity"] = 1

        result["solidity"] = regions[0].solidity
        intensity_pixels = intensity[coords]
        result["mean_intensity"] = np.mean(intensity_pixels)
        result["10th_percentile_intensity"] = np.percentile(
            intensity_pixels, 10
        )
        hue_pixels = hue[coords]
        result["mean_hue"] = np.mean(hue_pixels)
        saturation_pixels = saturation[coords]
        result["mean_saturation"] = np.mean(saturation_pixels)
