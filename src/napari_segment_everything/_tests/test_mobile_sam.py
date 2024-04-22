# import skimag example images
from skimage import data
from napari_segment_everything.sam_helper import (
    get_mobileSAMv2,
    get_bounding_boxes,
    get_bounding_boxes_trained,
)
import numpy as np
from skimage.measure import regionprops


def test_mobile_sam():
    # load a color examp
    image = data.coffee()

    bounding_boxes = get_bounding_boxes(image, imgsz=1024, device="cuda")
    segmentations = get_mobileSAMv2(image, bounding_boxes)
    for seg in segmentations:
        coords = np.where(seg)
        regions = regionprops(seg["segmentation"].astype("uint8"))

    assert len(segmentations) == 11


def test_bbox_trained():
    # load a color examp
    image = data.coffee()

    bounding_boxes = get_bounding_boxes_trained(image, device="cuda")
    segmentations = get_mobileSAMv2(image, bounding_boxes)
    return segmentations


seg = test_bbox_trained()
# seg, regions=test_mobile_sam()
