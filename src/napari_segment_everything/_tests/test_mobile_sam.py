# import skimag example images
from skimage import data
from napari_segment_everything.sam_helper import (
    get_mobileSAMv2,
    get_bounding_boxes,
)


def test_mobile_sam():
    # load a color examp
    image = data.coffee()

    bounding_boxes = get_bounding_boxes(image, imgsz=1024, device="cuda")
    segmentations = get_mobileSAMv2(image, bounding_boxes)

    assert len(segmentations) == 11


def test_bbox():
    # load a color examp
    image = data.coffee()
    bounding_boxes = get_bounding_boxes(
        image, detector_model="Finetuned", device="cuda", conf=0.01, iou=0.99
    )
    print(f"Length of bounding boxes: {len(bounding_boxes)}")
    segmentations = get_mobileSAMv2(image, bounding_boxes)
    return segmentations


seg = test_bbox()
# seg, regions=test_mobile_sam()
