# import skimag example images
import os

import requests
from gdown.parse_url import parse_url
from skimage import data

from napari_segment_everything.minimal_detection.prompt_generator import (
    RcnnDetector,
    YoloDetector,
)
from napari_segment_everything.sam_helper import (
    SAM_WEIGHTS_URL,
    add_properties_to_label_image,
    get_bounding_boxes,
    get_device,
    get_mobileSAMv2,
    get_sam_automatic_mask_generator,
    get_weights_path,
)

device = get_device()
if device == "mps":
    device = "cpu"

#%%
def test_urls():
    """
    Tests whether all the urls for the model weights exist and are accessible.
    """
    TIMEOUT = 1
    for name, url in SAM_WEIGHTS_URL.items():
        try:
            if url.startswith("https://drive.google.com/"):
                _, path_exists = parse_url(url)
                assert (
                    path_exists
                ), f"Google Drive URL path wasn't parsed correctly: {url}"
            else:
                req = requests.head(url, timeout=TIMEOUT)
                assert (
                    req.status_code == 200
                ), f"Failed to access URL: {url}, Status code: {req.status_code}"
        except requests.exceptions.Timeout:
            print(f"Request timed out for URL: {url}")


def test_mobile_sam():
    """
    Tests the mobileSAMv2 process pipeline
    """
    # load a color examp
    image = data.coffee()

    bounding_boxes = get_bounding_boxes(
        image,
        detector_model="YOLOv8",
        imgsz=1024,
        device=device,
        conf=0.4,
        iou=0.9,
    )
    segmentations = get_mobileSAMv2(image, bounding_boxes)

    assert len(segmentations) == 11


def test_bbox():
    """
    Test whether bboxes can be generated
    """
    image = data.coffee()
    bounding_boxes = get_bounding_boxes(
        image, detector_model="YOLOv8", device=device, conf=0.9, iou=0.90
    )
    print(f"Length of bounding boxes: {len(bounding_boxes)}")
    assert len(bounding_boxes) > 0


def test_RCNN():
    """
    Test RCNN object detection on CPU and CUDA devices.
    """
    image = data.coffee()
    model_path = str(get_weights_path("ObjectAwareModel_Cell_FT"))
    assert os.path.exists(model_path)
    rcnn = RcnnDetector(model_path, device=device)
    bbox = rcnn.get_bounding_boxes(image, conf=0.5, iou=0.2)
    assert len(bbox) == 6


def test_YOLO():
    """
    Test YOLO object detection on CPU and CUDA devices.
    """
    image = data.coffee()
    model_path = str(get_weights_path("ObjectAwareModel"))
    assert os.path.exists(model_path)
    yolo = YoloDetector(model_path, device=device)
    bbox = yolo.get_bounding_boxes(
        image, conf=0.5, iou=0.2, max_det=400, imgsz=1024
    )
    assert len(bbox) == 8


def test_weights_path():
    """
    Tests whether the weights directory existing on the operating system
    """
    weights_path = get_weights_path("default")
    assert os.path.exists(os.path.dirname(weights_path))


def test_labels():
    """
    Tests whether region properties can be generated for segmentations for different models
    """
    image = data.coffee()

    bbox_yolo = get_bounding_boxes(
        image,
        detector_model="YOLOv8",
        imgsz=1024,
        device=device,
        conf=0.4,
        iou=0.9,
    )
    bbox_rcnn = get_bounding_boxes(
        image,
        detector_model="Finetuned",
        imgsz=1024,
        device=device,
        conf=0.4,
        iou=0.9,
    )

    segmentations_rcnn = get_mobileSAMv2(image, bbox_rcnn)
    segmentations_yolo = get_mobileSAMv2(image, bbox_yolo)
    segmentations_vit_b = get_sam_automatic_mask_generator(
        "vit_b",
        points_per_side=4,
        pred_iou_thresh=0.2,
        stability_score_thresh=0.5,
        box_nms_thresh=0.1,
        crop_n_layers=0,
    ).generate(image)

    add_properties_to_label_image(image, segmentations_rcnn)
    add_properties_to_label_image(image, segmentations_yolo)
    add_properties_to_label_image(image, segmentations_vit_b)

    props_rcnn = segmentations_rcnn[0].keys()
    assert len(props_rcnn) == 10
    props_yolo = segmentations_yolo[0].keys()
    assert len(props_yolo) == 10
    props_vit_b = segmentations_vit_b[0].keys()
    assert len(props_vit_b) == 13

test_urls()
test_bbox()
test_mobile_sam()
test_RCNN()
test_YOLO()
test_weights_path()
test_labels()
