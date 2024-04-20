#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 23:58:30 2024

@author: ian
"""

import cv2
import torch
import numpy as np
from detect_and_segment import (
    create_OA_model,
    create_MS_model,
    detect_bbox,
    segment_everything,
)
from visualizers import (
    xyxy_boxes_to_shape,
    add_spaces_between_layers,
    show_anns,
    make_box_patches,
)
from .mobilesamv2 import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt

encoder_path = {
    "efficientvit_l2": "./weight/l2.pt",
    "tiny_vit": "./weight/mobile_sam.pt",
    "sam_vit_h": "./weight/sam_vit_h.pt",
}

objDetect = create_OA_model()
image_path = "cell_00227.png"
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
device = "cuda" if torch.cuda.is_available() else "cpu"
confidence = 0.2  # higher means objects need to have high confidence before being labeled
iou_threshold = 0.9  # higher means more overlapping objects gets detected

obj_results = detect_bbox(
    objDetect,
    filepath=image_path,
    device=device,
    imgsz=2048,
    conf=confidence,
    iou=iou_threshold,
    max_det=400,
)

print(f"Discovered {len(obj_results[0])} objects")
samV2 = create_MS_model()
image_encoder = sam_model_registry["efficientvit_l2"](
    encoder_path["efficientvit_l2"]
)
samV2.image_encoder = image_encoder
samV2.to(device=device)
samV2.eval()
predictor = SamPredictor(samV2)
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image)
bounding_boxes = obj_results[0].boxes.xyxy.cpu().numpy()
make_box_patches(ax, bounding_boxes)
predictor.set_image(image)

sam_mask = segment_everything(bounding_boxes, predictor, samV2)
show_anns(sam_mask, image)

cpu_annotations = sam_mask.cpu().numpy()
del (sam_mask, obj_results)
import gc

gc.collect()
torch.cuda.empty_cache()

# %%
num_masks = len(cpu_annotations)
label_image = np.zeros(
    (num_masks, cpu_annotations.shape[1], cpu_annotations.shape[2]),
    dtype="uint16",
)
label_bbox = label_image.copy()
bounding_boxes = bounding_boxes.astype("int")

import napari

for enum, ann in enumerate(cpu_annotations):
    label_image[enum, :, :] = ann.astype("uint16") * (enum + 1)
    # poly = xyxy_boxes_to_mask(label_bbox[enum], bounding_boxes[enum], enum + 1)
    # label_bbox[enum, :, :] = poly


viewer = napari.Viewer()
viewer.add_image(image)
# label_layer = viewer.add_labels(label_image, name="SAM 3D labels")
# label_layer.translate = (-len(label_image), 0, 0)
num_space = 1

# minx, miny, maxx, maxy = bounding_boxes[0]
# modeled after: https://github.com/napari/napari/blob/main/examples/nD_shapes.py
plane_slices = (np.arange(len(bounding_boxes))) * (num_space + 1)
planes = np.tile(
    plane_slices.reshape((len(bounding_boxes), 1, 1)),
    (1, 4, 1),
)

rectangles = []
for enum, bounding_box in enumerate(bounding_boxes):
    rectangle = xyxy_boxes_to_shape(bounding_box)
    rectangles.append(np.concatenate((planes[enum], rectangle), axis=1))

base_cols = ["red", "green", "blue", "yellow", "magenta", "cyan"]
colors = np.random.choice(base_cols, size=len(rectangles))

rect_layer = viewer.add_shapes(
    rectangles,
    shape_type="rectangle",
    edge_color=colors,
    name="Original_Box_Prompts",
)
rect_layer.translate = (-planes.max(), 0, 0)
rect_layer.blending = "minimum"

spaced_array = add_spaces_between_layers(num_space, label_image)
label_layer_spaced = viewer.add_labels(spaced_array, name="Segmentation_3D")

viewer.dims.ndisplay = 3
label_layer_spaced.translate = (-len(spaced_array), 0, 0)
