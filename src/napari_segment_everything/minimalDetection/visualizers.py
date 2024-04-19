#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 01:24:29 2024

@author: ian
"""

import numpy as np
import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch


def xyxy_boxes_to_mask(label_bbox, bounding_box, col):
    # converts a box from xyxy-coordinates to a filled polygon
    minx, miny, maxx, maxy = bounding_box
    box = np.array(
        [[[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny]]],
        dtype=np.int32,
    )
    poly = cv2.fillPoly(label_bbox, box, col)
    return poly


def xyxy_boxes_to_shape(bounding_box):
    minx, miny, maxx, maxy = bounding_box
    rectangle = np.array(
        [[miny, minx], [maxy, minx], [maxy, maxx], [miny, maxx]]
    )
    return rectangle


def add_spaces_between_layers(num_space, label_image):
    spacer = np.zeros(
        (label_image.shape[1], label_image.shape[2]), dtype="uint16"
    )
    spaces = (np.arange(1, label_image.shape[0])).astype("int")
    spaces = np.repeat(spaces, num_space)
    spaced_array = np.insert(label_image, spaces, spacer, axis=0)
    return spaced_array


def show_anns(annotation, image):
    areas = torch.sum(annotation, dim=(1, 2))
    sorted_indices = torch.argsort(areas, descending=True)
    anns = annotation[sorted_indices]
    plt.figure(figsize=(10, 10))
    background = np.ones_like(image) * 255
    plt.imshow(background)

    if len(anns) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((anns.shape[1], anns.shape[2], 4))
    img[:, :, 3] = 0
    for ann in range(anns.shape[0]):
        m = anns[ann].bool()
        m = m.cpu().numpy()
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask
    ax.imshow(img)


def make_box_patches(ax, bounding_boxes):
    for box in bounding_boxes:
        minc, minr, maxc, maxr = box
        rect = mpatches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        ax.add_patch(rect)
