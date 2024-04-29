#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .mobilesamv2 import sam_model_registry, SamPredictor
from typing import Any, Generator, List
import torch
import os
from segment_anything.utils.amg import calculate_stability_score
import gc

current_dir = os.path.dirname(__file__)


def create_MS_model():
    Prompt_guided_path = os.path.join(
        current_dir, "PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt"
    )
    PromptGuidedDecoder = sam_model_registry["PromptGuidedDecoder"](
        Prompt_guided_path
    )
    mobilesamv2 = sam_model_registry["vit_h"]()
    mobilesamv2.prompt_encoder = PromptGuidedDecoder["PromtEncoder"]
    mobilesamv2.mask_decoder = PromptGuidedDecoder["MaskDecoder"]
    return mobilesamv2


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(
        len(args[0]) % batch_size != 0
    )
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


def segment_from_bbox(bounding_boxes, predictor, mobilesamv2, device):
    """
    Segments everything given the bounding boxes of the objects and the mobileSAMv2 prediction model.
    Code from mobileSAMv2
    """
    input_boxes = predictor.transform.apply_boxes(
        bounding_boxes, predictor.original_size
    )  # Does this need to be transformed?
    if device == "cuda":
        input_boxes = torch.from_numpy(input_boxes).cuda()
    elif device == "cpu":
        input_boxes = torch.from_numpy(input_boxes)
    sam_mask = []

    predicted_ious = []
    stability_scores = []

    image_embedding = predictor.features
    image_embedding = torch.repeat_interleave(image_embedding, 400, dim=0)

    prompt_embedding = mobilesamv2.prompt_encoder.get_dense_pe()
    prompt_embedding = torch.repeat_interleave(prompt_embedding, 400, dim=0)

    for (boxes,) in batch_iterator(200, input_boxes):
        with torch.no_grad():
            image_embedding = image_embedding[0 : boxes.shape[0], :, :, :]
            prompt_embedding = prompt_embedding[0 : boxes.shape[0], :, :, :]
            sparse_embeddings, dense_embeddings = mobilesamv2.prompt_encoder(
                points=None,
                boxes=boxes,
                masks=None,
            )
            low_res_masks, pred_ious = mobilesamv2.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=prompt_embedding,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                simple_type=True,
            )
            low_res_masks = predictor.model.postprocess_masks(
                low_res_masks, predictor.input_size, predictor.original_size
            )
            mobilesamv2.threshold_offset = 1
            stability_score = (
                calculate_stability_score(
                    low_res_masks,
                    mobilesamv2.mask_threshold,
                    mobilesamv2.threshold_offset,
                )
                .cpu()
                .numpy()
            )
            sam_mask_pre = (low_res_masks > mobilesamv2.mask_threshold) * 1.0
            sam_mask.append(sam_mask_pre.squeeze(1))
            predicted_ious.extend(pred_ious.cpu().numpy().flatten().tolist())
            stability_scores.extend(stability_score.flatten().tolist())

    sam_mask = torch.cat(sam_mask)
    # predicted_ious = pred_ious.cpu().numpy()
    cpu_segmentations = sam_mask.cpu().numpy()
    del sam_mask

    gc.collect()
    torch.cuda.empty_cache()

    curr_anns = []
    for idx in range(len(cpu_segmentations)):
        ann = {
            "segmentation": cpu_segmentations[idx],
            "area": sum(sum(cpu_segmentations[idx])),
            "predicted_iou": predicted_ious[idx],
            "stability_score": stability_scores[idx],
        }
        if (
            cpu_segmentations[idx].max() < 1
        ):  # this means that bboxes won't always == segmentations
            continue
        curr_anns.append(ann)
    return curr_anns
