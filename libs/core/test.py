# Written by Roy Tseng
#
# Based on:
# --------------------------------------------------------
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import copy
import cv2
import numpy as np
import pycocotools.mask as mask_util

import torch
from torch.autograd import Variable

from core.config import cfg
from utils.timer import Timer
import utils.boxes as box_utils
import utils.blob as blob_utils
import utils.image as image_utils
from modeling.hook import seg_im_and_rois


def im_detect_all_realN_subnms(model, im, box_proposals=None, timers=None):
    if timers is None:
        timers = defaultdict(Timer)
        
    if type(box_proposals) != list:
        box_proposals = [box_proposals]
    
    pred_boxes = np.vstack(box_proposals)
    
    # Convert im list and rois list
    if len(box_proposals) > 1:
        sub_im, sub_box = seg_im_and_rois(im, box_proposals)
    else:
        sub_im = [im]
        sub_box = box_proposals
    
    timers['im_detect_bbox'].tic()
    if cfg.TEST.BBOX_AUG.ENABLED:
        scores = im_detect_bbox_aug(
            model, sub_im, sub_box)
    else:
        scores = im_detect_bbox(
            model, sub_im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, sub_box)
    timers['im_detect_bbox'].toc()
    
    # for mAP only
    if 'test' in cfg.TEST.DATASETS[0]:
        Score_nms = []
        Box_nms = []
        st = 0
        for B in box_proposals:
            nB = len(B)
            s_i, b_i = sub_box_nms(scores[st:st+nB, :], B)
            Score_nms.append(s_i)
            Box_nms.append(b_i)
            st = st + nB
        scores = np.vstack(Score_nms)
        pred_boxes = np.vstack(Box_nms)

    return {'scores': scores, 'boxes' : pred_boxes}

def im_detect_all_realN(model, im, box_proposals=None, timers=None):
    if timers is None:
        timers = defaultdict(Timer)
    
    if type(box_proposals) != list:
        box_proposals = [box_proposals]
        
    pred_boxes = np.vstack(box_proposals)
    
    # to im list and rois list
    if len(box_proposals) > 1:
        sub_im, sub_box = seg_im_and_rois(im, box_proposals)
    else:
        sub_im = [im]
        sub_box = box_proposals
    
    timers['im_detect_bbox'].tic()
    if cfg.TEST.BBOX_AUG.ENABLED:
        scores = im_detect_bbox_aug(
            model, sub_im, sub_box)
    else:
        scores = im_detect_bbox(
            model, sub_im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, sub_box)
    timers['im_detect_bbox'].toc()
    
    return {'scores': scores, 'boxes' : pred_boxes}

def im_detect_all_N1(model, im, box_proposals=None, timers=None):
    if timers is None:
        timers = defaultdict(Timer)

    if type(box_proposals) != list:
        box_proposals = [box_proposals]
        
    pred_boxes = np.vstack(box_proposals)
    
    # to im list and rois list
    sub_im = [im]
    sub_box = [np.vstack(box_proposals)]
    
    timers['im_detect_bbox'].tic()
    if cfg.TEST.BBOX_AUG.ENABLED:
        scores = im_detect_bbox_aug(
            model, sub_im, sub_box)
    else:
        scores = im_detect_bbox(
            model, sub_im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, sub_box)
    timers['im_detect_bbox'].toc()

    return {'scores': scores, 'boxes' : pred_boxes}


def im_detect_bbox(model, im, target_scale, target_max_size, boxes=None):
    """Prepare the bbox for testing"""

    inputs, im_scale = _get_blobs(im, boxes, target_scale, target_max_size)

    if cfg.PYTORCH_VERSION_LESS_THAN_040:
        inputs['data'] = Variable(torch.from_numpy(inputs['data']), volatile=True).cuda()
        inputs['rois'] = Variable(torch.from_numpy(inputs['rois']), volatile=True).cuda()
        inputs['labels'] = None
    else:
        inputs['data'] = torch.from_numpy(inputs['data']).cuda()
        inputs['rois'] = torch.from_numpy(inputs['rois']).cuda()
        inputs['labels'] = None
    
    return_dict = model(**inputs)

    # cls prob (activations after softmax)
    #num_ref = len(return_dict['refine_score'])
    #scores = return_dict['refine_score'][0].data.cpu().numpy().squeeze()
    #for i in range(1, num_ref):
    #    scores += return_dict['refine_score'][i].data.cpu().numpy().squeeze()
    #scores /= num_ref
    scores = return_dict['mil_score'].data.cpu().numpy().squeeze()
    
    # In case there is 1 proposal
    scores = scores.reshape([-1, scores.shape[-1]])
    #bg_scores = np.zeros_like(scores, shape=(scores.shape[0], 1))
    #scores = np.hstack((bg_scores, scores))
    
    return scores


def im_detect_bbox_aug(model, im, box_proposals=None):
    """Performs bbox detection with test-time augmentations.
    Function signature is the same as for im_detect_bbox.
    """
    assert not cfg.TEST.BBOX_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'
    assert not cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION', \
        'Coord heuristic must be union whenever score heuristic is union'
    assert not cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Score heuristic must be union whenever coord heuristic is union'

    # Collect detections computed under different transformations
    scores_ts = []
    #boxes_ts = []

    def add_preds_t(scores_t):
        #concat rois_list to a ndarray
        #new_boxes = np.empty((0, 4), dtype=np.float32)
        #for box_i in boxes_t:
        #    new_boxes = np.vstack((new_boxes, box_i))
        #boxes_ts.append(new_boxes)
        scores_ts.append(scores_t)

    # Perform detection on the horizontally flipped image
    if cfg.TEST.BBOX_AUG.H_FLIP:
        scores_hf = im_detect_bbox_hflip(
            model,
            im,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            box_proposals=box_proposals
        )
        add_preds_t(scores_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.BBOX_AUG.SCALES:
        max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
        scores_scl = im_detect_bbox_scale(
            model, im, scale, max_size, box_proposals
        )
        add_preds_t(scores_scl)

        if cfg.TEST.BBOX_AUG.SCALE_H_FLIP:
            scores_scl_hf = im_detect_bbox_scale(
                model, im, scale, max_size, box_proposals, hflip=True
            )
            add_preds_t(scores_scl_hf)

    # Perform detection at different aspect ratios
#     for aspect_ratio in cfg.TEST.BBOX_AUG.ASPECT_RATIOS:
#         scores_ar, boxes_ar = im_detect_bbox_aspect_ratio(
#             model, im, aspect_ratio, box_proposals
#         )
#         add_preds_t(scores_ar, boxes_ar)

#         if cfg.TEST.BBOX_AUG.ASPECT_RATIO_H_FLIP:
#             scores_ar_hf, boxes_ar_hf = im_detect_bbox_aspect_ratio(
#                 model, im, aspect_ratio, box_proposals, hflip=True
#             )
#             add_preds_t(scores_ar_hf, boxes_ar_hf)

    # Compute detections for the original image (identity transform) last to
    # ensure that the Caffe2 workspace is populated with blobs corresponding
    # to the original image on return (postcondition of im_detect_bbox)
    scores_i = im_detect_bbox(
        model, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, boxes=box_proposals
    )
    add_preds_t(scores_i)

    # Combine the predicted scores
    if cfg.TEST.BBOX_AUG.SCORE_HEUR == 'ID':
        scores_c = scores_i
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'AVG':
        scores_c = np.mean(scores_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION':
        scores_c = np.vstack(scores_ts)
    else:
        raise NotImplementedError(
            'Score heur {} not supported'.format(cfg.TEST.BBOX_AUG.SCORE_HEUR)
        )

#     # Combine the predicted boxes
#     if cfg.TEST.BBOX_AUG.COORD_HEUR == 'ID':
#         boxes_c = boxes_i
#     elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'AVG':
#         boxes_c = np.mean(boxes_ts, axis=0)
#     elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION':
#         boxes_c = np.vstack(boxes_ts)
#     else:
#         raise NotImplementedError(
#             'Coord heur {} not supported'.format(cfg.TEST.BBOX_AUG.COORD_HEUR)
#         )

    return scores_c


def im_detect_bbox_hflip(
        model, im_list, target_scale, target_max_size, box_proposals=None):
    """Performs bbox detection on the horizontally flipped image.
    Function signature is the same as for im_detect_bbox.
    """
    # Compute predictions on the flipped image
    im_list_hf = []
    for im in (im_list):
        im_list_hf.append(im[:, ::-1, :])
    im_width_list = [im_i.shape[1] for im_i in im_list]
    
    box_proposals_hf = box_utils.flip_boxes_list(box_proposals, im_width_list)
    
    scores_hf = im_detect_bbox(
        model, im_list_hf, target_scale, target_max_size, boxes=box_proposals_hf
    )

    # Invert the detections computed on the flipped image
    #boxes_inv = box_utils.flip_boxes_list(boxes_hf, im_width_list)

    return scores_hf


def im_detect_bbox_scale(
        model, im, target_scale, target_max_size, box_proposals=None, hflip=False):
    """Computes bbox detections at the given scale.
    Returns predictions in the original image space.
    """
    if hflip:
        scores_scl = im_detect_bbox_hflip(
            model, im, target_scale, target_max_size, box_proposals=box_proposals
        )
    else:
        scores_scl = im_detect_bbox(
            model, im, target_scale, target_max_size, boxes=box_proposals
        )
    return scores_scl


def im_detect_bbox_aspect_ratio(
        model, im, aspect_ratio, box_proposals=None, hflip=False):
    """Computes bbox detections at the given width-relative aspect ratio.
    Returns predictions in the original image space.
    """
    # Compute predictions on the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)

    box_proposals_ar = box_utils.aspect_ratio(box_proposals, aspect_ratio)

    if hflip:
        scores_ar, boxes_ar, _ = im_detect_bbox_hflip(
            model,
            im_ar,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            box_proposals=box_proposals_ar
        )
    else:
        scores_ar, boxes_ar, _, _ = im_detect_bbox(
            model,
            im_ar,
            cfg.TEST.SCALE,
            cfg.TEST.MAX_SIZE,
            boxes=box_proposals_ar
        )

    # Invert the detected boxes
    boxes_inv = box_utils.aspect_ratio(boxes_ar, 1.0 / aspect_ratio)

    return scores_ar, boxes_inv


def box_results_for_corloc(scores, boxes):  # NOTE: support single-batch
    """Returns bounding-box detection results for CorLoc evaluation.

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = cfg.MODEL.NUM_CLASSES + 1
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        max_ind = np.argmax(scores[:, j])
        cls_boxes[j] = np.hstack((boxes[max_ind, :].reshape(1, -1),
                               np.array([[scores[max_ind, j]]])))

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes


def box_results_with_nms_and_limit(scores, boxes):  # NOTE: support single-batch
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = cfg.MODEL.NUM_CLASSES + 1
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, :]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)
        if cfg.TEST.SOFT_NMS.ENABLED:
            nms_dets, _ = box_utils.soft_nms(
                dets_j,
                sigma=cfg.TEST.SOFT_NMS.SIGMA,
                overlap_thresh=cfg.TEST.NMS,
                score_thresh=0.0001,
                method=cfg.TEST.SOFT_NMS.METHOD
            )
        else:
            keep = box_utils.nms(dets_j, cfg.TEST.NMS)
            nms_dets = dets_j[keep, :]
        # Refine the post-NMS boxes using bounding-box voting
        if cfg.TEST.BBOX_VOTE.ENABLED:
            nms_dets = box_utils.box_voting(
                nms_dets,
                dets_j,
                cfg.TEST.BBOX_VOTE.VOTE_TH,
                scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
            )
        cls_boxes[j] = nms_dets

    # Limit to max_per_image detections **over all classes**
    if cfg.TEST.DETECTIONS_PER_IM > 0:
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(1, num_classes)]
        )
        if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
            image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
            for j in range(1, num_classes):
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes


def sub_box_nms(scores, boxes):  # NOTE: support single-batch
    num_classes = cfg.MODEL.NUM_CLASSES + 1
    cls_boxes = [[] for _ in range(num_classes)]
    keep_list = []
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, :]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)
        if cfg.TEST.SOFT_NMS.ENABLED:
            _, keep = box_utils.soft_nms(
                dets_j,
                sigma=cfg.TEST.SOFT_NMS.SIGMA,
                overlap_thresh=cfg.TEST.NMS,
                score_thresh=0.0001,
                method=cfg.TEST.SOFT_NMS.METHOD
            )
        else:
            keep = box_utils.nms(dets_j, cfg.TEST.NMS)
            #nms_dets = dets_j[keep, :]
        keep_list.append(inds[keep])
    #print(keep_list)
    keep_inds = np.unique(np.concatenate(keep_list)).astype(np.int)
    return scores[keep_inds, :], boxes[keep_inds, :]

def _get_rois_blob(im_rois, im_scales):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid with columns
            [level, x1, y1, x2, y2]
    """
    rois_blob = np.empty((0, 5), dtype=np.float32)
    for i, (R, im_scale) in enumerate(zip(im_rois, im_scales)):
        sub_rois, sub_levels = _project_im_rois(R, im_scale, i)
        sub_rois_blob = np.hstack((sub_levels, sub_rois))
        rois_blob = np.vstack((rois_blob, sub_rois_blob))
        
    return rois_blob


def _project_im_rois(im_rois, scales, idx):
    rois = im_rois.astype(np.float32, copy=False) * scales
    levels = np.ones((im_rois.shape[0], 1), dtype=rois.dtype) * idx
    return rois, levels


def _get_blobs(im_list, rois_list, target_scale, target_max_size):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scales = \
            blob_utils.get_image_blob(im_list, target_scale, target_max_size)
    if rois_list is not None:
        blobs['rois'] = _get_rois_blob(rois_list, im_scales)
    blobs['labels'] = np.zeros((1, cfg.MODEL.NUM_CLASSES), dtype=np.int32)
    return blobs, im_scales
