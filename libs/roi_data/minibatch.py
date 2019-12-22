import numpy as np
import numpy.random as npr
import cv2

from core.config import cfg
import utils.blob as blob_utils
from core.test import _get_blobs
from modeling.hook import seg_im_and_rois

def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data', 'rois', 'labels']
    return blob_names


def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    #blobs = {k: [] for k in get_minibatch_blob_names()}

    # now multiple sub-images
    #assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"
    
    # get raw input
    im = cv2.imread(roidb[0]['image'])
    assert im is not None, \
        'Failed to read image \'{}\''.format(roidb[i]['image'])
    if roidb[i]['flipped']:
        im = im[:, ::-1, :]
    labels, im_rois = _sample_rois(roidb[0], num_classes)
    break_points = [0] + list(accumulate([len(r_i) for r_i in im_rois]))
    
    # Convert to im list and rois list
    sub_im, sub_box = seg_im_and_rois(im, im_rois)
    
    scale_ind = np.random.randint(0, high=len(cfg.TRAIN.SCALES))
    target_size = cfg.TRAIN.SCALES[scale_ind]    
    inputs, im_scales = _get_blobs(sub_im, sub_box, target_scale, cfg.TRAIN.MAX_SIZE)
    inputs['labels'] = labels
    
    return inputs, break_points


def _sample_rois(roidb, num_classes):
    """Generate a random sample of RoIs"""
    labels = roidb['gt_classes']
    rois = roidb['boxes']

    if cfg.TRAIN.BATCH_SIZE_PER_IM > 0:
        batch_size = cfg.TRAIN.BATCH_SIZE_PER_IM
    else:
        batch_size = np.inf
    if type(rois) is list:
        rois_num = [len(i) for i in rois]
        total_num = float(sum(rois_num))
        if batch_size < total_num:
            sample_ratio = batch_size / total_num
            sample_num = [ int(j * sample_ratio) for j in rois_num]
            sampled_rois = []
            # sampling proportionally from sub-rois
            for sub_i, sub_R in enumerate(rois):
                sub_inds = npr.permutation(sub_R.shape[0])[:sample_num[sub_i]]
                sampled_rois.append(sub_R[sub_inds, :]
            rois = sampled_rois
    else:
        if batch_size < rois.shape[0]:
            rois_inds = npr.permutation(rois.shape[0])[:batch_size]
            rois = rois[rois_inds, :]

    return labels.reshape(1, -1), rois


# def _get_image_blob(roidb):
#     """Builds an input blob from the images in the roidb at the specified
#     scales.
#     """
#     num_images = len(roidb)
#     assert num_images == 1
#     # Sample random scales to use for each image in this batch
#     scale_inds = np.random.randint(
#         0, high=len(cfg.TRAIN.SCALES), size=num_images)
#     processed_ims = []
#     im_scales = []
#     for i in range(num_images):
#         im = cv2.imread(roidb[i]['image'])
#         assert im is not None, \
#             'Failed to read image \'{}\''.format(roidb[i]['image'])
#         # If NOT using opencv to read in images, uncomment following lines
#         # if len(im.shape) == 2:
#         #     im = im[:, :, np.newaxis]
#         #     im = np.concatenate((im, im, im), axis=2)
#         # # flip the channel, since the original one using cv2
#         # # rgb -> bgr
#         # im = im[:, :, ::-1]
#         if roidb[i]['flipped']:
#             im = im[:, ::-1, :]
#         target_size = cfg.TRAIN.SCALES[scale_inds[i]]
#         im, im_scale = blob_utils.prep_im_for_blob(
#             im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
#         im_scales.append(im_scale[0])
#         processed_ims.append(im[0])

#     # Create a blob to hold the input images [n, c, h, w]
#     blob = blob_utils.im_list_to_blob(processed_ims)

#     return blob, im_scales

# def _project_im_rois(im_rois, im_scale_factor):
#     """Project image RoIs into the rescaled training image."""
#     #rois = im_rois * im_scale_factor
#     rois = im_rois.astype(np.float32, copy=False) * im_scale_factor
#     return rois
