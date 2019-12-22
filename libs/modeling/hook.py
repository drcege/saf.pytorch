import math
import numpy as np

def seg_im_and_rois(im, rois):
    sub_ims = []
    sub_rois = []
    for R in rois:
        x1 = math.ceil(np.min(R[:, 0]))
        y1 = math.ceil(np.min(R[:, 1]))
        x2 = math.ceil(np.max(R[:, 2]))
        y2 = math.ceil(np.max(R[:, 3]))
        sub_ims.append(im[y1:y2+1, x1:x2+1, :])
        
        sub_R = R.copy()
        sub_R[:, [0,2]] -= x1
        sub_R[:, [1,3]] -= y1
        sub_rois.append(sub_R)

    return sub_ims, sub_rois

def agg_rois():
    pass