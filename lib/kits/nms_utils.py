import numpy as np
import pdb

def set_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    return emd_cpu_nms(dets, thresh)

def emd_cpu_nms(dets, base_thr, upp_thr = 1.):
    """Pure Python NMS baseline."""

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    numbers = dets[:, 5]

    sup = np.zeros(dets.shape[0])

    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)
    eps = 1e-8

    while order.size > 0:

        i = order[0]
        num = numbers[i]
        sup[i] = 0
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + eps)
        indices = np.where(np.logical_and(ovr > base_thr, sup[order[1:]] < 1))[0]
        loc = np.where(numbers[order[indices + 1]] == num)[0]
        sup[order[indices + 1]] = 1
        if loc.size:
            sup[order[indices[loc[0]] + 1]] = 0
        inds = np.where(sup[order[1:]] < 1)[0]
        order = order[inds + 1]

    keep = np.where(sup < 1)[0]
    return keep
    
def cpu_nms(dets, base_thr):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)

    keep = []
    eps = 1e-8
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + eps)

        inds = np.where(ovr <= base_thr)[0]
        indices = np.where(ovr > base_thr)[0]
        order = order[inds + 1]
    return np.array(keep)

