# encoding: utf-8
import os,sys
import torch
import pdb
def iou_loss(pred, gt, cls_gt, background=0):
    mask = 1 - cls_gt.eq(background).float()

    aog = torch.abs(gt[:, :, 2] - gt[:, :, 0] + 1) * torch.abs(
        gt[:, :, 3] - gt[:, :, 1] + 1)
    aop = Abs(pred[:, :, 2] - pred[:, :, 0] + 1) * Abs(
        pred[:, :, 3] - pred[:, :, 1] + 1)

    iw = Min(pred[:, :, 2], gt[:, :, 2]) - \
         Max(pred[:, :, 0], gt[:, :, 0]) + 1
    ih = Min(pred[:, :, 3], gt[:, :, 3]) - \
         Max(pred[:, :, 1], gt[:, :, 1]) + 1
    inter = Max(iw, 0) * Max(ih, 0)

    union = aog + aop - inter
    iou = Max(inter / union, 0)
    loss = - safelog(iou) * mask

    return loss.sum() / Max(mask.sum(), 1)

def sigmoid_cross_entropy_centerness(
        pred, label, cls_gt, background=0):
    """
        pred    : (minibatch, bboxcnt, class)
        gt      : (minibatch, bboxcnt)
    """
    mask = 1 - cls_gt.eq(background)
    not_neg_mask = (pred >= 0)
    loss = (pred * not_neg_mask - pred * label + \
            safelog(1 + Exp(-Abs(pred)))) * mask
    return loss.sum() / Max(mask.sum(), 1)

def _smooth_l1_base(pred, gt, sigma):
    sigma2 = sigma ** 2
    cond_point = 1 / sigma2
    x = pred - gt
    abs_x = torch.abs(x)
    in_mask = abs_x < cond_point
    out_mask = 1 - in_mask.float()
    in_value = 0.5 * (sigma * x) ** 2
    out_value = abs_x - 0.5 / sigma2
    value = in_value * in_mask.float() + out_value * out_mask
    return value


def _get_mask_of_label(label, background, ignore_label):
    
    mask_fg = 1 - label.eq(background).float()
    mask_ig = 1 - label.eq(ignore_label).float()
    mask = mask_fg * mask_ig
    return mask, mask_ig


def smooth_l1_loss_rpn(
        pred, gt, label, sigma=1, background=0, ignore_label=-1, axis=1):
    
    value = _smooth_l1_base(pred, gt, sigma)
    mask, mask_ig = _get_mask_of_label(label, background, ignore_label)
    loss = (value.sum(axis = axis) * mask).sum() / torch.clamp(mask_ig.sum(), 1)
    return loss


def smooth_l1_loss_retina(
        pred, gt, label, sigma=3, background=0, ignore_label=-1, axis=2):
    value = _smooth_l1_base(pred, gt, sigma)
    mask, mask_ig = _get_mask_of_label(label, background, ignore_label)
    loss = (value.sum(axis=axis) * mask).sum() / torch.clamp(mask.sum(), 1)
    return loss

def smooth_l1_loss_retina_retain_axis(pred, gt, label,
    sigma=3, background=0, ignore_label=-1, axis=2):
    value = _smooth_l1_base(pred, gt, sigma)
    mask, mask_ig = _get_mask_of_label(label, background, ignore_label)
    loss = (value.sum(axis=axis) * mask)
    return loss

def smooth_l1_loss(pred, target, beta: float):
    
    if beta < 1e-5:
        loss = torch.abs(input - target)
    else:
        abs_x = torch.abs(pred- target)
        in_mask = abs_x < beta
        loss = torch.where(in_mask, 0.5 * abs_x ** 2 / beta, abs_x - 0.5 * beta)
    return loss.sum(axis=1)

def iou_l1_loss(pred, max_overlaps, gt, ignore_label=-1, background=0):

    pred = pred.reshape(pred.shape[0], -1, max_overlaps.shape[2])
    abs_x = torch.abs(pred - max_overlaps)
    mask_bg = 1 - gt.eq(background).float()
    mask_ig = 1 - gt.eq(ignore_label).float()
    mask = mask_bg * mask_ig

    mask = mask.reshape(mask.shape[0], -1, pred.shape[2])
    loss = (abs_x * mask).sum() / torch.clamp(mask.sum(), 1)
    return loss

def softmax_loss(score, label, ignore_label = -1):

    max_score = score.max(dim=1, keepdim=True)[0].detach()
    score -= max_score
    log_prob = score - torch.log(torch.exp(score).sum(dim=1, keepdim=True))
    ignore_mask = 1 - label.eq(ignore_label).float()
    vlabel = (label * ignore_mask).unsqueeze(1)
    loss = -(torch.gather(log_prob, 1, vlabel.long()).view(-1) * ignore_mask).sum()
    loss /= torch.clamp(ignore_mask.sum(), 1)
    return loss

def softmax_loss_opr(score, label, ignore_label=-1):
    
    #  Softmax Loss for backend cls loss
    # NOTE: this style of code would introduce different gradients
    max_score = score.max(dim=1, keepdim=True)[0].detach()
    score -= max_score
    log_prob = score - torch.log(torch.exp(score).sum(dim=1,keepdim=True))
    mask = 1 - label.eq(ignore_label).float().unsqueeze(1)
    vlabel = label.unsqueeze(1) * mask
    loss = -(torch.gather(log_prob, 1, vlabel.long()) * mask).view(-1)
    return loss

def sigmoid_cross_entropy_retina_retain_axis(pred, label, ignore_label=-1,
    background=0, alpha=0.5, gamma=0):
    
    mask = 1 - label.eq(ignore_label).float()
    vlabel = label * mask

    device = pred.device
    n, c, m = pred.shape
    zero_mat = torch.zeros([n, c, m + 1]).to(device)
    one_mat = torch.ones([n, c]).to(device)

    zero_mat[:, :, vlabel.long()] = one_mat
    onehot = one_hot[:, :, 1:]
    pos_part = torch.pow(1 - pred, gamma) * onehot * torch.log(pred)
    neg_part = torch.pow(pred, gamma) * (1 - onehot) * torch.log(1 - pred)
    loss = -(alpha * pos_part + (1 - alpha) * neg_part).sum(axis=2) * mask
    
    return loss

def sigmoid_cross_entropy_retina(
        pred, label, ignore_label=-1, background=0, alpha=0.5, gamma=0):
    
    device = pred.device
    mask = 1 - label.eq(ignore_label).float()
    vlabel = label * mask

    n, m, c = pred.shape
    zero_mat = torch.zeros(n, m, c + 1).to(device)
    one_hot = torch.scatter(zero_mat, 2, vlabel.unsqueeze(2).long(), 1)
    onehot = one_hot[:, :, 1:]

    pos_part = torch.pow(1 - pred, gamma) * onehot * torch.log(pred)
    neg_part = torch.pow(pred, gamma) * (1 - onehot) * torch.log(1 - pred)
    loss = -(alpha * pos_part + (1 - alpha) * neg_part).sum(axis=2) * mask

    positive_mask = (label > 0)
    return loss.sum() / torch.clamp(positive_mask.sum(), 1)

def smooth_l1_loss_rcnn_opr(
        pred, gt, label, sigma = 1, background=0, ignore_label=-1):
    """
        pred    : (minibatch, class_num, 4)
        gt      : (minibatch, 4)
        label   : (minibatch,  )
    """
    
    broadcast_label = label.reshape(-1, 1).repeat(1, pred.shape[-1])
    broadcast_mask, broadcast_mask_ig = _get_mask_of_label(
        broadcast_label, background, ignore_label)
    vlabel = broadcast_label * broadcast_mask
   
    pred_corr = torch.gather(pred,1, vlabel.unsqueeze(1).long()).squeeze(1)
    value = _smooth_l1_base(pred_corr, gt, sigma)
    loss = (value * broadcast_mask).sum(dim=1)
    return loss


def smooth_l1_loss_rcnn_single(
        pred, gt, label, sigma, background=0, ignore_label=-1):
    """
        author: lrx & jbr
        pred    : (minibatch, 4)
        gt      : (minibatch, 4)
        label   : (minibatch, )
    """
    label_mask, label_mask_ig = _get_mask_of_label(
        label, background, ignore_label)
    label_mask = label_mask.reshape(label_mask.shape[0], 1)
    value = _smooth_l1_base(pred, gt, sigma)
    loss = (value * label_mask).sum() / pred.shape[0]
    return loss


def sum_ohem_loss(
        score, label, pred, gt, sigma, nr_ohem_sampling, background=0,
        ignore_label=-1):
    # NOTE: This is implementation of R-FCN, which is effective on COCO dataset
    # NOTE: this style of code would introduce different gradients
    # score -= score.max(axis=1, keepdims=True)
    max_score = ZeroGrad(score.max(axis=1, keepdims=True))
    score -= max_score
    log_prob = score - safelog(Exp(score).sum(axis=1, keepdims=True))
    mask = 1 - label.eq(ignore_label)
    vlabel = label * mask
    cls_loss = -(IndexingOneHot(log_prob, 1, vlabel) * mask)

    broadcast_label = label.reshape((label.shape[0], 1)) \
        .broadcast(Concat([label.shape[0], pred.shape[-1]], axis=0))
    broadcast_mask, broadcast_mask_ig = _get_mask_of_label(
        broadcast_label, background, ignore_label)
    vlabel = broadcast_label * broadcast_mask
    pred_corr = IndexingOneHot(pred, 1, vlabel)
    bbox_loss = (_smooth_l1_base(pred_corr, gt, sigma) * broadcast_mask) \
        .sum(axis=1)

    cls_bbox_loss = cls_loss + bbox_loss
    sort_index = Argsort(cls_bbox_loss.flatten().add_axis(0), ascending=False) \
        .outputs[1].reshape(-1)

    nr_index = Min(sort_index.shape[0], nr_ohem_sampling)
    sort_index = sort_index[:nr_index]
    sort_index = ZeroGrad(sort_index)

    final_cls_loss = cls_loss.ai[sort_index].sum() / nr_index
    final_bbox_loss = bbox_loss.ai[sort_index].sum() / nr_ohem_sampling

    return final_cls_loss, final_bbox_loss


def smooth_l1_loss_rcnn_ohem(
        pred, gt, label, sigma, nr_ohem_sampling, background=0,
        ignore_label=-1):
    broadcast_label = label.reshape((label.shape[0], 1)) \
        .broadcast(Concat([label.shape[0], pred.shape[-1]], axis=0))
    broadcast_mask, broadcast_mask_ig = _get_mask_of_label(
        broadcast_label, background, ignore_label)
    vlabel = broadcast_label * broadcast_mask
    pred_corr = IndexingOneHot(pred, 1, vlabel)

    loss = _smooth_l1_base(pred_corr, gt, sigma) * broadcast_mask
    sorted_loss = Argsort(loss.flatten().add_axis(0), ascending=False) \
        .outputs[0]
    index_sum = Min(nr_ohem_sampling, sorted_loss.shape[1])
    sum_loss = sorted_loss[0, :index_sum].sum() / nr_ohem_sampling
    return sum_loss


# not checked

def softmax_loss_ohem(score, label, nr_ohem_sampling, ignore_label=-1):
    # NOTE: this style of code would introduce different gradients
    # score -= score.max(axis=1, keepdims=True)
    max_score = ZeroGrad(score.max(axis=1, keepdims=True))
    score -= max_score
    log_prob = score - safelog(Exp(score).sum(axis=1, keepdims=True))
    mask = 1 - label.eq(ignore_label)
    vlabel = label * mask
    loss = -(IndexingOneHot(log_prob, 1, vlabel) * mask)
    sorted_loss = Argsort(loss.flatten().add_axis(0), ascending=False) \
        .outputs[0]
    index_sum = Min(nr_ohem_sampling, sorted_loss.shape[1])
    sum_loss = sorted_loss[0, :index_sum].sum() / nr_ohem_sampling
    # loss = -(IndexingOneHot(log_prob, 1, vlabel) * mask).sum() / mask.sum()
    return sum_loss


def sigmoid_cross_entropy_ohem(
        pred, label, ignore_label=-1, background=0, nr_ohem_sampling=256):
    mask = 1 - label.eq(ignore_label)
    vlabel = label * mask
    zero_mat = zeros(
        pred.shape[0], pred.shape[1], pred.shape[2] + 1, dtype='int32')
    one_mat = ones(pred.shape[0], pred.shape[1], 1, dtype='int32')
    one_hot = IndexingSetOneHot(zero_mat, 2, vlabel, one_mat)
    onehot = one_hot[:, :, 1:]
    pos_part = onehot * safelog(pred)
    neg_part = (1 - onehot) * safelog(1 - pred)
    # loss = -(alpha * pos_part + (1 - alpha) * neg_part).sum(axis=2) * mask
    pos_loss = -pos_part.sum(axis=2) * mask
    neg_loss = -neg_part.sum(axis=2) * mask

    sorted_negative_loss = \
        Argsort(neg_loss.flatten().add_axis(0), ascending=False).outputs[0]
    index_sum = Min(nr_ohem_sampling, sorted_negative_loss.shape[1])

    pos_mask = mask > 0

    pos_loss = pos_loss.sum() / Max(pos_mask.sum(), 1)
    neg_loss = sorted_negative_loss[0, :index_sum].sum() / index_sum
    # return loss.sum() / Max(positive_mask.sum(), 1)
    return pos_loss, neg_loss


def soft_focal_on_hard_focal_cross_entropy(
        pred, label, ignore_label=-1, background=0, alpha=0.5, gamma=0):
    pass

def mask_sigmoid_cross_entropy(
        pred, label, gt_masks):
    """
    author: lrx & jbr
    """
    # mask sigmoid cross entropy loss
    loss = Log(1.0 + Exp((1 - gt_masks * 2) * pred))
    loss = loss.mean(2)
    import numpy as np
    loss_valid = (label > 0).astype(np.float32)
    loss *= loss_valid
    return loss.sum() / (Max(loss_valid.sum(), 1))
