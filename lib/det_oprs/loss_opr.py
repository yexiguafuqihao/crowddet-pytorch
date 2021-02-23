# encoding: utf-8
import os,sys
import torch

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
