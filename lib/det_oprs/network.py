import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from config import config
from backbone.resnet50 import ResNet50
from backbone.fpn import FPN
from module.rpn import RPN
from layers.pooler import roi_pooler
from det_oprs.bbox_opr import bbox_transform_inv_opr
from fpn_roi_target import fpn_roi_target
from det_oprs.utils import get_padded_tensor
import loss_opr
from bbox_opr import restore_bbox
import pdb
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = ResNet50(config.backbone_freeze_at, False)
        self.FPN = FPN(self.resnet50, 2, 6)
        self.RPN = RPN(config.rpn_channel)
        self.RCNN = RCNN()

    def forward(self, image, im_info, gt_boxes=None):

        image = (image - torch.tensor(config.image_mean.reshape(1, -1, 1, 1)).type_as(image)) / (
                torch.tensor(config.image_std.reshape(1, -1, 1, 1)).type_as(image))
        image = get_padded_tensor(image, 64)
        if self.training:
            return self._forward_train(image, im_info, gt_boxes)
        else:
            return self._forward_test(image, im_info)

    def _forward_train(self, image, im_info, gt_boxes):
        
        loss_dict = {}
        fpn_fms = self.FPN(image)
        # fpn_fms stride: 64,32,16,8,4, p6->p2
        rpn_rois, rpn_loss_dict = self.RPN(fpn_fms, im_info, gt_boxes)
        # rcnn_rois, rcnn_labels, rcnn_bbox_targets = fpn_roi_target(
        #         rpn_rois, im_info, gt_boxes, top_k=1)
        rcnn_loss_dict = self.RCNN(fpn_fms, rpn_rois, gt_boxes, im_info)

        loss_dict.update(rpn_loss_dict)
        loss_dict.update(rcnn_loss_dict)
        return loss_dict

    def _forward_test(self, image, im_info):
        fpn_fms = self.FPN(image)
        rpn_rois = self.RPN(fpn_fms, im_info)
        pred_bbox = self.RCNN(fpn_fms, rpn_rois)
        return pred_bbox.cpu().detach()

class CascadeRCNN(nn.Module):

    def __init__(self, iou_thresh, nheads, stage):

        super().__init__()

        assert iou_thresh >= 0.5 and nheads > 0
        self.iou_thresh = iou_thresh
        self.nheads = nheads
        self.n = config.num_classes
        self.name = 'cascade_stage_{}'.format(stage)

        self.fc1 = nn.Linear(256 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()

        self.n = config.num_classes
        self.p = nn.Linear(1024, 5 * self.n * nheads)
        self._init_weights()

    def _init_weights(self):

        for l in [self.fc1, self.fc2, self.p]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

    def forward(self, fpn_fms, rois, gtboxes=None, im_info = None):

        rpn_fms = fpn_fms[1:]
        rpn_fms.reverse()
        if self.training:
            rcnn_rois, labels, bbox_targets = fpn_roi_target(rois, im_info, gtboxes, top_k=self.nheads)

        stride = [4, 8, 16, 32]
        pool5 = roi_pooler(rpn_fms, rcnn_rois, stride, (7, 7), "ROIAlignV2")
        pool5 = torch.flatten(pool5, start_dim=1)
        fc1 = self.relu(self.fc1(pool5))
        fc2 = self.relu(self.fc2(fc1))
        prob = self.p(fc2)

        loss = {}
        if self.training:
            # compute the loss function and then return 
            bbox_targets = bbox_targets.reshape(-1, 4) if self.nheads > 1 else bbox_targets
            labels = labels.view(-1)
            
            loss = self.compute_regular_loss(prob, bbox_targets, labels) if self.nheads < 2 else \
                self.compute_emd_loss(prob, bbox_targets, labels)
            return loss, prob
        else:

            # return the detection boxes and their scores
            return loss, prob
        
    def compute_emd_loss_opr(self, a, b, bbox_targets, labels):
        
        labels = labels.long().flatten()
        c = a.shape[1]
        prob = torch.stack([a, b], dim=1).reshape(-1, c)
        offsets, cls_score = prob[:, :-self.n], prob[:,-self.n:]
        cls_loss = loss_opr.softmax_loss_opr(cls_score, labels)
        n = offsets.shape[0]
        offsets = offsets.reshape(n, -1, 4)
        bbox_targets = bbox_targets.reshape(-1, 4)
        reg_loss = loss_opr.smooth_l1_loss_rcnn_opr(offsets, bbox_targets,
            labels, sigma = config.rcnn_smooth_l1_beta)

        vlabel = 1 - ((labels < 0).view(-1, 2).sum(axis=1) > 1).float()
        loss = (cls_loss + 2 * reg_loss).view(-1, 2).sum(dim=1) * vlabel
        return loss

    def compute_gemini_loss_opr(self, prob, bbox_targets, labels):
            
        prob = prob.reshape(prob.shape[0], 2, -1)
        n, _, c = prob.shape
        prob = prob.permute(1, 0, 2)
        a, b = prob[0], prob[1]
        loss0 = self.compute_det_loss_opr(a, b, bbox_targets, labels)
        loss1 = self.compute_det_loss_opr(b, a, bbox_targets, labels)
        loss = torch.stack([loss0, loss1], dim = 1)
        emd_loss = loss.min(axis=1)[0].sum()/max(loss.shape[0], 1)
        return emd_loss

    def compute_regular_loss(self, prob, bbox_targets, labels):

        offsets, cls_scores = prob[:,:-self.n], prob[:, -self.n:]
        n = offsets.shape[0]
        offsets = offsets.reshape(n, -1, 4)
        cls_loss = loss_opr.softmax_loss(cls_scores, labels)

        bbox_loss = loss_opr.smooth_l1_loss_rcnn_opr(offsets, bbox_targets,
            labels, config.rcnn_smooth_l1_beta)
        bbox_loss = bbox_loss.sum() / torch.clamp((labels > 0).sum(), 1)
        loss = {}
        loss['{}_cls_loss'.format(self.name)] = cls_loss
        loss['{}_bbox_loss'.format(self.name)] = bbox_loss
        return loss


class RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.n = config.num_classes

        self.iou_thresholds = [0.5, 0.6]
        self.nheads = [1, 2]
        self.heads = nn.ModuleList()
        for i, iou_thresh in enumerate(self.iou_thresholds):

            rcnn_head = CascadeRCNN(iou_thresh, self.nheads[i], i+1)
            self.heads.append(rcnn_head)
        
    @torch.no_grad()
    def recover_pred_boxes(self, rcnn_rois, prob):

        cls_score, bbox_pred = prob[:, -self.n:], prob[:, :-self.n]
        cls_prob = torch.softmax(cls_score, dim=1)
        pdb.set_trace()
        rois = rcnn_rois[:,1:5]
        n = bbox_pred.shape[0]
        bbox_pred = bbox_pred.reshape(n, -1, 4).permute(1, 0, 2)[1]
        # pred_bbox = self.restore_bbox(rois, bbox_pred)
        return pred_bbox, cls_prob

    def forward(self, fpn_fms, rcnn_rois, gtboxes = None, im_info=None):
        
        # input p2-p5
        for i, _ in enumerate(self.iou_thresholds):

            loss, prob = self.heads[i](fpn_fms=fpn_fms, rois=rcnn_rois, gtboxes = gtboxes, im_info = im_info)
            rcnn_rois, pred_scores = self.recover_pred_boxes(rcnn_rois, prob)


        if self.training:
            # loss for regression
            offsets, cls_score = prob[:, :-self.n], prob[:,-self.n:]

            n = offsets.shape[0]
            bbox_targets, labels = bbox_targets.reshape(-1, 4), labels.flatten()
            cls_loss = loss_opr.softmax_loss_opr(cls_score, labels)
            
            offsets = offsets.reshape(n, -1, 4)
            reg_loss = loss_opr.smooth_l1_loss_rcnn_opr(offsets, bbox_targets,
                labels, sigma = config.rcnn_smooth_l1_beta)

            cls_num = (labels > -1).sum()
            reg_num = (labels > 0).sum()
            reg_loss  = reg_loss.sum() / torch.clamp(reg_num, 1)
            cls_loss = cls_loss.sum() / torch.clamp(cls_num, 1)
            loss_dict = {}
            loss_dict['loss_rcnn_loc'] = reg_loss
            loss_dict['loss_rcnn_cls'] = cls_loss
            return loss_dict
        else:
            cls_score, bbox_pred = prob[:, -self.n:], prob[:, :-self.n]
            cls_prob = torch.softmax(cls_score, dim=1)
            
            rois = rcnn_rois[:,1:5]
            n = bbox_pred.shape[0]
            bbox_pred = bbox_pred.reshape(n, -1, 4).permute(1, 0, 2)[1]
            pred_bbox = self.restore_bbox(rois, bbox_pred)
            tag = torch.linspace(0, rcnn_rois.shape[0]-1, rcnn_rois.shape[0]).to(rois.device) + 1
            tag = tag.view(-1, 1)
            pred_bbox = torch.cat([pred_bbox, cls_prob[:, 1:], tag], dim=1)

            return pred_bbox
