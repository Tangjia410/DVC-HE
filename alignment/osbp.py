import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from modules.classifier import Classifier as ClassifierBase
from modules.grl import GradientReverseLayer
from modules.grl import WarmStartGradientReverseLayer
from utils.metric import binary_accuracy, accuracy
from alignment import loss as newloss


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(

            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),

        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
        self.grl = GradientReverseLayer()

    def forward(self, x: torch.Tensor, grad_reverse: Optional[bool] = False, need_fp=False):
        features = self.backbone(x)
        features = self.bottleneck(features)
        outputs = self.head(features)

        if grad_reverse:
            features = self.grl(features)
            outputs = self.head(features)
            return outputs, features

        if need_fp:
            outs = self.head(torch.cat((features, nn.Dropout(0.1)(features))))
            out, out_fp = outs.chunk(2)
            return out, out_fp

        if self.training:
            return outputs, features
        else:
            return outputs


class DomainAdversarialLoss(nn.Module):
    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean',
                 grl: Optional = None, sigmoid=True):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        if self.sigmoid:
            d_s, d_t = d.split([f_s.size(0), f_t.size(0)], dim=0)
            d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
            d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
            self.domain_discriminator_accuracy = 0.5 * (
                        binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))

            if w_s is None:
                w_s = torch.ones_like(d_label_s)
            if w_t is None:
                w_t = torch.ones_like(d_label_t)
            return 0.5 * (
                F.binary_cross_entropy(d_s, d_label_s, weight=w_s.view_as(d_s), reduction=self.reduction) +
                F.binary_cross_entropy(d_t, d_label_t, weight=w_t.view_as(d_t), reduction=self.reduction)
            )
        else:
            d_label = torch.cat((
                torch.ones((f_s.size(0),)).to(f_s.device),
                torch.zeros((f_t.size(0),)).to(f_t.device),
            )).long()
            if w_s is None:
                w_s = torch.ones((f_s.size(0),)).to(f_s.device)
            if w_t is None:
                w_t = torch.ones((f_t.size(0),)).to(f_t.device)
            self.domain_discriminator_accuracy = accuracy(d, d_label)
            loss = F.cross_entropy(d, d_label, reduction='none') * torch.cat([w_s, w_t], dim=0)
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            elif self.reduction == "none":
                return loss
            else:
                raise NotImplementedError(self.reduction)


def bce_loss(output, target):
    output_neg = 1 - output
    target_neg = 1 - target
    result = torch.mean(target * torch.log(output + 1e-6))
    result += torch.mean(target_neg * torch.log(output_neg + 1e-6))
    return -torch.mean(result)


def get_prototype_weight(center_feat, feat, num_classes):
    N, C = feat.shape
    class_numbers = num_classes
    feat_proto_distance = -torch.ones((N, class_numbers)).to(feat.device)
    for i in range(class_numbers):
        proto = center_feat[i].expand(N, -1)
        feat_proto_distance[:, i] = torch.norm(proto - feat, 2, dim=1, )
    feat_nearest_proto_distance, feat_nearest_proto = feat_proto_distance.min(dim=1, keepdim=True)
    feat_proto_distance = feat_proto_distance - feat_nearest_proto_distance
    weight = F.softmax(-feat_proto_distance, dim=1)

    entropy2 = newloss.Entropy(weight)
    prt_loss = torch.mean(entropy2)

    return weight, prt_loss


class Unknown_class_detection(nn.Module):
    def __init__(self, num_classes):
        super(Unknown_class_detection, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits_s1, logits_s2, pred_u_w_fp, mask_u_1, mask_u_2) -> torch.Tensor:
        bsz = logits_s1.size()[0]
        device = logits_s1.device
        unknown_class_ind = self.num_classes - 1

        mask_1 = (mask_u_1 >= unknown_class_ind).float()
        mask_2 = (mask_u_2 >= unknown_class_ind).float()
        unkonwn_nums_1 = mask_1.sum().item()
        unkonwn_nums_2 = mask_2.sum().item()

        label_np = [unknown_class_ind for i in range(bsz)]
        label = torch.Tensor(label_np).type(torch.int64).to(device)
        loss_u_s1 = F.cross_entropy(logits_s1, label, reduction='none').to(device)
        loss_u_s2 = F.cross_entropy(logits_s2, label, reduction='none').to(device)
        loss_u_w_fp = F.cross_entropy(pred_u_w_fp, label, reduction='none').to(device)
        loss_u = (loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5)

        loss = 0
        if unkonwn_nums_1 != 0.0:
            loss += (loss_u * mask_1).sum() / unkonwn_nums_1
        if unkonwn_nums_2 != 0.0:
            loss += (loss_u * mask_2).sum() / unkonwn_nums_2
        if unkonwn_nums_1 == 0.0 and unkonwn_nums_2 == 0.0:
            loss = torch.tensor([0])

        return loss


class Known_class_detection(nn.Module):
    def __init__(self, num_classes: float):
        super(Known_class_detection, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits_s1, logits_s2, pred_u_w_fp, mask_u_1, mask_u_2, pred_u_w):
        device = logits_s1.device
        unknown_class_ind = self.num_classes - 1
        max_prob, label_m = torch.max(pred_u_w, dim=1)
        mask_1 = (mask_u_1 < unknown_class_ind).float()
        mask_2 = (mask_u_2 < unknown_class_ind).float()
        konwn_nums_1 = mask_1.sum().item()
        konwn_nums_2 = mask_2.sum().item()

        loss_u_s1 = F.cross_entropy(logits_s1, label_m, reduction='none').to(device)
        loss_u_s2 = F.cross_entropy(logits_s2, label_m, reduction='none').to(device)
        loss_u_w_fp = F.cross_entropy(pred_u_w_fp, label_m, reduction='none').to(device)
        loss_u = (loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5)

        loss = 0
        if konwn_nums_1 != 0.0:
            loss += (loss_u * mask_1).sum() / konwn_nums_1
        if konwn_nums_2 != 0.0:
            loss += (loss_u * mask_2).sum() / konwn_nums_2
        if konwn_nums_1 == 0.0 and konwn_nums_2 == 0.0:
            loss = torch.tensor([0])

        return loss


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0] // 2
        features = F.normalize(features, dim=1)
        f_t1, f_t2 = torch.split(features, [batch_size, batch_size], dim=0)
        features = torch.cat([f_t1.unsqueeze(1), f_t2.unsqueeze(1)], dim=1)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
