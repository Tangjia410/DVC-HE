import numpy as np
import torch
import prettytable


__all__ = ['keypoint_detection']


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def compute_h_score(args, class_list, gt_label_all, pred_cls_all, open_flag=True):
    per_class_num = np.zeros((len(class_list)))
    per_class_correct = np.zeros_like(per_class_num)
    pred_label_all = torch.max(pred_cls_all, dim=1)[1]

    for i, label in enumerate(class_list):
        label_idx = torch.where(gt_label_all == label)[0]
        correct_idx = torch.where(pred_label_all[label_idx] == label)[0]
        per_class_num[i] = float(len(label_idx))
        per_class_correct[i] = float(len(correct_idx))

    per_class_acc = per_class_correct / (per_class_num + 1e-5)

    if open_flag:
        known_acc = per_class_acc[:-1].mean()
        unknown_acc = per_class_acc[-1]
        all_acc = per_class_acc.mean()
        h_score = 2 * known_acc * unknown_acc / (known_acc + unknown_acc + 1e-5)
    else:
        known_acc = per_class_correct.sum() / (per_class_num.sum() + 1e-5)
        unknown_acc = 0.0
        all_acc = 0.0
        h_score = 0.0

    return h_score, all_acc, known_acc, unknown_acc


def compute_osda_metrics(pred_cls_all, gt_label_all, class_list, eps=1e-5):
    pred = torch.max(pred_cls_all, dim=1)[1]
    gt = gt_label_all

    known_labels = torch.tensor(class_list[:-1], dtype=gt.dtype, device=gt.device)
    unk_label = class_list[-1]

    nk = int((gt.unsqueeze(1) == known_labels.unsqueeze(0)).any(dim=1).sum().item())
    nu = int((gt == unk_label).sum().item())
    mk = int(((pred.unsqueeze(1) == known_labels.unsqueeze(0)) &
              (gt.unsqueeze(1) == known_labels.unsqueeze(0))).any(dim=1).sum().item())
    mu = int(((pred == unk_label) & (gt == unk_label)).sum().item())

    os_star = mk / (nk + eps) if nk > 0 else 0.0
    uk = mu / (nu + eps) if nu > 0 else 0.0
    os = (mk + mu) / (nk + nu + eps) if (nk + nu) > 0 else 0.0
    h = 2.0 * os_star * uk / (os_star + uk + eps)

    return {
        'os_star': os_star,
        'uk': uk,
        'os': os,
        'h': h,
        'nk': nk,
        'nu': nu,
        'mk': mk,
        'mu': mu,
    }
