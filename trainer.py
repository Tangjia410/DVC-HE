import os
import time
import inspect
import argparse
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

import datasets
from utils import utils
from alignment.osbp import ImageClassifier as Classifier, DomainAdversarialLoss, SupConLoss, Known_class_detection, Unknown_class_detection, get_prototype_weight, bce_loss
from modules.domain_discriminator import DomainDiscriminator
from utils.data import ForeverDataIterator
from utils.meter import AverageMeter, ProgressMeter
from utils.metric import compute_h_score, compute_osda_metrics
from utils.eval_viz import plot_confusion_matrix, plot_tsne
from compare_model import main_maosdan

from models.MAOSDAN_shift import MAOSDAN_Shift
from models.OG_MoE import OG_MoE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEFAULT_CKPT = {
    'mfcl': 'best.pth',
    'maosdan_shift': 'maosdan_shift_best.pth',
    'og_moe': 'og_moe_best.pth',
}


def main(args: argparse.Namespace):
    print(args)

    _cudnn_bench = os.environ.get('CUDNN_BENCHMARK')
    if _cudnn_bench is not None:
        cudnn.benchmark = bool(int(_cudnn_bench))
    elif not cudnn.deterministic:
        cudnn.benchmark = True

    os.makedirs(os.path.join(args.log, "checkpoints"), exist_ok=True)

    data_loaders = getattr(datasets, args.data_name)

    try:
        loaders = data_loaders(args.data_dir, args.src_name, args.tgt_name, args.src_classes,
                               args.tgt_classes, args.batch_size,
                               getattr(args, 'workers', 4)).create_loaders()
    except TypeError:
        loaders = data_loaders(args.data_dir, args.src_name, args.tgt_name, args.src_classes,
                               args.tgt_classes, args.batch_size).create_loaders()
    (train_source_loader, train_target_loader,
     val_source_loader, val_target_loader, num_classes) = loaders

    phase = getattr(args, 'phase', 'train')
    if phase not in ('train', 'test'):
        raise ValueError("--phase 只支持 'train' 或 'test'，当前为 '{}'".format(phase))
    if phase == 'test':
        test_model(args, val_source_loader, val_target_loader, num_classes)
        return

    if args.method == 'mfcl':
        main_mfcl(args, train_source_loader, train_target_loader,
                  val_source_loader, val_target_loader, num_classes)
    elif args.method == 'maosdan_shift':
        main_maosdan(args, train_source_loader, train_target_loader,
                     val_source_loader, val_target_loader, num_classes,
                     model_cls=MAOSDAN_Shift)
    elif args.method == 'og_moe':
        from og_moe_trainer import main_og_moe
        main_og_moe(args, train_source_loader, train_target_loader,
                    val_source_loader, val_target_loader, num_classes)
    else:
        raise ValueError('unknown method: {}'.format(args.method))


def _unpack_batch(batch):
    if isinstance(batch, (tuple, list)):
        if isinstance(batch[0], (tuple, list)):
            return batch[0][0], batch[1]
        return batch[0], batch[1]
    raise TypeError('无法解析 batch 结构: {}'.format(type(batch)))


def _forward_extract(model, x):
    if hasattr(model, 'forward_eval'):
        return model.forward_eval(x)

    if hasattr(model, 'base') and not (hasattr(model, 'backbone') and hasattr(model, 'bottleneck')):
        model = model.base

    if hasattr(model, 'backbone') and hasattr(model, 'bottleneck'):
        f = model.bottleneck(model.backbone(x))

        if hasattr(model, 'compute_logits'):
            return f, model.compute_logits(f)
        if getattr(model, 'head', None) is not None:
            return f, model.head(f)
        if hasattr(model, 'class_layer'):
            return f, model.class_layer(f)

    if hasattr(model, 'backbone_layer') and hasattr(model, 'feat_embed_layer'):
        rois_c, _, rois_c_mu = model.feat_embed_layer(model.backbone_layer(x), deterministic=True)
        return rois_c_mu, model.class_layer(rois_c_mu, False)

    if hasattr(model, 'feature_extractor') and hasattr(model, 'classifier'):
        f = model.feature_extractor(x)
        return f, model.classifier(f)

    out = model(x)
    if isinstance(out, dict) and 'logits' in out:
        return out.get('feature'), out['logits']

    if isinstance(out, (tuple, list)) and len(out) >= 2:
        return out[1], out[0]
    raise RuntimeError('无法从模型 {} 中抽取特征与 logits'.format(model.__class__.__name__))


@torch.no_grad()
def extract_features_and_logits(model, loader, device_):
    was_training = model.training
    model.eval()
    feats_all, logits_all, labels_all = [], [], []
    for batch in loader:
        x, y = _unpack_batch(batch)
        x = x.to(device_)
        f, lg = _forward_extract(model, x)
        feats_all.append(f.float().cpu())
        logits_all.append(lg.float().cpu())
        labels_all.append(y.long().reshape(-1).cpu())
    if was_training:
        model.train()
    return torch.cat(feats_all, 0), torch.cat(logits_all, 0), torch.cat(labels_all, 0)


_MODEL_OPTIONAL_KWARGS = (
    'anchor_weight', 'anchor_conf', 'source_proto_conf', 'anchor_decay_epochs',

    'anchor_unk_margin', 'anchor_cov_cap', 'anchor_balance', 'anchor_openness_alpha',
    'openness_pi_u',

    'proto_gamma', 'reliability_gamma', 'drift_budget', 'dual_lr',
    'relation_weight', 'iters_per_epoch',

    'drift_scale', 'stable_warmup_epochs', 'stable_warmup_length',

    'mask2anomaly_tta', 'mask2anomaly_score_mode', 'mask2anomaly_um_thr',
    'mask2anomaly_decoder', 'mask2anomaly_decoder_dim', 'mask2anomaly_decoder_depth',
    'mask2anomaly_decoder_heads', 'mask2anomaly_decoder_ffn', 'mask2anomaly_queries',
    'mask2anomaly_decoder_dropout',

    'mask2anomaly_gate_max',

    'mask2anomaly_enc_c1', 'mask2anomaly_enc_c2', 'mask2anomaly_enc_c3',
    'mask2anomaly_top_ratio', 'mask2anomaly_fusion', 'mask2anomaly_anomaly_thr',
    'mask2anomaly_detach_anomaly_backbone', 'mask2anomaly_mask_margin',
    'mask2anomaly_match_cls_w', 'mask2anomaly_match_bce_w', 'mask2anomaly_match_dice_w',

    'mask2anomaly_adapt_w', 'mask2anomaly_adapt_adv_w', 'mask2anomaly_adapt_adv_hidden',
    'mask2anomaly_adapt_adv_known_only', 'mask2anomaly_adapt_tgt_ce_w',
    'mask2anomaly_adapt_tgt_ce_thr', 'mask2anomaly_adapt_tgt_ce_conf',

    'maosdan_shift_w', 'maosdan_shift_band_q', 'maosdan_shift_pull_frac',
    'maosdan_shift_dir', 'maosdan_shift_margin_target', 'maosdan_shift_eps_max',
    'maosdan_shift_eps_rel',
    'maosdan_shift_pull_w', 'maosdan_shift_push_w',
    'maosdan_shift_balance', 'maosdan_shift_gap_scale',
    'maosdan_shift_ev_closed', 'maosdan_shift_ev_knn',
    'maosdan_shift_ev_geom', 'maosdan_shift_knn_k', 'maosdan_shift_bank_size',
    'maosdan_shift_proto_momentum', 'maosdan_shift_radius_q', 'maosdan_shift_radius_clip',
    'maosdan_shift_min_support', 'maosdan_shift_min_side', 'maosdan_shift_start_epoch',
    'maosdan_shift_orig_w', 'maosdan_shift_balance_clip',
    'maosdan_shift_preserve_class', 'maosdan_shift_reliability_margin',
    'maosdan_shift_class_cap',
    'maosdan_shift_gate_pull_geom', 'maosdan_shift_gate_push_geom',
    'maosdan_shift_gate_pull_knn', 'maosdan_shift_gate_push_pu',

    'maosdan_shift_margin_pull_max', 'maosdan_shift_pull_proto_agree',
    'maosdan_shift_ramp_epochs', 'maosdan_shift_coverage_ref',

    'maosdan_shift_proto_gate_enable',

    'shift_log_enable', 'shift_log_dir', 'shift_log_samples', 'shift_log_skips',
    'shift_log_every_epoch', 'shift_log_every_iter', 'shift_log_max_samples',
    'shift_log_meta',

    'mfcl_shift_w', 'mfcl_shift_pi_center', 'mfcl_shift_pi_scale',
    'mfcl_shift_balance', 'mfcl_prior_w', 'mfcl_shift_proto_momentum',
    'mfcl_shift_radius_q', 'mfcl_shift_radius_clip', 'mfcl_shift_min_support',
    'mfcl_shift_start_epoch',
)


def _resolve_openness_pi_u(args):
    if getattr(args, 'openness_pi_u', None) is not None:
        return args
    try:
        n_tgt = len(args.tgt_classes)
        n_src = len(args.src_classes)
        args.openness_pi_u = max(0.0, min(1.0, (n_tgt - n_src) / float(max(n_tgt, 1))))
    except (TypeError, AttributeError):
        args.openness_pi_u = 0.0
    return args


def _model_init_kwargs(model_cls, args, names=_MODEL_OPTIONAL_KWARGS):
    kwargs = {}
    try:
        params = inspect.signature(model_cls.__init__).parameters
    except (TypeError, ValueError):
        return kwargs
    for name in names:
        value = getattr(args, name, None)
        if value is not None and name in params:
            kwargs[name] = value
    return kwargs


def build_model_for_method(args, num_classes):
    method = args.method

    if method == 'og_moe':
        from models.MAOSDAN_shift import MAOSDAN_Shift as _Shift
        _kw = _model_init_kwargs(_Shift, args)
        _n_t = max(len(getattr(args, 'tgt_classes', []) or []), 1)
        _n_s = len(getattr(args, 'src_classes', []) or [])
        _pi = max(0.0, min(1.0, (_n_t - _n_s) / float(_n_t)))
        return OG_MoE(getattr(args, 'arch', 'lenet'), num_classes,
                      bottleneck_dim=int(getattr(args, 'bottleneck_dim', 256)),
                      shared_backbone=bool(int(getattr(args, 'og_shared', 1))),
                      pi_u=_pi, shift_kwargs=_kw).to(device)

    backbone = utils.get_model(getattr(args, 'arch', 'lenet'))
    bottleneck_dim = args.bottleneck_dim

    if method == 'mfcl':
        return Classifier(backbone, num_classes, bottleneck_dim=bottleneck_dim).to(device)

    if method == 'maosdan_shift':
        kwargs = _model_init_kwargs(MAOSDAN_Shift, args)
        return MAOSDAN_Shift(backbone, num_classes, bottleneck_dim=bottleneck_dim,
                             **kwargs).to(device)

    raise ValueError('unknown method: {}'.format(method))


def load_model_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        for key in ('model_state_dict', 'state_dict', 'net'):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    def _report(missing, unexpected, dropped=()):
        if missing:
            print('  [warn] 检查点缺失参数 {} 个（保持随机初始化）'.format(len(missing)))
        if unexpected:
            print('  [warn] 检查点多余参数 {} 个（已忽略）'.format(len(unexpected)))
        if dropped:
            print('  [warn] 检查点形状不匹配的键 {} 个（已忽略）: {}'.format(
                len(dropped), list(dropped)[:5]))

    def _drop_shape_mismatch(state_dict, ref_state):
        keep, dropped = {}, []
        for k, v in state_dict.items():
            r = ref_state.get(k, None)
            if r is not None and tuple(getattr(v, 'shape', ())) != tuple(r.shape):
                dropped.append(k)
                continue
            keep[k] = v
        return keep, dropped

    if hasattr(model, 'network') and isinstance(model.network, nn.Sequential):
        own_keys = set(model.state_dict().keys())
        net_keys = set(model.network.state_dict().keys())
        ckpt_keys = set(ckpt.keys())
        if len(ckpt_keys & net_keys) > len(ckpt_keys & own_keys):
            net_ckpt, dropped = _drop_shape_mismatch(ckpt, model.network.state_dict())
            missing, unexpected = model.network.load_state_dict(net_ckpt, strict=False)
            _report(missing, unexpected, dropped)
            return model

    filtered, dropped = _drop_shape_mismatch(ckpt, model.state_dict())
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    _report(missing, unexpected, dropped)
    return model


def get_class_names(args, num_out):
    raw = None
    try:
        _dn = str(getattr(args, 'data_name', '') or '').lower()
        if _dn.startswith('rflymad'):
            from datasets.RflyMAD import CLASS_NAMES as _RF_NAMES
            raw = list(_RF_NAMES)
        else:
            from datasets.SDUST import RDBdata
            raw = list(RDBdata)
    except Exception:
        pass

    names = []
    for i in range(num_out):
        if i == num_out - 1:
            names.append('Unknown')
        elif raw is not None and i < len(raw):
            names.append(str(raw[i]))
        else:
            names.append(str(i))
    return names


def test_model(args, val_source_loader, val_target_loader, num_classes):
    ckpt_path = getattr(args, 'checkpoint', None) or os.path.join(
        args.log, 'checkpoints', DEFAULT_CKPT.get(args.method, 'best.pth'))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            '未找到检查点: {}\n请先用 --phase train 训练，或用 --checkpoint 显式指定权重路径。'.format(ckpt_path))

    print('\n================== TEST MODE ==================')
    print('method     : {}'.format(args.method))
    print('checkpoint : {}'.format(ckpt_path))

    model = build_model_for_method(args, num_classes)
    load_model_weights(model, ckpt_path)
    model.to(device).eval()

    eval_dir = os.path.join(args.log, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    tag = args.method

    print('\n=========== Final evaluation on target (open-set) ===========')
    feats_t, logits_t, gt_t = extract_features_and_logits(model, val_target_loader, device)

    num_out = int(logits_t.shape[1])
    unk_gt = num_classes - 1
    unk_idx = num_out - 1
    gt_t_map = torch.where(gt_t >= unk_gt, torch.tensor(unk_idx), gt_t)
    class_list = list(range(unk_gt)) + [unk_idx]

    h_score, all_acc, known_acc, unknown_acc = compute_h_score(
        args, class_list, gt_t_map, logits_t, open_flag=True)

    osda = compute_osda_metrics(logits_t, gt_t_map, class_list)
    print('target H-score : {:.4f}  known acc: {:.4f}  unknown acc: {:.4f}  all acc: {:.4f}'.format(
        h_score, known_acc, unknown_acc, all_acc))
    print('target OS*     : {:.4f}  UK       : {:.4f}  OS        : {:.4f}  (Nk={}, Nu={}, Mk={}, Mu={})'.format(
        osda['os_star'], osda['uk'], osda['os'],
        osda['nk'], osda['nu'], osda['mk'], osda['mu']))

    tsne_path = os.path.join(eval_dir, 'tsne_openset_{}.png'.format(tag))
    try:
        plot_tsne(feats_t.numpy(), gt_t_map.numpy(),
                  get_class_names(args, num_out), tsne_path,
                  title='t-SNE (target, open-set) - {}'.format(tag),
                  perplexity=getattr(args, 'tsne_perplexity', 30),
                  n_iter=getattr(args, 'tsne_iter', 1000),
                  max_samples=getattr(args, 'tsne_max_samples', 3000),
                  unknown_index=unk_idx)
        print('t-SNE 图已保存: {}'.format(tsne_path))
    except Exception as e:
        print('[warn] t-SNE 生成失败: {}'.format(e))

    print('\n=========== Final evaluation on source (closed-set) ===========')
    feats_s, logits_s, gt_s = extract_features_and_logits(model, val_source_loader, device)
    pred_s = logits_s.argmax(dim=1)
    total = gt_s.numel()
    correct = int((pred_s == gt_s).sum().item())
    print('source closed-set acc: {:.2f} ({}/{})'.format(
        100. * correct / max(total, 1), correct, total))

    names = get_class_names(args, num_out)
    cm_path = os.path.join(eval_dir, 'cm_closedset_{}.png'.format(tag))
    cm_norm_path = os.path.join(eval_dir, 'cm_closedset_{}_norm.png'.format(tag))
    try:
        plot_confusion_matrix(gt_s.numpy(), pred_s.numpy(), names, cm_path,
                              normalize=False,
                              title='Confusion Matrix (source, closed-set) - {}'.format(tag))
        plot_confusion_matrix(gt_s.numpy(), pred_s.numpy(), names, cm_norm_path,
                              normalize=True,
                              title='Confusion Matrix (normalized) - {}'.format(tag))
        print('混淆矩阵已保存: {}'.format(cm_path))
        print('归一化混淆矩阵已保存: {}'.format(cm_norm_path))
    except Exception as e:
        print('[warn] 混淆矩阵生成失败: {}'.format(e))

    print('\n================== TEST DONE ==================')
    return h_score


def _print_mfcl_stable_plus_epoch_diag(args, model, epoch):
    diag_fn = getattr(model, 'diagnostics', None)
    if not callable(diag_fn):
        return
    try:
        d = diag_fn()
    except Exception:
        return
    if not isinstance(d, dict) or 'drift_lambda' not in d:
        return

    tag = ' * [{}] epoch {} 后'.format(args.method, epoch)

    if 'stable_factor' in d:
        _cnt = int(d.get('anchor_epoch_count', 0) or 0) or int(d.get('anchor_count', 0) or 0)
        _total = int(d.get('anchor_epoch_total', 0) or 0) or int(d.get('anchor_batch', 0) or 0)
        _ratio = (float(_cnt) / float(_total)) if _total > 0 else \
            float(d.get('anchor_epoch_ratio', 0.0) or d.get('anchor_ratio', 0.0) or 0.0)
        print('[StablePlus] epoch={} factor={:.3f} anchor={}/{} anchor_ratio={:.4f} '
              'L_STA={:.5f} D_K={:.5f} lambda_beta_d={:.3f} '
              '(warmup={} ramp={} lambda={:.6f} eps={:.4f})'
              .format(epoch,
                      float(d.get('stable_factor', 1.0)),
                      _cnt, _total, _ratio,
                      float(d.get('anchor_loss', 0.0)),
                      float(d.get('epoch_drift_mean_running', 0.0)),
                      float(d.get('drift_scale', 1.0)),
                      d.get('stable_warmup_epochs'),
                      d.get('stable_warmup_length'),
                      float(d.get('drift_lambda', 0.0)),
                      float(d.get('drift_budget', 0.0))), flush=True)

    if int(d.get('stage', 1)) < 2:
        print('{} | stage 1：源域原型累积中 support={}（D_K / D_C / D_R / lambda / '
              'anchor ratio / L_STA 进 stage 2 才有意义）'
              .format(tag, d.get('prototype_support')), flush=True)
        return

    anchor_txt = 'anchor ratio={}/{}={:.1%}'.format(
        d['anchor_count'], d['anchor_batch'], d['anchor_ratio'])
    if d.get('anchor_epoch_ratio') is not None:
        anchor_txt += ' | anchor(epoch)={}/{}={:.1%}'.format(
            int(d.get('anchor_epoch_count', 0)),
            int(d.get('anchor_epoch_total', 0)),
            d.get('anchor_epoch_ratio', 0.0))

    print('{} | D_K={:.5f} D_C={:.5f} D_R={:.5f} lambda={:.6f} '
          '{} L_STA={:.5f} '
          '(eps={:.4f} beta={:.4f} n={})'
          .format(tag,
                  d['epoch_drift_mean_running'],
                  d['epoch_drift_center_running'],
                  d['epoch_drift_relation_running'],
                  d['drift_lambda'],
                  anchor_txt,
                  d['anchor_loss'],
                  d['drift_budget'], d['anchor_weight_effective'],
                  d['epoch_drift_count_running']), flush=True)


def main_mfcl(args: argparse.Namespace, train_source_loader, train_target_loader, val_source_loader, val_target_loader, num_classes,
              model_cls=None, ckpt_prefix=''):
    model_cls = Classifier if model_cls is None else model_cls
    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch)

    _resolve_openness_pi_u(args)
    model_kwargs = _model_init_kwargs(model_cls, args)
    if model_kwargs:
        print(' * [{}] 模型超参: {}'.format(args.method, model_kwargs))
    classifier = model_cls(backbone, num_classes,
                           bottleneck_dim=args.bottleneck_dim, **model_kwargs).to(device)
    domain_discri = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)
    print(classifier, domain_discri)

    ctl_fn = SupConLoss(temperature=0.3).cuda()
    unknown_csl_fn = Unknown_class_detection(num_classes).cuda()
    known_csl_fn = Known_class_detection(num_classes).cuda()
    domain_adv = DomainAdversarialLoss(domain_discri).to(device)

    parameter_list = classifier.get_parameters() + domain_discri.get_parameters()
    optimizer = SGD(parameter_list, args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)

    stage2_lr = getattr(args, 'stage2_lr', None)
    lr_base = {'lr': args.lr}
    if stage2_lr is not None:
        print(' * [{}] --stage2-lr={:.6g} 已启用：epoch > {} 后基学习率由 {:.6g} 切到 {:.6g}'
              .format(args.method, stage2_lr, args.pretrain_epoch, args.lr, stage2_lr))
    lr_scheduler = LambdaLR(
        optimizer,
        lambda x: lr_base['lr'] * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    stage2_lr_switched = False

    start_train = time.time()
    best_h_score = 0.
    for epoch in range(args.epochs):
        if hasattr(classifier, 'set_train_stage'):
            classifier.set_train_stage(1 if epoch <= args.pretrain_epoch else 2)

        if hasattr(classifier, 'set_epoch'):
            classifier.set_epoch(epoch)

        if stage2_lr is not None and epoch > args.pretrain_epoch and not stage2_lr_switched:
            lr_base['lr'] = float(stage2_lr)
            stage2_lr_switched = True
            print(' * [{}] epoch {}: 进入 stage2，基学习率已切到 {:.6g}'
                  .format(args.method, epoch, stage2_lr), flush=True)

        train(train_source_iter, train_target_iter, classifier, domain_adv, ctl_fn, unknown_csl_fn, known_csl_fn,
              optimizer, lr_scheduler, epoch, num_classes, args)

        _print_mfcl_stable_plus_epoch_diag(args, classifier, epoch)

        h_score = validate(args, classifier, val_target_loader)

        latest_path = os.path.join(os.path.join(args.log, "checkpoints"), ckpt_prefix + "latest.pth")
        best_path = os.path.join(os.path.join(args.log, "checkpoints"), ckpt_prefix + "best.pth")
        torch.save(classifier.state_dict(), latest_path)
        if h_score > best_h_score:
            shutil.copy(latest_path, best_path)
        best_h_score = max(h_score, best_h_score)

    print("best_h_score = {:3.2f}".format(best_h_score))
    end_train = time.time()
    print(f"训练总耗时: {end_train - start_train:.2f} 秒")
def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: Classifier,
          domain_adv: DomainAdversarialLoss, ctl_fn: nn.Module, unknown_csl_fn: nn.Module, known_csl_fn: nn.Module,
          optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, num_classes, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.2f')
    data_time = AverageMeter('Data', ':3.2f')
    losses = AverageMeter('Loss', ':3.2f')
    sce_losses = AverageMeter('Sce Loss', ':3.2f')
    esl_losses = AverageMeter('Esl Loss', ':3.2f')
    csl_losses = AverageMeter('Cst Loss', ':3.2f')
    prt_losses = AverageMeter('Prt Loss', ':3.2f')
    adv_losses = AverageMeter('Adv Loss', ':3.2f')
    ctl_losses = AverageMeter('Tdl Loss', ':3.2f')

    progress = ProgressMeter(args.iters_per_epoch, [losses, sce_losses, esl_losses, csl_losses, prt_losses, adv_losses, ctl_losses], prefix="Epoch: [{}]".format(epoch))

    model.train()
    domain_adv.train()

    shift_on = callable(getattr(model, 'shift_loss', None)) and bool(
        getattr(model, 'shift_enabled', False))
    if shift_on:
        _n_t = max(len(getattr(args, 'tgt_classes', []) or []), 1)
        _n_s = len(getattr(args, 'src_classes', []) or [])
        _pi_u = max(0.0, min(1.0, (_n_t - _n_s) / float(_n_t)))
        model.set_shift_epoch(epoch, _pi_u)
        print('[Shift-MFCL] epoch={} {}'.format(epoch, model.shift_describe()), flush=True)

    end = time.time()

    for i in range(args.iters_per_epoch):
        (x_s, x_s_s1, x_s_s2), labels_s = next(train_source_iter)
        (x_t_w, x_t_s1, x_t_s2), labels_t = next(train_target_iter)
        bsz = labels_s.shape[0]

        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        x_t = torch.cat([x_t_w, x_t_s1], dim=0)
        x_t = x_t.to(device)
        x_t_w = x_t_w.to(device)
        x_t_s1 = x_t_s1.to(device)
        x_t_s2 = x_t_s2.to(device)
        labels_t = labels_t.to(device)

        data_time.update(time.time() - end)

        y_s, f_s = model(x_s)

        if hasattr(model, 'observe_labeled_source'):
            model.observe_labeled_source(f_s, y_s, labels_s)

        y_t, f_t = model(x_t)
        y_t_w, y_t_s1 = torch.split(y_t, [bsz, bsz], dim=0)
        f_t_w, f_t_s1 = torch.split(f_t, [bsz, bsz], dim=0)

        shift_value = None
        if shift_on:
            model.observe_target(f_t_w, y_t_w)
            shift_value = model.shift_loss(f_t_w, y_t_w)

        softmax_s = nn.Softmax(dim=1)(y_s)
        softmax_t1 = nn.Softmax(dim=1)(y_t_w)
        softmax_t2 = nn.Softmax(dim=1)(y_t_s1)

        gt_list = labels_s.tolist()
        gt_list = np.unique(gt_list)

        if i == 0:
            center_feat = torch.randn(softmax_s.shape[1], f_s.shape[1]).cuda().detach()
            target_center_feat1 = torch.randn(softmax_t1.shape[1], f_t_w.shape[1]).cuda().detach()
            target_center_feat2 = torch.randn(softmax_t2.shape[1], f_t_w.shape[1]).cuda().detach()
        else:
            for c in range(len(gt_list)):
                c_idx = (labels_s == gt_list[c]).nonzero().squeeze()
                c_feat = torch.index_select(f_s, 0, c_idx)
                c_ctr = torch.mean(c_feat, dim=0)
                center_feat[gt_list[c], :] = 0.5 * center_feat[gt_list[c], :].detach() + (
                            1 - 0.5) * c_ctr.squeeze().detach()

            out_t_t1 = F.softmax(y_t_w, dim=1)
            pseudo_mask = out_t_t1.argmax(dim=1)
            gt_list = pseudo_mask.tolist()
            gt_list = np.unique(gt_list)
            for c in range(len(gt_list)):
                c_idx = (pseudo_mask == gt_list[c]).nonzero().squeeze()
                c_feat = torch.index_select(f_t_w, 0, c_idx)
                c_ctr = torch.mean(c_feat, dim=0)
                target_center_feat1[gt_list[c], :] = 0.5 * target_center_feat1[gt_list[c], :].detach() + (
                            1 - 0.5) * c_ctr.squeeze().detach()

            out_t_t2 = F.softmax(y_t_s1, dim=1)
            pseudo_mask = out_t_t2.argmax(dim=1)
            gt_list = pseudo_mask.tolist()
            gt_list = np.unique(gt_list)
            for c in range(len(gt_list)):
                c_idx = (pseudo_mask == gt_list[c]).nonzero().squeeze()
                c_feat = torch.index_select(f_t_s1, 0, c_idx)
                c_ctr = torch.mean(c_feat, dim=0)
                target_center_feat2[gt_list[c], :] = 0.5 * target_center_feat2[gt_list[c], :].detach() + (
                            1 - 0.5) * c_ctr.squeeze().detach()

        target_center1 = target_center_feat1.detach()
        target_center2 = target_center_feat2.detach()

        weight_1, _ = get_prototype_weight(target_center1, f_t_w, num_classes)
        weight_2, _ = get_prototype_weight(target_center2, f_t_w, num_classes)
        pred_weight_1 = y_t_w * weight_1
        mask_u_w_1 = pred_weight_1.argmax(dim=1)
        pred_weight_2 = y_t_s1 * weight_2
        mask_u_w_2 = pred_weight_2.argmax(dim=1)

        if epoch > args.pretrain_epoch:
            y_t_t, _ = model(x_t_w.to(device), grad_reverse=True)

            out_t_t = F.softmax(y_t_t, dim=1)
            prob1_t = torch.sum(out_t_t[:, :num_classes - 1], 1).view(-1, 1)
            prob2_t = out_t_t[:, num_classes - 1].contiguous().view(-1, 1)
            esl_loss = bce_loss(prob1_t, prob2_t)

            num_lb, num_ulb = x_s.shape[0], x_t_w.shape[0]
            preds, preds_fp = model(torch.cat((x_s, x_t_w)), need_fp=True)

            pred_s, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w = pred_u_w.detach()
            pred_u_w = F.softmax(pred_u_w, dim=1)

            pred_u_w_fp = preds_fp[num_lb:]
            pred_u_w_fp = F.softmax(pred_u_w_fp, dim=1)

            pred_u_s, _ = model(torch.cat((x_t_s1, x_t_s2)))
            pred_u_s1, pred_u_s2 = pred_u_s.chunk(2)
            pred_u_s1 = F.softmax(pred_u_s1, dim=1)
            pred_u_s2 = F.softmax(pred_u_s2, dim=1)

            unknown_csl = unknown_csl_fn(logits_s1=pred_u_s1, logits_s2=pred_u_s2, pred_u_w_fp=pred_u_w_fp,
                                         mask_u_1=mask_u_w_1, mask_u_2=mask_u_w_2).cuda()

            known_csl = known_csl_fn(logits_s1=pred_u_s1, logits_s2=pred_u_s2, pred_u_w_fp=pred_u_w_fp,
                                     mask_u_1=mask_u_w_1, mask_u_2=mask_u_w_2, pred_u_w=pred_u_w).cuda()

            _, prt_loss = get_prototype_weight(center_feat, f_t_w, num_classes)

            adv_loss = domain_adv(f_s, f_t_w)

        else:
            unknown_csl = torch.tensor([0]).cuda()
            known_csl = torch.tensor([0]).cuda()
            prt_loss = torch.tensor([0]).cuda()
            esl_loss = torch.tensor([0]).cuda()
            adv_loss = torch.tensor([0]).cuda()

        sce_loss = F.cross_entropy(y_s, labels_s)

        ctl_loss = ctl_fn(f_t)

        csl_loss = unknown_csl + known_csl

        loss = sce_loss + esl_loss + csl_loss + 0.5 * (prt_loss + adv_loss + ctl_loss)
        if shift_value is not None:
            loss = loss + shift_value

        losses.update(loss.item(), labels_s.size(0))
        sce_losses.update(sce_loss.item(), labels_s.size(0))
        esl_losses.update(esl_loss.item(), labels_s.size(0))
        csl_losses.update(csl_loss.item(), labels_s.size(0))
        prt_losses.update(prt_loss.item(), labels_s.size(0))
        adv_losses.update(adv_loss.item(), labels_s.size(0))
        ctl_losses.update(ctl_loss.item(), labels_s.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    if shift_on:
        print('[Shift-MFCL] epoch={} {}'.format(epoch, model.on_shift_epoch_end()),
              flush=True)


def validate(args, model, dataloader, src_flg=False):
    model.eval()

    gt_label_stack = []
    pred_cls_stack = []

    if src_flg:
        class_list = args.src_classes
        open_flg = False
    else:
        class_list = args.src_classes + [max(args.src_classes) + 1]
        open_flg = True

    for imgs_test, imgs_label in dataloader:
        imgs_test = imgs_test.cuda()

        pred_cls = model(imgs_test)
        gt_label_stack.append(imgs_label)
        pred_cls_stack.append(pred_cls.cpu())

    gt_label_all = torch.cat(gt_label_stack, dim=0)
    pred_cls_all = torch.cat(pred_cls_stack, dim=0)

    h_score, all_acc, known_acc, unknown_acc = compute_h_score(args, class_list, gt_label_all, pred_cls_all, open_flg)

    print(' * All {all:.5f} Known {known:.5f} Unknown {unknown:.5f} H-score {h_score:.5f}'
          .format(all=all_acc, known=known_acc, unknown=unknown_acc, h_score=h_score))

    return h_score


