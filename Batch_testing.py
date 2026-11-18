import os
import sys
import json
import time
import random
import argparse
import traceback
import contextlib
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import torch

import datasets
from utils.metric import compute_h_score, compute_osda_metrics
from trainer import (
    main_mfcl, main_maosdan,
    build_model_for_method, load_model_weights, extract_features_and_logits,
    DEFAULT_CKPT, device,
)
from models.MAOSDAN_shift import MAOSDAN_Shift

_spec = importlib.util.spec_from_file_location(
    'dvc_he', Path(__file__).resolve().parent / 'DVC-HE.py')
entry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(entry)

METHODS = ['og_moe']

SRC_CONDITIONS = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [0, 1, 2, 3, 4, 5, 6, 7],
    [0, 1, 2, 3, 4, 5, 6],
    [0, 1, 2, 3, 4, 5],

    [0, 1, 2, 3, 4],
    [0, 1, 2, 3],
    [0, 1, 2],
    [0, 1],
]
TGT_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
SRC_NAME = [2]
TGT_NAME = [3]
DATA_DIR = os.environ.get(
    'DVC_HE_DATA_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))


RUN_SEEDS = [202401, 202402, 202403, 202404, 202405]

LOG_ROOT = 'logs/batch'
RESULT_XLSX = 'results/batch_results.xlsx'
RESULT_JSON = 'results/batch_results.json'


class _Tee(object):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


@contextlib.contextmanager
def tee_console(log_path):
    d = os.path.dirname(os.path.abspath(log_path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(log_path, 'a', encoding='utf-8', buffering=1) as fp:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(old_out, fp), _Tee(old_err, fp)
        try:
            yield log_path
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def _append_record(path, record):
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    except Exception as e:
        print('[warn] 写 run 记录失败: {} ({})'.format(path, e), flush=True)


def _read_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _write_json(path, obj):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        print('[warn] 写 {} 失败: {}'.format(path, e), flush=True)


def _remove_file(path):
    try:
        if os.path.isfile(path):
            os.remove(path)
    except Exception as e:
        print('[warn] 删除 {} 失败: {}'.format(path, e), flush=True)


DONE_MARKER = 'DONE.json'


def _fingerprint(args):
    ignore = ('log', 'phase', 'checkpoint',
              'tsne_perplexity', 'tsne_iter', 'tsne_max_samples')
    payload = json.dumps({k: v for k, v in vars(args).items() if k not in ignore},
                         sort_keys=True, default=str)
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]


def main_maosdan_shift(args, train_source_loader, train_target_loader,
                       val_source_loader, val_target_loader, num_classes):
    return main_maosdan(args, train_source_loader, train_target_loader,
                        val_source_loader, val_target_loader, num_classes,
                        model_cls=MAOSDAN_Shift)


def main_og_moe(args, train_source_loader, train_target_loader,
                val_source_loader, val_target_loader, num_classes):
    from og_moe_trainer import main_og_moe as _train
    return _train(args, train_source_loader, train_target_loader,
                  val_source_loader, val_target_loader, num_classes)


MAIN_FUNCS = {
    'mfcl': main_mfcl,
    'maosdan_shift': main_maosdan_shift,
    'og_moe': main_og_moe,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_args(method, src_classes, log_dir, seed, extra=''):
    build_parser = entry.build_parser
    apply_method_defaults = entry.apply_method_defaults

    argv = ['--method', method, '--log', log_dir, '--seed', str(seed)]
    if extra:
        argv += extra.split()
    args = build_parser().parse_args(argv)

    apply_method_defaults(args)
    args.src_classes = list(src_classes)
    args.tgt_classes = list(TGT_CLASSES)
    args.src_name = list(SRC_NAME)
    args.tgt_name = list(TGT_NAME)
    args.workers = 4
    if DATA_DIR:
        args.data_dir = DATA_DIR
    args.phase = 'train'
    args.checkpoint = None
    return args


def _key_hparams(args):
    keys = ('lr', 'stage2_lr', 'pretrain_epoch', 'epochs',
            'anchor_weight', 'anchor_conf', 'anchor_decay_epochs', 'source_proto_conf',

            'proto_gamma', 'reliability_gamma', 'drift_budget', 'dual_lr', 'relation_weight',

            'maosdan_pretrain_epoch', 'maosdan_ramp_epochs', 'maosdan_aug_source_weight',
            'maosdan_stable_warmup', 'maosdan_stable_grad_clip',

            'maosdan_open_threshold', 'maosdan_open_alpha', 'maosdan_open_max',

            'trans_weight', 'adaptive_osda', 'adaptive_gain',
            'trans_weight_min', 'trans_weight_max',

            'maosdan_shift_w', 'maosdan_shift_band_q', 'maosdan_shift_pull_frac',
            'maosdan_shift_dir', 'maosdan_shift_margin_target', 'maosdan_shift_balance',
            'maosdan_shift_pull_w', 'maosdan_shift_push_w', 'maosdan_shift_start_epoch')
    return ' '.join('{}={}'.format(k, getattr(args, k, '-')) for k in keys)


def build_loaders(src_classes):
    data_loaders = getattr(datasets, 'SDUST')
    try:
        loaders = data_loaders(DATA_DIR, SRC_NAME, TGT_NAME, list(src_classes),
                               TGT_CLASSES, 32, 4).create_loaders()
    except TypeError:
        loaders = data_loaders(DATA_DIR, SRC_NAME, TGT_NAME, list(src_classes),
                               TGT_CLASSES, 32).create_loaders()
    return loaders


def find_ckpt(args):
    ck = getattr(args, 'checkpoint', None)
    if ck and os.path.isfile(ck):
        return ck
    ckpt_dir = os.path.join(args.log, 'checkpoints')
    name = DEFAULT_CKPT.get(args.method, 'best.pth')
    p = os.path.join(ckpt_dir, name)
    if os.path.isfile(p):
        return p

    if os.path.isdir(ckpt_dir):
        pth = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
        if pth:
            pth.sort(key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)))
            return os.path.join(ckpt_dir, pth[-1])
    return None


def _h_at_threshold(score, thr, gt_known, known_ok):
    score = score.float().cpu()
    gt_known = gt_known.cpu()
    known_ok = known_ok.cpu()
    pred_unk = score > float(thr)
    hit = (~pred_unk) & known_ok & gt_known
    os_star = float(hit.sum()) / max(int(gt_known.sum()), 1)
    uk = float((pred_unk & ~gt_known).sum()) / max(int((~gt_known).sum()), 1)
    h = 0.0 if (os_star + uk) <= 0 else 2.0 * os_star * uk / (os_star + uk)
    return h, os_star, uk, float(pred_unk.float().mean())


def _oracle_h_score(score, gt_known, known_ok, grid=None):
    if grid is None:
        grid = torch.arange(0.0, 1.0001, 0.01)
    score = score.float().cpu()
    gt_known = gt_known.cpu()
    known_ok = known_ok.cpu()
    best = -1.0
    for q in grid:
        pred_unk = score > torch.quantile(score, float(q))
        hit = (~pred_unk) & known_ok & gt_known
        os_star = float(hit.sum()) / max(int(gt_known.sum()), 1)
        uk = float((pred_unk & ~gt_known).sum()) / max(int((~gt_known).sum()), 1)
        if os_star + uk > 0:
            best = max(best, 2.0 * os_star * uk / (os_star + uk))
    return best


def _auc_rank(score_id, gt_known):
    s = score_id.float().cpu()
    pos = (~gt_known.cpu())
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = torch.argsort(s, descending=True)
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(1, s.numel() + 1, dtype=s.dtype)
    uniq, inv, cnt = torch.unique(s, return_inverse=True, return_counts=True)
    for i in range(uniq.numel()):
        if int(cnt[i]) > 1:
            m = (inv == i)
            ranks[m] = ranks[m].mean()
    return (float(ranks[pos].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


_NCI_HEAD_PATHS = (
    ('head',),
    ('viper', 'encoder', 'head'),
    ('encoder', 'head'),
    ('base', 'head'),
)


def _nci_head_weight(model):
    for path in _NCI_HEAD_PATHS:
        obj = model
        for p in path:
            obj = getattr(obj, p, None)
            if obj is None:
                break
        w = getattr(obj, 'weight', None) if obj is not None else None
        if w is not None and w.dim() == 2:
            return w.detach()
    return None


def _nci_score(feats, logits, w, mu, alpha):
    f = feats.float()

    w = w.to(f.device)

    K = int(logits.shape[1]) - 1
    if K <= 0 or int(w.shape[0]) < K:
        raise ValueError('NCI: logits 列数 {} 与 head 行数 {} 不匹配'
                         .format(int(logits.shape[1]), int(w.shape[0])))
    y_hat = logits[:, :K].argmax(dim=1)
    d = f - mu.unsqueeze(0)
    denom = d.norm(dim=1).clamp_min(1e-12)
    w_y = w[:K][y_hat].to(f.dtype)
    return (w_y * d).sum(dim=1) / denom + float(alpha) * f.abs().sum(dim=1)


@torch.no_grad()
def evaluate(args, val_target_loader, num_classes, src_loader=None):
    model = build_model_for_method(args, num_classes)
    ckpt_path = find_ckpt(args)
    if ckpt_path is None:
        raise FileNotFoundError('未找到检查点: {}/checkpoints/'.format(args.log))
    load_model_weights(model, ckpt_path)
    model.to(device).eval()

    feats_t, logits_t, gt_t = extract_features_and_logits(model, val_target_loader, device)

    ens_with = getattr(args, 'ensemble_with', None)
    _ens_w_eff = None
    if ens_with and str(ens_with).lower() not in ('none', ''):
        import copy as _copy
        ens_with = str(ens_with)
        _args2 = _copy.deepcopy(args)
        _args2.method = ens_with
        _m2 = build_model_for_method(_args2, num_classes)
        _state = os.path.basename(os.path.normpath(args.log))
        _root = getattr(args, 'ensemble_log_root', None) or 'logs_rf1_methods'
        _p2 = os.path.join(_root, ens_with, _state, 'checkpoints',
                           DEFAULT_CKPT.get(ens_with, 'best.pth'))
        if not os.path.isfile(_p2):
            _p2 = os.path.join(_root, _state, 'checkpoints',
                               DEFAULT_CKPT.get(ens_with, 'best.pth'))
        if not os.path.isfile(_p2):
            raise FileNotFoundError('集成检查点不存在: {}'.format(_p2))
        load_model_weights(_m2, _p2)
        _m2.to(device).eval()
        _f2, _l2, _g2 = extract_features_and_logits(_m2, val_target_loader, device)
        _w_arg = getattr(args, 'ensemble_w', 0.3)
        if str(_w_arg).strip().lower() == 'auto':
            _n_t = max(len(getattr(args, 'tgt_classes', []) or []), 1)
            _n_s = len(getattr(args, 'src_classes', []) or [])
            _pi_u = max(0.0, min(1.0, (_n_t - _n_s) / float(_n_t)))
            if _pi_u <= 0.45:
                _w = 0.1
            elif _pi_u < 0.55:
                _w = 0.5
            else:
                _w = 0.9
            print('[ensemble] auto: pi_u={:.3f} -> w({})={:.2f}'.format(
                _pi_u, args.method, _w), flush=True)
        else:
            _w = float(_w_arg)
        _p1 = torch.softmax(logits_t.float(), dim=1)
        _p2s = torch.softmax(_l2.float(), dim=1)
        _fused = _w * _p1 + (1.0 - _w) * _p2s
        logits_t = torch.log(_fused.clamp_min(1e-12))
        _ens_w_eff = _w
        print('[ensemble] {} w={:.2f} + {} w={:.2f} | ckpt={}'.format(
            args.method, _w, ens_with, 1.0 - _w, _p2), flush=True)

    num_out = int(logits_t.shape[1])
    unk_gt = num_classes - 1
    unk_idx = num_out - 1
    gt_t_map = torch.where(gt_t >= unk_gt, torch.tensor(unk_idx), gt_t)
    class_list = list(range(unk_gt)) + [unk_idx]

    h_score, all_acc, known_acc, unknown_acc = compute_h_score(
        args, class_list, gt_t_map, logits_t, open_flag=True)
    osda = compute_osda_metrics(logits_t, gt_t_map, class_list)

    out = {
        'os_star': float(osda['os_star']),
        'uk': float(osda['uk']),
        'os': float(osda['os']),
        'h': float(osda['h']),
        'known_acc': float(known_acc),
        'unknown_acc': float(unknown_acc),
        'all_acc': float(all_acc),
    }
    if ens_with and str(ens_with).lower() not in ('none', ''):
        out['ens_with'] = str(ens_with)
        out['ens_w'] = float(_ens_w_eff) if _ens_w_eff is not None else None

    if src_loader is not None and int(getattr(args, 'mgpl_per_class', 0) or 0) >= 1:
        from models.multiple_gauss_proto import MultipleGaussianPrototypes
        feats_s, _logits_s, gt_s = extract_features_and_logits(model, src_loader, device)
        mg = MultipleGaussianPrototypes(
            unk_gt,
            per_class=int(getattr(args, 'mgpl_per_class', 1) or 1),
            radius_q=float(getattr(args, 'mgpl_radius_q', None) or 0.9),
            radius_clip=float(getattr(args, 'mgpl_radius_clip', None) or 3.0),
            shrink=float(getattr(args, 'mgpl_shrink', None) or 0.1),
            temp=float(getattr(args, 'mgpl_temp', None) or 0.1),
            seed=int(getattr(args, 'seed', 0) or 0),
            metric=str(getattr(args, 'mgpl_metric', None) or 'cos'),
        )
        mg.observe_source(feats_s, gt_s)
        if mg.on_epoch_end():
            _d_s, ratio_s, _c_s, _j_s = mg.score(feats_s)
            g = getattr(args, 'mgpl_gamma', None)
            if g is None or float(g) <= 0:
                q = float(getattr(args, 'mgpl_calib_q', None) or 0.95)
                gamma = float(torch.quantile(ratio_s, q))
            else:
                gamma = float(g)
            logits_m = mg.open_set_logits(feats_t, gamma).to(logits_t.device).float()
            osda_m = compute_osda_metrics(logits_m, gt_t_map, class_list)
            dmin_t, ratio_t, _c_t, _j_t = mg.score(feats_t)
            gt_known = (gt_t < unk_gt)
            known_ok = (logits_t[:, :unk_gt].argmax(dim=1) == gt_t)

            tgt_cls = getattr(args, 'tgt_classes', None)
            src_cls = getattr(args, 'src_classes', None)
            n_tgt = len(tgt_cls) if isinstance(tgt_cls, (list, tuple)) else 10
            n_src = len(src_cls) if isinstance(src_cls, (list, tuple)) else unk_gt
            pi_u = max(0.0, min(1.0, (n_tgt - n_src) / float(max(n_tgt, 1))))

            thr_d = torch.quantile(dmin_t, 1.0 - pi_u)
            h_d, os_d, uk_d, pu_d = _h_at_threshold(dmin_t, thr_d, gt_known, known_ok)

            thr_r = torch.quantile(ratio_t, 1.0 - pi_u)
            h_r, os_r, uk_r, pu_r = _h_at_threshold(ratio_t, thr_r, gt_known, known_ok)
            out.update({
                'mgpl_J': int(mg.J),
                'mgpl_gamma': gamma,
                'os_star_mgpl': float(osda_m['os_star']),
                'uk_mgpl': float(osda_m['uk']),
                'h_mgpl': float(osda_m['h']),
                'oracle_h_mgpl': _oracle_h_score(ratio_t, gt_known, known_ok),
                'pi_u': pi_u,
                'theta_dmin_piu': float(thr_d),
                'h_dmin_piu': h_d,
                'os_star_dmin_piu': os_d,
                'uk_dmin_piu': uk_d,
                'pred_unk_dmin_piu': pu_d,
                'oracle_h_dmin': _oracle_h_score(dmin_t, gt_known, known_ok),
                'h_ratio_piu': h_r,
                'os_star_ratio_piu': os_r,
                'uk_ratio_piu': uk_r,
                'ratio_src_p50': float(ratio_s.median()),
                'ratio_tgt_p50': float(ratio_t.median()),
            })

    nci_alpha = getattr(args, 'nci_alpha', None)
    w_all = None if nci_alpha is None else _nci_head_weight(model)
    if nci_alpha is not None and src_loader is not None and w_all is not None:
        feats_s, logits_s, _gt_s = extract_features_and_logits(model, src_loader, device)
        K_known = int(logits_t.shape[1]) - 1
        if K_known <= 0 or int(w_all.shape[0]) < K_known:
            raise ValueError('NCI: logits 列数 {} 与 head 行数 {} 不匹配'
                             .format(int(logits_t.shape[1]), int(w_all.shape[0])))
        mu = feats_s.float().mean(dim=0)
        alpha = float(nci_alpha)
        q = float(getattr(args, 'nci_thr_q', None) or 0.05)
        s_src = _nci_score(feats_s, logits_s, w_all, mu, alpha)
        s_tgt = _nci_score(feats_t, logits_t, w_all, mu, alpha)

        tau = float(torch.quantile(s_src, q))
        scale = max(float(s_src.std()), 1e-6)

        k_known = torch.log_softmax(logits_t[:, :K_known].float(), dim=1)
        u_col = k_known.max(dim=1, keepdim=True).values + ((tau - s_tgt) / scale).unsqueeze(1)
        logits_nci = torch.cat([k_known, u_col], dim=1).to(logits_t.dtype)
        osda_nci = compute_osda_metrics(logits_nci, gt_t_map, class_list)

        gt_known = (gt_t < unk_gt)
        known_ok = (logits_t[:, :K_known].argmax(dim=1) == gt_t)
        h_chk, os_chk, uk_chk, pu_chk = _h_at_threshold(-s_tgt, -tau, gt_known, known_ok)
        out.update({
            'nci_alpha': alpha,
            'nci_thr_q': q,
            'nci_tau': tau,
            'nci_scale': scale,
            'nci_os_star': float(osda_nci['os_star']),
            'nci_uk': float(osda_nci['uk']),
            'nci_os': float(osda_nci['os']),
            'nci_h': float(osda_nci['h']),
            'nci_h_check': h_chk,
            'nci_os_star_check': os_chk,
            'nci_uk_check': uk_chk,
            'nci_pred_unk': pu_chk,
            'nci_auc': _auc_rank(s_tgt, gt_known),
            'nci_oracle_h': _oracle_h_score(-s_tgt, gt_known, known_ok),
            'nci_src_p50': float(s_src.median()),
            'nci_tgt_p50': float(s_tgt.median()),
            'nci_s_known_p50': float(s_tgt[gt_known].median()) if bool(gt_known.any()) else float('nan'),
            'nci_s_unknown_p50': float(s_tgt[~gt_known].median()) if bool((~gt_known).any()) else float('nan'),
        })

    return out


def run_batch(methods, cond_idxs, runs, skip_existing, only_eval, extra='', tag='',
              reuse_any=False, seed_offset=0):
    os.makedirs('results', exist_ok=True)
    os.makedirs(LOG_ROOT, exist_ok=True)
    sfx = '_{}'.format(tag) if tag else ''
    console_log = os.path.join(LOG_ROOT, '_console{}_{}.log'.format(sfx, time.strftime('%Y%m%d_%H%M%S')))
    print('整批控制台日志: {}'.format(os.path.abspath(console_log)), flush=True)
    print('调参用 run 记录: {}'.format(
        os.path.abspath(os.path.join(LOG_ROOT, '_records{}.jsonl'.format(sfx)))), flush=True)
    with tee_console(console_log):
        _run_batch_impl(methods, cond_idxs, runs, skip_existing, only_eval, extra, sfx,
                        reuse_any, seed_offset)


def _run_batch_impl(methods, cond_idxs, runs, skip_existing, only_eval, extra='', sfx='',
                    reuse_any=False, seed_offset=0):
    record_path = os.path.join(LOG_ROOT, '_records{}.jsonl'.format(sfx))

    results = {}
    for method in methods:
        results.setdefault(method, {})

    total = len(methods) * len(cond_idxs) * runs
    done = 0
    t0 = time.time()

    for ci in cond_idxs:
        src_classes = SRC_CONDITIONS[ci]
        n_known = len(src_classes)
        loaders = build_loaders(src_classes)
        (train_source_loader, train_target_loader,
         val_source_loader, val_target_loader, num_classes) = loaders
        print('\n################ 数据条件 #{}/{}  src_classes={} (已知{}类/未知{}类) num_classes={}'
              .format(ci + 1, len(SRC_CONDITIONS), src_classes, n_known,
                      10 - n_known, num_classes), flush=True)

        for method in methods:
            for r in range(runs):
                seed = RUN_SEEDS[r + seed_offset]
                log_dir = os.path.join(LOG_ROOT, method, 'src{}'.format(n_known), 'seed{}'.format(seed))
                os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)
                args = build_args(method, src_classes, log_dir, seed, extra)
                if r == 0:
                    print('    [{}|src{}] 生效超参: {}'
                          .format(method, n_known, _key_hparams(args)), flush=True)

                done += 1
                run_tag = '{}/src{}/run{}'.format(method, n_known, r + seed_offset + 1)
                print('[{}/{}] >>> {}'.format(done, total, run_tag), flush=True)

                ckpt_path = find_ckpt(args)
                fp = _fingerprint(args)
                marker = os.path.join(log_dir, DONE_MARKER)
                prev = _read_json(marker)
                marker_ok = bool(prev) and prev.get('fingerprint') == fp
                if skip_existing and ckpt_path and not marker_ok and not reuse_any:
                    print('  [warn] 已有检查点是{}（无完成标记），{}: {}'.format(
                        '被中断的未完成训练' if not prev else '用别的超参训练的',
                        '--only-eval 只评估它，不重训' if only_eval else '按默认策略重新训练',
                        os.path.relpath(log_dir)), flush=True)
                need_train = (not only_eval) and not (
                    skip_existing and ckpt_path and (marker_ok or reuse_any))

                run_log = os.path.join(log_dir, 'train_{}_{}.txt'.format(method, args.epochs))
                t_run = time.time()
                m, err = None, None
                with tee_console(run_log):
                    print('=' * 78)
                    print('run      : {}'.format(run_tag))
                    print('时间     : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
                    print('条件     : src_classes={} (已知{}类/未知{}类)'
                          .format(src_classes, n_known, 10 - n_known))
                    print('seed     : {}    need_train={}  (False 表示 --only-eval 或复用已有检查点)'
                          .format(seed, need_train))
                    print('ckpt     : {}'.format(ckpt_path if ckpt_path else '(尚无检查点)'))
                    print('生效超参 : {}'.format(_key_hparams(args)))
                    print('extra    : {}'.format(extra if extra else '(none)'))
                    print('超参指纹 : {}'.format(fp))
                    print('=' * 78, flush=True)

                    if need_train:
                        _remove_file(marker)
                        set_seed(seed)
                        try:
                            MAIN_FUNCS[method](args, train_source_loader, train_target_loader,
                                               val_source_loader, val_target_loader, num_classes)
                        except Exception as e:
                            err = e
                            print('  [ERROR] 训练失败 {}: {}'.format(run_tag, e), flush=True)
                            traceback.print_exc()

                    if err is None:
                        try:
                            m = evaluate(args, val_target_loader, num_classes)
                        except Exception as e:
                            err = e
                            print('  [ERROR] 评估失败 {}: {}'.format(run_tag, e), flush=True)
                            traceback.print_exc()

                    if m is not None:
                        print('  -> OS*={:.4f} UK={:.4f} OS={:.4f} H={:.4f}'
                              .format(m['os_star'], m['uk'], m['os'], m['h']), flush=True)
                elapsed = time.time() - t_run

                if need_train and err is None and m is not None:
                    _write_json(marker, {
                        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'method': method, 'cond_idx': ci, 'n_known': n_known,
                        'run': r + 1, 'seed': seed, 'epochs': args.epochs,
                        'fingerprint': fp, 'hparams': _key_hparams(args),
                        'metrics': m, 'elapsed_s': round(elapsed, 1),
                    })

                _append_record(record_path, {
                    'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'method': method, 'cond_idx': ci, 'src_classes': list(src_classes),
                    'n_known': n_known, 'n_unknown': 10 - n_known,
                    'run': r + 1, 'seed': seed, 'need_train': bool(need_train),
                    'extra': extra, 'hparams': _key_hparams(args), 'fingerprint': fp,
                    'metrics': m, 'elapsed_s': round(elapsed, 1),
                    'error': None if err is None else repr(err),
                    'log': os.path.abspath(run_log),
                    'args': vars(args),
                })

                if err is not None:
                    results[method].setdefault(ci, []).append(None)
                    continue

                results[method].setdefault(ci, []).append(m)
                print('  -> OS*={:.4f} UK={:.4f} OS={:.4f} H={:.4f}  (单run {:.1f}s, 累计 {:.1f}s)'
                      .format(m['os_star'], m['uk'], m['os'], m['h'], elapsed,
                              time.time() - t0), flush=True)

                save_results(results, cond_idxs)

    save_results(results, cond_idxs)
    write_excel(results, cond_idxs, methods)
    print('\n==== 完成 ==== 总耗时 {:.1f}s  结果已写入 {} 与 {}'.format(
        time.time() - t0, RESULT_XLSX, RESULT_JSON), flush=True)


def save_results(results, cond_idxs):
    with open(RESULT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _safe_sheet_title(t):
    for ch in '*?:/\\[]':
        t = t.replace(ch, '_')
    return t


def _mean(vals, key):
    xs = [v[key] for v in vals if v is not None]
    return float(np.mean(xs)) if xs else None


def write_excel(results, cond_idxs, methods):
    cond_labels = ['K={}(U={})'.format(len(SRC_CONDITIONS[ci]), 10 - len(SRC_CONDITIONS[ci]))
                   for ci in cond_idxs]

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment
        HAVE_XLSX = True
    except ImportError:
        HAVE_XLSX = False

    metric_sheets = [('OS*', 'os_star'), ('UK', 'uk'), ('OS', 'os'), ('H-score', 'h')]

    if HAVE_XLSX:
        wb = Workbook()
        wb.remove(wb.active)

        ws0 = wb.create_sheet('Summary')
        ws0.append(['Batch OSDA 测试结果'])
        ws0.append(['行=模型, 列=数据条件(已知类数 K / 未知类数 U), 值=多次运行均值(默认3次)'])
        ws0.append(['目标任务类固定为 0..9; 源任务类见各列 K'])
        ws0.append(['指标: OS* 已知类准确率, UK 未知类准确率, OS 总体准确率, H H-score(OS*与UK调和均值)'])
        ws0.append([])

        for title, key in metric_sheets:
            ws = wb.create_sheet(_safe_sheet_title(title))
            ws.append(['Method'] + cond_labels)
            for c in ws[1]:
                c.font = Font(bold=True)
            for method in methods:
                row = [method]
                for ci in cond_idxs:
                    vals = results.get(method, {}).get(ci, [])
                    m = _mean(vals, key)
                    row.append(round(m, 4) if m is not None else None)
                ws.append(row)

            for col in ws.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 8), 22)

        wsr = wb.create_sheet('Raw')
        wsr.append(['Method', 'Condition(src)', 'Known', 'Unknown', 'Run',
                    'OS*', 'UK', 'OS', 'H-score'])
        for method in methods:
            for ci in cond_idxs:
                src = SRC_CONDITIONS[ci]
                vals = results.get(method, {}).get(ci, [])
                for r, v in enumerate(vals, 1):
                    if v is None:
                        wsr.append([method, str(src), len(src), 10 - len(src), r] + [None] * 4)
                    else:
                        wsr.append([method, str(src), len(src), 10 - len(src), r,
                                    round(v['os_star'], 4), round(v['uk'], 4),
                                    round(v['os'], 4), round(v['h'], 4)])

        wb.save(RESULT_XLSX)
        print('Excel 已保存: {}'.format(RESULT_XLSX), flush=True)
    else:
        import csv
        for title, key in metric_sheets:
            fname = 'results/batch_{}.csv'.format(title.replace('*', 'star').replace('-', ''))
            with open(fname, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['Method'] + cond_labels)
                for method in methods:
                    row = [method]
                    for ci in cond_idxs:
                        m = _mean(results.get(method, {}).get(ci, []), key)
                        row.append(round(m, 4) if m is not None else None)
                    w.writerow(row)
            print('CSV 已保存(未安装 openpyxl): {}'.format(fname), flush=True)


def parse_args():
    p = argparse.ArgumentParser(description='批量测试 OSDA 模型')

    p.add_argument('--methods', nargs='+', default=METHODS, choices=sorted(MAIN_FUNCS),
                   help='要测试的模型（默认 {}）'.format(METHODS))
    p.add_argument('--cond-idx', nargs='+', type=int, default=list(range(len(SRC_CONDITIONS))),
                   help='数据条件索引 0..7（对应 |src|=9..2）')
    p.add_argument('--runs', type=int, default=len(RUN_SEEDS),
                   help='每个条件重复次数（默认 5）')
    p.add_argument('--seed-offset', type=int, default=0,
                   help='从 RUN_SEEDS 的第几个种子开始取（默认 0）。'
                        '配合 --runs 可实现「先跑前 3 个种子、之后再补后 2 个」，'
                        '例如 --seed-offset 3 --runs 2 → seed 202404/202405')
    p.add_argument('--no-skip', action='store_true',
                   help='关闭检查点复用，强制重新训练')
    p.add_argument('--only-eval', action='store_true',
                   help='只评估已存在的检查点，不训练')
    p.add_argument('--extra', default='',
                   help='额外透传给 DVC-HE.py parser 的参数串，例如 '
                        '--extra "--anchor-weight 0.275 --anchor-conf 0.85 --anchor-decay-epochs 6.0"')
    p.add_argument('--tag', default='',
                   help='结果文件名后缀，例如 mfcl_stable → results/batch_results_mfcl_stable.xlsx/json；'
                        '不传则用原文件名（会覆盖已有结果）')
    p.add_argument('--reuse-any', action='store_true',
                   help='忽略「完成标记 / 超参指纹」，只要目录里有检查点就复用（兼容加标记之前的老实验目录；'
                        '默认行为是：没有标记或超参变过就重新训练）')
    return p.parse_args()


if __name__ == '__main__':
    a = parse_args()
    seed_offset = max(0, a.seed_offset)
    if seed_offset >= len(RUN_SEEDS):
        raise SystemExit('--seed-offset {} 超出 RUN_SEEDS 范围 (0..{})'
                         .format(seed_offset, len(RUN_SEEDS) - 1))
    runs = min(a.runs, len(RUN_SEEDS) - seed_offset)
    if runs + seed_offset > len(RUN_SEEDS):
        print('[warn] --seed-offset {} + --runs {} 超出 RUN_SEEDS，已截断为 {} 个种子'
              .format(seed_offset, a.runs, runs), flush=True)
    cond_idxs = [i for i in a.cond_idx if 0 <= i < len(SRC_CONDITIONS)]

    if a.tag:
        _sfx = '_{}'.format(a.tag)
        RESULT_XLSX = 'results/batch_results{}.xlsx'.format(_sfx)
        RESULT_JSON = 'results/batch_results{}.json'.format(_sfx)

    run_batch(a.methods, cond_idxs, runs,
              skip_existing=not a.no_skip, only_eval=a.only_eval,
              extra=a.extra, tag=a.tag, reuse_any=a.reuse_any,
              seed_offset=seed_offset)
