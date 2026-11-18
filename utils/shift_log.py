import csv
import json
import math
import os

import numpy as np

__all__ = [
    'ShiftLogger', 'even_subsample', 'stream_key',
    'EPOCH_COLUMNS', 'SAMPLE_COLUMNS', 'SKIP_COLUMNS', 'META_COLUMNS',
]


META_COLUMNS = [
    'run_key',
    'dataset',
    'source_domain',
    'target_domain',
    'known_classes',
    'unknown_classes',
    'seed',
    'method',
]


EPOCH_COLUMNS = META_COLUMNS + [
    'epoch',
    'shift_calls',
    'shift_steps',
    'n_samples',
    'n_band',
    'n_pull',
    'n_push',
    'n_cross_pull',
    'n_cross_push',
    'n_at_delta',
    'n_cls_switch',
    'n_class_rejected',
    'n_cap_pull',
    'n_cap_push',
    'gate_in_pull',
    'gate_out_pull',
    'gate_in_push',
    'gate_out_push',
    'eps_pull_mean',
    'eps_push_mean',
    'L_pull',
    'L_push',
    'L_shift',
    'lam_pull',
    'lam_push',
    'lam_pull_raw',
    'lam_push_raw',
    'r_pred_unk',
    'gap',
    'gW',
    'gz',
    'pull_per_cls',
    'push_cs_per_cls',
    'known_acc',
    'unknown_acc',
    'os_acc',
    'h_score',

    'n_active_iters',
    'pull_coverage',
    'push_coverage',
    'cross_success_rate_pull',
    'cross_success_rate_push',
    'band_coverage',
    'push_gate_enabled',
]


SAMPLE_COLUMNS = META_COLUMNS + [
    'epoch',
    'iteration',
    'stream_key',
    'side',
    'in_batch',
    'margin_before',
    'margin_after',
    'margin_change',
    'closed_pred',
    'prototype_pred',
    'shift_pred',
    'prototype_distance',
    'prototype_radius',
    'geom_ratio',
    'unknown_prob',
    'knn_unknown_prob',
    'in_boundary_band',
    'pass_geometry_gate',
    'pass_margin_gate',
    'pass_class_agreement',
    'pass_crossing_check',
    'pass_class_cap',
    'selected_final',
    'stage',
    'shift_distance',
    'reached_delta',
    'used_fallback',
    'target_margin',
    'candidate_label',
    'subsampled',
]


SKIP_COLUMNS = META_COLUMNS + [
    'epoch',
    'iteration',
    'reason',
    'n_samples',
    'n_band',
    'n_cross_pull',
    'n_cross_push',
    'lam_pull_raw',
    'lam_push_raw',
    'r_pred_unk',
    'gap',
]


SKIP_REASONS = (
    'before_start_epoch',
    'no_prototype',
    'batch_too_small',
    'band_too_small',
    'no_candidate',
    'loss_not_finite',
)


def stream_key(epoch, iteration, in_batch):
    return int(epoch) * 1000000 + int(iteration) * 1000 + int(in_batch)


def even_subsample(n, k):
    if k is None or k <= 0 or n <= k:
        return list(range(int(n)))
    idx = np.linspace(0, n - 1, int(k)).round().astype(np.int64)
    return np.unique(idx).tolist()


def _fmt(v):
    if v is None:
        return ''
    if isinstance(v, bool):
        return '1' if v else '0'
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ''
        return '%.10g' % v
    if isinstance(v, (list, tuple)):
        return ';'.join(str(x) for x in v)
    return str(v)


class ShiftLogger(object):
    def __init__(self, out_dir, meta, enable_samples=True, enable_skips=True,
                 every_epoch=1, verbose=True):
        self.out_dir = str(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self.meta = {k: _fmt(meta.get(k)) for k in META_COLUMNS}
        self.enable_samples = bool(enable_samples)
        self.enable_skips = bool(enable_skips)
        try:
            self.every_epoch = max(1, int(every_epoch))
        except (TypeError, ValueError):
            self.every_epoch = 1

        self.path_epoch = os.path.join(self.out_dir, 'epoch_metrics.csv')
        self.path_sample = os.path.join(self.out_dir, 'shift_samples.csv')
        self.path_skip = os.path.join(self.out_dir, 'skipped_iters.csv')

        self._n_epoch_rows = 0
        self._n_sample_rows = 0
        self._n_skip_rows = 0
        self._closed = False

        self._fh = {}
        self._fh['epoch'] = self._open(self.path_epoch, EPOCH_COLUMNS)
        if self.enable_samples:
            self._fh['sample'] = self._open(self.path_sample, SAMPLE_COLUMNS)
        if self.enable_skips:
            self._fh['skip'] = self._open(self.path_skip, SKIP_COLUMNS)

        if verbose:
            print('[shift-log] 输出目录: {}'.format(os.path.abspath(self.out_dir)))

    @staticmethod
    def _open(path, columns):
        has_body = os.path.exists(path) and os.path.getsize(path) > 0
        fp = open(path, 'a', encoding='utf-8', newline='', buffering=262144)
        w = csv.writer(fp, lineterminator='\n')
        if not has_body:
            w.writerow(columns)
        return (fp, w, columns)

    def _append(self, key, row_dict):
        fp, w, columns = self._fh[key]
        w.writerow([_fmt(row_dict.get(c)) for c in columns])

    def _flush_epoch(self):
        ent = self._fh.get('epoch')
        if ent is not None:
            try:
                ent[0].flush()
            except (OSError, ValueError):
                pass

    def log_epoch_row(self, row):
        if self._closed or 'epoch' not in self._fh:
            return
        ep = row.get('epoch')

        if self.every_epoch > 1 and ep is not None:
            if (int(ep) + 1) % self.every_epoch != 0:
                return
        full = dict(self.meta)
        full.update(row)
        self._append('epoch', full)
        self._flush_epoch()
        self._n_epoch_rows += 1

    def log_sample_rows(self, rows):
        if self._closed or not self.enable_samples or not rows:
            return
        fp, w, columns = self._fh['sample']
        meta = self.meta

        w.writerows([[_fmt(r.get(c, meta.get(c))) for c in columns] for r in rows])
        self._n_sample_rows += len(rows)

    def log_skip_row(self, row):
        if self._closed or not self.enable_skips:
            return
        full = dict(self.meta)
        full.update(row)
        self._append('skip', full)
        self._n_skip_rows += 1

    def stats(self):
        return {'epoch_rows': self._n_epoch_rows,
                'sample_rows': self._n_sample_rows,
                'skip_rows': self._n_skip_rows}

    def close(self):
        if self._closed:
            return
        for fp, _w, _c in self._fh.values():
            try:
                fp.flush()
                fp.close()
            except (OSError, ValueError):
                pass
        self._fh = {}
        self._closed = True

    @staticmethod
    def dump_run_config(path, cfg, extra=None):
        payload = dict(extra or {})
        payload['config'] = dict(cfg or {})
        payload['_note_stream_key'] = (
            'shift_samples.csv 的 stream_key = epoch*1e6 + iter*1e3 + in_batch，'
            '是训练流定位符，不是全局数据集索引；全局索引见 eval_samples.npz 的 sample_ids。')
        payload['_note_target_labels'] = (
            '目标域真实标签仅用于离线评估与 eval_samples.npz 导出，'
            '不进入任何训练损失、样本选择或日志选择流程。')
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2, sort_keys=True)
        return path
