import math

import torch
import torch.nn.functional as F

from alignment.osbp import ImageClassifier

__all__ = ['MAOSDAN_Shift']


_SHIFT_SRC_CAP = 8192
_SHIFT_BANK_CAP_MULT = 4


class MAOSDAN_Shift(ImageClassifier):
    def __init__(
        self,
        backbone,
        num_classes,
        bottleneck_dim=256,
        maosdan_shift_w=0.5,
        maosdan_shift_band_q=1.0,
        maosdan_shift_pull_frac=0.5,
        maosdan_shift_dir='decision',
        maosdan_shift_margin_target=1.0,
        maosdan_shift_eps_max=2.0,
        maosdan_shift_eps_rel=0.5,
        maosdan_shift_pull_w=None,
        maosdan_shift_push_w=None,
        maosdan_shift_balance='reliability',
        maosdan_shift_gap_scale=0.15,
        maosdan_shift_ev_closed=0.45,
        maosdan_shift_ev_knn=0.45,
        maosdan_shift_ev_geom=0.10,
        maosdan_shift_knn_k=8,
        maosdan_shift_bank_size=1024,
        maosdan_shift_proto_momentum=0.9,
        maosdan_shift_radius_q=0.9,
        maosdan_shift_radius_clip=3.0,
        maosdan_shift_min_support=8,
        maosdan_shift_min_side=1,
        maosdan_shift_start_epoch=1,
        maosdan_shift_orig_w=0.0,
        maosdan_shift_balance_clip=4.0,
        maosdan_shift_preserve_class=True,
        maosdan_shift_reliability_margin=0.0,
        maosdan_shift_class_cap=2.0,
        maosdan_shift_gate_pull_geom=1.0,
        maosdan_shift_gate_push_geom=0.0,
        maosdan_shift_gate_pull_knn=0.0,
        maosdan_shift_gate_push_pu=0.0,

        maosdan_shift_margin_pull_max=0.5,
        maosdan_shift_pull_proto_agree=True,
        maosdan_shift_ramp_epochs=10,
        maosdan_shift_coverage_ref=0.10,

        maosdan_shift_proto_gate_enable=True,

        shift_log_enable=False,
        shift_log_dir=None,
        shift_log_samples=False,
        shift_log_skips=True,
        shift_log_every_epoch=1,
        shift_log_every_iter=0,
        shift_log_max_samples=512,
        shift_log_meta=None,
        **kwargs
    ):
        super().__init__(backbone, num_classes, bottleneck_dim=bottleneck_dim, **kwargs)

        self.maosdan_shift_w = float(maosdan_shift_w)
        self.maosdan_shift_band_q = float(maosdan_shift_band_q)
        self.maosdan_shift_pull_frac = float(maosdan_shift_pull_frac)
        self.maosdan_shift_dir = str(maosdan_shift_dir)
        self.maosdan_shift_margin_target = float(maosdan_shift_margin_target)
        self.maosdan_shift_eps_max = float(maosdan_shift_eps_max)
        self.maosdan_shift_eps_rel = float(maosdan_shift_eps_rel)
        self.maosdan_shift_pull_w = None if maosdan_shift_pull_w is None else float(maosdan_shift_pull_w)
        self.maosdan_shift_push_w = None if maosdan_shift_push_w is None else float(maosdan_shift_push_w)
        self.maosdan_shift_balance = str(maosdan_shift_balance)
        self.maosdan_shift_gap_scale = float(maosdan_shift_gap_scale)
        self.maosdan_shift_ev_closed = float(maosdan_shift_ev_closed)
        self.maosdan_shift_ev_knn = float(maosdan_shift_ev_knn)
        self.maosdan_shift_ev_geom = float(maosdan_shift_ev_geom)
        self.maosdan_shift_knn_k = int(maosdan_shift_knn_k)
        self.maosdan_shift_bank_size = int(maosdan_shift_bank_size)
        self.maosdan_shift_proto_momentum = float(maosdan_shift_proto_momentum)
        self.maosdan_shift_radius_q = float(maosdan_shift_radius_q)
        self.maosdan_shift_radius_clip = float(maosdan_shift_radius_clip)
        self.maosdan_shift_min_support = int(maosdan_shift_min_support)
        self.maosdan_shift_min_side = int(maosdan_shift_min_side)
        self.maosdan_shift_start_epoch = int(maosdan_shift_start_epoch)
        self.maosdan_shift_orig_w = float(maosdan_shift_orig_w)
        self.maosdan_shift_balance_clip = float(maosdan_shift_balance_clip)
        self.maosdan_shift_preserve_class = bool(maosdan_shift_preserve_class)
        self.maosdan_shift_reliability_margin = float(maosdan_shift_reliability_margin)
        self.maosdan_shift_class_cap = float(maosdan_shift_class_cap)
        self.maosdan_shift_gate_pull_geom = float(maosdan_shift_gate_pull_geom)
        self.maosdan_shift_gate_push_geom = float(maosdan_shift_gate_push_geom)
        self.maosdan_shift_gate_pull_knn = float(maosdan_shift_gate_pull_knn)
        self.maosdan_shift_gate_push_pu = float(maosdan_shift_gate_push_pu)
        self.maosdan_shift_margin_pull_max = float(maosdan_shift_margin_pull_max)
        self.maosdan_shift_pull_proto_agree = bool(maosdan_shift_pull_proto_agree)
        self.maosdan_shift_ramp_epochs = int(maosdan_shift_ramp_epochs)
        self.maosdan_shift_coverage_ref = float(maosdan_shift_coverage_ref)
        self.maosdan_shift_proto_gate_enable = bool(maosdan_shift_proto_gate_enable)

        self._shift_agree_eff = bool(maosdan_shift_pull_proto_agree
                                     and maosdan_shift_proto_gate_enable)

        self._shift_K = int(num_classes) - 1

        self._shift_mu = None
        self._shift_rad = None
        self._shift_src_feat = []
        self._shift_src_lab = []

        self._shift_bank_z = None
        self._shift_bank_pu = None
        self._shift_tgt_feat = []
        self._shift_tgt_pu = []

        self._shift_epoch = -1
        self._shift_pi_u = 0.5
        self._shift_lam_pull = 0.0
        self._shift_lam_push = 0.0

        self._shift_r_ema = None

        self._shift_last = {}

        self._shift_acc_reset()

        self._shift_log_enable = bool(shift_log_enable)
        self._shift_log_samples_on = bool(shift_log_samples)
        self._shift_log_skips_on = bool(shift_log_skips)
        self._shift_log_every_iter = max(0, int(shift_log_every_iter or 0))
        self._shift_log_max_samples = int(shift_log_max_samples or 0)
        self._shift_log_meta = dict(shift_log_meta or {})

        self._shift_log_sample_buf = []

        self._shift_log_epoch_pending = None

        self._shift_iter_in_epoch = 0
        self._shift_logger = None
        if self._shift_log_enable:
            from utils.shift_log import ShiftLogger
            out_dir = shift_log_dir or 'logs/boundary_analysis'
            self._shift_logger = ShiftLogger(
                out_dir, self._shift_log_meta,
                enable_samples=self._shift_log_samples_on,
                enable_skips=self._shift_log_skips_on,
                every_epoch=int(shift_log_every_epoch or 1))

    @property
    def shift_enabled(self):
        return self.maosdan_shift_w > 0.0

    def _shift_lam(self, r=None):
        w0 = self.maosdan_shift_w
        mode = self.maosdan_shift_balance
        if mode == 'reliability':
            lam_p = w0
            lam_q = w0
        elif mode == 'gate':
            rr = self._shift_r_ema if r is None else float(r)
            if rr is None:
                lam_p = 0.0
                lam_q = 0.0
            else:
                gap = float(rr) - self._shift_pi_u
                s = max(1e-6, self.maosdan_shift_gap_scale)
                lam_p = w0 * min(1.0, max(0.0, gap) / s)
                lam_q = w0 * min(1.0, max(0.0, -gap) / s)
        elif mode == 'prior':
            lam_p = 2.0 * w0 * (1.0 - self._shift_pi_u)
            lam_q = 2.0 * w0 * self._shift_pi_u
        else:
            lam_p = w0
            lam_q = w0
        if self.maosdan_shift_pull_w is not None:
            lam_p = self.maosdan_shift_pull_w
        if self.maosdan_shift_push_w is not None:
            lam_q = self.maosdan_shift_push_w
        return lam_p, lam_q

    def set_shift_epoch(self, epoch, pi_u):
        self._shift_epoch = int(epoch)
        self._shift_pi_u = float(pi_u)
        self._shift_lam_pull, self._shift_lam_push = self._shift_lam()

        self._shift_iter_in_epoch = 0
        return self._shift_lam_pull, self._shift_lam_push

    def shift_describe(self):
        return ('w={:.3f} dir={} band_q={:.2f} pull_frac={:.2f} delta={:.2f} eps_max={:.2f} '
                'balance={} ev=(closed {:.2f}, knn {:.2f}, geom {:.2f}) knn_k={} '
                'start_epoch={} orig_w={:.2f} gate_pull={:.2f} gate_push={:.2f} min_side={} '
                'pi_u={:.3f} '
                'lambda_pull={:.4f} lambda_push={:.4f} mu_ready={} bank={}'
                ' DEPRECATED(ev_*,pull_frac,reliability_margin)'
                .format(self.maosdan_shift_w, self.maosdan_shift_dir,
                        self.maosdan_shift_band_q, self.maosdan_shift_pull_frac,
                        self.maosdan_shift_margin_target, self.maosdan_shift_eps_max,
                        self.maosdan_shift_balance,
                        self.maosdan_shift_ev_closed, self.maosdan_shift_ev_knn,
                        self.maosdan_shift_ev_geom, self.maosdan_shift_knn_k,
                        self.maosdan_shift_start_epoch, self.maosdan_shift_orig_w,
                        self.maosdan_shift_gate_pull_geom, self.maosdan_shift_gate_push_geom,
                        self.maosdan_shift_min_side,
                        self._shift_pi_u, self._shift_lam_pull, self._shift_lam_push,
                        int(self._shift_mu is not None),
                        0 if self._shift_bank_z is None else int(self._shift_bank_z.size(0))))

    @torch.no_grad()
    def observe_labeled_source(self, features, logits=None, labels=None):
        if not self.shift_enabled or features is None or labels is None:
            return
        z = F.normalize(features.detach().float(), p=2, dim=1).cpu()
        y = labels.detach().reshape(-1).long().cpu()
        keep = (y >= 0) & (y < self._shift_K)
        if not bool(keep.any()):
            return

        have = sum(t.size(0) for t in self._shift_src_feat)
        if have >= _SHIFT_SRC_CAP:
            return
        z, y = z[keep], y[keep]
        room = _SHIFT_SRC_CAP - have
        self._shift_src_feat.append(z[:room])
        self._shift_src_lab.append(y[:room])

    @torch.no_grad()
    def _shift_fit_prototypes(self):
        if not self._shift_src_feat:
            return False
        z = torch.cat(self._shift_src_feat, 0)
        y = torch.cat(self._shift_src_lab, 0)
        K = self._shift_K
        if z.size(0) < 4 * K:
            return False

        if self._shift_mu is None:
            self._shift_mu = torch.zeros(K, z.size(1), dtype=torch.float32)

        mom = max(0.0, min(1.0, self.maosdan_shift_proto_momentum))
        rad = torch.zeros(K, dtype=torch.float32)
        valid = []
        for c in range(K):
            sel = z[y == c]
            if sel.size(0) < self.maosdan_shift_min_support:
                continue
            mu_c = F.normalize(sel.mean(dim=0), p=2, dim=0)
            if self._shift_mu[c].norm() < 1e-8:
                self._shift_mu[c] = mu_c
            else:
                self._shift_mu[c] = F.normalize(
                    mom * self._shift_mu[c] + (1.0 - mom) * mu_c, p=2, dim=0)
            d = (1.0 - torch.mm(sel, self._shift_mu[c].unsqueeze(1)).squeeze(1)).clamp(min=0.0)
            r = float(torch.quantile(d, self.maosdan_shift_radius_q))
            med = float(d.median())
            if self.maosdan_shift_radius_clip > 0 and med > 0:
                r = min(r, self.maosdan_shift_radius_clip * med)
            rad[c] = max(r, 1e-6)
            valid.append(rad[c])

        if valid:
            med_r = float(torch.stack(valid).median())
            rad = torch.where(rad > 0.0, rad, torch.full_like(rad, med_r))
        else:
            rad.fill_(1.0)
        self._shift_rad = rad.clamp_min(1e-6)
        return True

    @torch.no_grad()
    def observe_target(self, features, logits):
        if not self.shift_enabled or features is None or logits is None:
            return

        have = sum(t.size(0) for t in self._shift_tgt_feat)
        if have >= _SHIFT_BANK_CAP_MULT * max(2, self.maosdan_shift_bank_size):
            return
        z = F.normalize(features.detach().float(), p=2, dim=1).cpu()
        pu = F.softmax(logits.detach().float(), dim=1)[:, self._shift_K].cpu()
        self._shift_tgt_feat.append(z)
        self._shift_tgt_pu.append(pu)

    def _shift_acc_reset(self):
        self._shift_acc = {
            'calls': 0, 'steps': 0, 'skip_band': 0, 'skip_gate': 0,
            'n_samples': 0, 'n_band': 0,
            'n_cross_pull': 0, 'n_cross_push': 0,
            'n_pull': 0, 'n_push': 0, 'n_cls_switch': 0, 'n_at_delta': 0,
            'n_class_rejected': 0, 'n_small_pull': 0, 'n_small_push': 0,
            'n_cap_pull': 0, 'n_cap_push': 0,
            'gate_in_pull': 0, 'gate_out_pull': 0,
            'gate_in_push': 0, 'gate_out_push': 0,
            'lam_pull_raw': None, 'lam_push_raw': None,
            'cov': 0.0, 'm_p50': 0.0,
            'eps_pull_mean': 0.0, 'eps_push_mean': 0.0,
            'L_pull': 0.0, 'L_push': 0.0, 'L_shift': 0.0,
            'gW': 0.0, 'gz': 0.0,
            'pull_per_cls': None, 'push_cs_per_cls': None,
            'r_pred_unk': None, 'gap': None,
            'lam_pull': 0.0, 'lam_push': 0.0,
        }

    def _shift_acc_add(self, st):
        a = self._shift_acc
        a['calls'] += 1
        a['n_samples'] += int(st.get('n_samples', 0))
        if 'skip' in st:
            if st.get('skip') == 'band':
                a['skip_band'] += 1
            else:
                a['skip_gate'] += 1
        else:
            a['steps'] += 1
        for k in ('n_band', 'n_cross_pull', 'n_cross_push', 'n_pull', 'n_push',
                  'n_cls_switch', 'n_class_rejected', 'n_small_pull', 'n_small_push',
                  'n_cap_pull', 'n_cap_push',
                  'gate_in_pull', 'gate_out_pull', 'gate_in_push', 'gate_out_push'):
            a[k] += int(st.get(k, 0))

        v_delta = st.get('n_at_delta')
        if v_delta is not None and int(v_delta) >= 0:
            a['n_at_delta'] += int(v_delta)
        for k in ('cov', 'm_p50', 'eps_pull_mean', 'eps_push_mean',
                  'L_pull', 'L_push', 'L_shift', 'gW', 'gz'):
            a[k] += float(st.get(k, 0.0))
        for k in ('pull_per_cls', 'push_cs_per_cls'):
            v = st.get(k)
            if v:
                if a[k] is None:
                    a[k] = [0] * len(v)
                for i, x in enumerate(v):
                    if i < len(a[k]):
                        a[k][i] += int(x)
        for k in ('r_pred_unk', 'gap', 'lam_pull', 'lam_push',
                  'lam_pull_raw', 'lam_push_raw'):
            if st.get(k) is not None:
                a[k] = float(st[k])

    def shift_loss(self, features, logits):
        if not self.shift_enabled:
            return None

        self._shift_iter_in_epoch += 1
        if self._shift_epoch < self.maosdan_shift_start_epoch:
            self._shift_log_skip('before_start_epoch')
            return None
        if self._shift_mu is None or features is None or logits is None:
            self._shift_log_skip('no_prototype')
            return None

        K = self._shift_K
        B = int(features.size(0))
        if B < max(4, 2 * self.maosdan_shift_min_side):
            self._shift_log_skip('batch_too_small', n_samples=B)
            return None

        with torch.no_grad():
            z_c = F.normalize(features.detach().float().cpu(), p=2, dim=1)
            lp = logits.detach().float().cpu()
            m = lp[:, K] - lp[:, :K].max(dim=1).values
            chat = lp[:, :K].argmax(dim=1)
            pu = F.softmax(lp, dim=1)[:, K]

            mu = self._shift_mu
            d = 1.0 - torch.mm(z_c, mu.t())
            dmin, cstar = d.min(dim=1)
            rad = self._shift_rad.clamp_min(1e-6)
            geom = dmin / rad[cstar]

            if (self._shift_bank_z is not None
                    and self._shift_bank_z.size(0) >= max(2, self.maosdan_shift_knn_k)):
                bz = self._shift_bank_z
                kk = min(self.maosdan_shift_knn_k, bz.size(0))
                sim = torch.mm(z_c, bz.t())
                knn = self._shift_bank_pu[sim.topk(kk, dim=1).indices].mean(dim=1)
            else:
                knn = torch.zeros(B, dtype=torch.float32)

            r_batch = float((lp.argmax(dim=1) == K).float().mean())
            r_use = r_batch if self._shift_r_ema is None else 0.5 * self._shift_r_ema + 0.5 * r_batch
            self._shift_r_ema = r_use
            lam_p_b, lam_q_b = self._shift_lam(r_use)

            q = min(max(self.maosdan_shift_band_q, 0.0), 1.0)
            thr = float(torch.quantile(m.abs().float(), q))
            band = m.abs() <= thr
            bidx = band.nonzero(as_tuple=False).squeeze(1)
            n_band = int(bidx.numel())
            if n_band < 2 * self.maosdan_shift_min_side:
                self._shift_acc_add({'skip': 'band', 'n_band': n_band, 'n_samples': B})
                self._shift_log_skip('band_too_small', n_samples=B, n_band=n_band)
                return None

            gp, kp = self.maosdan_shift_gate_pull_geom, self.maosdan_shift_gate_pull_knn
            gq, pq = self.maosdan_shift_gate_push_geom, self.maosdan_shift_gate_push_pu

            agree_eff = bool(self.maosdan_shift_pull_proto_agree)
            if not self.maosdan_shift_proto_gate_enable:
                gp, kp, gq, pq = 0.0, 0.0, 0.0, 0.0
                agree_eff = False
            self._shift_agree_eff = agree_eff
            if gp > 0.0:
                side_pull = band & (m > 0) & (geom <= gp)

                if self.maosdan_shift_margin_pull_max > 0.0:
                    side_pull = side_pull & (m <= self.maosdan_shift_margin_pull_max)

                if agree_eff:
                    side_pull = side_pull & (chat == cstar)
                if kp > 0.0:
                    side_pull = side_pull & (knn <= kp)
            else:
                side_pull = band & (m > 0)
                if self.maosdan_shift_margin_pull_max > 0.0:
                    side_pull = side_pull & (m <= self.maosdan_shift_margin_pull_max)
            if gq > 0.0:
                side_push = band & (m < 0) & (geom >= gq)
                if pq > 0.0:
                    side_push = side_push & (pu >= pq)
            else:
                side_push = band & (m < 0)

            f_raw = features.detach().float().cpu()
            if self._shift_log_enable:
                disp_p, cross_p, lab_p, reach_p, dbg_p = self._shift_disp(
                    f_raw, lp, side='pull', debug=True)
                disp_q, cross_q, lab_q, reach_q, dbg_q = self._shift_disp(
                    f_raw, lp, side='push', debug=True)
            else:
                disp_p, cross_p, lab_p, reach_p = self._shift_disp(f_raw, lp, side='pull')
                disp_q, cross_q, lab_q, reach_q = self._shift_disp(f_raw, lp, side='push')
                dbg_p = dbg_q = None

            cand_p = side_pull & cross_p
            cand_q = side_push & cross_q

            pre_cap_p = cand_p.clone() if self._shift_log_enable else None
            pre_cap_q = cand_q.clone() if self._shift_log_enable else None
            class_rejected = 0
            if self.maosdan_shift_preserve_class:
                changed = cand_p & (lab_p != chat)
                class_rejected = int(changed.sum())
                cand_p = cand_p & (lab_p == chat)

            if lam_p_b <= 0.0:
                cand_p = torch.zeros_like(cand_p)
            if lam_q_b <= 0.0:
                cand_q = torch.zeros_like(cand_q)

            cap_drop_p, cap_drop_q = 0, 0
            if self.maosdan_shift_class_cap > 0.0:
                cand_p, cap_drop_p = self._shift_class_cap(cand_p, lab_p, -geom)
                cand_q, cap_drop_q = self._shift_class_cap(cand_q, lab_q, pu)

            small_pull = int(cand_p.sum()) if int(cand_p.sum()) < self.maosdan_shift_min_side else 0
            small_push = int(cand_q.sum()) if int(cand_q.sum()) < self.maosdan_shift_min_side else 0
            if small_pull:
                cand_p = torch.zeros_like(cand_p)
            if small_push:
                cand_q = torch.zeros_like(cand_q)
            ip = cand_p.nonzero(as_tuple=False).squeeze(1)
            iq = cand_q.nonzero(as_tuple=False).squeeze(1)
            if (ip.numel() < self.maosdan_shift_min_side
                    and iq.numel() < self.maosdan_shift_min_side):
                self._shift_acc_add({'skip': 'cross/gate', 'n_band': n_band,
                                     'n_samples': B,
                                     'n_cross_pull': int(cand_p.sum()),
                                     'n_cross_push': int(cand_q.sum()),
                                     'r_pred_unk': r_use,
                                     'gap': r_use - self._shift_pi_u,
                                     'lam_pull': lam_p_b, 'lam_push': lam_q_b})
                self._shift_log_skip('no_candidate', n_samples=B, n_band=n_band,
                                     n_cross_pull=int(cand_p.sum()),
                                     n_cross_push=int(cand_q.sum()),
                                     r_pred_unk=r_use,
                                     gap=r_use - self._shift_pi_u,
                                     lam_pull_raw=lam_p_b, lam_push_raw=lam_q_b)
                return None

            lam_p_eff = lam_p_b if ip.numel() >= self.maosdan_shift_min_side else 0.0
            lam_q_eff = lam_q_b if iq.numel() >= self.maosdan_shift_min_side else 0.0

            ramp = (min(1.0, max(0.0, (self._shift_epoch - self.maosdan_shift_start_epoch + 1)
                     / float(self.maosdan_shift_ramp_epochs)))
                    if self.maosdan_shift_ramp_epochs > 0 else 1.0)
            cov_p = float(ip.numel()) / float(B)
            cov_q = float(iq.numel()) / float(B)
            if self.maosdan_shift_coverage_ref > 0.0:
                cov_scale_p = min(1.0, cov_p / self.maosdan_shift_coverage_ref)
                cov_scale_q = min(1.0, cov_q / self.maosdan_shift_coverage_ref)
            else:
                cov_scale_p = cov_scale_q = 1.0
            lam_p_eff *= ramp * cov_scale_p
            lam_q_eff *= ramp * cov_scale_q

            is_pull = torch.zeros(B, dtype=torch.bool)
            is_push = torch.zeros(B, dtype=torch.bool)
            is_pull[ip] = True
            is_push[iq] = True

            disp_p = disp_p * is_pull.float().unsqueeze(1)
            disp_q = disp_q * is_push.float().unsqueeze(1)

            w_pull_sample = self._shift_balance_weights(lab_p, is_pull)
            w_push_sample = self._shift_balance_weights(lab_q, is_push)

            stats_cpu = {
                'n_band': n_band, 'n_samples': B,
                'n_pull': int(is_pull.sum()),
                'n_push': int(is_push.sum()),
                'cov': n_band / float(B),
                'm_p50': float(m.abs().median()),
                'n_cross_pull': int(cand_p.sum()), 'n_cross_push': int(cand_q.sum()),
                'n_cls_switch': int((lab_p[is_pull] != chat[is_pull]).sum()),
                'n_class_rejected': class_rejected,
                'n_small_pull': small_pull, 'n_small_push': small_push,
                'n_cap_pull': cap_drop_p, 'n_cap_push': cap_drop_q,

                'n_at_delta': (int((reach_p & is_pull).sum())
                               if reach_p is not None else -1),
                'r_pred_unk': r_use, 'gap': r_use - self._shift_pi_u,
                'lam_pull': lam_p_eff, 'lam_push': lam_q_eff,
                'lam_pull_raw': lam_p_b, 'lam_push_raw': lam_q_b,
                'gate_in_pull': int((band & (m > 0)).sum()),
                'gate_out_pull': int(side_pull.sum()),
                'gate_in_push': int((band & (m < 0)).sum()),
                'gate_out_push': int(side_push.sum()),
                'eps_pull_mean': (float(disp_p[is_pull].norm(dim=1).mean())
                                  if bool(is_pull.any()) else 0.0),
                'eps_push_mean': (float(disp_q[is_push].norm(dim=1).mean())
                                  if bool(is_push.any()) else 0.0),
                'pull_per_cls': torch.bincount(lab_p[is_pull], minlength=K).tolist(),
                'push_cs_per_cls': torch.bincount(cstar[is_push], minlength=K).tolist(),
            }

            if self._shift_log_enable:
                self._log_shift_samples(
                    m=m, chat=chat, cstar=cstar, geom=geom, dmin=dmin,
                    rad_c=rad[cstar], pu=pu, knn=knn, band=band,
                    side_pull=side_pull, side_push=side_push,
                    cross_p=cross_p, cross_q=cross_q,
                    pre_cap_p=pre_cap_p, pre_cap_q=pre_cap_q,
                    lab_p=lab_p, lab_q=lab_q, is_pull=is_pull, is_push=is_push,
                    reach_p=reach_p, reach_q=reach_q,
                    dbg_p=dbg_p, dbg_q=dbg_q,
                    lam_p_b=lam_p_b, lam_q_b=lam_q_b, gp=gp, gq=gq)

            dev = features.device
            disp_p = disp_p.detach().to(dev)
            disp_q = disp_q.detach().to(dev)
            w_pull_dev = w_pull_sample.detach().to(dev)
            w_push_dev = w_push_sample.detach().to(dev)
            chat_dev = lab_p.to(dev)
            unk_dev = torch.full((B,), K, dtype=torch.long, device=dev)
            pull_mask = is_pull.to(dev).float()
            push_mask = is_push.to(dev).float()

        z_pull = features + disp_p
        z_push = features + disp_q
        lg_pull = self.head(z_pull)
        lg_push = self.head(z_push)
        ce_pull = F.cross_entropy(lg_pull, chat_dev, reduction='none')
        ce_push = F.cross_entropy(lg_push, unk_dev, reduction='none')

        l_pull = (w_pull_dev * pull_mask * ce_pull).sum()
        l_push = (w_push_dev * push_mask * ce_push).sum()

        if self.maosdan_shift_orig_w > 0.0 and bool(is_pull.any()):
            ce_orig = F.cross_entropy(self.head(features), chat_dev, reduction='none')
            l_pull = l_pull + self.maosdan_shift_orig_w * (w_pull_dev * pull_mask * ce_orig).sum()

        if not (bool(torch.isfinite(l_pull)) and bool(torch.isfinite(l_push))):
            self._shift_log_skip('loss_not_finite')
            return None

        total = lam_p_eff * l_pull + lam_q_eff * l_push
        if not bool(torch.isfinite(total)):
            self._shift_log_skip('loss_not_finite')
            return None

        with torch.no_grad():
            g_p = (F.softmax(lg_pull.detach(), dim=1)
                   - F.one_hot(chat_dev, K + 1).float()
                   ) * (w_pull_dev * pull_mask).unsqueeze(1)
            g_q = (F.softmax(lg_push.detach(), dim=1)
                   - F.one_hot(unk_dev, K + 1).float()
                   ) * (w_push_dev * push_mask).unsqueeze(1)
            Wh = self.head.weight.detach()
            gW = float(((g_p.t() @ z_pull.detach()) * lam_p_eff
                        + (g_q.t() @ z_push.detach()) * lam_q_eff).norm())
            gz = float(((g_p @ Wh).norm(dim=1) * lam_p_eff
                        + (g_q @ Wh).norm(dim=1) * lam_q_eff).sum())
            stats_cpu.update({
                'L_pull': float(l_pull.detach()), 'L_push': float(l_push.detach()),
                'L_shift': float(total.detach()), 'gW': gW, 'gz': gz,
            })
            self._shift_acc_add(stats_cpu)
        return total

    def _shift_disp(self, f_cpu, lp_cpu, side, debug=False):
        K = self._shift_K
        W = self.head.weight.detach().float().cpu()
        delta = self.maosdan_shift_margin_target
        eps_cap = float(self.maosdan_shift_eps_max)
        lg = lp_cpu
        chat = lg[:, :K].argmax(dim=1)

        if self.maosdan_shift_dir == 'geom':
            zh = F.normalize(f_cpu, p=2, dim=1)
            if side == 'pull':
                v = self._shift_mu[chat] - zh
            else:
                cstar = (1.0 - zh @ self._shift_mu.t()).min(dim=1).indices
                v = zh - self._shift_mu[cstar]
            nv = v.norm(dim=1).clamp_min(1e-6)
            eps = torch.minimum(self.maosdan_shift_eps_rel * nv,
                                torch.tensor(eps_cap, dtype=nv.dtype))
            disp = eps[:, None] * (v / nv[:, None])
            reached_delta = None
        else:
            v = W[chat] - W[K]
            if side == 'push':
                v = -v
            nv = v.norm(dim=1).clamp_min(1e-6)
            u = v / nv[:, None]
            a = u @ W.t()
            D = lg[:, K].unsqueeze(1) - lg[:, :K]
            S = a[:, K].unsqueeze(1) - a[:, :K]
            tiny = 1e-12
            cap_t = torch.tensor(eps_cap, dtype=D.dtype)

            def _solve(target):
                if side == 'pull':
                    dn = torch.where(S < -tiny, S, torch.ones_like(S))
                    cand = torch.where(
                        S < -tiny, (-target - D) / dn,
                        torch.where(D <= -target, torch.zeros_like(D),
                                    torch.full_like(D, float('inf'))))
                    e = cand.min(dim=1).values
                    return e.clamp(min=0.0), torch.isfinite(e)
                dp = torch.where(S > tiny, S, torch.ones_like(S))
                lo = torch.where(S > tiny, (target - D) / dp, torch.zeros_like(D))
                bad = ((S <= tiny) & (D < target)).any(dim=1)
                return lo.max(dim=1).values.clamp(min=0.0), ~bad

            e_d, f_d = _solve(delta)

            floor = max(1e-3, 0.05 * delta)
            m0 = D.min(dim=1).values
            need = (m0 > -floor) if side == 'pull' else (m0 < floor)
            e_0, f_0 = _solve(floor)
            use_d = need & f_d & (e_d <= cap_t)
            use_0 = need & (~use_d) & f_0 & (e_0 <= cap_t)
            eps = torch.where(use_d, e_d,
                              torch.where(use_0, e_0, torch.zeros_like(e_d)))
            reached_delta = use_d
            disp = eps[:, None] * u

        lg2 = lg + disp @ W.t()
        m_end = lg2[:, K] - lg2[:, :K].max(dim=1).values
        chat_end = lg2[:, :K].argmax(dim=1)
        moved = disp.norm(dim=1) > 1e-9
        if side == 'pull':
            crossed = (m_end <= -1e-3) & moved
        else:
            crossed = (m_end >= 1e-3) & moved
        label = chat_end if side == 'pull' else chat
        if debug:
            _m_before = lg[:, K] - lg[:, :K].max(dim=1).values
            _floor = max(1e-3, 0.05 * float(delta))
            _fallback = (None if reached_delta is None
                         else (moved & (~reached_delta)))
            _target = (None if reached_delta is None
                       else torch.where(reached_delta,
                                        torch.full_like(m_end, float(delta)),
                                        torch.full_like(m_end, _floor)))
            _dbg = {
                'm_before': _m_before,
                'm_after': m_end,
                'disp_norm': disp.norm(dim=1),
                'chat_end': chat_end,
                'crossed': crossed,
                'reached_delta': reached_delta,
                'used_fallback': _fallback,
                'target_margin': _target,
            }
            return disp, crossed, label, reached_delta, _dbg
        return disp, crossed, label, reached_delta

    def _shift_class_cap(self, cand, key, score):
        kappa = float(self.maosdan_shift_class_cap)
        n_sel = int(cand.sum())
        if kappa <= 0.0 or n_sel < 2:
            return cand, 0
        K = self._shift_K
        cap = max(1, int(math.ceil(kappa * n_sel / float(K))))
        keep = torch.zeros_like(cand)
        for c in range(K):
            ci = (cand & (key == c)).nonzero(as_tuple=False).squeeze(1)
            if ci.numel() == 0:
                continue
            if ci.numel() > cap:
                ci = ci[torch.sort(score[ci], descending=True, stable=True).indices[:cap]]
            keep[ci] = True
        capped = cand & keep
        return capped, n_sel - int(capped.sum())

    def _shift_balance_weights(self, chat, mask):
        K = self._shift_K
        w = torch.zeros_like(chat, dtype=torch.float32)
        if not bool(mask.any()):
            return w
        cnt = torch.bincount(chat[mask], minlength=K).float()
        inv = torch.zeros(K, dtype=torch.float32)
        nz = cnt > 0
        if bool(nz.any()):
            inv[nz] = 1.0 / cnt[nz]
            inv[nz] = inv[nz].clamp(max=self.maosdan_shift_balance_clip * float(inv[nz].min()))
        w = inv[chat] * mask.float()
        w = w / w.sum().clamp_min(1e-6)
        return w

    def _shift_log_skip(self, reason, **fields):
        if not (self._shift_log_enable and self._shift_logger is not None):
            return
        row = {'epoch': self._shift_epoch,
               'iteration': self._shift_iter_in_epoch,
               'reason': reason}
        row.update(fields)
        self._shift_logger.log_skip_row(row)

    def _log_shift_samples(self, m, chat, cstar, geom, dmin, rad_c, pu, knn, band,
                           side_pull, side_push, cross_p, cross_q,
                           pre_cap_p, pre_cap_q, lab_p, lab_q,
                           is_pull, is_push, reach_p, reach_q,
                           dbg_p, dbg_q, lam_p_b, lam_q_b, gp, gq):
        if not (self._shift_log_enable and self._shift_logger is not None):
            return
        it = int(self._shift_iter_in_epoch)
        if self._shift_log_every_iter > 0 and (it % self._shift_log_every_iter) != 0:
            return

        from utils.shift_log import even_subsample, stream_key

        dir_p = (band & (m > 0)).nonzero(as_tuple=False).squeeze(1).tolist()
        dir_q = (band & (m < 0)).nonzero(as_tuple=False).squeeze(1).tolist()
        if not dir_p and not dir_q:
            return

        cap = int(self._shift_log_max_samples or 0)
        n_p, n_q = len(dir_p), len(dir_q)
        if cap > 0 and (n_p + n_q) > cap:
            tot = float(n_p + n_q)
            k_p = max(1, int(round(cap * n_p / tot))) if n_p else 0
            k_q = max(1, int(round(cap * n_q / tot))) if n_q else 0
            use_p = [dir_p[i] for i in even_subsample(n_p, k_p)]
            use_q = [dir_q[i] for i in even_subsample(n_q, k_q)]
            sub_flag = 1
        else:
            use_p, use_q = dir_p, dir_q
            sub_flag = 0

        m_l = m.tolist()
        chat_l = chat.tolist()
        cstar_l = cstar.tolist()
        geom_l = geom.tolist()
        dmin_l = dmin.tolist()
        rad_l = rad_c.tolist()
        pu_l = pu.tolist()
        knn_l = knn.tolist()
        sp_l = side_pull.tolist()
        sq_l = side_push.tolist()
        crp_l = cross_p.tolist()
        crq_l = cross_q.tolist()
        pcp_l = pre_cap_p.tolist()
        pcq_l = pre_cap_q.tolist()
        lpl_l = lab_p.tolist()
        lql_l = lab_q.tolist()
        ip_l = is_pull.tolist()
        iq_l = is_push.tolist()
        mov_l = {'pull': dbg_p['m_before'].tolist(), 'push': dbg_q['m_before'].tolist()}
        maf_l = {'pull': dbg_p['m_after'].tolist(), 'push': dbg_q['m_after'].tolist()}
        dnm_l = {'pull': dbg_p['disp_norm'].tolist(), 'push': dbg_q['disp_norm'].tolist()}
        cls_l = {'pull': dbg_p['chat_end'].tolist(), 'push': chat_l}
        rd_l = {'pull': (reach_p.tolist() if reach_p is not None else None),
                'push': (reach_q.tolist() if reach_q is not None else None)}
        fb_l = {'pull': (dbg_p['used_fallback'].tolist()
                         if dbg_p['used_fallback'] is not None else None),
                'push': (dbg_q['used_fallback'].tolist()
                         if dbg_q['used_fallback'] is not None else None)}
        tm_l = {'pull': (dbg_p['target_margin'].tolist()
                         if dbg_p['target_margin'] is not None else None),
                'push': (dbg_q['target_margin'].tolist()
                         if dbg_q['target_margin'] is not None else None)}

        gpm = float(self.maosdan_shift_margin_pull_max)

        agree_on = bool(getattr(self, '_shift_agree_eff',
                                self.maosdan_shift_pull_proto_agree))
        rows = []
        for side, idxs in (('pull', use_p), ('push', use_q)):
            sel_l = ip_l if side == 'pull' else iq_l
            gate_l = sp_l if side == 'pull' else sq_l
            cross_l = crp_l if side == 'pull' else crq_l
            cap_l = pcp_l if side == 'pull' else pcq_l
            lab_l = lpl_l if side == 'pull' else lql_l
            for i in idxs:
                d0 = mov_l[side][i]
                d1 = maf_l[side][i]
                if side == 'pull':
                    g_ok = bool(gp > 0.0 and geom_l[i] <= gp)
                    m_ok = bool(gpm <= 0.0 or m_l[i] <= gpm)
                    a_ok = bool((not agree_on) or (chat_l[i] == cstar_l[i]))
                else:
                    g_ok = bool(gq > 0.0 and geom_l[i] >= gq)
                    m_ok = True
                    a_ok = True
                ok_gate = bool(gate_l[i])
                ok_cross = bool(cross_l[i])
                ok_sel = bool(sel_l[i])
                st = 1
                if ok_gate:
                    st = 2
                if ok_gate and ok_cross:
                    st = 3
                if ok_sel:
                    st = 4
                rows.append({
                    'epoch': self._shift_epoch,
                    'iteration': it,
                    'stream_key': stream_key(self._shift_epoch, it, i),
                    'side': side,
                    'in_batch': i,
                    'margin_before': d0,
                    'margin_after': d1,
                    'margin_change': d1 - d0,
                    'closed_pred': chat_l[i],
                    'prototype_pred': cstar_l[i],
                    'shift_pred': cls_l[side][i],
                    'prototype_distance': dmin_l[i],
                    'prototype_radius': rad_l[i],
                    'geom_ratio': geom_l[i],
                    'unknown_prob': pu_l[i],
                    'knn_unknown_prob': knn_l[i],
                    'in_boundary_band': 1,
                    'pass_geometry_gate': g_ok,
                    'pass_margin_gate': m_ok,
                    'pass_class_agreement': a_ok,
                    'pass_crossing_check': ok_cross,
                    'pass_class_cap': bool(cap_l[i]),
                    'selected_final': ok_sel,
                    'stage': st,
                    'shift_distance': dnm_l[side][i],
                    'reached_delta': (None if rd_l[side] is None else bool(rd_l[side][i])),
                    'used_fallback': (None if fb_l[side] is None else bool(fb_l[side][i])),
                    'target_margin': (None if tm_l[side] is None else tm_l[side][i]),
                    'candidate_label': lab_l[i],
                    'subsampled': sub_flag,
                })
        self._shift_logger.log_sample_rows(rows)

    def _build_epoch_row(self, a):
        n_st = max(1, int(a.get('steps') or 0))
        n_sm = max(1, int(a.get('n_samples') or 0))
        n_xp = int(a.get('n_cross_pull') or 0)
        n_xq = int(a.get('n_cross_push') or 0)
        n_p = int(a.get('n_pull') or 0)
        n_q = int(a.get('n_push') or 0)
        ks = ('calls', 'skip_band', 'skip_gate', 'n_band', 'm_p50', 'n_cls_switch',
              'n_at_delta', 'n_class_rejected', 'n_small_pull', 'n_small_push',
              'n_cap_pull', 'n_cap_push', 'gate_in_pull', 'gate_out_pull',
              'gate_in_push', 'gate_out_push', 'eps_pull_mean', 'eps_push_mean',
              'L_pull', 'L_push', 'L_shift', 'gW', 'gz')
        v = {k: a.get(k) for k in ks}
        for k in ('m_p50', 'eps_pull_mean', 'eps_push_mean', 'L_pull', 'L_push',
                  'L_shift', 'gW', 'gz'):
            v[k] = (None if v[k] is None else float(v[k]) / n_st)
        row = {
            'epoch': self._shift_epoch,
            'shift_calls': v['calls'],
            'shift_steps': int(a.get('steps') or 0),
            'n_samples': int(a.get('n_samples') or 0),
            'n_band': v['n_band'],
            'n_pull': n_p,
            'n_push': n_q,
            'n_cross_pull': n_xp,
            'n_cross_push': n_xq,
            'n_at_delta': v['n_at_delta'],
            'n_cls_switch': v['n_cls_switch'],
            'n_class_rejected': v['n_class_rejected'],
            'n_cap_pull': v['n_cap_pull'],
            'n_cap_push': v['n_cap_push'],
            'gate_in_pull': v['gate_in_pull'],
            'gate_out_pull': v['gate_out_pull'],
            'gate_in_push': v['gate_in_push'],
            'gate_out_push': v['gate_out_push'],
            'eps_pull_mean': v['eps_pull_mean'],
            'eps_push_mean': v['eps_push_mean'],
            'L_pull': v['L_pull'],
            'L_push': v['L_push'],
            'L_shift': v['L_shift'],
            'lam_pull': a.get('lam_pull'),
            'lam_push': a.get('lam_push'),
            'lam_pull_raw': a.get('lam_pull_raw'),
            'lam_push_raw': a.get('lam_push_raw'),
            'r_pred_unk': a.get('r_pred_unk'),
            'gap': a.get('gap'),
            'gW': v['gW'],
            'gz': v['gz'],
            'pull_per_cls': a.get('pull_per_cls') or [],
            'push_cs_per_cls': a.get('push_cs_per_cls') or [],
            'n_active_iters': int(a.get('steps') or 0),
            'pull_coverage': n_p / float(n_sm),
            'push_coverage': n_q / float(n_sm),
            'cross_success_rate_pull': (n_p / float(n_xp)) if n_xp > 0 else None,
            'cross_success_rate_push': (n_q / float(n_xq)) if n_xq > 0 else None,
            'band_coverage': int(a.get('n_band') or 0) / float(n_sm),
            'push_gate_enabled': 1 if self.maosdan_shift_gate_push_geom > 0.0 else 0,
        }
        return row

    def record_val_stats(self, all_acc, known_acc, unknown_acc, h_score):
        if not (self._shift_log_enable and self._shift_logger is not None):
            return
        row = self._shift_log_epoch_pending
        if row is None:
            return
        row = dict(row)
        row.update({'all_acc': all_acc, 'known_acc': known_acc,
                    'unknown_acc': unknown_acc, 'h_score': h_score})
        self._shift_logger.log_epoch_row(row)
        self._shift_log_epoch_pending = None

    @torch.no_grad()
    def on_shift_epoch_end(self):
        if not self.shift_enabled:
            return ''
        self._shift_fit_prototypes()

        if self._shift_tgt_feat:
            z = torch.cat(self._shift_tgt_feat, 0)
            pu = torch.cat(self._shift_tgt_pu, 0)
            n = z.size(0)
            cap = max(2, self.maosdan_shift_bank_size)
            if n > cap:
                idx = torch.linspace(0, n - 1, cap).round().long().unique()
                z, pu = z[idx], pu[idx]
            self._shift_bank_z = z
            self._shift_bank_pu = pu
        self._shift_tgt_feat, self._shift_tgt_pu = [], []

        self._shift_src_feat, self._shift_src_lab = [], []

        a = dict(self._shift_acc)
        self._shift_last = a
        self._shift_acc_reset()

        if self._shift_log_enable and self._shift_logger is not None:
            self._shift_log_epoch_pending = self._build_epoch_row(a)
        if a['calls'] == 0:
            return ('no-active-step (未到 start_epoch / 原型未就绪) n_bank={}'.format(
                0 if self._shift_bank_z is None else int(self._shift_bank_z.size(0))))
        n_st = max(1, a['steps'])
        n_sm = max(1, a['n_samples'])
        return ('steps={}/{} skip(band={},cross/gate={}) '
                'band={:.1f} ({:.3f}) cross=({},{}) pull={} push={} switch={} at_delta={} '
                'r={:.3f} gap={:+.3f} '
                '|m|p50={:.3f} '
                'eps_pull={:.4f} eps_push={:.4f} '
                'L_pull={:.4f} L_push={:.4f} L_shift={:.4f} '
                'gW={:.4e} gz={:.4e} lam=({:.4f},{:.4f}) lam_raw=({:.4f},{:.4f}) n_bank={} '
                'pull_per_cls={} push_cs_per_cls={} rejected_cls={} small=({},{}) cap_dropped=({},{})'
                ' gate=({},{},{},{})'
                .format(a['steps'], a['calls'], a['skip_band'], a['skip_gate'],
                        a['n_band'] / float(max(1, a['calls'])),
                        a['n_band'] / float(n_sm),
                        a['n_cross_pull'], a['n_cross_push'],
                        a['n_pull'], a['n_push'],
                        a['n_cls_switch'], a['n_at_delta'],
                        float('nan') if a['r_pred_unk'] is None else a['r_pred_unk'],
                        float('nan') if a['gap'] is None else a['gap'],
                        a['m_p50'] / n_st,
                        a['eps_pull_mean'] / n_st, a['eps_push_mean'] / n_st,
                        a['L_pull'] / n_st, a['L_push'] / n_st, a['L_shift'] / n_st,
                        a['gW'] / n_st, a['gz'] / n_st,
                        a['lam_pull'], a['lam_push'],
                        float('nan') if a['lam_pull_raw'] is None else a['lam_pull_raw'],
                        float('nan') if a['lam_push_raw'] is None else a['lam_push_raw'],
                        0 if self._shift_bank_z is None else int(self._shift_bank_z.size(0)),
                        a['pull_per_cls'] or [], a['push_cs_per_cls'] or [],
                        a['n_class_rejected'], a['n_small_pull'], a['n_small_push'],
                        a['n_cap_pull'], a['n_cap_push'],
                        a['gate_in_pull'], a['gate_out_pull'],
                        a['gate_in_push'], a['gate_out_push']))

    @torch.no_grad()
    def shift_diagnostics(self):
        d = {
            'shift_enabled': bool(self.shift_enabled),
            'shift_epoch': self._shift_epoch,
            'pi_u': self._shift_pi_u,
            'lambda_pull': self._shift_lam_pull,
            'lambda_push': self._shift_lam_push,
            'mu_ready': self._shift_mu is not None,
            'bank_size': 0 if self._shift_bank_z is None else int(self._shift_bank_z.size(0)),
        }
        d.update(self._shift_last)
        return d

    def close_shift_log(self):
        self._shift_log_sample_buf = []
        self._shift_log_epoch_pending = None
        if self._shift_logger is not None:
            stats = self._shift_logger.stats()
            self._shift_logger.close()
            return stats
        return None


def _zscore(x):
    if x.numel() < 4:
        return torch.zeros_like(x)
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-6)
