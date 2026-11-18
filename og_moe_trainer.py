import time

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from utils.data import ForeverDataIterator
from alignment.osbp import (DomainAdversarialLoss, Known_class_detection,
                            SupConLoss, Unknown_class_detection,
                            get_prototype_weight, bce_loss,
                            GradientReverseLayer)
from modules.domain_discriminator import DomainDiscriminator
from models.OG_MoE import OG_MoE


def _pi_u(args):
    n_t = max(len(getattr(args, 'tgt_classes', []) or []), 1)
    n_s = len(getattr(args, 'src_classes', []) or [])
    return max(0.0, min(1.0, (n_t - n_s) / float(n_t)))


def _update_tgt_center(center, logits, feats, momentum=0.5):
    pseudo = logits.argmax(dim=1)
    for c in torch.unique(pseudo).tolist():
        ctr = feats[pseudo == c].mean(dim=0)
        center[c] = momentum * center[c].detach() + (1.0 - momentum) * ctr.detach()


def _full_mfcl_expert_a(args, model, out_s, out_t, out_ts, x_t_s2, y_s, epoch,
                        grl, ctl_fn, domain_adv, unknown_csl_fn, known_csl_fn,
                        state, i):
    device = out_s['fA'].device
    n_cls = model.num_classes
    K = model.num_known
    sce = F.cross_entropy(out_s['logitsA'], y_s)
    ctl = ctl_fn(torch.cat([out_t['fA'], out_ts['fA']], dim=0))

    if i == 0:
        state['center'] = torch.randn(n_cls, out_s['fA'].size(1), device=device)
        state['tgt1'] = torch.randn(n_cls, out_s['fA'].size(1), device=device)
        state['tgt2'] = torch.randn(n_cls, out_s['fA'].size(1), device=device)
    else:
        for c in torch.unique(y_s).tolist():
            ctr = out_s['fA'][y_s == c].mean(dim=0)
            state['center'][c] = 0.5 * state['center'][c].detach() + 0.5 * ctr.detach()
        _update_tgt_center(state['tgt1'], out_t['logitsA'], out_t['fA'])
        _update_tgt_center(state['tgt2'], out_ts['logitsA'], out_ts['fA'])

    if epoch <= args.pretrain_epoch:
        zero = torch.zeros((), device=device)
        return sce + 0.5 * ctl, zero, zero, zero, zero

    pA_t = F.softmax(model.expertA.head(grl(out_t['fA'])), dim=1)
    esl = bce_loss(pA_t[:, :K].sum(1, keepdim=True), pA_t[:, K:K + 1])

    fA_s2, logitsA_s2 = model.forward_a(x_t_s2)
    weight1, _ = get_prototype_weight(state['tgt1'].detach(), out_t['fA'], n_cls)
    weight2, _ = get_prototype_weight(state['tgt2'].detach(), out_t['fA'], n_cls)
    mask1 = (out_t['logitsA'] * weight1).argmax(dim=1)
    mask2 = (out_ts['logitsA'] * weight2).argmax(dim=1)

    fp_drop = torch.nn.Dropout(0.1)
    fp_out = model.expertA.head(torch.cat(
        [out_t['fA'], fp_drop(out_t['fA'])], dim=0))
    _, logitsA_fp = fp_out.chunk(2)
    logits_s1 = F.softmax(out_ts['logitsA'], dim=1)
    logits_s2 = F.softmax(logitsA_s2, dim=1)
    pred_u_w = F.softmax(out_t['logitsA'], dim=1)
    pred_u_w_fp = F.softmax(logitsA_fp, dim=1)
    csl = unknown_csl_fn(logits_s1=logits_s1, logits_s2=logits_s2,
                         pred_u_w_fp=pred_u_w_fp,
                         mask_u_1=mask1, mask_u_2=mask2).to(device)
    csl = csl + known_csl_fn(logits_s1=logits_s1, logits_s2=logits_s2,
                             pred_u_w_fp=pred_u_w_fp,
                             mask_u_1=mask1, mask_u_2=mask2,
                             pred_u_w=pred_u_w).to(device)

    _, prt = get_prototype_weight(state['center'].detach(), out_t['fA'], n_cls)
    adv = domain_adv(out_s['fA'], out_t['fA'])
    return sce + esl + csl + 0.5 * (ctl + prt + adv), esl, csl, prt, adv


def train_one_epoch(args, model, src_iter, tgt_iter, ctl_fn, optimizer, scheduler,
                    epoch, domain_adv=None, unknown_csl_fn=None,
                    known_csl_fn=None):
    import trainer
    device = trainer.device
    model.train()
    K = model.num_known
    model.set_shift_epoch(epoch, _pi_u(args))
    if domain_adv is not None:
        domain_adv.train()
    grl = GradientReverseLayer()
    expert_a = str(getattr(args, 'og_expert_a', 'full')).lower()
    full_a = expert_a == 'full'
    proto_state = {}

    meters = {k: 0.0 for k in ('loss', 'A', 'B', 'gate', 'shift', 'csl', 'adv')}
    t0 = time.time()
    for i in range(args.iters_per_epoch):
        (x_s, _, _), y_s = next(src_iter)
        (x_t, x_t_s1, x_t_s2), _ = next(tgt_iter)
        x_s = x_s.to(device)
        y_s = y_s.to(device).long()
        x_t = x_t.to(device)
        x_t_s1 = x_t_s1.to(device)
        x_t_s2 = x_t_s2.to(device)

        out_s = model.forward_all(x_s)
        out_t = model.forward_all(x_t)
        out_ts = model.forward_all(x_t_s1)

        if hasattr(model.expertB, 'observe_labeled_source'):
            model.expertB.observe_labeled_source(out_s['fB'], out_s['logitsB'], y_s)

        if full_a:
            lossA, esl, csl, prt, adv = _full_mfcl_expert_a(
                args, model, out_s, out_t, out_ts, x_t_s2, y_s, epoch, grl,
                ctl_fn, domain_adv, unknown_csl_fn, known_csl_fn,
                proto_state, i)
        else:
            K = model.num_known
            n_cls = model.num_classes
            sce = F.cross_entropy(out_s['logitsA'], y_s)
            pA_t = F.softmax(model.expertA.head(grl(out_t['fA'])), dim=1)
            esl = bce_loss(pA_t[:, :K].sum(1, keepdim=True), pA_t[:, K:K + 1])
            ctl = ctl_fn(torch.cat([out_t['fA'], out_ts['fA']], dim=0))
            center = torch.zeros(n_cls, out_s['fA'].size(1), device=device)
            for c in torch.unique(y_s).tolist():
                center[c] = out_s['fA'][y_s == c].mean(dim=0)
            _, prt = get_prototype_weight(center, out_t['fA'], n_cls)
            lossA = sce + esl + 0.5 * (ctl + prt)
            csl = torch.zeros((), device=device)
            adv = torch.zeros((), device=device)

        cls_b = F.cross_entropy(out_s['logitsB'], y_s)

        pB_t = F.softmax(model.expertB.head(grl(out_t['fB'])), dim=1)
        trans = bce_loss(pB_t[:, :K].sum(1, keepdim=True), pB_t[:, K:K + 1])
        model.expertB.observe_target(out_t['fB'], out_t['logitsB'])
        shift_value = model.expertB.shift_loss(out_t['fB'], out_t['logitsB'])
        lossB = cls_b + trans + (shift_value if shift_value is not None else 0.0)

        fused_s, g_s, _, _ = model.fuse(out_s['logitsA'], out_s['logitsB'], _pi_u(args))
        fused_t, g_t, _, _ = model.fuse(out_t['logitsA'], out_t['logitsB'], _pi_u(args))
        loss_src = F.nll_loss(torch.log(fused_s.clamp_min(1e-12)), y_s)
        loss_prior = (fused_t[:, K].mean() - _pi_u(args)) ** 2
        pk = fused_t[:, :K]
        pk = pk / pk.sum(1, keepdim=True).clamp_min(1e-12)
        loss_ent = -(pk * torch.log(pk.clamp_min(1e-12))).sum(1).mean()
        loss_bal = (g_t.mean() - 0.5) ** 2
        loss_gate = loss_src + loss_prior + 0.1 * loss_ent + 0.1 * loss_bal

        loss = lossA + lossB + loss_gate

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        meters['loss'] += float(loss)
        meters['A'] += float(lossA)
        meters['B'] += float(lossB)
        meters['gate'] += float(loss_gate)
        meters['shift'] += float(shift_value) if shift_value is not None else 0.0
        meters['csl'] += float(csl)
        meters['adv'] += float(adv)
        if i % max(int(args.print_freq), 1) == 0:
            print('OG-MoE epoch {} it {}/{} loss={:.3f} A={:.3f} B={:.3f} '
                  'gate={:.3f} shift={:.3f} csl={:.3f} adv={:.3f} g={:.3f}'.format(
                      epoch, i, args.iters_per_epoch, float(loss), float(lossA),
                      float(lossB), float(loss_gate),
                      float(shift_value) if shift_value is not None else 0.0,
                      float(csl), float(adv), float(g_t.mean())), flush=True)
    n = max(int(args.iters_per_epoch), 1)
    print('OG-MoE epoch {} done | loss={:.3f} A={:.3f} B={:.3f} gate={:.3f} '
          'shift={:.3f} csl={:.3f} adv={:.3f} | {:.1f}s'.format(
              epoch, meters['loss'] / n, meters['A'] / n, meters['B'] / n,
              meters['gate'] / n, meters['shift'] / n, meters['csl'] / n,
              meters['adv'] / n, time.time() - t0), flush=True)


def main_og_moe(args, train_source_loader, train_target_loader,
                val_source_loader, val_target_loader, num_classes):
    import trainer
    device = trainer.device
    from models.MAOSDAN_shift import MAOSDAN_Shift

    pi_u = _pi_u(args)
    shift_kwargs = trainer._model_init_kwargs(MAOSDAN_Shift, args)
    model = OG_MoE(args.arch, num_classes, bottleneck_dim=args.bottleneck_dim,
                   shared_backbone=bool(int(getattr(args, 'og_shared', 1))),
                   pi_u=pi_u, shift_kwargs=shift_kwargs).to(device)
    print(model, flush=True)
    expert_a = str(getattr(args, 'og_expert_a', 'full')).lower()
    print(' * [og_moe] shared_backbone={} expert_a={} pi_u={:.3f} '
          'shift_kwargs_keys={}'.format(
              bool(int(getattr(args, 'og_shared', 1))), expert_a, pi_u,
              sorted(shift_kwargs)), flush=True)

    param_list = model.get_parameters()
    domain_adv = None
    unknown_csl_fn = None
    known_csl_fn = None
    if expert_a == 'full':
        domain_discri = DomainDiscriminator(
            in_feature=model.expertA.features_dim, hidden_size=1024).to(device)
        domain_adv = DomainAdversarialLoss(domain_discri).to(device)
        unknown_csl_fn = Unknown_class_detection(num_classes).to(device)
        known_csl_fn = Known_class_detection(num_classes).to(device)
        param_list = param_list + domain_discri.get_parameters()

    optimizer = SGD(param_list, args.lr, momentum=args.momentum,
                    weight_decay=args.wd, nesterov=True)
    scheduler = LambdaLR(
        optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    ctl_fn = SupConLoss(temperature=0.3).to(device)

    src_iter = ForeverDataIterator(train_source_loader)
    tgt_iter = ForeverDataIterator(train_target_loader)

    ckpt_dir = '{}/checkpoints'.format(args.log)
    import os
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, 'og_moe_best.pth')
    latest_path = os.path.join(ckpt_dir, 'og_moe_latest.pth')

    best_h = 0.0
    for epoch in range(args.epochs):
        train_one_epoch(args, model, src_iter, tgt_iter, ctl_fn, optimizer,
                        scheduler, epoch, domain_adv=domain_adv,
                        unknown_csl_fn=unknown_csl_fn,
                        known_csl_fn=known_csl_fn)
        print('[OG-MoE] epoch={} {}'.format(epoch, model.expertB.on_shift_epoch_end()),
              flush=True)
        h = trainer.validate(args, model, val_target_loader)
        torch.save(model.state_dict(), latest_path)
        if h > best_h:
            best_h = h
            torch.save(model.state_dict(), best_path)
        print(' * OG-MoE epoch {} val H={:.4f} (best {:.4f})'.format(epoch, h, best_h),
              flush=True)

    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    print('\\n=========== Final evaluation on target (open-set) ===========')
    final_h = trainer.validate(args, model, val_target_loader)
    print('=========== Final evaluation on source (closed-set) ===========')
    trainer.validate(args, model, val_source_loader, src_flg=True)
    print('best_h_score = {:.2f}'.format(best_h))
    return final_h
