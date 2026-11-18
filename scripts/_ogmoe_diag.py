import importlib.util
import os
from pathlib import Path

import torch

import datasets
import trainer
from utils.metric import compute_osda_metrics

_spec = importlib.util.spec_from_file_location(
    'dvc_he', Path(__file__).resolve().parents[1] / 'DVC-HE.py')
entry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(entry)


def metrics(logits, gt, K):
    gt_map = torch.where(gt >= K, torch.tensor(K), gt)
    class_list = list(range(K)) + [K]
    d = compute_osda_metrics(logits, gt_map, class_list)
    return d['os_star'], d['uk'], d['h']


def main():
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument('--log-root', default='logs_ogmoe_shared')
    _p.add_argument('--state', default='src6(known6)')
    _p.add_argument('--data-name', default='SDUST', choices=['SDUST', 'RflyMAD1'])
    _p.add_argument('--shared', type=int, default=0)
    _a, _ = _p.parse_known_args()
    if _a.data_name == 'SDUST':
        src, tgt = 2, 3
    else:
        src, tgt = 0, 1
    ap = entry.build_parser().parse_args([
        '--method', 'og_moe', '--data_name', _a.data_name,
        '--src_name', str(src), '--tgt_name', str(tgt), '--states', _a.state,
        '--epochs', '50', '-i', '500', '--seed', '202401',
        '--og-shared', str(_a.shared), '--log-root', _a.log_root])
    entry.resolve_dataset_defaults(ap)
    entry.apply_method_defaults(ap)
    ap.src_classes = list(entry.states_for(_a.data_name)[_a.state])
    ap.log = os.path.join(_a.log_root, _a.state)
    dl = getattr(datasets, _a.data_name)
    (_, _, _, tv, nc) = dl(ap.data_dir, ap.src_name, ap.tgt_name,
                           list(ap.src_classes), ap.tgt_classes, 32, 0).create_loaders()
    model = trainer.build_model_for_method(ap, nc)
    ck = os.path.join(ap.log, 'checkpoints', 'og_moe_best.pth')
    if not os.path.isfile(ck):
        ck = os.path.join(ap.log, 'checkpoints', 'og_moe_latest.pth')
    model.load_state_dict(torch.load(ck, map_location=trainer.device))
    model.to(trainer.device).eval()
    K = nc - 1
    A, B, F, G, Y = [], [], [], [], []
    with torch.no_grad():
        for x, y in tv:
            x = x.to(trainer.device)
            out = model.forward_all(x)
            fused, g, _, _ = model.fuse(out['logitsA'], out['logitsB'])
            A.append(out['logitsA'].cpu()); B.append(out['logitsB'].cpu())
            F.append(torch.log(fused.clamp_min(1e-12)).cpu())
            G.append(g.cpu()); Y.append(y)
    y = torch.cat(Y)
    for name, L in (('ExpertA(MFCL)', A), ('ExpertB(Shift)', B),
                    ('Fused(gate)', F)):
        logits = torch.cat(L)
        os_star, uk, h = metrics(logits, y, K)
        print('%-20s OS*=%.4f UK=%.4f H=%.4f' % (name, os_star, uk, h))
    g = torch.cat(G)
    print('gate g: mean=%.4f p10=%.4f p90=%.4f' % (
        g.mean(), torch.quantile(g, 0.1), torch.quantile(g, 0.9)))


if __name__ == '__main__':
    main()
