import argparse
import csv
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


def _metrics(logits, gt, K):
    gt_map = torch.where(gt >= K, torch.tensor(K), gt)
    d = compute_osda_metrics(logits, gt_map, list(range(K)) + [K])
    return d['os_star'], d['uk'], d['h']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='logs_ogmoe_fullmfcl/full_ogmoe_mfcl_results.csv')
    p.add_argument('--out', default='logs_ogmoe_fullmfcl/expert_vs_fused.csv')
    a = p.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding='utf-8')))
    out_rows = []
    for row in rows:
        dataset, state = row['dataset'], row['state']
        root = os.path.dirname(os.path.dirname(row['log']))
        src, tgt = (2, 3) if dataset == 'SDUST' else (0, 1)
        args = entry.build_parser().parse_args([
            '--method', 'og_moe', '--data_name', dataset,
            '--src_name', str(src), '--tgt_name', str(tgt),
            '--states', state, '--epochs', row['epochs'], '-i', '100',
            '--seed', row['seed'], '--og-shared', '1', '--log-root', root])
        entry.resolve_dataset_defaults(args)
        entry.apply_method_defaults(args)
        args.src_classes = list(entry.states_for(dataset)[state])
        args.log = os.path.join(root, state)

        dl = getattr(datasets, dataset)
        (_, _, _, tgt_va, ncls) = dl(
            args.data_dir, args.src_name, args.tgt_name, list(args.src_classes),
            args.tgt_classes, 32, 0).create_loaders()
        model = trainer.build_model_for_method(args, ncls)
        model.load_state_dict(torch.load(row['checkpoint'], map_location=trainer.device))
        model.to(trainer.device).eval()

        A, B, F, G, Y = [], [], [], [], []
        with torch.no_grad():
            for x, y in tgt_va:
                x = x.to(trainer.device)
                o = model.forward_all(x)
                fused, g, _, _ = model.fuse(o['logitsA'], o['logitsB'])
                A.append(o['logitsA'].cpu())
                B.append(o['logitsB'].cpu())
                F.append(torch.log(fused.clamp_min(1e-12)).cpu())
                G.append(g.cpu())
                Y.append(y)
        y = torch.cat(Y)
        K = ncls - 1
        mA = _metrics(torch.cat(A), y, K)
        mB = _metrics(torch.cat(B), y, K)
        mF = _metrics(torch.cat(F), y, K)
        g = torch.cat(G)
        out_rows.append({
            'dataset': dataset, 'state': state,
            'h_A': '%.4f' % mA[2], 'h_B': '%.4f' % mB[2],
            'h_fused': '%.4f' % mF[2],
            'g_mean': '%.4f' % float(g.mean()),
            'g_p10': '%.4f' % float(torch.quantile(g, 0.1)),
            'g_p90': '%.4f' % float(torch.quantile(g, 0.9)),
        })
        print('%-9s %-18s A=%.4f B=%.4f F=%.4f g=%.3f' % (
            dataset, state, mA[2], mB[2], mF[2], float(g.mean())), flush=True)
        with open(a.out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
    print('CSV ->', os.path.abspath(a.out))


if __name__ == '__main__':
    main()
