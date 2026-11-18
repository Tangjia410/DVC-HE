import argparse
import csv
import os
import re
import subprocess
import sys
import time

PY = sys.executable
ROOT = os.path.join('logs_ogmoe_fullmfcl')
SEED = 202401
SDUST_STATES = ['src6(known6)', 'src9(known9)', 'src8(known8)', 'src7(known7)',
                'src5(known5)', 'src4(known4)', 'src3(known3)', 'src2(known2)']
RF1_STATES = ['K6(pi_u=0.143)', 'K5(pi_u=0.286)', 'K4(pi_u=0.429)',
              'K3(pi_u=0.571)', 'K2(pi_u=0.714)']
FIELDS = ['dataset', 'state', 'known_classes', 'seed', 'epochs',
          'os_star', 'uk', 'os', 'h', 'checkpoint', 'log', 'success']


def progress(msg):
    os.makedirs(ROOT, exist_ok=True)
    line = '{}  {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), msg)
    print(line, flush=True)
    with open(os.path.join(ROOT, 'progress_single.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def log_path(out_root, state, epochs):
    return os.path.join(out_root, state, 'og_moe_{}.txt'.format(epochs))


def ckpt_path(out_root, state):
    return os.path.join(out_root, state, 'checkpoints', 'og_moe_best.pth')


def parse_metrics(path):
    if not os.path.isfile(path):
        return None
    txt = open(path, encoding='utf-8', errors='ignore').read()
    m = re.findall(r'\] OS\*=([0-9.]+) UK=([0-9.]+) OS=([0-9.]+) H=([0-9.]+)', txt)
    if not m:
        return None
    return dict(zip(('os_star', 'uk', 'os', 'h'), map(float, m[-1])))


def run_one(dataset, state, epochs, iters, src_name, tgt_name):
    out_root = os.path.join(ROOT, dataset, 'seed{}'.format(SEED))
    lg = log_path(out_root, state, epochs)
    ck = ckpt_path(out_root, state)
    res = parse_metrics(lg)
    if res is not None and os.path.isfile(ck):
        progress('[SKIP] {} {} H={:.4f}'.format(dataset, state, res['h']))
        return res
    os.makedirs(os.path.join(out_root, state), exist_ok=True)
    cmd = [PY, '-u', 'DVC-HE.py', '--method', 'og_moe', '--data_name', dataset,
           '--src_name', str(src_name), '--tgt_name', str(tgt_name),
           '--states', state, '--epochs', str(epochs), '-i', str(iters),
           '--seed', str(SEED), '--og-shared', '1', '--og-expert-a', 'full',
           '--log-root', out_root]
    with open(os.path.join(out_root, state, 'cmd.txt'), 'w', encoding='utf-8') as f:
        f.write(' '.join(cmd) + '\n')
    progress('[START] {} {}'.format(dataset, state))
    t0 = time.time()
    with open(os.path.join(out_root, state, 'console.out'), 'w', encoding='utf-8') as fp:
        rc = subprocess.call(cmd, stdout=fp, stderr=subprocess.STDOUT)
    sec = int(time.time() - t0)
    res = parse_metrics(lg)
    if res is None:
        txt = open(os.path.join(out_root, state, 'console.out'), encoding='utf-8',
                   errors='ignore').read()
        m = re.findall(r'\] OS\*=([0-9.]+) UK=([0-9.]+) OS=([0-9.]+) H=([0-9.]+)', txt)
        if m:
            res = dict(zip(('os_star', 'uk', 'os', 'h'), map(float, m[-1])))
    ok = int(rc == 0 and res is not None)
    if res is None:
        res = {'os_star': float('nan'), 'uk': float('nan'),
               'os': float('nan'), 'h': float('nan')}
    progress('[DONE {}] {} {} H={:.4f} ({}s)'.format(
        'OK' if ok else 'FAIL', dataset, state, res['h'], sec))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['SDUST', 'RflyMAD1'])
    p.add_argument('--states', nargs='+', default=None)
    a = p.parse_args()

    os.makedirs(ROOT, exist_ok=True)
    rows = []
    jobs = []
    for state in SDUST_STATES:
        jobs.append(('SDUST', state, 50, 500, 2, 3))
    for state in RF1_STATES:
        jobs.append(('RflyMAD1', state, 20, 200, 0, 1))
    if a.datasets:
        jobs = [j for j in jobs if j[0] in a.datasets]
    if a.states:
        jobs = [j for j in jobs if j[1] in a.states]

    csv_path = os.path.join(ROOT, 'full_ogmoe_mfcl_results.csv')
    for dataset, state, epochs, iters, sn, tn in jobs:
        res = run_one(dataset, state, epochs, iters, sn, tn)
        known = int(state.split('known')[1].split(')')[0]) if 'known' in state \
            else 6 - int(state.split('K')[1].split('(')[0]) if state.startswith('K') else ''
        out_root = os.path.join(ROOT, dataset, 'seed{}'.format(SEED))
        rows.append({
            'dataset': dataset, 'state': state, 'known_classes': str(known),
            'seed': str(SEED), 'epochs': str(epochs),
            'os_star': '%.4f' % res['os_star'], 'uk': '%.4f' % res['uk'],
            'os': '%.4f' % res['os'], 'h': '%.4f' % res['h'],
            'checkpoint': os.path.abspath(ckpt_path(out_root, state)),
            'log': os.path.abspath(log_path(out_root, state, epochs)),
            'success': '1' if res['h'] == res['h'] else '0',
        })
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    progress('[ALL DONE] {} runs, csv={}'.format(len(rows), os.path.abspath(csv_path)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
