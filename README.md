# DVC-HE

Reference implementation of **DVC-HE**, an open-set domain adaptation method for
cross-working-condition fault diagnosis.

DVC-HE combines a shared-backbone dual-expert architecture with an
openness-aware gated fusion, so that a single checkpoint can classify known
fault types and reject unknown ones on an unlabelled target domain.

## Method

| Component | Implementation | Objective |
|---|---|---|
| Backbone | 1-D CNN (`models/digits.py`), shared by both experts when `--og-shared 1` | — |
| Expert A | `alignment/osbp.py::ImageClassifier` | full MFCL: `SCE + ESL + CSL + 0.5 * (CTL + PRT + ADV)` |
| Expert B | `models/MAOSDAN_shift.py::MAOSDAN_Shift` | `cls + trans + shift` (reliability-weighted bidirectional decision-boundary correction) |
| Gate | MLP over `[p_A, p_B, H(p_A), H(p_B), max p_A, max p_B, pi_u]` | `g` in `(0, 1)` |
| Fusion | `p = g * p_A + (1 - g) * p_B` | — |

`pi_u` is the openness prior, computed from the source and target class counts.

Baselines kept in the same code path:

- `mfcl` — Expert A alone, the closed/open-set baseline.
- `maosdan_shift` — Expert B alone.
- `og_moe` — the full DVC-HE model.

## Repository layout

```
DVC-HE.py            command-line entry point
trainer.py           method dispatch, MFCL training loop, evaluation
og_moe_trainer.py    end-to-end training of the gated dual-expert model
Batch_testing.py     multi-seed / multi-condition batch runner
alignment/           backbone classifier, OSBP losses (SCE/ESL/CSL/CTL/PRT/ADV)
models/              LeNet backbone, MAOSDAN-Shift expert, DVC-HE model
modules/             domain discriminator, gradient reversal layer, heads
utils/               data iterators, metrics (H-score / OS* / UK / OS), plots
scripts/             single-seed sweep, per-expert diagnostics
```

## Requirements

```
python >= 3.9
torch
timm
numpy
scikit-learn
matplotlib
prettytable
openpyxl        # optional, only for the Excel report in Batch_testing.py
```

## Data

Datasets are not redistributed here. Point the code at a local copy either with
an environment variable:

```bash
export DVC_HE_DATA_DIR=/path/to/datasets
```

or per run with `--data_dir`.

Open-set conditions are selected with `--states`, e.g. `src4(known4)` keeps four
known classes out of nine, `K3(pi_u=0.571)` is the RflyMAD variant.

## Usage

Train the full model on one SDUST condition:

```bash
python DVC-HE.py --method og_moe --data_name SDUST \
  --src_name 2 --tgt_name 3 --states 'src4(known4)' \
  --epochs 50 -i 500 --seed 202401 \
  --og-shared 1 --og-expert-a full \
  --log-root logs/DVC-HE
```

Train the two baselines:

```bash
python DVC-HE.py --method mfcl --data_name SDUST \
  --src_name 2 --tgt_name 3 --states 'src4(known4)' \
  --epochs 50 -i 500 --seed 202401 --log-root logs/mfcl

python DVC-HE.py --method maosdan_shift --data_name SDUST \
  --src_name 2 --tgt_name 3 --states 'src4(known4)' \
  --epochs 50 -i 500 --seed 202401 --log-root logs/maosdan_shift
```

Evaluate a trained checkpoint:

```bash
python DVC-HE.py --method og_moe --data_name SDUST \
  --src_name 2 --tgt_name 3 --states 'src4(known4)' --phase test \
  --og-shared 1 --log-root logs/DVC-HE
```

Batch run over conditions and seeds:

```bash
python Batch_testing.py --methods og_moe mfcl maosdan_shift --runs 5
```

Reported metrics are H-score, OS*, UK and OS, printed per run and written to
`logs/` together with t-SNE and confusion-matrix figures.


