# ML Olympiad — Water Quality Prediction

Final solution for the Kaggle *ML Olympiad — Water Quality Prediction* regression task.

| | RMSLE |
|---|---:|
| Local OOF (5-fold, seed 42) | **0.052083** |
| Kaggle Public | **0.05650** |
| Kaggle Private | **0.05479** |

Kaggle submission ref `57913552`. `submission/final_submission.csv` is the exact file
that was submitted (sha256 `e4092c9b…`), produced by the 30 fitted models described below.

## What this is

A fixed, five-member ensemble over three gradient-boosting libraries. The full pipeline
is in this repository: feature engineering, member training, weight selection and
prediction. There is no hyper-parameter search, no leaderboard feedback loop, and no
automatic submission.

- **63 engineered features**: raw fields, categorical crosses, support counts,
  hierarchical target statistics, and conditional residuals.
- **Target** is `log1p(result)`; predictions return through `expm1` and are clipped to `[0, 1]`.
- **30 fitted models**: 5 member configurations × (5 fold models + 1 full-data model).
- **Frozen weights**: selected once on out-of-fold predictions and then left alone.

Method details, including the leakage controls and the limitations of the reported
scores, are in [METHOD.md](METHOD.md). A longer, figure-heavy narrative — from the
exploratory analysis through to the ensemble — is in [WRITEUP.md](WRITEUP.md).

## Pipeline

Three stages, run in order. Each one reads what the previous stage wrote.

| stage | script | reads | writes |
|---|---|---|---|
| 1. train members | `final_ensemble/train.py` | `data/*.csv`, `members.json` | `checkpoints/`, `runs/` |
| 2. select and blend | `final_ensemble/blend.py` | `runs/`, `members.json` | `blend/` |
| 3. rebuild from models | `final_ensemble/predict.py` | `checkpoints/`, `blend/manifest.json` | `outputs/` |

Stage 2 and stage 3 should agree: stage 2 combines the per-member test predictions
saved during training, stage 3 reloads all 30 models and recomputes the same blend
from scratch.

```
final_ensemble/xgb_features.py        base feature builder: crosses, counts, target statistics, residuals
final_ensemble/ensemble_features.py   hierarchical priors layered on top of the base builder
final_ensemble/model.py               the five fixed member definitions
final_ensemble/train.py               stage 1, trains 30 models
final_ensemble/blend.py               stage 2, constrained weight selection and blending
final_ensemble/predict.py             stage 3, rebuild the prediction from the fitted models
final_ensemble/verify.py              checks the shipped artefacts against recorded values
final_ensemble/members.json           frozen weights and round counts
final_ensemble/common.py              shared I/O and metric helpers
submission/                           the exact submitted CSV
reference/                            OOF predictions and recorded metrics
figures/                              the 16 figures used by the writeup
WRITEUP.md                            full solution writeup
data/                                 put train.csv / test.csv here (not tracked)
```

## Quickstart

The competition data is not redistributed here. Download `train.csv`, `test.csv` and
`sample_submission.csv` from the Kaggle competition page into `data/`
(see [data/README.md](data/README.md)).

```bash
pip install -r requirements.txt

# Check the shipped submission without any data or training
python final_ensemble/verify.py

# Full pipeline: about 3.5 minutes on CPU
python final_ensemble/train.py      # 30 models -> checkpoints/, runs/
python final_ensemble/blend.py      # frozen weights -> blend/
python final_ensemble/predict.py    # rebuild -> outputs/submission.csv
```

To re-solve the weights instead of using the frozen ones:

```bash
python final_ensemble/blend.py --refit
```

On this five-member pool that produces slightly different weights (largest component
difference 0.0045, OOF `0.052083561` versus `0.052083415`). The frozen weights came
from an earlier solve over a larger member pool whose zero-weight members were then
dropped, so `--refit` is a diagnostic, not a reproduction path.

To re-derive the round counts from validation instead of applying the recorded ones:

```bash
python final_ensemble/train.py --early-stopping
```

## Reproducing the exact submission

Unpack [`checkpoints.zip`](https://github.com/ClaiiiiiirrrrrreeeeeBlue/water-quality-prediction-ml-olympiad/releases/download/v1.0-submission/checkpoints.zip)
into `./checkpoints` and run `final_ensemble/predict.py`. On the machine that produced
this repository the replayed file matches `submission/final_submission.csv` to within
`5e-9` per row; the residual comes from floating-point accumulation order across
library versions.

Running the full pipeline from scratch reaches the same result: a fresh
`train.py` + `blend.py` run with the frozen rounds reproduces the submitted
predictions to within `5e-9` and the OOF RMSLE to within `5e-12`.

## Honest notes

- The competition had already closed when this work was done. Public and Private
  scores were visible during development, so they are recorded results, not a blind test.
- Weights were fitted on the same 5-fold OOF split that is used to report 0.052083,
  so that number is a development estimate.
- The gap between OOF (0.0521) and Private (0.0548) is mostly distribution shift:
  35.56% of test rows contain a `categoryC` never seen in training, versus 4.46% of
  OOF rows. Reweighting the OOF errors to the test unseen rate predicts 0.0562,
  close to the observed Public 0.0565.
- The ensemble improves on the best single model (CatBoost, Private 0.05496) by
  0.00017. That is a small margin and is not claimed to be a stable advantage.

## Licence

MIT — see [LICENSE](LICENSE).
