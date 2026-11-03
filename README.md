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

A fixed, five-member ensemble over three gradient-boosting libraries. Everything is
deterministic and reproduced from source: there is no hyper-parameter search, no
leaderboard feedback loop, and no automatic submission.

- **63 engineered features**: raw fields, categorical crosses, support counts,
  hierarchical target statistics, and conditional residuals.
- **Target** is `log1p(result)`; predictions return through `expm1` and are clipped to `[0, 1]`.
- **30 fitted models**: 5 member configurations × (5 fold models + 1 full-data model).
- **Frozen weights**: selected once on out-of-fold predictions and then left alone.

Method details, including the leakage controls and the limitations of the reported
scores, are in [METHOD.md](METHOD.md).

## Layout

```
final_ensemble/xgb_features.py        base feature builder: crosses, counts, target statistics, residuals
final_ensemble/ensemble_features.py   hierarchical priors layered on top of the base builder
final_ensemble/model.py               the five fixed member definitions
final_ensemble/train.py               trains the members, writes OOF and a submission
final_ensemble/predict.py             replays a submission from saved checkpoints
final_ensemble/verify.py              checks the shipped artefacts against recorded values
final_ensemble/members.json        frozen weights and round counts
submission/                the exact submitted CSV
reference/                 OOF predictions and recorded metrics
data/                      put train.csv / test.csv here (not tracked)
```

## Quickstart

The competition data is not redistributed here. Download `train.csv`, `test.csv` and
`sample_submission.csv` from the Kaggle competition page into `data/`
(see [data/README.md](data/README.md)).

```bash
pip install -r requirements.txt

# 1. Check the shipped submission without any data or training
python final_ensemble/verify.py

# 2. Train all five members (~15 min on CPU) and write outputs/submission.csv
python final_ensemble/train.py

# 3. Or replay from the release checkpoints, without retraining
#    (unpack the release asset into ./checkpoints first)
python final_ensemble/predict.py --output outputs/submission.csv
```

`final_ensemble/train.py` uses the round counts in `final_ensemble/members.json` by default, so it
trains exactly the models that were submitted. Pass `--early-stopping` to re-derive
those counts from validation instead.

## Reproducing the exact submission

Unpack the release asset `checkpoints.zip` into `./checkpoints` and run
`final_ensemble/predict.py`. On the machine that produced this repository the replayed file
matches `submission/final_submission.csv` to within `5e-9` per row; the difference
comes from floating-point accumulation order across CPU/GPU and library versions.

A fresh `final_ensemble/train.py` run is statistically equivalent but not byte-identical: the
data, seeds and hyper-parameters are fixed, while the underlying library builds are not.

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
