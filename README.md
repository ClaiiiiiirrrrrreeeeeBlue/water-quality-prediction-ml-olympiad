# ML Olympiad Water Quality Prediction

A clean reproduction of the final submission for the Kaggle ML Olympiad Water Quality Prediction competition.

## Final result

The submitted solution is a fixed five-member ensemble using 63 engineered features:

- Local OOF RMSLE: **0.052083**
- Kaggle Public RMSLE: **0.05650**
- Kaggle Private RMSLE: **0.05479**
- Student-account submission: **57913552**

The solution was submitted successfully from the student account. This repository contains the final implementation, the final prediction file, and the reports that support the result. It intentionally excludes competition data, model binaries, temporary logs and credentials.

## Repository contents

- `final_ensemble/` — final feature engineering, model definitions, fixed member configuration, training and prediction scripts.
- `reports/ensemble_results.md` — final ensemble design and online result.
- `reports/xgb_feature_engineering.md` — core feature engineering and leakage controls.
- `reports/single_algorithm_comparison.md` — XGBoost, LightGBM and CatBoost comparison using the same features.
- `reports/参考与署名.md` — reference notebooks and attribution.
- `submission/final_submission.csv` — the exact final prediction file used for the student-account submission.
- `requirements.txt`, `environment.yml` — reproducible Python environment definitions.

## Final method

The feature pipeline combines categorical crosses, fold-local frequency statistics, hierarchical smoothed target statistics, conditional residuals and category-availability dropout. All target-derived features are fitted inside the training folds with inner cross-fitting.

The final prediction combines five fixed members:

- two CatBoost configurations;
- two LightGBM configurations;
- one XGBoost configuration.

Each member mixes its five-fold prediction average with its full-data model prediction in log1p space. The member weights are frozen in `final_ensemble/members.json`; they are not adjusted from the public or private leaderboard after submission.

## Reproducing locally

```bash
conda env create -f environment.yml
conda activate WQP_Teach
# Place train.csv, test.csv and sample_submission.csv in ./data/
python final_ensemble/train.py
python final_ensemble/predict.py
```

The training scripts expect the competition CSV files under `./data/`. The final CSV in `submission/` is provided for review and provenance. A fresh local run may differ slightly with package versions or hardware.

## Reports

- [Final ensemble report](reports/ensemble_results.md)
- [Feature engineering report](reports/xgb_feature_engineering.md)
- [Single-algorithm comparison](reports/single_algorithm_comparison.md)
- [Sources and attribution](reports/参考与署名.md)

This repository presents a completed competition result and its implementation details. It does not claim a theoretical optimum.
