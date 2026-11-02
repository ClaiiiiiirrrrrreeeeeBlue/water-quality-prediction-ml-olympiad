# ML Olympiad Water Quality Prediction

这是一个面向高中生教学的 Kaggle 表格回归项目：从简单基线出发，逐步学习数据分析、无泄漏特征工程、三类 Boosting 算法和固定权重集成。

## Final result

The final teaching version is a fixed five-member ensemble using 63 engineered features:

- Local OOF RMSLE: **0.052083**
- Kaggle Public RMSLE: **0.05650**
- Kaggle Private RMSLE: **0.05479**
- Student-account submission: **57913552**

The final ensemble was submitted successfully. The repository intentionally excludes Kaggle tokens, competition data, model binaries and temporary logs. The included `submission/final_submission.csv` is the final prediction artifact; model training requires placing the competition CSV files in a local `data/` directory.

## Teaching path

1. `teaching/01_constant` — constant reference
2. `teaching/02_unit_grouping` — grouped statistic
3. `teaching/03_random_forest` — first complete tree model
4. `teaching/04_xgboost_basic` — raw-feature control
5. `teaching/05_xgboost_residual` — conditional residual features
6. `teaching/06_xgboost_enhanced` — residual scaling and category dropout
7. `teaching/07_xgboost_hierarchical` — hierarchical smoothing with XGBoost
8. `teaching/08_lightgbm_hierarchical` — same features, LightGBM
9. `teaching/09_catboost_hierarchical` — same features, CatBoost
10. `final_ensemble` — frozen five-member blend

Every lesson contains a README, notebook and source files. The reports in `reports/` document the data analysis, feature engineering, validation, permutation importance and online submissions.

## Running locally

```bash
conda env create -f environment.yml
conda activate WQP_Teach
# Put train.csv, test.csv and sample_submission.csv under ./data/
python final_ensemble/train.py
python final_ensemble/predict.py
```

For a quick walkthrough, open `final_ensemble/lesson.ipynb` or any numbered lesson notebook in JupyterLab. The historical leaderboard numbers are recorded as evidence from completed submissions; a new local run may differ slightly with hardware or package versions.

## Reports

- [Open-source solution analysis](reports/open_source_data_insights.md)
- [Baseline results](reports/baseline_results.md)
- [XGBoost feature engineering](reports/xgb_feature_engineering.md)
- [Ensemble results](reports/ensemble_results.md)
- [Permutation importance](reports/permutation_importance.md)
- [Single-algorithm comparison](reports/single_algorithm_comparison.md)
- [Sources and attribution](reports/参考与署名.md)

This is an educational reproduction and competition project, not a claim of a theoretical optimum.
