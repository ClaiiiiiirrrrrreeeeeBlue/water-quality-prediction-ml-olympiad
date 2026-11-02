# 相同特征工程下的单算法线上对照

提交日期：2026-10-07。三份文件均来自各自的全量单模型，全部评分完成。

共同条件：63 列层级平滑与残差特征，目标 `log1p(result)`，外层五折种子 42，特征内层交叉拟合种子 42，类别信息遮蔽 20%，普通组平滑强度 10，稀疏组合残差平滑强度 3。模型训练轮数为五折最佳轮数中位数。没有平均折模型，也没有跨算法融合。

| 算法 | OOF RMSLE | Public RMSLE | Private RMSLE | 全量轮数 | 提交 ID |
|---|---:|---:|---:|---:|---:|
| xgboost | 0.052474 | 0.05675 | 0.05554 | 477 | 56897881 |
| catboost | 0.052566 | 0.05659 | 0.05496 | 1177 | 56897883 |
| lightgbm | 0.052510 | 0.05716 | 0.05541 | 298 | 56897886 |
| 最佳集成（此前提交） | 0.052083 | 0.05650 | **0.05479** | 30 个模型 | 56703068 |

XGB 深度 5；CatBoost 深度 6；LightGBM 为 31 叶子。CatBoost 单模型 Private 最好，为 0.05496；集成的 Private 优势为 0.00017，差距较小，不能据此宣称稳定显著优于单模型。LightGBM 的 OOF 最低，却没有获得最佳 Private，体现本地与测试子集排名差异。

XGB 的种子 42 对照文件在前次清理中删除，因此按同一配置用 CPU 重新训练。此前 GPU 版本 OOF 为 0.052562，本次 CPU 重训为 0.052474；报告和提交回执采用本次实际结果，不沿用旧分数。CatBoost / LightGBM 使用保留的原始全量预测文件。

之前 Private 0.05686 的 XGB 使用前一阶段特征工程（没有本次层级平滑且类别遮蔽 30%），不可将它替代上述横向对照。

提交到最初用于本项目的 Kaggle 账号，该账号能看到原始最佳集成提交。仅在本次提交进程中使用原先提供的凭据，没有改写本机当前凭据文件。本轮使用 3 次提交机会。

产物：

- `submissions/single_algorithms/xgboost.csv`、`catboost.csv`、`lightgbm.csv`：已提交文件。
- `artifacts/kaggle/single_algorithm_receipts.json`：提交 ID、分数、配置和文件 SHA256。
- `artifacts/kaggle/single_algorithm_scores.csv`：比较成绩表。
- `artifacts/ensemble/5fold/xgb_hier_d5_s42_drop20_smooth10/`：重训 XGB 的模型、特征处理器、OOF 和指标。

集成方案详情见 [集成实验报告](ensemble_results.md)。
