# 置换重要性与特征删减实验

日期：2026-10-07。已完成当前 63 列特征的验证折置换分析、删减重训及 3 次 Kaggle 提交。结论：LightGBM 单模仅有极小的线上改善，最佳集成 Private **仍为 0.05479**，没有明确继续降低。

## 方法

使用此前同一套层级平滑、残差特征的三种模型：CatBoost 深度 6、LightGBM 31 叶子、XGB 深度 5，模型种子 42、外层五折种子 42。加载每折已训练模型，只在该折未参与训练的 2,600 行上计算，未在训练集上做重要性评估。

定义：`重要性 = 置换后的验证 RMSLE - 原验证 RMSLE`。正值表示打乱后变差，模型依赖这一列；负值表示这次打乱略有改善，不自动表示删除并重训会更好。

每种模型检查 63 个单列和 12 个特征组，每个验证折重复 3 次。三种算法合计 3,375 个置换评分；每项汇总 15 个折/重复组合的均值、标准差及正贡献比例，另外记录合并 MSLE 后计算的整体 RMSLE 差。

分组包含：原始数值、composition、demographics、featureF 及非零标记、原始类别、交叉类别、频数、目标均值、目标标准差、四个条件残差、原始 C 信号及全部 C 信号。同组使用同一个行排列，保留组内关联。

这里置换的是**已经生成的模型输入列**，不会重新计算其他衍生特征；因而可能产生与原始类别或单位不一致的人工组合。结果描述固定模型对这些输入的依赖，不是因果贡献，也不是删列后的预期提升。相关特征会相互掩盖，分组置换用于补充单列结果，仍需重训验证。参见 [scikit-learn 官方说明](https://scikit-learn.org/stable/modules/permutation_importance)及[相关特征示例](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html)。

## 主要发现

以下为跨折、重复置换的平均 RMSLE 增量：

| 输入或特征组 | CatBoost | LightGBM | XGBoost |
|---|---:|---:|---:|
| `te_A_E` | 0.096507 | 0.146781 | 0.129966 |
| `te_unit_E` | 0.020531 | 0.008766 | 0.009557 |
| 四个条件残差一起置换 | 0.011962 | 0.014982 | 0.014968 |
| 目标标准差组 | 0.010530 | 0.001629 | 0.002007 |
| 频数特征组 | 0.004079 | 0.000427 | 0.000469 |
| 全部原始数值组 | 0.000715 | 0.000314 | 0.000306 |

条件均值和残差是主要信号。CatBoost 比另外两种模型更依赖标准差和频数。虽然很多原始数值列单独贡献很小，整组置换仍使三种模型变差，不能简单称为“所有数值特征都无用”。

负贡献或接近零的列依赖具体算法：CatBoost 的 `featureE` 最明显，平均增量约 -0.0000412，15 次中仅 2 次为正；LightGBM 的 `compositionG` 约 -0.0000110；XGB 的 `freq_unit`、`compositionF` 分别约 -0.0000068、-0.0000063。完整排名与波动见 CSV。

![各算法最高和最低的单列置换重要性](../artifacts/permutation/importance.png)

## 重训验证

保持特征映射的生成方式、模型参数和早停设置不变，生成 63 列后仅筛选模型输入。删除原始列的模型输入时，仍可保留由其他原始字段生成的衍生特征。

三个共同策略：

1. 保守删除：平均贡献为负且正贡献比例不超过 40%。
2. 非正贡献删除：删除平均贡献不大于零的列。
3. 保留重要性最高的 45 列。45 是候选数量，不是已证明的最优数量；每种算法的保留清单不同。

另对 CatBoost 单独删除 `featureE`，以及删除 `featureF` / 非零标记组。共 11 个完整随机五折删减实验。

| 算法 | 原 63 列 | 保守删除 | 非正贡献删除 | Top45 |
|---|---:|---:|---:|---:|
| CatBoost | 0.052566 | 0.052966（58 列） | 0.052809（53 列） | **0.052545** |
| LightGBM | 0.052510 | 0.052576（59 列） | 0.052546（50 列） | **0.052490** |
| XGBoost | **0.052474** | 0.052478（54 列） | 0.052493（47 列） | 0.052487 |

CatBoost 删除 `featureE` 后为 0.052707，删除 featureF 组后为 0.052702，都更差。**置换贡献最负的列，直接删除并重训仍可能更差**，这是本轮最值得讲解的反例。

CatBoost Top45 的五折中有四折改善，但第一折变差，合并 OOF 只改善约 0.000021。LightGBM Top45 改善约 0.000020。

LightGBM 另换 seed=2026 的外层五折检查：63 列为 0.052951021，45 列为 0.052951587，几乎持平且没有改善。换划分只是稳定性诊断，并非独立数据：置换分析已经使用全部训练数据的各个验证块来筛选特征。本轮 OOF 是开发分数，不是严格嵌套、一次盲测结果。

## 实际提交结果

| 方案 | OOF RMSLE | Public | Private | 提交 ID |
|---|---:|---:|---:|---:|
| LightGBM 原 63 列 | 0.052510 | 0.05716 | 0.05541 | 56897886 |
| LightGBM Top45 | 0.052490 | 0.05706 | **0.05539** | 56899408 |
| CatBoost 原 63 列 | 0.052566 | 0.05659 | **0.05496** | 56897883 |
| CatBoost Top45 | 0.052545 | 0.05675 | 0.05530 | 56899450 |
| 原最佳集成 | 0.052083 | 0.05650 | **0.05479** | 56703068 |
| 原集成 + CatBoost Top45 | 0.052071 | 0.05648 | **0.05479** | 56899454 |

全部提交状态 `COMPLETE`。新增融合的权重在查看其线上分数前，根据 OOF 固定：原集成 85.7747%，CatBoost Top45 成员 14.2253%；新成员内部为 50% 折模型对数平均、50% 全量模型对数预测。使用与原集成相同的 log1p 空间融合，优化权重的正则强度为 1e-7。

LightGBM 改善仅 0.00002，无法据此宣称稳健显著提升；新融合 Private 与原集成按平台显示精度完全相同，Public 改善 0.00002。原方案更简单，继续保留原 30 模型集成为主方案，不覆盖原始提交文件。

本轮只新增 3 次提交。结论是发现了可压缩的低贡献输入，但没有获得明确的整体 Private 上分。不能用低重要性或更低 OOF 代替真实线上结果。

## 文件与复现

- `src/permutation_importance_wqp.py`：验证折单列/分组置换。
- `src/train_permutation_ablation.py`：固定参数的特征删减消融。
- `src/predict_permutation_model.py`：加载保留的 Top45 模型与列掩码推理。
- `artifacts/permutation/{cat,lgb,xgb}/importance.csv`、`detail.csv`、`config.json`：排名、逐折重复记录、方法配置。
- `artifacts/permutation/all_importance.csv`、`consensus_importance.csv`：三算法汇总。
- `artifacts/permutation/ablation_results.csv`：11 个删减实验及 2 个换划分对照。
- `artifacts/permutation/blend_manifest.json`：新融合的冻结权重。
- `artifacts/kaggle/permutation_scores.csv`、`permutation_submission_receipts.json`：实际线上成绩和文件 SHA256。
- `submissions/permutation/`：3 个实际提交文件。

只保留 CatBoost / LightGBM Top45 的模型与预测作为实际提交的对照证据；未入选方案保留指标与列清单，清理大体积 checkpoint 和预测。需要重训未入选实验时使用 `--force`。

```bash
conda activate WQP
python src/permutation_importance_wqp.py --family cat
python src/permutation_importance_wqp.py --family lgb
python src/permutation_importance_wqp.py --family xgb
python src/train_permutation_ablation.py --family cat --selections top45
python src/train_permutation_ablation.py --family lgb --selections top45
python src/predict_permutation_model.py --member cat_top45 --output /tmp/cat_top45.csv
```

模型重载预测、保存 OOF 的重算分数和已提交文件 SHA256 均已核对。原最佳集成的冻结文件与依赖保持不变。
