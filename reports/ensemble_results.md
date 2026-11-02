# 集成学习阶段：Private RMSLE 0.05479

> 清理说明：仅保留 7 个关键实验及最佳集成依赖。文中历史消融数字仍供教学参考，未入选实验文件已删除；当前文件与复现入口见 [项目索引](../README.md)。

实验日期：2026-09-30。目标 Private RMSLE < 0.055，已通过真实 Kaggle 提交达成。本轮观察到的最佳值为 **0.05479**，不是这个任务的理论上限。

> 2026-10-07 补交了同一套特征下的单算法模型：XGB Private 0.05554、CatBoost 0.05496、LightGBM 0.05541。详见 [单算法线上对照](single_algorithm_comparison.md)。

## 线上结果与最终候选

| 阶段 / 候选 | 本地 OOF RMSLE | Public | Private | 提交 ID |
|---|---:|---:|---:|---:|
| 上一阶段 XGB 单模型 | 0.053468 | 0.05813 | 0.05686 | 56702392 |
| 第一轮：OOF 权重 + 折模型平均 | 0.052083 | 0.05669 | 0.05489 | 56703064 |
| 第一轮：未见类别重加权 + 折模型平均 | 0.052134 | 0.05664 | 0.05484 | 56703066 |
| **第一轮：OOF 权重 + 折模型/全量模型混合** | **0.052083** | **0.05650** | **0.05479** | **56703068** |
| 第二轮：OOF 权重 + 折模型/全量模型混合 | 0.051954 | 0.05642 | 0.05487 | 56703172 |
| 第二轮：重加权 + 折模型/全量模型混合 | 0.052002 | 0.05639 | 0.05489 | 56703174 |
| 第二轮：已见/未见类别分别选权重 | 0.051950 | 0.05662 | 0.05494 | 56703176 |

本轮新增 6 次提交，全部 `COMPLETE`。账号今天合计 21 次，剩余 79 次。清理后仅保留最佳提交回执 `artifacts/kaggle/ensemble_submission_receipts.json`，包含文件 SHA256；六个候选的历史成绩保留在上表。

最终提交文件为 `submissions/ensemble_best.csv`，与提交 56703068 的文件字节一致。它使用 5 个非零权重成员配置，每个成员包含 5 个折模型及 1 个全量模型，实际调用 **30 个已训练模型**。

![不同阶段的真实 Private 分数](../artifacts/ensemble/private_score_progress.png)

可用于讲课或导出的矢量图：`artifacts/ensemble/private_score_progress.svg`。

## 本轮进一步挖掘的数据特征

复用上一阶段的类别交叉、折内频数、均值与标准差、A/E 条件残差、单位标准化残差及 A×C / unit×C 残差。新的主要改进是**层级平滑和未见类别的回退策略**。

以前 A×E、unit×E 等组统计统一向全局均值收缩。对于不同尺度的单位、稀疏 A 类别，这可能把本应很大或很小的组拉向不合适的全局水平。

本轮改变为：

- A 的均值向其关联单位的均值收缩；少数 A 对应多单位时，对组内单位先验取平均。
- A×E、A×D、A×F 向 A 均值收缩。
- unit×E 向对应单位均值收缩。
- 未见 A 回退到单位均值；未见交叉组回退到对应父组均值。
- 条件残差重新基于层级 A×E 均值计算，其余交叉拟合原则保持一致。

例如：`m(A×E) = (sum(t in A×E) + s × m(A)) / (count(A×E) + s)`，其中 `t=log1p(result)`。

关键单模型对照，均为相同随机五折、相同训练预算：

| 算法配置 | 原先的全局平滑 | 层级平滑 |
|---|---:|---:|
| XGB，深度 5 | 0.053277 | 0.052562 |
| CatBoost，深度 6 | 0.053475 | 0.052566 |
| LightGBM，31 叶子 | 0.053183 | 0.052510 |

这是跨三种算法一致的本地增益，支持“分组先验应与数据层级匹配”的判断。

后续降低平滑强度（普通组从 10 改为 3，A×C / unit×C 残差分母先验从 3 改为 1）使 XGB / LightGBM 的 OOF 略有改善，但对 CatBoost 不是一致改善，且最终融合 Private 未超过第一轮。

还做了类别数字后缀的两折敏感性实验。它只有小幅局部变化，没有扩展为最终成员。匿名编号的排序不等于物理意义，不能未经验证解释为地理坐标、污染物顺序或时间。

## 集成设计与权重

本轮曾训练 18 个完整五折成员配置，清理后仅保留最佳融合实际使用的 5 个成员。实验覆盖 XGB、CatBoost、LightGBM，不同深度/叶数、平滑强度及两个 XGB 随机种子。统一外层 `KFold(5, shuffle=True, random_state=42)`，使每一行 OOF 可直接对应。

所有目标相关特征都在每个外层训练折内再做五折交叉拟合；所有外层验证标签均不进入该折特征映射。已通过标签扰动审计：改变某一训练行的标签，该行所在内层验证块的目标编码保持不变。

每个成员保留：

1. 该成员的 13,000 行 OOF。
2. 五个外层模型的测试预测，在 `log1p` 空间等权平均。
3. 用五折最佳轮数的中位数，在全部训练数据上训练的全量模型预测。

第一轮 OOF 最优权重（非负，和为 1）如下：

| 成员 | 权重 |
|---|---:|
| CatBoost，层级平滑，深度 6 | 30.293% |
| CatBoost，层级平滑，深度 7 | 11.887% |
| LightGBM，层级平滑，31 叶子 | 29.737% |
| LightGBM，原平滑，31 叶子 | 14.464% |
| XGB，层级平滑，深度 5，种子 2026 | 13.619% |

按算法汇总约为 CatBoost 42.18%、LightGBM 44.20%、XGB 13.62%。其余第一轮成员的最优权重为零，未纳入最终推理。

权重通过约束优化选择，使 OOF 对数目标的均方误差最小；另加 `1e-5 × sum((w-uniform_prior)^2)` 的弱正则。优化器为 SLSQP，权重限制为非负、总和为 1。没有根据 Private 分数手工调整这些权重。

对每个成员 m，记其折模型平均后的对数预测为 `b_m`，全量模型对数预测为 `f_m`。最终：

`prediction = clip(expm1(sum(w_m × (0.5 × b_m + 0.5 × f_m))), 0, 1)`。

50% 折模型平均 + 50% 全量模型是预先设定的候选之一，没有继续扫比例来刷 Private。其意义是平衡全量训练的信息量与多个折模型平均后的稳定性。

误差相关矩阵在 `artifacts/ensemble/blends/round1/error_correlations.csv`。用于判断互补性的是模型残差相关性，而不是仅查看预测相关性；预测高度相关仍可能存在可融合的误差差异。

## 未见类别与融合验证

随机 OOF 中未见 C 的行占约 4.46%，测试中约 35.56%。因此本轮除总体 OOF 权重外，还比较：

- 按测试未见类别比例，对已见/未见 C 的 OOF 误差重加权。
- 在两个子集分别拟合融合权重，测试时按 C 是否见过选择。
- 每种成员的折模型平均、全量模型、二者 50/50 混合。

第一轮总体 OOF 优化的本地分数为 0.052083；另做元层五折交叉拟合权重，得到约 0.052168，差异不大。第二轮对应为约 0.051954 / 0.052051。

这些元层诊断复用了已训练的基模型 OOF，并非将全部基模型重新做严格嵌套训练。基模型早停、成员筛选和权重选择也使用了本地验证，因此不能把 0.052083 视为一次独立盲测分数。

测试分布重加权只使用 C 的可观测出现关系，不使用测试标签。它只调整一个分布差异，不能保证准确代表完整测试风险。分组门控候选没有胜过固定权重候选。

## 对“机器学习上限”的解释

本轮可报告的结论是：**现有数据、这批特征与三类 boosting 的集成，已实现 Private 0.05479**，相对单 XGB 的 0.05686 降低约 3.64%。

第二轮 OOF 更低但 Private 略差，说明接近该量级后，局部验证改进不一定稳定反映在线上。六个融合候选的 Private 在 0.05479–0.05494 之间，不能把其中很小的差距夸大为稳定优势。

这不是已证明的全局最优、不可超越界限或统计意义上的性能上限；没有穷尽所有模型和特征。Private 在比赛结束后公开且已经被查看，最终候选也依据线上结果选出，因此应如实描述为竞赛开发中的最佳观测结果。

## 复现与文件

最佳方案可在 CPU 加载已训练模型推理，无需重新训练：

```bash
conda activate WQP
python src/predict_ensemble.py --round round1 --candidate oof_opt_mix --output artifacts/ensemble/replayed.csv
```

训练第一轮成员、生成融合的命令如下。脚本会跳过已有 `metrics.json` 的已完成成员；第二轮成员另有 `smooth=3, pair-smooth=1`，不要把所有成员自动重算出的权重误当作第一轮冻结权重。

```bash
python src/train_ensemble_members.py --models xgb_more_d4 xgb_more_d5 xgb_hier_d4 xgb_hier_d5 --folds 5 --device cuda:0
python src/train_ensemble_members.py --models xgb_hier_d4 xgb_hier_d5 --folds 5 --device cpu --seed 2026
python src/train_ensemble_members.py --models cat_more_d6 cat_hier_d6 cat_hier_d7 --folds 5
python src/train_ensemble_members.py --models lgb_more_31 lgb_hier_15 lgb_hier_31 --folds 5
```

冻结的第一轮权重和成员路径已保存到 `artifacts/ensemble/blends/round1/manifest.json`，直接使用推理命令即可准确复现。当前 `blend_ensemble.py` 默认汇总目录中全部已完成成员，用于新实验，已提交方案使用冻结 manifest。

主要文件：

- `src/ensemble_features.py`：层级先验与新特征处理；上一阶段 `xgb_features.py` 保持不变。
- `src/train_ensemble_members.py`：统一 OOF、折模型平均及全量训练。
- `src/blend_ensemble.py`：约束权重、分布重加权和门控对照。
- `src/predict_ensemble.py`：从冻结 manifest 和已训练模型复现。
- `artifacts/ensemble/5fold/summary.csv`（仅保留最佳集成的 5 个成员）：18 个完整五折成员的成绩。
- `artifacts/ensemble/blends/round1/manifest.json`：最佳方案冻结权重与成员。
- `artifacts/ensemble/best_candidate.json`：最佳提交、源代码和模型 SHA256、复现误差。
- `artifacts/kaggle/ensemble_online_scores.csv`：最佳候选的真实线上成绩（其余历史成绩见正文）。
- `submissions/ensemble_best.csv`：最终提交文件。

已核对完整成员 OOF、所有正式融合 OOF 的重算分数、测试 ID / 行数 / 有限性 / 数值范围，以及最佳文件 SHA256；重载 30 个模型后在 CPU 复现 7,000 行预测。具体最大差异保存在 best_candidate.json。

课堂路线可从 `unit` 分组预测，到随机森林、单 XGB 特征工程，再到这一版的层级统计、误差互补和 OOF 融合。模型调用是工具，数据结构发现、泄漏控制、失败实验及对分数差异的解释才是项目重点。
