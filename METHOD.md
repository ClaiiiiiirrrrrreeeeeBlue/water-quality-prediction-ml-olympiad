# Method

## Task

13,000 training rows and 7,000 test rows, 26 anonymised input fields (19 numeric,
7 categorical including `unit`). The target `result` lives in `[0, 1]` and is heavily
skewed towards zero (median 0.0279, mean 0.1442). The competition metric is RMSLE,
which is why the target is modelled as `t = log1p(result)` and inverted with `expm1`.

The field names are anonymised. `A`, `C`, `E` and `unit` are treated as grouping
codes only; no claim is made about what physical quantity they represent.

## Data structure that motivated the feature set

These are statistics of the training CSV, not assumptions:

- `unit` has 19 levels with very different target levels — mean `result` is 0.0019
  for `unit_6` and 0.6717 for `unit_7`. Two levels have 7 and 4 rows.
- After dropping `featureF`, the numeric fields take only 24 distinct joint patterns,
  and each `categoryE` maps to exactly one of them.
- 164 of 167 `categoryA` levels map to a single `unit`.
- 2,211 of 2,212 `categoryC` levels map to a single `categoryE`.
- `categoryC` has 2,212 training levels; 694 of them are absent from training but
  appear in test, covering **35.56%** of test rows. The equivalent OOF fraction is 4.46%.
- `featureF` is non-zero for 590 training rows and zero for every test row.

The first three observations say the categorical fields form a hierarchy
(`unit → A → A×E`, `C → E`). The last two say that a model relying on `categoryC`
must survive that column being unavailable at prediction time.

## Feature set (63 columns)

| Group | Count | Notes |
|---|---:|---|
| Raw fields | 26 | 19 numeric + 7 categorical |
| Crosses | 5 | `A×E`, `A×D`, `A×F`, `unit×E`, `B×D×F` |
| Marker flags | 2 | per-row count of missing numeric fields; `featureF` non-zero indicator |
| Support | 12 | `log1p(count)` for the 7 raw categoricals and the 5 crosses |
| Target means | 7 | `unit`, `A`, `E`, `A×E`, `A×D`, `A×F`, `unit×E` |
| Target std | 7 | same seven groups |
| Conditional residuals | 4 | see below |

All target-derived columns are computed from `t = log1p(result)`.

### Smoothed group means

For a group `g` with prior `p(g)` and strength `s = 10`:

```
m(g) = (Σ t_i over g + s · p(g)) / (count(g) + s)
```

Rare groups shrink towards their prior; unseen groups fall back to it. The final
solution uses a **hierarchical** prior rather than the global mean:

| group | shrinks towards |
|---|---|
| `A` | `unit` mean |
| `A×E`, `A×D`, `A×F` | `A` mean |
| `unit×E` | `unit` mean |
| `E`, `unit` | global mean |

Rare combinations therefore borrow strength from the right parent instead of being
pulled towards a global level that may be orders of magnitude away. For one sampled
`A×E` group (count 351) the hierarchical estimate is 0.000659 while the global-prior
estimate is 0.003901 — a 3.6× difference on the same group.

### Conditional residual

A `categoryC` level mixes different `A` and `unit` contexts, so its raw mean is
contaminated by which measurements happen to include it. The residual removes that
baseline first:

```
r_i        = t_i − m(A_i × E_i)
C_residual = Σ r_i over C=c / (count(c) + 10)
```

Unseen `C` falls back to 0, meaning "no evidence yet that this group deviates from
its baseline". Three companions are added:

- `r_i / (std(t | unit_i) + 0.03)` aggregated by `C`, so high-variance units do not dominate;
- `r` aggregated by `A×C` and by `unit×C` with prior strength 3, for finer granularity.

### Category-availability dropout

During training, a fixed random 20% of rows have `categoryC` set to missing,
`freq_categoryC` set to zero, and all four residual columns zeroed. At validation,
test and inference time nothing is masked; genuinely unseen levels simply fall back.

This is covariate regularisation for the fact that 35.56% of test rows carry an
unseen `categoryC`. It uses no test labels and no test IDs.

## Leakage control

Target statistics are generated with two nested levels of splitting:

1. An outer `KFold(5, shuffle=True, random_state=42)` defines the evaluation folds.
2. Inside each outer training fold, the training rows are split again into five inner
   folds. A row's target-derived features are computed only from the other four inner
   folds, so a row never contributes to its own encoding.
3. The outer validation rows and the test rows are transformed with statistics fitted
   on the whole outer training fold — never on validation or test labels.

The final full-data model repeats the inner cross-fitting for its training features
and uses statistics fitted on all training labels only for the test transform.

**Audit performed:** adding `+2` to a single row's target and regenerating features
leaves that row's coded columns unchanged, leaves every other row in its inner block
unchanged, and changes rows outside that block. That is the expected behaviour under
correct cross-fitting, and it was verified for both the base and hierarchical builders.

Frequency columns use no labels, so they are computed on the fold as a whole.

## Members and weights

| member | library | depth / leaves | feature mode | seed | weight |
|---|---|---|---:|---:|---:|
| `catboost_depth6` | CatBoost | depth 6 | hierarchical | 42 | 30.293% |
| `catboost_depth7` | CatBoost | depth 7 | hierarchical | 42 | 11.887% |
| `lightgbm_hierarchical` | LightGBM | 31 leaves | hierarchical | 42 | 29.737% |
| `lightgbm_global_prior` | LightGBM | 31 leaves | global prior | 42 | 14.464% |
| `xgboost_depth5` | XGBoost | depth 5 | hierarchical | 2026 | 13.619% |

`lightgbm_global_prior` deliberately keeps the older non-hierarchical prior; it is a
diversity source, not an oversight. Round counts were selected by early stopping
inside each fold, taking the median for the full-data model; they are frozen in
`final_ensemble/members.json`.

Each member produces two predictions in `log1p` space — the mean over its five fold
models (`b_m`) and its full-data model (`f_m`) — and the final prediction is

```
ŷ = clip(expm1(Σ_m w_m · (0.5 · b_m + 0.5 · f_m)), 0, 1)
```

`w` is the solution of a non-negative least-squares problem on OOF predictions, with
weights constrained to sum to 1 and a weak `1e-5` pull towards uniform weights.

## Results

| stage | OOF RMSLE | Private |
|---|---:|---:|
| raw 26 features, XGBoost | 0.058795 | 0.05951 |
| + conditional residual | 0.055119 | 0.05796 |
| + residual bundle and dropout | 0.053468 | 0.05686 |
| hierarchical priors, CatBoost single model | 0.052566 | 0.05496 |
| five-member ensemble | **0.052083** | **0.05479** |

Where the gain actually comes from, measured on OOF:

| step | OOF RMSLE | change |
|---|---:|---:|
| best single member | 0.052510 | — |
| equal-weight average of 5 members | 0.052104 | −0.00041 |
| OOF-optimised weights | 0.052083 | −0.00002 |

The weight optimisation is worth about one twentieth of a plain average. The real
improvement is in the features, not in stacking models. Member OOF residuals
correlate 0.955–0.994, and within a member the fold average correlates 0.999 with
the full-data prediction, so there is little error diversity left to exploit.

## Limitations

- **The 0.5/0.5 fold/full mix has no OOF evidence.** OOF is built from fold models
  only; the full-data prediction never enters it. The mix was a pre-registered
  candidate whose only evidence is the leaderboard (Private 0.05489 for fold-average
  only versus 0.05479 for the mix).
- **Weights were fitted on the reported split.** 0.052083 is a development score,
  not an independent estimate.
- **The weights were not re-solved on the five retained members.** Re-running the
  optimisation on these five gives slightly different weights (max difference 0.0045,
  OOF 0.052083561); the frozen values come from an earlier solve over a larger member
  pool, with the zero-weight members dropped.
- **Cold-start rows dominate the error.** OOF RMSLE is 0.05146 on rows whose
  `categoryC` was seen during training and 0.06399 on unseen ones.
- **Public and Private were visible.** The competition had closed; this is a
  competition result, not a blind academic evaluation, and it is not a theoretical bound.
