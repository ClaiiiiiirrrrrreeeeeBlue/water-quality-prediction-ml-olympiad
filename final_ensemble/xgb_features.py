"""特征工程：类别交叉、频数、折内目标统计和条件残差。"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

CAT = [f"category{x}" for x in "ABCDEF"] + ["unit"]
CROSSES = {
    "A_E": ["categoryA", "categoryE"],
    "A_D": ["categoryA", "categoryD"],
    "A_F": ["categoryA", "categoryF"],
    "unit_E": ["unit", "categoryE"],
    "B_D_F": ["categoryB", "categoryD", "categoryF"],
}
TE_KEYS = ["unit", "categoryA", "categoryE", "A_E", "A_D", "A_F", "unit_E"]


class FeatureBuilder:
    def __init__(self, mode="base", smooth=10, seed=42, dropout=0):
        self.mode, self.smooth, self.seed, self.dropout = mode, smooth, seed, dropout

    def basic(self, frame):
        z = frame.drop(columns=["id", "result"], errors="ignore").copy()
        for c in CAT:
            z[c] = z[c].fillna("__MISSING__").astype(str)
        if self.mode != "base":
            num = [c for c in z if c not in CAT]
            z["missing_numeric"] = z[num].isna().sum(axis=1).astype(np.float32)
            z["featureF_nonzero"] = (z.featureF.fillna(0) != 0).astype(np.float32)
            for name, columns in CROSSES.items():
                z[name] = z[columns].agg("|".join, axis=1)
        return z

    def stats_fit(self, z, y):
        global_mean = float(np.mean(y))
        maps = {}
        for key in TE_KEYS:
            table = pd.DataFrame({"key": z[key].to_numpy(), "y": y}).groupby("key").y.agg(["sum", "count", "std"])
            maps[key] = {
                "mean": (table["sum"] + self.smooth * global_mean) / (table["count"] + self.smooth),
                "std": table["std"].fillna(0),
            }
        # A/E-conditional residual: distinguish group_C effects from its target mix.
        if "residual" in self.mode:
            base = z.A_E.map(maps["A_E"]["mean"]).to_numpy()
            residual = y - base
            s = pd.DataFrame({"key": z.categoryC.to_numpy(), "r": residual}).groupby("key").r.agg(["sum", "count"])
            maps["C_residual"] = s["sum"] / (s["count"] + self.smooth)
            if "more" in self.mode:
                scale = z.unit.map(maps["unit"]["std"]).fillna(0).to_numpy() + 0.03
                normal = residual / scale
                s = pd.DataFrame({"key": z.categoryC.to_numpy(), "r": normal}).groupby("key").r.agg(["sum", "count"])
                maps["C_normal_residual"] = s["sum"] / (s["count"] + self.smooth)
                for key, labels in [
                    ("A_C_residual", z.categoryA + "|" + z.categoryC),
                    ("unit_C_residual", z.unit + "|" + z.categoryC),
                ]:
                    s = pd.DataFrame({"key": labels.to_numpy(), "r": residual}).groupby("key").r.agg(["sum", "count"])
                    maps[key] = s["sum"] / (s["count"] + 3)
        return global_mean, maps

    def stats_transform(self, z, fitted):
        mean, maps = fitted
        result = pd.DataFrame(index=z.index)
        for key in TE_KEYS:
            result[f"te_{key}"] = z[key].map(maps[key]["mean"]).fillna(mean)
            result[f"te_std_{key}"] = z[key].map(maps[key]["std"]).fillna(0)
        if "C_residual" in maps:
            result["te_C_residual"] = z.categoryC.map(maps["C_residual"]).fillna(0)
        if "C_normal_residual" in maps:
            result["te_C_normal_residual"] = z.categoryC.map(maps["C_normal_residual"]).fillna(0)
            result["te_A_C_residual"] = (z.categoryA + "|" + z.categoryC).map(maps["A_C_residual"]).fillna(0)
            result["te_unit_C_residual"] = (z.unit + "|" + z.categoryC).map(maps["unit_C_residual"]).fillna(0)
        return result.astype(np.float32)

    def fit_transform(self, frame, y):
        z = self.basic(frame)
        self.cat_columns = list(z.select_dtypes(include="object").columns)
        self.levels = {c: z[c].unique().tolist() for c in self.cat_columns}
        self.counts = {c: z[c].value_counts() for c in self.cat_columns}
        if "te" in self.mode:
            inner = KFold(5, shuffle=True, random_state=self.seed)
            encoded = None
            self.inner_fold_ids = np.empty(len(z), dtype=int)
            for k, (tr, va) in enumerate(inner.split(z)):
                fitted = self.stats_fit(z.iloc[tr], np.asarray(y)[tr])
                values = self.stats_transform(z.iloc[va], fitted)
                if encoded is None:
                    encoded = pd.DataFrame(np.nan, index=z.index, columns=values.columns, dtype=np.float32)
                encoded.iloc[va] = values.to_numpy()
                self.inner_fold_ids[va] = k
            self.target_stats = self.stats_fit(z, np.asarray(y))
            z = pd.concat([z, encoded], axis=1)
        z = self.finish(z)
        if self.dropout:
            # Covariate regularization: simulate unavailable group_C information.
            mask = np.random.default_rng(self.seed).random(len(z)) < self.dropout
            z.loc[mask, "categoryC"] = np.nan
            if "freq_categoryC" in z:
                z.loc[mask, "freq_categoryC"] = 0
            for c in ["te_C_residual", "te_C_normal_residual", "te_A_C_residual", "te_unit_C_residual"]:
                if c in z:
                    z.loc[mask, c] = 0
        return z

    def transform(self, frame):
        z = self.basic(frame)
        if "te" in self.mode:
            z = pd.concat([z, self.stats_transform(z, self.target_stats)], axis=1)
        return self.finish(z)

    def finish(self, z):
        z = z.copy()
        if self.mode != "base":
            for c in self.cat_columns:
                counts = z[c].map(self.counts[c]).fillna(0)
                z[f"freq_{c}"] = np.log1p(counts).astype(np.float32)
        for c in self.cat_columns:
            z[c] = pd.Categorical(z[c], categories=self.levels[c])
        for c in z.select_dtypes(include="number"):
            z[c] = z[c].astype(np.float32)
        return z
