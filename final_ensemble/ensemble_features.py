"""层级平滑：稀疏交叉组合向父组均值收缩。"""
import numpy as np
import pandas as pd
from xgb_features import FeatureBuilder, TE_KEYS


class EnsembleFeatureBuilder(FeatureBuilder):
    def __init__(self, mode="te_residual_more", smooth=10, seed=42, dropout=0, pair_smooth=3):
        super().__init__(mode, smooth, seed, dropout)
        self.pair_smooth=pair_smooth

    def stats_fit(self, z, y):
        if "hier" not in self.mode:
            return super().stats_fit(z, y)
        mean, maps = super().stats_fit(z, y)
        for key, parent in [("categoryA", "unit"), ("A_E", "categoryA"),
                            ("A_D", "categoryA"), ("A_F", "categoryA"), ("unit_E", "unit")]:
            prior = z[parent].map(maps[parent]["mean"]).fillna(mean).to_numpy()
            table = pd.DataFrame({"key": z[key].to_numpy(), "y": y, "prior": prior}).groupby("key").agg(
                total=("y", "sum"), count=("y", "count"), prior=("prior", "mean"))
            maps[key]["mean"] = (table.total + self.smooth * table.prior) / (table["count"] + self.smooth)
        base = z.A_E.map(maps["A_E"]["mean"]).to_numpy()
        residual = y - base
        normal = residual / (z.unit.map(maps["unit"]["std"]).fillna(0).to_numpy() + 0.03)
        for key, labels, values, smooth in [
            ("C_residual", z.categoryC, residual, self.smooth),
            ("C_normal_residual", z.categoryC, normal, self.smooth),
            ("A_C_residual", z.categoryA + "|" + z.categoryC, residual, self.pair_smooth),
            ("unit_C_residual", z.unit + "|" + z.categoryC, residual, self.pair_smooth),
        ]:
            table = pd.DataFrame({"key": labels.to_numpy(), "r": values}).groupby("key").r.agg(["sum", "count"])
            maps[key] = table["sum"] / (table["count"] + smooth)
        return mean, maps

    def stats_transform(self, z, fitted):
        result = super().stats_transform(z, fitted)
        if "hier" not in self.mode:
            return result
        mean, maps = fitted
        result["te_categoryA"] = z.categoryA.map(maps["categoryA"]["mean"]).fillna(result.te_unit)
        for key, parent in [("A_E", "categoryA"), ("A_D", "categoryA"), ("A_F", "categoryA"), ("unit_E", "unit")]:
            result[f"te_{key}"] = z[key].map(maps[key]["mean"]).fillna(result[f"te_{parent}"])
        return result.astype(np.float32)
