import numpy as np
import pandas as pd

class IdentityFeatures:
    """本课不额外产生特征；保留统一的 fit/transform 课堂接口。"""
    def fit_transform(self, frame, target):
        return frame.copy()
    def transform(self, frame):
        return frame.copy()

def make_features():
    return IdentityFeatures()

def fit_model(frame, target, valid_frame=None, valid_target=None, rounds=None):
    global_mean = float(np.mean(target))
    stats = pd.DataFrame({'unit': frame.unit.to_numpy(), 'target': target}).groupby('unit').target.agg(['sum', 'count'])
    # 样本多时相信本组，样本少时更相信全局；20 是先验强度。
    means = (stats['sum'] + 20 * global_mean) / (stats['count'] + 20)
    return {'means': means, 'global_mean': global_mean}

def predict_model(estimator, frame):
    # 测试时遇到新 unit，使用训练折的全局均值。
    return frame.unit.map(estimator['means']).fillna(estimator['global_mean']).to_numpy()

def best_rounds(estimator):
    return 0  # 分组统计没有“树的轮数”。
