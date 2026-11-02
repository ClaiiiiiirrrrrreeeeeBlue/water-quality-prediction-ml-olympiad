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
    # target 已经是 log1p(result)，且这里只收到当前训练折。
    return {'mean': float(np.mean(target))}

def predict_model(estimator, frame):
    return np.full(len(frame), estimator['mean'])

def best_rounds(estimator):
    return 0  # 分组统计没有“树的轮数”。
