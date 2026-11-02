"""原始 26 列的类型整理：不生成目标编码或其他新特征。"""
import numpy as np
import pandas as pd

CAT = [f'category{x}' for x in 'ABCDEF'] + ['unit']

class FeatureBuilder:
    def basic(self, frame):
        result = frame.drop(columns=['id', 'result'], errors='ignore').copy()
        for column in CAT:
            result[column] = result[column].fillna('__MISSING__').astype(str)
        return result

    def fit_transform(self, frame, target):
        result = self.basic(frame)
        self.levels = {column: result[column].unique().tolist() for column in CAT}
        return self.finish(result)

    def transform(self, frame):
        return self.finish(self.basic(frame))

    def finish(self, result):
        result = result.copy()
        for column in CAT:
            result[column] = pd.Categorical(result[column], categories=self.levels[column])
        for column in result.select_dtypes(include='number'):
            result[column] = result[column].astype(np.float32)
        return result
