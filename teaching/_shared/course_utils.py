"""课程间只共享数据读取、评分与保存；模型和特征代码均在各课内。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'

def load_data(directory=DATA_DIR):
    directory = Path(directory)
    train = pd.read_csv(directory / 'train.csv')
    test = pd.read_csv(directory / 'test.csv')
    sample = pd.read_csv(directory / 'sample_submission.csv')
    if not sample.id.equals(test.id):
        raise ValueError('提交模板与测试集 ID 顺序不一致。')
    return train, test

def rmsle(y_true, y_pred):
    # 本比赛目标非负。log1p(x) 就是 log(1+x)。
    a = np.log1p(np.asarray(y_true))
    b = np.log1p(np.maximum(np.asarray(y_pred), 0))
    return float(np.sqrt(np.mean((a-b)**2)))

def save_outputs(directory, train, test, oof, predictions, scores):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(oof).all() or not np.isfinite(predictions).all():
        raise ValueError('预测中出现 NaN 或无穷大。')
    pd.DataFrame({'id': train.id, 'result': train.result,
                  'prediction': oof}).to_csv(directory/'oof.csv', index=False)
    pd.DataFrame({'id': test.id, 'result': predictions}).to_csv(
        directory/'submission.csv', index=False)
    (directory/'metrics.json').write_text(
        json.dumps(scores, ensure_ascii=False, indent=2)+'\n')
