"""加载本课 checkpoint；不训练，不连接 Kaggle。"""
from pathlib import Path
import argparse
import pickle
import numpy as np
import pandas as pd
from model import predict_model  # 同时提供 pickle 所需的本地类/模块。

HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=HERE.parent/'data/test.csv')
    parser.add_argument('--checkpoint', type=Path, default=HERE/'checkpoint.pkl')
    parser.add_argument('--output', type=Path, default=HERE/'outputs/prediction.csv')
    args = parser.parse_args()
    raw = pd.read_csv(args.input)
    with args.checkpoint.open('rb') as file:
        bundle = pickle.load(file)
    result = np.zeros(len(raw))
    for features, estimator in bundle['estimators']:
        frame = features.transform(raw.drop(columns=['id']))
        result += np.clip(np.expm1(predict_model(estimator, frame)), 0, 1) / len(bundle['estimators'])
    if not np.isfinite(result).all():
        raise ValueError('预测出现无效数值。')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'id':raw.id, 'result':result}).to_csv(args.output, index=False)
    print('完成预测：', len(raw), '行；', args.output.resolve())

if __name__ == '__main__':
    main()
