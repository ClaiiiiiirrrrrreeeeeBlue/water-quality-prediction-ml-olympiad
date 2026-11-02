"""固定权重的最终预测；全部依赖来自本课程文件夹。"""
from pathlib import Path
import argparse
import pickle
import numpy as np
import pandas as pd
from model import MEMBERS, predict_log

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=HERE.parent/'data/test.csv')
    parser.add_argument('--checkpoint-dir',type=Path,default=HERE/'checkpoints')
    parser.add_argument('--output',type=Path,default=HERE/'outputs/prediction.csv')
    args=parser.parse_args()
    raw=pd.read_csv(args.input)
    X=raw.drop(columns=['id'])
    final_log=np.zeros(len(raw))
    count=0
    for member in MEMBERS:
        directory=args.checkpoint_dir/member['key']
        bag_log=np.zeros(len(raw))
        for fold in range(5):
            with (directory/f'fold{fold}_builder.pkl').open('rb') as file:
                features=pickle.load(file)
            with (directory/f'fold{fold}_model.pkl').open('rb') as file:
                estimator=pickle.load(file)
            if member['family']=='xgb':estimator.set_params(device='cpu',n_jobs=8)
            bag_log+=predict_log(estimator,features.transform(X),member['family'])/5
            count+=1
        with (directory/'full_builder.pkl').open('rb') as file:
            features=pickle.load(file)
        with (directory/'full_model.pkl').open('rb') as file:
            estimator=pickle.load(file)
        if member['family']=='xgb':estimator.set_params(device='cpu',n_jobs=8)
        full_log=predict_log(estimator,features.transform(X),member['family'])
        final_log+=member['weight']*(0.5*bag_log+0.5*full_log)
        count+=1
    predictions=np.clip(np.expm1(final_log),0,1)
    if not np.isfinite(predictions).all():raise ValueError('存在无效预测。')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'id':raw.id,'result':predictions}).to_csv(args.output,index=False)
    print('使用',count,'个已训练模型完成',len(raw),'行预测。')

if __name__=='__main__':main()
