from pathlib import Path
import sys
import pickle
import numpy as np
from sklearn.model_selection import KFold
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'_shared'))
from course_utils import DATA_DIR, load_data, rmsle, save_outputs
from model import MEMBERS, make_features, fit_model, predict_log, best_rounds


def train_member(member, X, target, T, folds, output_dir):
    directory = Path(output_dir)/'checkpoints'/member['key']
    directory.mkdir(parents=True, exist_ok=True)
    oof_log = np.zeros(len(X))
    bag_log = np.zeros(len(T))
    rounds = []
    for fold, (train_index, valid_index) in enumerate(folds):
        features = make_features(member)
        A = features.fit_transform(X.iloc[train_index], target[train_index])
        B = features.transform(X.iloc[valid_index])
        estimator = fit_model(member, A, target[train_index], B, target[valid_index])
        oof_log[valid_index] = predict_log(estimator, B, member['family'])
        bag_log += predict_log(estimator, features.transform(T), member['family'])/len(folds)
        rounds.append(best_rounds(estimator, member['family']))
        for name, value in [('builder', features), ('model', estimator)]:
            with (directory/f'fold{fold}_{name}.pkl').open('wb') as file:
                pickle.dump(value, file)
        print(member['key'], 'fold', fold+1, 'rounds', rounds[-1])
    # 轮数来自验证；全量模型使用全部训练数据拟合。
    features = make_features(member)
    A = features.fit_transform(X, target)
    count = int(np.median(rounds))
    estimator = fit_model(member, A, target, rounds=count)
    full_log = predict_log(estimator, features.transform(T), member['family'])
    for name, value in [('builder', features), ('model', estimator)]:
        with (directory/f'full_{name}.pkl').open('wb') as file:
            pickle.dump(value, file)
    return oof_log, 0.5*bag_log+0.5*full_log, count

def run(data_dir=DATA_DIR, output_dir=HERE/"outputs"):
    train, test = load_data(data_dir)
    X = train.drop(columns=['id','result'])
    T = test.drop(columns=['id'])
    target = np.log1p(train.result.to_numpy())
    folds = list(KFold(5, shuffle=True, random_state=42).split(X))

    oof_log = np.zeros(len(train))
    test_log = np.zeros(len(test))
    training_rounds = {}
    for member in MEMBERS:
        member_oof, member_test, count = train_member(member, X, target, T, folds, output_dir)
        # 权重是冻结的历史开发结果，本课不再读取榜单调权重。
        oof_log += member['weight']*member_oof
        test_log += member['weight']*member_test
        training_rounds[member['key']] = count

    oof = np.clip(np.expm1(oof_log),0,1)
    predictions = np.clip(np.expm1(test_log),0,1)
    score = rmsle(train.result, oof)
    metrics = {'oof_rmsle':score, 'rounds':training_rounds,
               'weights':'members.json 中的冻结权重',
               'note':'这些权重历史上使用过相同验证划分，OOF 不是独立盲测。'}
    save_outputs(output_dir, train, test, oof, predictions, metrics)
    print('本次 OOF RMSLE:', score)
    return metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=DATA_DIR)
    parser.add_argument('--output-dir',type=Path,default=HERE/'outputs')
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)
