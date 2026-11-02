from pathlib import Path
import sys
import pickle
import numpy as np
from sklearn.model_selection import KFold
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '_shared'))
from course_utils import DATA_DIR, load_data, rmsle, save_outputs
from model import make_features, fit_model, predict_model, best_rounds

def run(data_dir=DATA_DIR, output_dir=HERE/"outputs"):
    train, test = load_data(data_dir)
    X = train.drop(columns=['id', 'result'])
    T = test.drop(columns=['id'])
    target = np.log1p(train.result.to_numpy())
    print('训练集 / 测试集：', train.shape, test.shape)


    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X))
    oof = np.zeros(len(train))
    test_mean = np.zeros(len(test))
    fold_scores, round_counts, estimators = [], [], []
    for fold, (train_index, valid_index) in enumerate(folds):
        # 只在训练折拟合特征；验证折只 transform，不能泄漏标签。
        features = make_features()
        A = features.fit_transform(X.iloc[train_index], target[train_index])
        B = features.transform(X.iloc[valid_index])
        estimator = fit_model(A, target[train_index], B, target[valid_index])
        valid_prediction = np.clip(np.expm1(predict_model(estimator, B)), 0, 1)
        oof[valid_index] = valid_prediction
        score = rmsle(train.result.iloc[valid_index], valid_prediction)
        fold_scores.append(score)
        round_counts.append(best_rounds(estimator))
        print(f'Fold {fold + 1}: RMSLE={score:.6f}')
        C = features.transform(T)
        test_mean += np.clip(np.expm1(predict_model(estimator, C)), 0, 1) / len(folds)
        estimators.append((features, estimator))


    score = rmsle(train.result, oof)
    print(f'完整 OOF RMSLE: {score:.6f}')

    predictions = test_mean
    prediction_rule = 'fold_mean_raw'
    final_rounds = None


    metrics = {'oof_rmsle': score, 'fold_scores': fold_scores,
               'validation': 'KFold(5, shuffle=True, random_state=42)',
               'prediction_rule': prediction_rule, 'final_rounds': final_rounds}
    save_outputs(output_dir, train, test, oof, predictions, metrics)
    with (Path(output_dir) / 'checkpoint.pkl').open('wb') as file:
        pickle.dump({'prediction_rule': prediction_rule, 'estimators': estimators}, file)
    print('已保存到：', Path(output_dir).resolve())
    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=HERE/'outputs')
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)
