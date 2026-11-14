"""v7, v2, 50:50의 공식 Validation과 실험 단위 교차검증을 비교합니다."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import model as M
from resnet18_v2.model import validate_frame


def read_split(data_dir, split, columns, encoder, image_jobs):
    root = data_dir / split
    frame = pd.read_csv(root / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    validate_frame(frame, require_labels=True)
    print(f'[{split}] {len(frame):,}행 / {frame.experiment_id.nunique()}개 실험 피처 계산 중', flush=True)
    features = M.build_features(
        frame, [root, data_dir], columns, encoder, image_jobs=image_jobs, split=split,
    )
    labels = pd.to_numeric(frame['label']).to_numpy(dtype=int)
    return frame, labels, features


def macro_f1(actual, predicted):
    return f1_score(actual, predicted, labels=M.LABELS, average='macro', zero_division=0)


def report(title, actual, predictions):
    scores = pd.DataFrame({
        name: f1_score(actual, predicted, labels=M.LABELS, average=None, zero_division=0)
        for name, predicted in zip(M.MODEL_NAMES, predictions)
    }, index=[f'{stage}단계' for stage in M.LABELS])
    scores.loc['Macro F1'] = scores.mean()
    print(f'\n{title} ({len(actual):,}행)')
    print(scores.to_string(float_format='{:.4f}'.format))
    for actual_stage, predicted_stage in ((2, 3), (3, 2), (4, 5), (5, 4)):
        errors = [
            int(((actual == actual_stage) & (predicted == predicted_stage)).sum())
            for predicted in predictions
        ]
        print(f'{actual_stage}→{predicted_stage}: ' +
              ' | '.join(f'{name} {count:,}건' for name, count in zip(M.MODEL_NAMES, errors)))


def report_validation(frame, actual, predictions):
    report(f'공식 Validation 전체 · {frame.experiment_id.nunique()}개 실험', actual, predictions)
    stage5 = actual == 5
    if not stage5.any():
        return

    longest = frame.loc[stage5, 'experiment_id'].value_counts().idxmax()
    selected = frame['experiment_id'].eq(longest).to_numpy(dtype=bool)
    keep = ~selected
    if keep.any():
        report('5단계 행이 가장 많은 실험을 제외한 Validation',
               actual[keep], tuple(predicted[keep] for predicted in predictions))

    times = pd.to_numeric(frame['relative_time_s']).to_numpy()
    within_stage5 = selected & stage5
    start = times[within_stage5].min()
    print(f'\n긴 실험 {longest} · 5단계 {within_stage5.sum():,}행 · 시작 {start:g}초')
    for name, predicted in zip(M.MODEL_NAMES, predictions):
        hits = within_stage5 & (predicted == 5)
        delay = f'{times[hits].min() - start:g}초' if hits.any() else '감지 없음'
        misses = (within_stage5 & (predicted == 4)).sum()
        early = (selected & (times < start) & (predicted >= 5)).sum()
        print(f'{name} | 첫 5단계 감지 지연 {delay} | 5→4 {misses:,}건 | 시작 전 5·6 예측 {early:,}건')


def run_cv(groups, actual, features, models, splitter):
    name = type(splitter).__name__
    oof = np.empty((len(M.MODEL_NAMES), len(actual)), dtype=int)
    counts = np.zeros(len(actual), dtype=int)
    fold_scores = []
    print(f'\n{name} · Train {groups.nunique()}개 실험', flush=True)
    for fold, (fit, holdout) in enumerate(splitter.split(features[0], actual, groups), start=1):
        if not set(groups.iloc[fit]).isdisjoint(groups.iloc[holdout]):
            raise ValueError('동일한 실험이 학습과 검증에 포함됐습니다.')
        print(f'fold {fold}/{splitter.n_splits} 학습 중', flush=True)
        fitted = tuple(clone(model).fit(values.iloc[fit], actual[fit])
                       for model, values in zip(models, features))
        predicted = M.predict_all(fitted, tuple(values.iloc[holdout] for values in features))
        oof[:, holdout] = predicted
        counts[holdout] += 1
        scores = [macro_f1(actual[holdout], labels) for labels in predicted]
        fold_scores.append(scores)
        score_text = ' | '.join(f'{member} {score:.4f}' for member, score in zip(M.MODEL_NAMES, scores))
        print(f'fold {fold} · {groups.iloc[holdout].nunique()}개 실험 | {score_text}', flush=True)

    if not np.all(counts == 1):
        raise ValueError('모든 Train 행은 폴드 밖에서 정확히 한 번 예측되어야 합니다.')
    report(f'{name} OOF', actual, oof)
    means = np.mean(fold_scores, axis=0)
    print('폴드별 Macro F1 평균 | ' +
          ' | '.join(f'{member} {score:.4f}' for member, score in zip(M.MODEL_NAMES, means)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--weights_dir', type=Path, default=M.WEIGHTS_DIR)
    parser.add_argument('--cv', choices=('group', 'stratified', 'both'))
    parser.add_argument('--image_jobs', type=int, default=4)
    args = parser.parse_args()
    if args.image_jobs < 1:
        parser.error('--image_jobs는 1 이상이어야 합니다.')

    models, columns, encoder = M.load_models(args.weights_dir)
    val, actual, features = read_split(args.data_dir, 'validation', columns, encoder, args.image_jobs)
    report_validation(val, actual, M.predict_all(models, features))
    if args.cv:
        train, actual, features = read_split(args.data_dir, 'train', columns, encoder, args.image_jobs)
        if not set(train['experiment_id']).isdisjoint(val['experiment_id']):
            raise ValueError('Train과 Validation에 동일한 experiment_id가 있습니다.')
        del encoder
        groups = train['experiment_id']
        count = min(5, groups.nunique())
        if count < 2:
            raise ValueError('교차검증에는 적어도 두 개의 Train 실험이 필요합니다.')
        if args.cv in ('group', 'both'):
            run_cv(groups, actual, features, models, GroupKFold(n_splits=count))
        if args.cv in ('stratified', 'both'):
            splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=42)
            run_cv(groups, actual, features, models, splitter)


if __name__ == '__main__':
    main()
