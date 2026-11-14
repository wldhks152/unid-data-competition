"""저장된 v7과 v2로 50:50 앙상블 제출 CSV를 생성합니다."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import model as M
from resnet18_v2.model import validate_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--image_root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample_submission', type=Path)
    parser.add_argument('--weights_dir', type=Path, default=M.WEIGHTS_DIR)
    parser.add_argument('--image_jobs', type=int, default=4)
    args = parser.parse_args()
    if args.image_jobs < 1:
        parser.error('--image_jobs는 1 이상이어야 합니다.')

    frame = pd.read_csv(args.input, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    validate_frame(frame)
    if frame.empty:
        labels = np.empty(0, dtype=int)
    else:
        models, columns, encoder = M.load_models(args.weights_dir)
        features = M.build_features(
            frame, [args.image_root, args.image_root.parent], columns, encoder,
            image_jobs=args.image_jobs, split='test',
        )
        _, _, labels = M.predict_all(models, features)
    submission = pd.DataFrame({'sample_id': frame['sample_id'], 'label': labels})
    if args.sample_submission is not None:
        order = pd.read_csv(args.sample_submission, dtype={'sample_id': 'string'})['sample_id']
        if not order.is_unique or set(order) != set(submission['sample_id']):
            raise ValueError('제출 양식과 입력 CSV의 sample_id가 다릅니다.')
        submission = submission.set_index('sample_id').loc[order].reset_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f'저장: {args.output} ({len(submission):,}행)')


if __name__ == '__main__':
    main()
