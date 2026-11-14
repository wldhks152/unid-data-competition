# unid-data-competition

Uni-DTHON — 센서 시계열로 배터리 열폭주 단계(1~6)를 분류하는 솔루션 코드입니다.
평가지표는 **Macro F1**입니다.

## 대회 규칙 요약 (코드에 반영된 부분)
| 규칙 | 반영 위치 |
|---|---|
| 같은 실험의 **현재·과거 관측만** 사용 (미래 관측 금지) | `src/features.py`: 누적 통계와 과거 방향 시간창만 사용. `train.py`와 `predict.py`가 누수를 자동 검사 |
| 외부 데이터 · 사전학습 모델 사용 금지 | 제공 CSV만 사용하고, 모델은 처음부터 학습 |
| 코드 제출 | `train.py` → `predict.py`로 재현. 시드 고정, LightGBM `deterministic=True` |
| Test는 5/10/15초 간격이 섞여 있음 | 시간창은 행 수가 아닌 초 단위로 계산. 학습할 때 10·15초로 솎아낸 데이터를 함께 사용 |

## 데이터
대회 데이터는 저장소에 넣지 않습니다. `data/` 아래에 넣으면 CSV를 하위 폴더까지 찾아서 읽습니다 (`.gitignore`로 제외됨).
```
data/
└── battery_data_light/
    ├── train/train.csv            (+ images/)
    ├── validation/validation.csv  (+ images/)
    ├── test/test.csv              (+ images/)
    └── sample_submission.csv
```

## 환경
```bash
pip install -r requirements.txt
```

## 실행 방법
```bash
python eda.py --data data                          # 데이터 점검 (단조성, 간격, 조건 분포)
python train.py --data data --run lgb_v1           # 실험 단위 5-fold CV → outputs/lgb_v1/
python tune_pp.py --run lgb_v1                      # OOF로 후처리 비교 (raw/+w/cummax/hmm) → pp.json
python predict.py --data data --run lgb_v1 --pp hmm   # → outputs/lgb_v1/submission_hmm.csv
```
- `--pp none|cummax|hmm`: 후처리 방식을 고릅니다. `tune_pp.py` 출력에서 **가중 점수가 가장 높은 방식**을 쓰세요.
  - label은 모든 실험에서 단조 증가해요. `hmm`은 과거 확률만 누적하는 좌→우 forward 필터예요. cummax처럼 한 번 튄 예측이 고정되는 문제가 덜해요.
`predict.py` 옵션
- `--no-tune`: OOF로 튜닝한 클래스 배율 `w`를 적용하지 않습니다.

## 구조
```
src/data.py       로드, 상수, 다운샘플링
src/features.py   과거 정보만 쓰는 피처와 누수 검사
src/model.py      LightGBM, Macro-F1 클래스 배율 튜닝
src/postprocess.py  단조 단계 HMM forward 필터 (인과적)
eda.py / train.py / tune_pp.py / predict.py
EXPERIMENTS.md    실험·제출 기록 (제출 30회 제한)
```

## 모델 가중치
`outputs/<run>/bundle.joblib`은 `train.py`로 재생성할 수 있습니다. 100MB를 넘는 가중치는 외부 드라이브에 보관합니다.

- 가중치 다운로드: (링크 추가 예정)
