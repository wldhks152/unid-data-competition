# 1팀 · LightGBM v7 + ResNet18 CatBoost v2

Uni-DTHON 배터리 열폭주 6단계 분류의 앙상블 v9입니다. 각 모델이 낸 1~6단계 확률을 같은 순서로 맞춰 50:50으로 평균하고, 가장 높은 확률의 단계를 출력합니다. DAKER에서 평가한 파일은 `submission_ensemble_v9.csv`입니다.

## 모델 구성

| 모델 | 입력과 역할 |
|---|---|
| LightGBM v7 | 센서, 열화상 밝기 통계, 최근 120초 센서 피처 등 43개 |
| ResNet18 CatBoost v2 | 기본 피처 33개, 동결 ResNet18의 현재 열화상 임베딩 512개, 현재 센서 13개 |
| 앙상블 v9 | 두 모델의 확률을 같은 비중으로 평균한 뒤 최대 확률 단계 선택 |

시계열 피처는 같은 실험의 현재와 이전 관측으로 계산합니다. 제출 추론은 저장된 두 모델과 전처리를 사용합니다.

## 검증 결과

| Macro F1 | LightGBM v7 | ResNet18 CatBoost v2 | 앙상블 v9 |
|---|---:|---:|---:|
| 공식 Validation 전체, 19개 실험 | 0.8047 | 0.7456 | 0.7834 |
| 5단계 행이 가장 많은 실험 제외, 18개 실험 | 0.8993 | 0.9014 | 0.9048 |
| Train GroupKFold 5폴드 OOF | 0.8840 | 0.8877 | 0.8904 |
| Train StratifiedGroupKFold 5폴드 OOF, seed 42 | 0.8809 | 0.8899 | 0.8904 |
| DAKER Public | 0.8538848716 | 0.9305697537 | 0.8642385297 |

Train CV는 78개 실험을 `experiment_id` 단위로 분할했습니다. 두 모델을 같은 학습 폴드에서 각각 학습한 뒤 검증 폴드의 확률을 평균해 OOF로 집계했습니다.

Validation에서 5단계 행이 가장 많은 실험의 `5→4` 오류는 v7 299건, v2 533건, v9 393건이었습니다. 그 실험에서 첫 5단계 감지 지연은 각각 75초, 2,865초, 75초였습니다. 해당 실험을 제외한 점수는 실험 하나가 전체 Validation에 미치는 영향을 확인하기 위한 진단입니다.

제공 GPU 서버의 자가 점검에서 v9 CSV 3,451행을 재현했습니다. 반복 실행, 셔플, 실험 분리, 미래 행 제거에서 공통 행의 예측이 일치했습니다. 빈 이미지 검사에서는 예측의 4.93%가 바뀌었습니다.

## 실행 환경과 가중치

대회 제공 GPU 서버의 기존 Python 3.10 CUDA 가상환경에서 실행합니다. 라이브러리 버전은 `requirements.txt`를 참고하세요.

| 경로 | 파일 |
|---|---|
| `weights/lightgbm_v7.joblib` | GPU 서버에서 Validation과 제출에 사용한 v7의 `model.joblib` |
| `weights/resnet18_v2/model.joblib` | 팀 ResNet18 CatBoost v2의 원본 학습 가중치 |
| `weights/resnet18_v2/encoder/` | ResNet18 manifest 및 체크포인트 조각 3개 |

## 검증 및 제출 파일 재현

```bash
source ~/venv/bin/activate
DATA_DIR="$HOME/data/battery_data_light"

python -u validate.py --data_dir "$DATA_DIR" --cv both

python -u test.py \
  --input "$DATA_DIR/test/test.csv" \
  --image_root "$DATA_DIR/test" \
  --sample_submission "$DATA_DIR/sample_submission.csv" \
  --output submission_ensemble_v9.csv

python -u check_submission.py --data_dir "$DATA_DIR" --final submission_ensemble_v9.csv
```

운영진 재현 명령에는 `test.py --input … --image_root … --output …` 세 인자만 필요합니다. `--sample_submission`을 지정하면 공식 양식의 ID 순서로 저장합니다. `validate.py`에서 `--cv`를 생략하면 공식 Validation만 평가합니다.

## 모델 출처

- 팀 ResNet18 CatBoost v2의 코드와 가중치를 사용했습니다. [원본 커밋](https://github.com/wldhks152/unid-data-competition/commit/f92f029bb8bc57a5e486ac81b004931db0df38c9)
- 이미지 인코더는 ImageNet-1K 사전학습 ResNet18 `IMAGENET1K_V1`을 동결해 사용합니다. 구조 코드 라이선스는 `third_party/LICENSE.torchvision`에 있습니다. [PyTorch 모델 문서](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) · [공식 가중치](https://download.pytorch.org/models/resnet18-f37072fd.pth)
