# 1팀 — 고정 ResNet18 + 추가 센서 + CatBoost v2

Public 0.9156013487을 기록한 `uni_d_resnet18_catboost_v1`을 기준으로, **현재 시점 센서 13개만 추가**하는 비교 실험입니다. 기존 모델은 별도 폴더에 보존합니다. 이미지 CNN은 ImageNet 사전학습 상태로 고정하며 공동학습하지 않습니다.

배포본에는 소스만 있습니다. 실제 대회 데이터·학습 가중치·예측 CSV는 포함하지 않습니다. 실제 학습·추론은 대회에서 제공한 팀 서버에서만 실행하세요.

## 변경한 입력

```text
기존 수치·이미지 통계·인과 시계열 특징 33개
+ 고정 ImageNet ResNet18 특징          512개
+ 새 원시 센서 현재 값                  13개
= 총                                  558개
```

앞의 **545개 특징은 v1과 값·순서·dtype가 같도록 유지**합니다. 추가 센서는 다음 순서입니다.

```text
meas_ch4, meas_co, meas_co2, meas_hcn, meas_hcl,
meas_hf, meas_n2o, meas_no, meas_no2, meas_so2,
thermal_t, atmp_t, indu_t
```

`thermal_t`는 제공된 열화상 온도 수치이며 이미지 픽셀별 온도가 아닙니다. 센서의 숫자 변환 실패·무한값은 NaN으로 처리하고 기존 median imputer를 그대로 적용합니다. `indu_t`의 구조적 결측도 이 방식으로 처리합니다. 전처리 통계는 각 fold의 학습 부분에서만 fit합니다. 특정 fold에서 한 특징이 전부 결측이면 기존 SimpleImputer 동작에 따라 내부적으로 제외될 수 있으며 같은 학습 파이프라인을 검증·추론에 그대로 적용합니다.

이번에는 배터리 조건(`bat_class`, `bat_cc`, `soc_status`, `tr_method`), 중복 온도 합(`meas_t1_plus_t2`), Test에 값이 없는 것으로 안내된 `indu_t.1`을 추가하지 않습니다. 새 센서의 이동 평균·최댓값·변화율도 추가하지 않습니다. 기존 33개 특징에 포함된 과거 30초 특징만 유지합니다.

## 유지한 학습·검증 설정

| 항목 | 설정 |
|---|---|
| 이미지 | RGB letterbox 224, ImageNet mean/std, 증강 없음, encoder 고정 |
| encoder 추출·추론 | FP32, 고정 batch=1, eval/inference_mode, 결정적 연산 |
| 전처리 | median SimpleImputer → StandardScaler |
| 분류기 | CatBoost CPU, 600 trees, depth=6, learning_rate=0.05, l2_leaf_reg=3 |
| 클래스 가중치 | none |
| seed | 42 |
| CV | Train의 experiment_id 기준 GroupKFold 5개, v1과 같은 분할 방식 |
| 최종 학습 | Train 전체만 사용 |
| Validation | 학습·전처리 fit·early stopping에 사용하지 않고 평가 |

검증할 가설은 **추가 센서가 실제 3단계를 2단계로, 실제 5단계를 4단계로 예측하는 오류를 줄이는가**입니다. 가스 센서나 특정 변수가 해당 오류를 해결한다고 미리 가정하지 않습니다.

참고할 v1 결과:

| 지표 | 고정 ResNet18+CatBoost v1 |
|---|---:|
| Train GroupKFold Macro F1 | 0.8820 ± 0.0369 |
| Validation Macro F1 | 0.7477 |
| Public | 0.9156013487 |
| Validation 실제3→예측2 | 590 / 3066 |
| Validation 실제5→예측4 | 545 / 1134 |

v1 Validation의 2단계 precision/recall은 0.5768/0.8696, 4단계는 0.3687/0.9091, 5단계는 0.9203/0.4277입니다. v2 로그에서 이 값들과 Macro F1·각 fold 결과를 함께 비교하세요. 일부 혼동 감소가 다른 오분류 증가와 맞바뀔 수 있습니다.

## 서버에서 학습

이 폴더를 서버의 `/home/ubuntu/jaebok/uni_d_resnet18_catboost_v2`에 올리고 기존 venv·tmux에서 실행합니다.

```bash
source /home/ubuntu/venv/bin/activate
cd /home/ubuntu/jaebok/uni_d_resnet18_catboost_v2
python -u train.py --data_dir /home/ubuntu/data/battery_data_light --pretrained_dir /home/ubuntu/jaebok/uni_d_resnet18_catboost_v1/weights
```

`--pretrained_dir`는 **고정 ResNet18+CatBoost v1의 weights 폴더**입니다. 내부 공식 encoder 파일을 검증한 뒤 새 v2의 `weights/encoder/`에 복사합니다. v1의 코드·모델·가중치는 수정하지 않습니다. 공동학습 NN의 `weights`를 이 옵션에 넣으면 안 됩니다.

기존 encoder가 없다면 `--pretrained_dir`를 빼세요. 학습 준비 시 공식 ImageNet 가중치를 다운로드합니다. 복사·다운로드는 학습 준비 단계에서만 하며 `test.py`는 인터넷을 사용하지 않습니다. 완성된 v2는 v1 폴더 없이 독립적으로 추론합니다.

완료 로그에 다음 내용이 나옵니다.

- Train·Validation 각각 558개 특징
- CV 1~5의 Macro F1과 평균·표준편차
- 최종 Validation Macro F1
- 단계별 precision·recall·F1·support
- `actual_3_predicted_2`, `actual_5_predicted_4` 등 주요 혼동 개수

학습된 `weights/model.joblib`이 이미 있으면 덮어쓰지 않고 중단합니다. 새 학습은 `--model_out weights_run2/model.joblib`처럼 별도 경로로 실행할 수 있습니다. 기본 설정의 결과부터 비교하세요.

별도 경로로 학습했다면 추론에도 `--model weights_run2/model.joblib`을 지정해야 합니다. 공식 자가 점검은 기본 `weights/model.joblib`을 사용하므로 최종 선택한 모델과 그 encoder를 기본 `weights/`에 준비한 뒤 CSV를 다시 만들고 검사하세요.

## 환경

기존 v1이 실행된 venv를 그대로 사용하세요. 기존 실행 기록은 Python 3.10.12, Linux, RTX 5080 16GB, torch 2.11.0+cu128입니다. v1 학습 기록에 나온 numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, pillow 12.3.0, joblib 1.6.0, catboost 1.2.8을 requirements에 기재했습니다. 대회 안내에 따라 torch는 고정하지 않았습니다.

현재 venv가 이미 작동한다면 패키지를 다시 설치할 필요가 없습니다. 설치가 필요한 경우 `python -m pip install -r requirements.txt`를 사용할 수 있습니다. torchvision 런타임 의존성은 없습니다. 실제 실행한 패키지 버전·OS·장치·encoder 해시는 `weights/model.metrics.json`에 기록됩니다. 기존 모델과 공정하게 비교하려면 같은 환경·장치에서 실행하세요.

## Test 예측과 공식 검사

```bash
python test.py --input /home/ubuntu/data/battery_data_light/test/test.csv --image_root /home/ubuntu/data/battery_data_light/test --output submission_resnet18_catboost_v2.csv
python check_submission.py --data_dir /home/ubuntu/data/battery_data_light --final submission_resnet18_catboost_v2.csv
```

필수 세 인자만으로 실행됩니다. 저장한 모델·전처리를 로드해 입력 CSV의 행과 해당 이미지로만 예측합니다. 추가 센서 13개는 해당 행의 현재 값만 사용합니다. 기존 과거 통계는 같은 실험에서 현재 이하 시점만 사용하며, 예측 CSV는 입력 sample_id 순서로 저장합니다.

스케일러 재fit, 실험 전체 통계·길이·종료 시각, 미래 관측, Test 캐시·의사라벨은 사용하지 않습니다. 저장한 encoder 장치를 재사용하며 CUDA가 없다고 조용히 CPU로 바꾸지 않습니다. 이미지 전처리·가중치 hash·특징 목록·558개 특징 수와 v2 모델 ID를 검증합니다.

공식 검사에서 반복·셔플·실험 분리·미래 제거와 이미지 사용 검사가 모두 통과해야 합니다. v2 성능과 실제 이미지 영향은 서버에서 검사하기 전에는 보장할 수 없습니다. 최종 CSV는 반드시 이 `test.py`로 만든 파일을 사용하세요. 최고 후보 v1은 그대로 보존하고 새 결과를 비교하세요.

## Validation 상세 비교

이전에 받은 `analyze_validation.py`가 `/home/ubuntu/jaebok`에 있다면 다음처럼 v1·v2를 비교할 수 있습니다. 이것은 선택 사항이며 학습 로그에도 단계별 점수가 나옵니다.

```bash
cd /home/ubuntu/jaebok
python -u analyze_validation.py \
  --model resnet18_catboost_v1=/home/ubuntu/jaebok/uni_d_resnet18_catboost_v1 \
  --model resnet18_catboost_v2=/home/ubuntu/jaebok/uni_d_resnet18_catboost_v2
```

입력 행·실험·정답·예측이 들어간 파일은 서버에서만 확인하고 AI에 보내거나 공개 GitHub에 올리지 마세요. 공유할 때는 Macro F1·단계별 성능·혼동행렬 같은 집계만 사용하세요.

## 제출·백업 파일

```text
model.py
baseline_features.py
encoder.py
train.py
test.py
check_submission.py
requirements.txt
README.md
third_party/LICENSE.torchvision
weights/                         # 서버 학습 후 생성
  model.joblib                    # CatBoost + 학습한 imputer/scaler + 메타데이터
  model.metrics.json
  encoder/
    manifest.json
    resnet18-f37072fd.pth.part001
    resnet18-f37072fd.pth.part002
    resnet18-f37072fd.pth.part003
```

공식 encoder는 손실 없이 최대 16MiB씩 분할되며, 모든 조각과 manifest가 필요합니다. v1·v2가 같은 사전학습 모델을 사용해도 v2 `weights/`를 전체 보관하면 독립 재현됩니다. 데이터·이미지·CSV·캐시·개인 키는 공개 저장소에 올리지 마세요. 실험용 브랜치 예: `experiments/resnet18_catboost_v2`.

## 사전학습 모델과 출처

- `ResNet18_Weights.IMAGENET1K_V1`: [공식 PyTorch 모델 설명](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- 사전학습 가중치: [공식 다운로드](https://download.pytorch.org/models/resnet18-f37072fd.pth), 실제 SHA-256은 manifest와 model.joblib에 저장
- 구조 출처: [torchvision ResNet 구현](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py). BSD-3-Clause 구현 라이선스 전문은 `third_party/LICENSE.torchvision`에 동봉
- 사전학습 가중치에는 원 학습 데이터의 조건이 적용될 수 있음: [torchvision 안내](https://github.com/pytorch/vision#pre-trained-model-license)

제공된 대회 규정의 공개 범용 사전학습 모델 허용 조항에 따라 같은 공식 ImageNet 가중치를 사용합니다. 별도 외부 학습 데이터는 추가하지 않습니다.

## 배포 전 확인 범위

로컬에서는 실제 대회 데이터·실제 가중치를 읽거나 학습·추론하지 않았습니다. 합성 입력으로 기존 545열의 정확한 보존, 새 13열 값·인덱스 정렬·결측, 반복·셔플·실험 분리·미래 제거 일관성을 확인했습니다. 모의 encoder·분류기를 연결한 실제 test.py의 CSV 입출력·가중치 경로·추가 센서 반영 검사도 통과했습니다. 공식 encoder 복사 함수는 합성 체크포인트로 원본 불변·손상 거부·중단 처리 검사를 통과했습니다. 실제 학습 성능·GPU 실행·공식 제출 검사는 제공 서버에서 확인해야 합니다.
