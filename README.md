# 1팀 · ResNet18 이미지 특징 + 센서 + CatBoost v1

기존 CatBoost v1을 보관하고, **같은 33개 특징에 고정 ResNet18 특징 512개만 추가**하는 비교 실험입니다.
총 입력은 545개이며, 6단계 분류는 CatBoost가 수행합니다. ResNet18은 이미지 특징 추출만 하고 대회 데이터로 학습하지 않습니다.
제공 서버에서 학습과 공식 제출 검사를 완료했습니다. 사용자 확인 기준 Public Macro F1은 **0.9156013487**입니다. 현재 후보 파일명은 **submission_resnet18_catboost_v1.csv**이며 최종 선택은 대회 사이트에서 별도로 확정합니다.

## 비교 조건

| 항목 | 이번 설정 |
|---|---|
| 기존 특징 | 센서 5개 + 흑백 이미지 통계 6개 + 과거 30초 특징 21개 + relative_time_s 1개 = 33개 |
| 새 특징 | 현재 RGB 이미지의 ResNet18 avgpool 특징 512개 |
| 분류기 | CatBoost MultiClass, 600 trees, depth 6, learning rate 0.05, L2 3.0 |
| 클래스 가중치 / 시드 | 없음 / 42 |
| 분류기 실행 | CPU, 기본 4 threads |
| 이미지 모델 | ImageNet 사전학습 ResNet18, 전체 동결, eval, float32 |
| 특징 추출 | 기본 auto 장치; 아래 실행 예시는 CUDA를 명시. 고정 batch 1, CPU intra-op threads 1 |
| 전처리 | 기존 median imputer + StandardScaler, fold의 학습 부분에서만 fit |
| 검증 | Train의 experiment_id 기준 GroupKFold 5개 + 제공 Validation |
| 저장 모델 학습 | Train만 사용. Validation/Test는 모델·전처리 학습에 사용하지 않음 |

## 실행 결과

| 모델 | Train GroupKFold Macro F1 | Validation Macro F1 | Public Macro F1 |
|---|---:|---:|---:|
| CatBoost v1 | 0.8732 ± 0.0300 | 0.7505 | 0.8827502081 |
| ResNet18 + CatBoost v1 | **0.8820 ± 0.0369** | 0.7477 | **0.9156013487** |

현재 모델의 fold별 Macro F1은 0.9009, 0.8962, 0.9106, 0.8932, 0.8091입니다.
공식 check_submission.py 전체 통과를 사용자 실행 결과로 확인했습니다. 형식·반복 실행·최종 후보 파일 재현·행 순서 변경·실험 분리·미래 행 제거·이미지 사용 검사를 포함합니다.
빈 이미지의 정확한 라벨 변경률은 이 문서에 별도로 기록하지 않았습니다.
Public은 전체 Test 중 공개 평가 부분의 점수이며 Private 점수는 아직 확인하지 않았습니다.

## 비교 실험 범위

새 센서, 배터리 조건, 클래스 가중치, CatBoost 하이퍼파라미터는 이번 실험에서 추가로 바꾸지 않습니다.
고정 사전학습 특징은 각 샘플별로 추출하므로 CV 전에 계산할 수 있지만, imputer와 scaler는 각 fold의 학습 부분에만 fit합니다.

## 이미지 전처리

RGB로 변환한 뒤 종횡비를 유지해 긴 변을 224픽셀로 맞춥니다. PIL BILINEAR 보간을 사용하고, 크기는 floor(value + 0.5)로 반올림합니다.
224×224 캔버스 중앙에 넣으며 고정 RGB (124, 116, 104)로 여백을 채웁니다. 가운데를 잘라내지 않아 원본 이미지 전체를 유지합니다.
float32 / 255 후 ImageNet mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225)로 정규화합니다.
랜덤 crop, flip, 색상 변경은 사용하지 않습니다. `encoder.py`의 TRANSFORM_SPEC과 가중치 manifest에 이 설정을 저장합니다.
가색 영상의 픽셀을 실제 온도로 환산하는 과정은 없습니다.

## 서버 실행 순서

이 `uni_d_resnet18_catboost_v1` 폴더를 서버의 `/home/ubuntu/jaebok`에 올립니다.
기존 CatBoost v1 폴더에 덮어쓰지 않고 새 폴더에서 실행합니다.
학습과 추론은 대회 제공 서버에서만 수행합니다. 기존 tmux 터미널을 사용하세요.

```bash
source /home/ubuntu/venv/bin/activate
cd /home/ubuntu/jaebok/uni_d_resnet18_catboost_v1
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
nvidia-smi
```

제공 서버의 GPU 사용 상황을 팀원과 확인하고, CUDA가 True이면 학습을 시작합니다.
기존 대회 venv에는 이전 실험의 6개 패키지와 GPU용 torch가 있으므로 추가 설치 명령이 필요하지 않습니다.
torchvision 런타임은 사용하지 않습니다. 기존 GPU용 torch를 교체하는 설치는 하지 마세요.

```bash
python -u train.py --data_dir /home/ubuntu/data/battery_data_light --device cuda
```

첫 실행은 공식 ResNet18 가중치 약 45MB를 내려받습니다. 그다음 Train·Validation 이미지 특징 추출 진행 상황을 500장마다 출력합니다.
특징 추출이 끝나면 GPU 메모리를 반환하고 CPU에서 CatBoost 교차검증과 최종 학습을 진행합니다. 입력 특징이 545개여서 이전 실험보다 학습 시간이 길 수 있습니다.
CPU로 특징 추출하려면 처음 학습부터 `--device cpu`를 사용합니다. 추론도 학습 때 기록한 장치를 그대로 사용하며 자동으로 다른 장치로 전환하지 않습니다.

## 추론과 공식 점검

학습 완료 후 같은 폴더에서 실행합니다. `test.py`는 네트워크로 가중치를 받지 않고 저장된 로컬 가중치만 사용합니다.

```bash
python test.py --input /home/ubuntu/data/battery_data_light/test/test.csv --image_root /home/ubuntu/data/battery_data_light/test --output submission_resnet18_catboost_v1.csv
python check_submission.py --data_dir /home/ubuntu/data/battery_data_light --final submission_resnet18_catboost_v1.csv
```

필수 인자는 --input, --image_root, --output 세 개뿐입니다. 기본 모델은 스크립트 폴더의 `weights/model.joblib`입니다.
모델 옆의 encoder manifest와 모든 가중치 조각이 필요합니다. 다른 모델 위치는 선택 인자 `--model`로 지정할 수 있습니다.
입력 CSV에 있는 행과 이미지만 읽으며, 입력 행 순서와 sample_id를 유지합니다. label은 1~6 정수입니다.
Test fit, 전체 Test 캐시, 미래 관측, 다른 실험의 시계열 정보를 사용하지 않습니다.
ResNet18은 eval·batch 1로 사용하고 결정적 연산을 켜며 TF32를 끕니다. 같은 서버 환경에서 공식 재현성 검사를 반드시 확인하세요.
공식 점검은 추론을 7번 실행하므로 추가 시간이 걸립니다. 빈 이미지 검사에서 예측 라벨도 달라지는지 확인합니다.

## 저장 파일과 GitHub 백업

```text
weights/
├── model.joblib
├── model.metrics.json
└── encoder/
    ├── manifest.json
    ├── resnet18-f37072fd.pth.part001
    ├── resnet18-f37072fd.pth.part002
    └── resnet18-f37072fd.pth.part003
```

공식 가중치를 수치 변경 없이 바이너리 파일 조각으로 나눕니다. 각 조각은 최대 16MiB여서 GitHub 웹의 파일당 25MB 한도 안에 들어갑니다.
추론 시 메모리에서 정확한 원본 바이트로 합치며 조각별·전체 SHA-256을 검사합니다. FP16 변환이나 손실 압축은 하지 않습니다.
**GitHub 백업에는 코드와 weights 폴더 전체가 필요합니다.** model.joblib만 올리거나 encoder 조각 일부를 빼면 재현할 수 없습니다.
데이터·이미지·예측 CSV·노트북 출력은 공개 저장소에 올리지 않습니다.

## 파일 구성

- model.py: 기존 33개와 ResNet18 512개 특징 결합, 검증, 분류기 연결
- baseline_features.py: 기존 CatBoost v1의 특징·전처리·분류기 코드
- encoder.py: ResNet18 구현, 전처리, 공식 가중치 준비, 오프라인 검증·추출
- train.py / test.py: 학습 및 재학습 없는 추론 진입점
- check_submission.py: 운영진이 제공한 점검 스크립트
- requirements.txt: 기존 실제 환경의 6개 패키지 버전과 버전을 고정하지 않은 torch
- third_party/LICENSE.torchvision: ResNet18 구현의 라이선스 고지

실제 Python·OS·패키지 버전, GPU, CUDA/cuDNN, 특징 추출 장치·설정, fold별 점수, 단계별 F1, 혼동행렬과 공식 가중치의 전체 SHA-256은 `weights/model.metrics.json`에 기록합니다.

## 사전학습 모델 출처·버전·라이선스

- 모델 / 가중치 버전: **ResNet18 / ResNet18_Weights.IMAGENET1K_V1** (ImageNet-1K 사전학습)
- 공식 모델 안내: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18
- 공식 가중치 URL: https://download.pytorch.org/models/resnet18-f37072fd.pth
- 고정 파일명 / 공식 해시 접두부: resnet18-f37072fd.pth / f37072fd. 실제 전체 SHA-256은 준비 후 manifest와 metrics에 저장합니다.
- 구현 출처: torchvision의 ResNet18 계층·state_dict 키와 호환되게 torch.nn으로 구현했습니다. torchvision을 설치하지 않고도 공식 체크포인트를 strict load합니다.
- 구현 라이선스: torchvision **BSD-3-Clause**. 전문은 third_party/LICENSE.torchvision에 포함했습니다. 공식 출처: https://github.com/pytorch/vision/blob/main/LICENSE
- 사전학습 가중치는 학습 데이터셋에서 유래한 별도 이용 조건이 적용될 수 있습니다. 구현의 BSD 라이선스가 가중치·ImageNet 데이터의 모든 이용 조건을 대신한다는 주장은 하지 않습니다. torchvision 안내: https://github.com/pytorch/vision#pre-trained-model-license

대회 시작 전 공개된 범용 사전학습 모델만 사용하며, 외부 데이터를 추가하거나 외부 추론 API에 대회 데이터를 보내지 않습니다.
OT 기준 최종 마감은 11월 15일 13:00 리더보드, 13:30 코드·PPT 메일 제출입니다.

## 작성 시 소프트웨어 점검 범위

합성 이미지로 RGB·padding·정규화, 가중치 조각 누락·변조 검출을 확인했습니다.
기존 33개 특징의 보존과 545개 특징의 행 정렬, 순서 변경·실험 분리·미래 제거도 확인했습니다.
신경망·분류기를 명시적인 모의 객체로 대체한 CLI 검사에서 CSV 형식·기본/사용자 지정 모델 경로·오프라인 오류 처리를 확인했습니다.
이후 제공 서버에서 실제 이미지 특징 추출·학습을 실행했으며, 위 실행 결과와 공식 제출 검사 전체 통과를 사용자 확인으로 기록했습니다.
