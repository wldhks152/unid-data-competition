# Uni-D 배터리 열폭주 6단계 분류

**최종 모델: LightGBM(과거 방향 시계열·사건 피처) 0.4 + DINOv2 이미지 CatBoost 0.4 + 인과 GRU 시퀀스 모델 0.2 확률 앙상블 → 단계 누적최대(cummax) 후처리.** 열화상 이미지와 센서를 함께 사용해요.

## 팀 정보
- 팀명: <팀명>
- 최종 선택 제출 파일: <리더보드에서 선택한 파일명>, Private 복원 대상

## 개발 환경
- OS: <예: Ubuntu 22.04>  (`lsb_release -d`로 확인)
- Python: 3.10
- GPU: NVIDIA GPU 1장 (학습: LightGBM OpenCL, CatBoost GPU, PyTorch CUDA). 추론은 GPU가 없으면 CPU로 동작
- 라이브러리: `requirements.txt` 참고

## 파일 구성
| 파일 | 역할 |
|---|---|
| `src/features.py` | 센서·이미지 통계에 대한 과거 방향 시계열 피처 생성, 누수 검사 |
| `src/images.py` | 열화상 PNG의 의사색 팔레트 위치(보라→빨강→흰색)를 상대 온도로 환산해 분포 통계 계산 (사전학습 모델 없음) |
| `src/model.py` | LightGBM 정의, Macro-F1용 클래스 배율 튜닝, 후처리(cummax) |
| `src/ensemble.py` | 우리 LightGBM과 팀원 CatBoost 확률을 α:(1-α)로 섞음 |
| `src/team_catboost.py` | 팀원 CatBoost v1의 피처·모델 코드 (브랜치 `experiments/catboost_v1`에서 그대로 가져옴) |
| `src/postprocess.py` | 단조 단계 HMM forward 필터 (선택적 후처리) |
| `src/data.py` | 데이터 로드, 상수, 측정 간격 다운샘플링 |
| `src/members.py` | 스태킹 멤버 학습·예측 (lgb, dino 등). 같은 실험 단위 fold 사용 |
| `src/dino.py` | DINOv2-small 동결 임베딩 추출 (배치 크기 고정, 행 순서와 무관) |
| `src/rn18.py`, `src/rn18_encoder.py` | 이미지 임베딩 + 팀원 피처 CatBoost 멤버 (dino 멤버도 이 코드로 추론) |
| `seq/`, `train_seq.py` | 인과 GRU 시퀀스 모델 (과거 행만 보는 단방향 GRU) 학습·추론 |
| `stack.py` | Train 5-fold + Validation 홀드아웃으로 멤버별 OOF와 가중치 저장 |
| `tools/make_final.py` | 멤버·블렌드 가중치·후처리를 `model.joblib` 하나로 고정 |
| `tools/check_submission.py` | 제출 전 재현성·이미지 사용 점검 |
| `tools/report_metrics.py` | 발표용 단계별 F1·혼동행렬 (집계값만) |
| `tools/get_dinov2.py` | DINOv2-small 가중치 내려받기 (revision 고정, 해시 기록) |
| `train.py` | (초기 버전) Train+Validation K-fold 학습 |
| `test.py` | 입력 CSV에 들어 있는 행만으로 추론하고 제출 파일 생성, 누수 자동 검사 |
| `eda.py` | 데이터 점검 |
| `model.joblib` | 최종 가중치 (LightGBM 6개 + DINOv2 CatBoost 6개 + GRU 6개 + 피처 목록 + 블렌드 가중치 + 후처리 설정) |
| `weights/` | 동결 사전학습 이미지 인코더 가중치 (`dinov2-small/`, `encoder/`=ResNet18) |

## 최종 모델 구성
| 멤버 | 가중치 | 입력 | 모델 |
|---|---|---|---|
| `lgb` | 0.4 | 센서 기반 과거 방향 시계열·사건 피처 306개 (이미지는 `dino` 멤버가 담당) (아래 "시계열 피처 구성") | LightGBM, 5/10/15초 솎아낸 행으로 증강 학습 |
| `dino` | 0.4 | 팀원 CatBoost v1 피처 33개 + 현재 행 센서 13개 + 열화상 DINOv2-small 임베딩 384차원 | CatBoost (GPU) |
| `seq_gru` | 0.2 | 실험별 센서 시퀀스 (현재값·첫 관측 대비 변화·직전 대비 변화율, Δt, 경과시간, 실험 조건) | 단방향 2층 GRU (hidden 128), 처음부터 학습. 각 시점 출력은 그 시점까지의 행만 봄 |

- 세 멤버의 클래스 확률을 위 가중치로 더한 뒤 argmax, 실험 안에서 지금까지 예측한 단계의 누적 최댓값(cummax)을 최종 단계로 써요. 과거 예측만 사용해요.
- 각 멤버는 fold 모델 5개 + Train 전체 모델 1개(GRU는 fold마다 같은 설정)의 확률 평균으로 추론해요.
- `model.joblib`에 세 멤버의 가중치, 피처 목록, 블렌드 가중치, 후처리 설정이 모두 들어 있어요. 재학습 없이 `test.py`만으로 결과를 재현해요.

## 실행 방법
```bash
pip install -r requirements.txt
python tools/get_dinov2.py          # DINOv2-small 가중치 1회 내려받기 (저장소에 이미 포함되어 있으면 생략)

# 학습 (Train만 학습, Validation은 홀드아웃)
python stack.py --data_dir <battery_data_light 경로> --splits train --holdout_valid \
                --members lgb dino --out_dir runs/stack_trv
python train_seq.py --data_dir <battery_data_light 경로> --model gru --out_dir runs/stack_trv \
                    --folds_json runs/stack_trv/folds.json --holdout_valid
python tools/make_final.py --members lgb=runs/stack_trv/lgb.joblib dino=runs/stack_trv/dino.joblib \
                           seq_gru=runs/stack_trv/seq_gru.joblib --weights 0.4 0.4 0.2 --post cummax --out model.joblib

# 추론
python test.py --input <battery_data_light 경로>/test/test.csv \
               --image_root <battery_data_light 경로>/test \
               --model model.joblib --output submission.csv \
               --sample_submission <battery_data_light 경로>/sample_submission.csv
```
- `test.py`는 실행할 때마다 자동으로 누수 검사를 해요. 일부 실험을 임의 지점에서 잘라 다시 예측하고, 남은 행의 예측이 같은지 확인해요.
- 입력 CSV가 잘려 있으면 `sample_submission` 순서를 따르되 **입력에 있는 행만** 출력해요.

## 학습·검증 구성
- **학습 데이터: Train 실험만.** 제공 Validation 19개 실험은 학습에 한 번도 넣지 않고 마지막 홀드아웃으로만 썼어요.
  - Train+Validation을 모두 학습에 넣은 버전은 Public 점수가 매번 더 낮았어요 (예: 0.8773 vs Train만 0.8879). 그래서 Validation은 모델 선택 기준으로 남겼어요.
- **검증 방법**
  1. Train 실험 단위 Stratified 5-fold (층화 기준 tr_method×bat_class, 칸이 너무 작으면 tr_method, `runs/stack_trv/folds.json`). 같은 실험의 행이 학습과 검증에 동시에 들어가지 않아요.
  2. Validation 홀드아웃: Train 전체로 학습한 모델로 Validation 19개 실험을 예측 (6번째 fold).
  - Test에는 5/10/15초 간격이 섞여 있어요. 그래서 검증 실험을 10초·15초로 솎아낸 버전으로도 평가하고, 9:1:1로 가중 평균해요.
- **모델 선택 기준**: Train 5-fold, Validation 홀드아웃 두 점수 중 **낮은 쪽**이 가장 높은 조합. Public은 Test의 30%·11개 실험이라 변동이 커서 보조 지표로만 썼어요.

| 조합 (cummax) | Train 5-fold | Validation 홀드아웃 | Public |
|---|---|---|---|
| LightGBM 0.5 + 팀원 v2(ResNet18+센서) 0.5 | 0.9059 | 0.8860 | 0.8879 |
| LightGBM 0.3 + DINOv2 CatBoost 0.7 | 0.8953 | 0.8875 | 0.8899 |
| **LightGBM 0.4 + DINOv2 CatBoost 0.4 + GRU 0.2 (최종)** | **0.9030** | **0.9105** | **0.8917** |

### 최종 모델 단계별 성능 (5초 간격, cummax 후)
| 단계 | 1 | 2 | 3 | 4 | 5 | 6 | Macro F1 | 9:1:1 가중 |
|---|---|---|---|---|---|---|---|---|
| Train 5-fold OOF (78개 실험) | 0.9996 | 0.8483 | 0.9067 | 0.9017 | 0.9034 | 0.8589 | 0.9031 | 0.9030 |
| Validation 홀드아웃 (19개 실험) | 1.0000 | 0.8468 | 0.9456 | 0.8472 | 0.9277 | 0.9131 | 0.9134 | 0.9105 |

- 가장 많이 틀리는 곳은 **2↔3단계 경계**예요. Train OOF에서 정답 2를 3으로 733행, 정답 3을 2로 845행 예측했어요. 그다음은 4↔5와 5↔6 경계예요. 1단계는 거의 완벽해요.
- 혼동행렬 전체는 `reports/metrics.json`에 있어요 (`tools/report_metrics.py`로 재생성).

- 시도했지만 채택하지 않은 것: 단계군별 가중치, seed 평균(LightGBM·CatBoost·GRU 3-seed), 2↔3단계 전용 보정 모델, TCN 시퀀스 모델, ResNet50 이미지 특징, XGBoost·우리 피처 CatBoost 추가, 로지스틱 회귀 메타 스태킹(최저 0.843), 순서형(누적) LightGBM·XGBoost·GRU, 추가 인과 피처 12개(최근 30/60초 변동성·전압-온도 결합; 검증 최저 0.9059였지만 Public 0.8587로 하락), 짧은 창 피처 12개(직전 관측 대비·15초 변화율; lgb 설정 XGBoost에서 효과 없음, GRU 채널로도 블렌드 하락), 벤팅 기준 전압·압력 피처 6개, 벤팅 이후 기준선 대비 이탈 점수 피처(온도·전압·압력의 수준과 30초 기울기 z-score, 최대·합·초과 누적; XGBoost·CatBoost 모두 Validation 하락), ExtraTrees, MiniRocket(GPU 합성곱의 길이별 수치 차이로 뒷부분 삭제 검사 실패), TabNet, Train+Validation 학습. 모두 두 검증 점수의 최저점을 올리지 못했어요.
- 조건별 오류 분석: `python tools/error_analysis.py --stack_dir runs/stack_trv --data_dir <경로>`. 가장 약한 그룹은 39Ah 셀(0.749)로, 4단계가 평균 33초·5단계가 8초로 매우 짧고 급격해 30초 이상 창 피처가 전환을 늦게 잡아요(4단계 77행 중 64행을 3으로 예측).
- 조건별 2↔3 오류의 원인 (같은 도구, `--group bat_cc=73 tr_method=2`):
  - tr_method=2: 2·3단계에 전압이 크게 오르고(+2.5V, +7.5V) 압력도 커지지만, 3단계 승온은 작아요(dT1/dt 0.16, 나머지 1.47). 2→3 신호가 온도가 아니라 전압·압력이에요.
  - 73Ah: 3단계가 짧고(263초, 나머지 746초) 높은 온도(93°C, 나머지 62°C)에서 시작하며 압력 상승이 커요(+54).
  - 벤팅 기준 전압·압력 피처 6개를 넣어 봤지만 두 그룹 점수가 오르지 않았어요. 전압 관련 정보는 이미 피처에 있고, 2↔3 오류는 정보 부족보다 경계가 그룹마다 다른 신호로 정해지는 데서 생기는 것으로 판단했어요.
- 견고성 검증: `python tools/robustness.py --stack_dir runs/stack_trv --data_dir <경로>` (실험 8·11개 무작위 추출 5,000회 점수 분포, fold별 편차, 조건별 점수)
- 단계별 F1과 혼동행렬: `python tools/report_metrics.py --members lgb dino seq_gru --weights 0.4 0.4 0.2` (`reports/metrics.json`, 집계값만 저장)

## 시계열 피처 구성
같은 `experiment_id` 안에서 `relative_time_s` 순으로 정렬한 뒤, **현재 행과 그 이전 행만** 사용해요.
- (선택, 최종 lgb에서는 끔) 이미지 팔레트 통계 11종(상대온도의 평균·분위수·최댓값, 청록/초록/노랑/빨강/흰색 이상 면적 비율). Validation에서 이득이 없어 최종 lgb에는 넣지 않았고, 이미지 정보는 DINOv2 멤버가 담당해요.
- 센서 17종(전압, 온도, 압력, 가스 10종, 주위·열화상·가열 온도) 각각에 대해:
  - 현재값, 첫 관측 대비 변화량
  - 지금까지의 누적 최댓값·최솟값, 최댓값 대비 하락폭, 최솟값 대비 상승폭
  - 과거 30·60·120초 구간의 변화율(초당)과 구간 평균 대비 편차
  - 시간 구간은 **행 수가 아니라 초 단위**로 계산해요. 그래서 측정 간격이 달라도 같은 의미를 가져요.
- 온도·압력·전압이 지금까지의 최고점을 찍은 뒤 지난 시간
- **열폭주 사건 피처**: 열폭주 진행 단계(벤팅 → 내부단락 → 열폭주)와 단계별 센서 중앙값에 근거해 임계값을 정했어요. 사건마다 "지금까지 발생했는지"와 "처음 발생한 뒤 지난 시간"을 피처로 써요.
  - 압력 급등, 전압 저하·최고점 대비 하락, 과충전 전압 초과
  - 온도 임계 초과, 30초 온도 상승률 임계 초과, 열화상 온도 급락, 가스 첫 검출
  - 발생한 사건 수
- 실험 조건 4종(bat_class, bat_cc, soc_status, tr_method), 경과시간
- 학습 데이터 증강: Train 실험을 10초·15초 간격으로 솎아낸 버전을 함께 학습해요.
- 후처리: label이 실험 안에서 단조 증가한다는 점을 이용해 **지금까지의 예측 단계 누적 최댓값**을 써요. 과거 예측만 사용해요.
- **미래 시점 관측과 실험 전체 기준 통계(전체 평균, 종료 시각, 전체 행 수 등)는 사용하지 않습니다.**

## 사용한 사전학습 모델
GRU 시퀀스 모델과 LightGBM은 사전학습 가중치 없이 대회 데이터로 처음부터 학습했어요. 사전학습 모델은 이미지 특징 추출에만 썼고, 가중치는 동결했어요.

### ResNet18 (비교 실험용, 최종 모델에는 미사용)
| 항목 | 내용 |
|---|---|
| 모델 | ResNet18, ImageNet-1K 사전학습 (torchvision `ResNet18_Weights.IMAGENET1K_V1`) |
| 출처 | https://download.pytorch.org/models/resnet18-f37072fd.pth (torchvision 공식 배포, 대회 시작 전 공개) |
| 버전 | torchvision `IMAGENET1K_V1` 가중치, 파일 SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| 라이선스 | 모델 구조 코드: torchvision BSD-3-Clause (`third_party/LICENSE.torchvision`). 가중치: torchvision 배포 조건을 따름 |
| 사용 방식 | 가중치를 **동결**하고 fc 층을 뺀 512차원 평균풀링 특징만 사용 (대회 데이터로 미세조정하지 않음). 이미지마다 독립 계산 |
| 저장 위치 | `weights/encoder/` (16MiB 이하 조각 3개 + `manifest.json`의 해시로 검증, 추론 때 내려받지 않음) |
| 코드 | `src/rn18_encoder.py` (전처리·검증), `src/rn18.py` (스태킹 멤버: 팀원 CatBoost 피처 33개 + 임베딩 512개) |

### DINOv2-small (최종 모델 `dino` 멤버에서 사용)
| 항목 | 내용 |
|---|---|
| 모델 | DINOv2 ViT-S/14 (`facebook/dinov2-small`, Meta AI) |
| 출처 | https://huggingface.co/facebook/dinov2-small (대회 시작 전 공개) |
| 버전 | `weights/dinov2-small/manifest.json`의 revision(커밋 해시)과 SHA-256 |
| 라이선스 | Apache-2.0 |
| 사용 방식 | 가중치 **동결**, pooler 출력 384차원만 특징으로 사용 (미세조정 없음). 이미지마다 독립 계산 |
| 저장 위치 | `weights/dinov2-small/` (`manifest.json`의 revision·SHA-256으로 검증, 추론 때 내려받지 않음) |
| 코드 | `src/dino.py`, 내려받기 `tools/get_dinov2.py` (학습 전 1회, 추론 때는 내려받지 않음) |

### ResNet50 (비교 실험용, 최종 채택 시 기재 유지)
| 항목 | 내용 |
|---|---|
| 모델 | ResNet50, ImageNet-1K 사전학습 (torchvision `ResNet50_Weights.IMAGENET1K_V2`) |
| 출처 | https://download.pytorch.org/models/resnet50-11ad3fa6.pth (torchvision 공식 배포, 대회 시작 전 공개) |
| 버전 | torchvision `IMAGENET1K_V2` 가중치, 파일 SHA-256은 `weights/resnet50/manifest.json` (`11ad3fa6…`로 시작) |
| 라이선스 | 모델 구조 코드: torchvision BSD-3-Clause (`third_party/LICENSE.torchvision`). 가중치: torchvision 배포 조건을 따름 |
| 사용 방식 | 가중치 **동결**, fc 층을 뺀 전역 평균풀링 2048차원 특징만 사용 (미세조정 없음). 이미지마다 독립 계산 |
| 저장 위치 | `weights/resnet50/` (16MiB 이하 조각 + `manifest.json`의 조각별·전체 SHA-256으로 검증, 추론 때 내려받지 않음) |
| 코드 | `src/r50.py` (torchvision과 같은 층 이름으로 구조 정의, 전처리·검증), 내려받기 `tools/get_resnet50.py` |

## 제출 전 점검
```bash
python tools/check_submission.py --input <경로>/validation/validation.csv --image_root <경로>/validation --model model.joblib
```
행 셔플, 실험 단위 분할, 뒷부분 삭제 조건에서 예측이 같은지, 빈 이미지로 바꾸면 예측이 달라지는지(이미지 사용 확인) 검사해요. 집계값만 출력해요.
