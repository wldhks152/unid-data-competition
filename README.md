# Uni-D 배터리 열폭주 6단계 분류

**현재 후보 (최종 아님, 계속 개선 중): LightGBM(과거 방향 시계열·사건 피처) 0.5 + 팀원 CatBoost v1 0.5 확률 앙상블 → 클래스 배율 + 단계 누적최대(cummax) 후처리**

## 팀 정보
- 팀명: <팀명>
- 최종 선택 제출 파일: <리더보드에서 선택한 파일명>, Private 복원 대상

## 개발 환경
- OS: <예: Ubuntu 22.04>  (`lsb_release -d`로 확인)
- Python: 3.10
- GPU: 사용하지 않음 (LightGBM CPU 학습)
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
| `train.py` | Train+Validation으로 실험 단위 K-fold 학습·평가, 가중치 저장 |
| `test.py` | 입력 CSV에 들어 있는 행만으로 추론하고 제출 파일 생성 |
| `tune_pp.py` | (선택) OOF로 후처리 방식 비교·적용 |
| `eda.py` | 데이터 점검 |
| `ensemble_eval.py` | (비교용) 두 모델 단독·앙상블 비율별 Validation 점수 |
| `model.joblib` | 학습된 가중치 (LightGBM fold 모델 5개 + 팀원 CatBoost 1개 + 피처 목록 + 앙상블 비율 + 클래스 배율 + 후처리 설정). 재학습 없이 `test.py`만으로 결과 재현 |

## 실행 방법
```bash
pip install -r requirements.txt

# 학습 (Train+Validation 전체 사용, LightGBM + CatBoost 앙상블)
python train.py --data_dir <battery_data_light 경로> --model_out model.joblib

# 추론
python test.py --input <battery_data_light 경로>/test/test.csv \
               --image_root <battery_data_light 경로>/test \
               --model model.joblib --output submission.csv \
               --sample_submission <battery_data_light 경로>/sample_submission.csv
```
- `test.py`는 실행할 때마다 자동으로 누수 검사를 해요. 일부 실험을 임의 지점에서 잘라 다시 예측하고, 남은 행의 예측이 같은지 확인해요.
- 입력 CSV가 잘려 있으면 `sample_submission` 순서를 따르되 **입력에 있는 행만** 출력해요.

## 학습·검증 구성
- 학습 데이터: Train+Validation (실험 97개)
- 검증 방법: Train+Validation을 합친 뒤 **실험 단위 Stratified K-fold(5)**. 층화 기준은 tr_method×bat_class예요. 같은 실험의 행이 학습과 검증에 동시에 들어가지 않아요.
  - Test에는 5/10/15초 간격이 섞여 있어요. 그래서 검증 실험을 10초·15초로 솎아낸 버전으로도 평가하고, 9:1:1로 가중 평균해요.
- LightGBM: 5개 fold 모델의 확률 평균
- CatBoost (팀원 v1, 피처 33개): Train+Validation 전체로 1회 학습
- 앙상블: 확률을 LightGBM 0.5 : CatBoost 0.5로 평균 → 클래스 배율 → cummax
- 검증 점수 (제공 Validation 19개 실험, 두 모델 모두 Train만 학습, `python train.py ... --holdout_valid`):

| 모델 | Validation Macro F1 (5/10/15초 9:1:1 가중) |
|---|---|
| 팀원 CatBoost v1 단독 | 0.750 |
| LightGBM 단독 (+w) | 0.897 |
| **앙상블 α=0.5 (+w+cummax)** | **0.903** |
| 앙상블 α=0.25 (+w+cummax) | 0.905 |

  α=0.25와 0.5의 차이가 작고 0.25는 Validation에서 고른 값이라, 치우치지 않은 0.5를 채택했어요.

## 시계열 피처 구성
같은 `experiment_id` 안에서 `relative_time_s` 순으로 정렬한 뒤, **현재 행과 그 이전 행만** 사용해요.
- 이미지 통계 11종(팔레트 상대온도의 평균·분위수·최댓값, 청록/초록/노랑/빨강/흰색 이상 면적 비율): 현재값과 첫 프레임 대비 변화만 사용해요.
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
- 클래스 배율: Macro F1을 높이려고 OOF 확률로 클래스별 배율을 튜닝해요.
- **미래 시점 관측과 실험 전체 기준 통계(전체 평균, 종료 시각, 전체 행 수 등)는 사용하지 않습니다.**

## 사용한 사전학습 모델
| 항목 | 내용 |
|---|---|
| 모델 | ResNet18, ImageNet-1K 사전학습 (torchvision `ResNet18_Weights.IMAGENET1K_V1`) |
| 출처 | https://download.pytorch.org/models/resnet18-f37072fd.pth (torchvision 공식 배포, 대회 시작 전 공개) |
| 버전 | torchvision `IMAGENET1K_V1` 가중치, 파일 SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| 라이선스 | 모델 구조 코드: torchvision BSD-3-Clause (`third_party/LICENSE.torchvision`). 가중치: torchvision 배포 조건을 따름 |
| 사용 방식 | 가중치를 **동결**하고 fc 층을 뺀 512차원 평균풀링 특징만 사용 (대회 데이터로 미세조정하지 않음). 이미지마다 독립 계산 |
| 저장 위치 | `weights/encoder/` (16MiB 이하 조각 3개 + `manifest.json`의 해시로 검증, 추론 때 내려받지 않음) |
| 코드 | `src/rn18_encoder.py` (전처리·검증), `src/rn18.py` (스태킹 멤버: 팀원 CatBoost 피처 33개 + 임베딩 512개) |

### DINOv2-small (실험 중, 최종 채택 시 기재 유지)
| 항목 | 내용 |
|---|---|
| 모델 | DINOv2 ViT-S/14 (`facebook/dinov2-small`, Meta AI) |
| 출처 | https://huggingface.co/facebook/dinov2-small (대회 시작 전 공개) |
| 버전 | `weights/dinov2-small/manifest.json`의 revision(커밋 해시)과 SHA-256 |
| 라이선스 | Apache-2.0 |
| 사용 방식 | 가중치 **동결**, pooler 출력 384차원만 특징으로 사용 (미세조정 없음). 이미지마다 독립 계산 |
| 코드 | `src/dino.py`, 내려받기 `tools/get_dinov2.py` (학습 전 1회, 추론 때는 내려받지 않음) |

## 제출 전 점검
```bash
python tools/check_submission.py --input <경로>/validation/validation.csv --image_root <경로>/validation --model model.joblib
```
행 셔플, 실험 단위 분할, 뒷부분 삭제 조건에서 예측이 같은지, 빈 이미지로 바꾸면 예측이 달라지는지(이미지 사용 확인) 검사해요. 집계값만 출력해요.
