# Uni-D 배터리 열폭주 6단계 분류

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
| `src/images.py` | 열화상 PNG에서 밝기·붉은 영역 통계 계산 (사전학습 모델 없음) |
| `src/model.py` | LightGBM 정의, Macro-F1용 클래스 배율 튜닝, 후처리(cummax) |
| `src/postprocess.py` | 단조 단계 HMM forward 필터 (선택적 후처리) |
| `src/data.py` | 데이터 로드, 상수, 측정 간격 다운샘플링 |
| `train.py` | Train+Validation으로 실험 단위 K-fold 학습·평가, 가중치 저장 |
| `test.py` | 입력 CSV에 들어 있는 행만으로 추론하고 제출 파일 생성 |
| `tune_pp.py` | (선택) OOF로 후처리 방식 비교·적용 |
| `eda.py` | 데이터 점검 |
| `model.joblib` | 학습된 가중치 (fold 모델 5개 + 피처 목록 + 클래스 배율 + 후처리 설정). 재학습 없이 `test.py`만으로 결과 재현 |

## 실행 방법
```bash
pip install -r requirements.txt

# 학습 (Train+Validation 전체 사용, 약 수 분)
python train.py --data_dir <battery_data_light 경로> --model_out model.joblib

# (선택) 후처리 비교 후 최고 방식 적용
python tune_pp.py --model model.joblib --apply

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
- 최종 모델: 5개 fold 모델의 확률 평균
- 검증 점수: <OOF Macro F1, model_cv.json의 oof_macro_f1>

## 시계열 피처 구성
같은 `experiment_id` 안에서 `relative_time_s` 순으로 정렬한 뒤, **현재 행과 그 이전 행만** 사용해요.
- 센서 17종(전압, 온도, 압력, 가스 10종, 주위·열화상·가열 온도)과 이미지 통계 10종 각각에 대해:
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
- 없음
