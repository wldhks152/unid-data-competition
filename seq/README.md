# 처음부터 학습하는 시퀀스 모델 (seq/)

사전학습 가중치·외부 데이터 없음. 모든 모델은 같은 실험의 현재·과거 행만 사용하고, `train_seq.py`가 실행마다
실험을 임의 지점에서 잘라 다시 예측해 확률이 같은지 검사해요. 기존 파이프라인 파일은 수정하지 않아요.

| 모델 | 구조 | 인과성 |
|---|---|---|
| `gru` | 입력 투영 → 단방향 GRU 2층 → [h, cummax(h)] → 선형 | 단방향 순환, cummax는 과거만 |
| `tcn` | dilated Conv1d (1..64) 잔차 블록, 시점별 LayerNorm → [h, cummax(h)] | 왼쪽 패딩만 |
| `rocket` | MiniRocket 커널 84개 × dilation 6개, 최근 24행 PPV·누적 PPV·누적 최댓값 → 로지스틱 회귀 | 왼쪽 패딩, 과거 방향 풀링 |

입력 채널: 센서 17종 × (slog 현재값, 첫 관측 대비 변화, 5초당 변화율) + log1p(Δt) + 경과시간 + 조건 4종.
결측은 forward-fill만, 표준화 통계는 fold 학습 실험으로만. 학습은 5/10/15초 다운샘플 6개 버전을 함께 써요.

산출물은 VALIDATION.md 계약을 따라요: `oof_<name>.csv`(p1..p6, `write_oof` 검사), `<name>.joblib`
(`{"kind": "custom", "module": "seq.member", "models": [...]}`), fold는 `experiment_folds` 또는 `--folds_json`.

```bash
python train_seq.py --synthetic 20 --model gru --out_dir runs/syn --epochs 5      # 동작 확인
python train_seq.py --data_dir <경로> --model gru --out_dir runs/seq --folds_json runs/stack/folds.json
python blend.py --stack_dir runs/stack --members lgb cat team seq_gru=runs/seq --no_bundle
```
