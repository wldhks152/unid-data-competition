# 검증 규칙과 OOF 계약

모든 브랜치(피처 · 시퀀스 모델 · 후처리)는 아래 규칙을 지켜야 `blend.py`에서 함께 비교하고 섞을 수 있어요.

## 1. 고정 fold
- `src/evaluation.py`의 `experiment_folds(tv)`만 써요. Train+Validation 97개 실험, **실험 단위** StratifiedKFold(5), 층화 기준 `tr_method×bat_class`, seed 42.
- `stack.py`가 `runs/stack/folds.json`에 검증 실험 목록을 저장해요. 다른 브랜치는 `load_folds("runs/stack/folds.json")`로 같은 분할을 써도 돼요.
- 같은 실험의 행(다운샘플 버전 포함)은 항상 한 fold에만 들어가요.

## 2. 공통 점수 `score_fn(o, pred)`
- 5초 / 10초 / 15초 버전(`ver` = `k1o0` / `k2o0` / `k3o0`)별 Macro F1을 **9:1:1**로 가중해요 (Test 간격 구성).
- 클래스 배율 튜닝(`tune_w`), 블렌드 선택, 보고 모두 이 함수 하나를 써요. 버전을 합친 pooled F1은 쓰지 않아요.
- 3개 버전이 모두 없으면 오류를 내요 (빠진 버전을 조용히 재가중하지 않음).

## 3. OOF 파일 형식 (`oof_<이름>.csv`)
| 열 | 내용 |
|---|---|
| `sample_id`, `ver` | 행 키. `(sample_id, ver)`는 중복 없음 |
| `grp` | 원래 실험 id (다운샘플 전) |
| `fold` | 이 행을 예측한 fold 번호 (위 고정 분할) |
| `y` | 정답 단계 1~6 |
| `experiment_id`, `relative_time_s` | 버전별 실험 id(`__k2o0` 등 접미사)와 버전 안에서 0부터 다시 센 시간. 후처리(cummax 등)가 이 두 열로 시간순 정렬해요 |
| `p1`..`p6` | 단계 1~6 확률, 행마다 합 1 |

- 만들 때는 `make_oof(P, F, ver, fold)` → `write_oof(o, path)`를 쓰면 형식 검사가 같이 돼요.
- 평가 버전 3개(`k1o0`, `k2o0`, `k3o0`)를 모두 넣어 주세요. 다운샘플은 `src.data.subsample(tv, k, 0)`.
- 확률은 **그 행 시점까지의 관측만** 써서 만들어야 해요 (미래 행 금지, 실험 종료 시각·전체 길이 금지).

## 4. 모델 파일 (`<이름>.joblib`)
- `stack.py` 멤버(`lgb`, `cat`, `team`)는 자동으로 만들어져요.
- 다른 브랜치 모델은 `{"kind": "custom", "module": "src.<모듈>", ...}` 형태로 저장하고, 그 모듈에
  `predict_proba(member, F, df, timg) -> (n, 6)` 를 두면 돼요. `F`는 `make_features(df)` 결과(실험·시간 정렬), `df`는 입력 원본 행이에요. 반환 행 순서는 `F`와 같아야 해요.
- Test 추론은 fold 모델 5개의 확률 평균을 권장해요 (OOF와 같은 모델).

## 5. 후처리 플러그인
- `POSTS = {이름: fn(frame, P, w) -> 라벨}` 을 가진 모듈을 `blend.py --post_module src.<모듈>`로 넘기면 후보에 들어가요.
- `frame`은 OOF/피처 프레임(`experiment_id`, `relative_time_s` 포함), `P`는 블렌드 확률, `w`는 클래스 배율이에요. 과거 행만 써야 해요 (forward 필터는 OK, smoothing은 금지).

## 6. 점수 보고 규칙
- **dev 점수**: 전체 OOF에서 비율·w·후처리를 고르고 같은 OOF로 잰 점수. 선택 편향 때문에 낙관적이라 참고용이에요.
- **outer 점수 (메타 교차검증, base 학습 미분리)**: fold마다 나머지 4개 fold의 OOF로 메타 설정(비율·w·후처리)을 고르고 남은 fold에 적용한 점수예요. dev보다 덜 낙관적이라 **비교 기준으로** 써요.
- outer는 독립 성능 추정이 아니고, 선택 편향 제거를 보장하지도 않아요. 다른 fold의 OOF를 만든 base 모델이 남은 fold 실험으로도 학습됐기 때문이에요. 완전한 nested CV는 base 모델까지 바깥 fold 없이 다시 학습해야 해요.
- OOF 검사: `write_oof`/`read_oof`가 `tools/audit_oof.py`(독립 검사기)를 함께 실행해요. 단독 실행은 `python tools/audit_oof.py --oof <파일> --report <json>`, 테스트는 `python tests/test_oof_audit.py`.
- `src/features.py`의 사건 임계값(45°C, 146.5°C, 242°C, 2.45V 등)은 Train+Validation 라벨 분석으로 정했어요. 그래서 이 피처를 쓰는 CV는 피처 선택까지 독립인 검증이 아니에요. 같은 이유로 라벨을 보고 정한 새 임계값은 문서에 남겨 주세요.
- Public LB는 일부 실험만으로 매겨지는 것으로 보여 노이즈가 커요. 모델 선택 기준으로 쓰지 않아요.

## 실행 순서
```bash
python stack.py --data_dir <경로> --out_dir runs/stack --members lgb cat team   # 멤버 OOF + fold 모델
# 다른 브랜치: runs/<폴더>/oof_<이름>.csv, <이름>.joblib 생성 (위 계약)
python blend.py --stack_dir runs/stack --members lgb cat team seq=runs/seq \
                [--post_module src.causal_filter] --model_out model_stack.joblib
python test.py --input <경로>/test/test.csv --image_root <경로>/test --model model_stack.joblib \
               --output submission.csv --sample_submission <경로>/sample_submission.csv
```
