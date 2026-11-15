"""피처 생성 함수 모음 — 최종 모델(lgb 0.4 + dino 0.4 + seq_gru 0.2)이 쓰는 피처 함수를 한곳에서 보여 준다.

train.py(학습)와 test.py(추론)는 모두 아래 함수를 같은 방식으로 호출한다. 실제 구현은 src/·seq/에 있고,
이 파일은 그 함수들을 다시 내보내는 진입점이다. 모든 피처는 같은 experiment_id 안에서
relative_time_s 순으로 정렬한 뒤 현재 행과 그 이전 행만 사용한다 (미래 행·실험 전체 통계 사용 없음).

멤버별 피처
  lgb      build_sensor_features(df)
           센서 변화량·누적 최대·30/60/120초 기울기·열폭주 사건 피처 (src/features.make_features)
  dino     image_embeddings(df, image_root) → build_image_features(df, team_img, emb, sensors=True)
           동결 DINOv2-small 임베딩 384차원 + 팀원 CatBoost 피처 33개 + 현재 행 센서 13개 (src/rn18.build_X)
  seq_gru  build_sequences(df)
           실험별 센서 시퀀스 (과거 방향, seq/prep.build_sequences)
후처리     confirm_labels(F, P)
           4~6단계 상승 재확인 후 누적 최댓값 (src/late.confirm_labels)

학습 때 계산한 전처리(스케일러·피처 목록·클래스 순서 등)는 model.joblib에 저장하고, test.py는 다시 fit하지 않는다.
"""
from src.dino import embeddings as image_embeddings  # noqa: F401  동결 DINOv2-small, 이미지마다 독립 계산
from src.ensemble import team_image_features  # noqa: F401  열화상 PNG 통계 (팀원 CatBoost 피처용, 행마다 독립)
from src.features import make_features as build_sensor_features  # noqa: F401
from src.late import confirm_labels  # noqa: F401
from src.rn18 import build_X as build_image_features  # noqa: F401
from seq.prep import build_sequences  # noqa: F401

__all__ = ["build_sensor_features", "image_embeddings", "team_image_features",
           "build_image_features", "build_sequences", "confirm_labels"]
