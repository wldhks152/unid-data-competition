"""blend.py / test.py용 custom 멤버 — 기존 피처 + 벤팅 기준 물리 피처(src/physics.py)로 학습한 LightGBM fold 모델.
member = {"kind": "custom", "module": "src.physics_member", "models": [...], "feats": [...], "physics": [...]}
(ablation_physics.py --save_model 이 만든다). 반환 행 순서는 F(make_features 결과, 실험·시간 정렬)와 같다."""
import numpy as np

from .evaluation import CLASSES
from .features import make_features


def predict_proba(member, F, df, timg=None):
    Fp = make_features(df, physics=member["physics"]).reset_index(drop=True)
    assert (Fp["sample_id"].to_numpy() == F["sample_id"].to_numpy()).all(), "행 순서가 F와 다릅니다"
    out = np.zeros((len(Fp), len(CLASSES)))
    for m in member["models"]:
        P = m.predict_proba(Fp[member["feats"]])
        for j, c in enumerate(m.classes_):
            out[:, int(c) - 1] += P[:, j]
    return out / len(member["models"])
