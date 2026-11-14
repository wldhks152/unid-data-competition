"""blend.py / test.py용 custom 멤버 (VALIDATION.md §4).
joblib: {"kind": "custom", "module": "seq.member", "models": [fold×seed 모델], "name": ..., "cfg": ...}
predict_proba(member, F, df, timg) -> (len(F), 6), 열 순서 1..6, 행 순서는 F와 같음.
df(입력 원본 행)만으로 실험별 시퀀스를 만들어 예측하므로, 각 행은 같은 실험의 현재·과거 행만 사용한다."""
import numpy as np
import torch

from . import core
from .prep import build_sequences

_DEVICE = None


def device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def predict_proba(member, F, df, timg=None):
    seqs = build_sequences(df)
    P = core.predict_ensemble(member["models"], seqs, device())
    sid = np.concatenate([np.asarray(s["sample_id"]).astype(str) for s in seqs])
    pos = {s: i for i, s in enumerate(sid)}
    allP = np.concatenate(P)
    return allP[[pos[str(s)] for s in F["sample_id"].to_numpy()]]
