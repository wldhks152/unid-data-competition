"""blend.py 후처리 플러그인 — 인과 HSMM forward 필터 (VALIDATION.md 5절 계약).

    python tune_filter.py --oof runs/ens_oof_oof.csv --export_params src/causal_filter_params.json
    python blend.py ... --post_module src.causal_filter

POSTS[이름](frame, P, w) -> 라벨 1..6. frame에 experiment_id, relative_time_s 필요. 과거 행만 사용.
  hsmm         : argmax(필터 사후확률 · w)
  hsmm_cummax  : 위 결과의 실험별 누적최대
  hsmm_median  : P(단계 ≥ k) ≥ thr 인 최대 k 의 누적최대 (w 미사용)
파라미터(tau, 체류시간 위험률, 건너뛰기 확률)는 tune_filter.py가 nested CV로 고르고
전체 Train+Val label로 추정해 JSON에 저장한 값이다.
주의: blend.py의 outer 점수에서 이 사전분포는 남은 fold의 label도 본 값이라 아주 약간 낙관적이다.
      필터 자체의 선택 편향 없는 점수는 tune_filter.py의 outer 표를 기준으로 한다.
경로 바꾸기: 환경변수 CAUSAL_FILTER_PARAMS
"""
import json
import os

import numpy as np

from .postprocess import decode, hsmm_filter

CLASSES = np.arange(1, 7)
_PATH = os.environ.get("CAUSAL_FILTER_PARAMS", os.path.join(os.path.dirname(__file__), "causal_filter_params.json"))
_cache = {}


def params():
    if "p" not in _cache:
        if not os.path.exists(_PATH):
            raise FileNotFoundError(f"{_PATH} 없음 → 먼저 python tune_filter.py --oof <OOF> --export_params {_PATH}")
        with open(_PATH, encoding="utf-8") as f:
            _cache["p"] = json.load(f)
    return _cache["p"]


def _post(frame, P):
    fr = frame.reset_index(drop=True)
    return fr, hsmm_filter(fr, np.asarray(P, float), params())


def hsmm(frame, P, w):
    fr, A = _post(frame, P)
    return CLASSES[decode(A, fr, "argmax", w)]


def hsmm_cummax(frame, P, w):
    fr, A = _post(frame, P)
    return CLASSES[decode(A, fr, "cummax", w)]


def hsmm_median(frame, P, w):
    fr, A = _post(frame, P)
    return CLASSES[decode(A, fr, "median", thr=params().get("thr", 0.5))]


POSTS = {"hsmm": hsmm, "hsmm_cummax": hsmm_cummax, "hsmm_median": hsmm_median}
