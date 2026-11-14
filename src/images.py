"""열화상 이미지 피처 (v2) — 의사색 팔레트의 '색 위치'를 상대 온도로 환산해 통계를 낸다.

이미지는 무지개 팔레트: 차가움 자주/보라 → 파랑 → 청록 → 초록 → 노랑 → 빨강 → 흰색(가장 뜨거움).
밝기(gray)는 이 순서와 단조 관계가 아니어서(보라·청록은 밝고 파랑은 어두움) 온도를 잘 반영하지 못한다.
→ HSV 색상(hue)으로 팔레트 위치(pt: 0=차가움 ~ 1=빨강, 흰색은 1.15)를 구하고 그 분포 통계를 쓴다.
각 행은 그 시점 이미지 1장만 사용하므로 미래 정보가 없다. 사전학습 모델 미사용.
"""
import os

import numpy as np
import pandas as pd

IMG_COLS = [
    "img_pt_mean", "img_pt_p50", "img_pt_p90", "img_pt_p99", "img_pt_max",
    "img_frac_cyan", "img_frac_green", "img_frac_yellow", "img_frac_red", "img_frac_white",
    "img_pt_std",
]
CACHE_VERSION = "v2"


def palette_temp(rgb):
    """(h, w, 3) uint8 RGB → (h*w,) 팔레트 위치. 0=자주/보라(차가움) … 1=빨강, 1.15=흰색."""
    from PIL import Image
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.float32).reshape(-1, 3)
    hue = hsv[:, 0] * (360.0 / 255.0)
    s, v = hsv[:, 1] / 255.0, hsv[:, 2] / 255.0
    pt = np.clip((300.0 - hue) / 300.0, 0.0, 1.0)  # 300°(자주)=0, 180°(청록)=0.4, 120°(초록)=0.6, 60°(노랑)=0.8, 0°(빨강)=1
    pt[(hue > 300) & (hue < 340)] = 0.0           # 자주~분홍: 차가운 쪽
    pt[hue >= 340] = 1.0                           # 360° 근처 빨강
    pt[(s < 0.2) & (v > 0.8)] = 1.15               # 흰색: 가장 뜨거움 (포화)
    pt[v < 0.12] = 0.0                             # 거의 검정
    return pt


def _stats(path):
    from PIL import Image
    try:
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
    except (FileNotFoundError, OSError):
        return [np.nan] * len(IMG_COLS)
    pt = palette_temp(rgb)
    p50, p90, p99 = np.percentile(pt, [50, 90, 99])
    return [
        pt.mean(), p50, p90, p99, pt.max(),
        (pt >= 0.4).mean(), (pt >= 0.6).mean(), (pt >= 0.8).mean(), (pt >= 0.95).mean(), (pt > 1.05).mean(),
        pt.std(),
    ]


def image_features(df, image_root, cache_dir=None, tag=None):
    """df의 각 행(image_path 칼럼)에 대해 이미지 통계를 계산해 sample_id 인덱스 DataFrame으로 반환."""
    cache = os.path.join(cache_dir, f"img{CACHE_VERSION}_{tag}.csv") if cache_dir and tag else None
    if cache and os.path.exists(cache):
        c = pd.read_csv(cache, index_col="sample_id")
        if set(df.sample_id) <= set(c.index) and set(IMG_COLS) <= set(c.columns):
            return c.loc[df.sample_id, IMG_COLS]
    if "image_path" in df.columns:
        paths = [os.path.join(image_root, p) for p in df.image_path]
    else:
        paths = [os.path.join(image_root, "images", f"{s}.png") for s in df.sample_id]
    rows = []
    for i, p in enumerate(paths):
        rows.append(_stats(p))
        if (i + 1) % 5000 == 0:
            print(f"    이미지 {i + 1:,}/{len(paths):,}")
    out = pd.DataFrame(rows, columns=IMG_COLS, index=pd.Index(df.sample_id.values, name="sample_id"))
    miss = out[IMG_COLS[0]].isna().mean()
    if miss > 0:
        print(f"  [경고] 이미지 {miss:.1%}를 읽지 못했습니다 (image_root={image_root}) → 해당 행은 NaN")
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        out.to_csv(cache)
    return out


def add_image_features(df, image_root, **kw):
    f = image_features(df, image_root, **kw)
    return df.drop(columns=[c for c in IMG_COLS if c in df.columns]).join(f, on="sample_id")
