"""열화상 이미지 통계 피처 — 각 행은 '그 시점의 이미지 1장'만 사용하므로 미래 정보가 없다.

이미지는 크기가 제각각인 RGB 의사색(pseudo-color) PNG. 색→온도 환산 기준이 없으므로
밝기(gray) 분포와 '붉은 영역' 비율 같은 단순 통계를 쓴다. 사전학습 모델 미사용.
같은 이미지 폴더를 다시 읽지 않도록 cache_dir에 sample_id별 결과를 캐시한다.
"""
import os

import numpy as np
import pandas as pd

IMG_COLS = [
    "img_mean", "img_std", "img_max", "img_p95", "img_p99", "img_hot_ratio",
    "img_r_mean", "img_g_mean", "img_b_mean", "img_red_ratio",
]


def _stats(path):
    from PIL import Image
    try:
        with Image.open(path) as im:
            a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    except (FileNotFoundError, OSError):
        return [np.nan] * len(IMG_COLS)
    g = a.mean(axis=2).ravel()
    r, gr, b = a[..., 0].ravel(), a[..., 1].ravel(), a[..., 2].ravel()
    p95, p99 = np.percentile(g, [95, 99])
    return [
        g.mean(), g.std(), g.max(), p95, p99, (g > 0.8).mean(),
        r.mean(), gr.mean(), b.mean(), ((r > 0.6) & (r > gr + 0.15) & (r > b + 0.15)).mean(),
    ]


def image_features(df, image_root, cache_dir=None, tag=None):
    """df의 각 행(image_path 칼럼)에 대해 이미지 통계를 계산해 sample_id 인덱스 DataFrame으로 반환."""
    cache = os.path.join(cache_dir, f"img_{tag}.csv") if cache_dir and tag else None
    if cache and os.path.exists(cache):
        c = pd.read_csv(cache, index_col="sample_id")
        if set(df.sample_id) <= set(c.index):
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
    miss = out.img_mean.isna().mean()
    if miss > 0:
        print(f"  [경고] 이미지 {miss:.1%}를 읽지 못했습니다 (image_root={image_root}) → 해당 행은 NaN")
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        out.to_csv(cache)
    return out


def add_image_features(df, image_root, **kw):
    f = image_features(df, image_root, **kw)
    return df.drop(columns=[c for c in IMG_COLS if c in df.columns]).join(f, on="sample_id")
