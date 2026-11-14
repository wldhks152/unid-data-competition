"""동작 확인용 합성 데이터 (실제 데이터와 같은 칼럼 이름, 단조 증가 label 1..6, 5초 간격)."""
import numpy as np
import pandas as pd

from src.data import SENSORS


def make_synthetic(n_exp=30, seed=0):
    g = np.random.default_rng(seed)
    rows = []
    for e in range(n_exp):
        tr = int(g.integers(1, 3))
        bc = int(g.integers(1, 4))
        dur = g.integers(15, 60, size=6)
        lab = np.repeat(np.arange(1, 7), dur)
        n = len(lab)
        t = np.arange(n) * 5.0
        base = {c: g.normal(0, 1) for c in SENSORS}
        X = {c: np.cumsum(g.normal(0, 0.2, n)) + base[c] for c in SENSORS}
        X["meas_t1"] = 25 + 3 * lab + np.cumsum(np.where(lab >= 3, 0.3, 0.05)) + g.normal(0, 0.5, n)
        X["meas_t2"] = X["meas_t1"] * 0.9 + g.normal(0, 0.5, n)
        X["meas_p"] = np.where(lab >= 2, 30, 0) + g.normal(0, 1, n)
        X["meas_v"] = np.where(lab >= 4, 0.3, 3.9) + g.normal(0, 0.05, n)
        X["meas_co"] = np.where(lab >= 3, np.exp(g.normal(4, 1, n)), 0)
        X["thermal_t"] = X["meas_t1"] - np.where(lab >= 5, 40, 0)
        X["meas_t1"] = X["meas_t1"] + np.where(lab >= 6, 200, 0)
        df = pd.DataFrame(X)
        df.loc[g.random(n) < 0.02, "meas_hf"] = np.nan
        df["experiment_id"] = f"exp{e:03d}"
        df["sample_id"] = [f"s{e:03d}_{i:05d}" for i in range(n)]
        df["relative_time_s"] = t
        df["label"] = lab
        df["tr_method"], df["bat_class"], df["bat_cc"], df["soc_status"] = tr, bc, float(g.choice([4.8, 50, 280])), 100
        rows.append(df)
    return pd.concat(rows, ignore_index=True)
