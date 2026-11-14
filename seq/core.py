"""fold 하나 학습 · 예측 (GRU / TCN / Rocket 공통 인터페이스)."""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import nets
from .prep import Scaler
from .rocket import CausalRocket, LinearHead

CLASSES = np.arange(1, 7)  # 확률 칼럼 순서 고정: 1..6


def class_weights(seqs, power=1.0):
    y = np.concatenate([s["y"] for s in seqs])
    cnt = np.array([(y == c).sum() for c in CLASSES], dtype=float)
    w = (cnt.sum() / np.maximum(cnt, 1) / len(CLASSES)) ** power
    w[cnt == 0] = 0.0
    return w / w[cnt > 0].mean()


def _pad(batch, device, noise=0.0):
    L = max(len(z) for z, _ in batch)
    C = batch[0][0].shape[1]
    X = np.zeros((len(batch), L, C), dtype=np.float32)
    Y = np.full((len(batch), L), -100, dtype=np.int64)
    for i, (z, y) in enumerate(batch):
        X[i, :len(z)] = z
        if y is not None:
            Y[i, :len(z)] = y - 1
    X = torch.as_tensor(X, device=device)
    if noise:
        X = X + noise * torch.randn_like(X)
    return X, torch.as_tensor(Y, device=device)


# ---------------------------------------------------------------- 신경망 (GRU/TCN)
def ord_probs(logits):
    """순서형(누적 로짓) 출력 → 6클래스 확률. logits[..., k-2] = logit P(단계 ≥ k), k=2..6 (6번째 출력은 안 씀).
    P(≥k)가 k에 대해 줄어들도록 누적 최소로 보정한다."""
    q = torch.sigmoid(logits[..., :len(CLASSES) - 1].float())
    q = torch.cummin(q, dim=-1).values
    one = torch.ones_like(q[..., :1])
    zero = torch.zeros_like(q[..., :1])
    return (torch.cat([one, q], -1) - torch.cat([q, zero], -1)).clamp_min(0)


def _ord_loss(out, Y, w):
    """누적 로짓 손실: k=2..6마다 BCE(logit_k, y ≥ k), 행마다 클래스 가중."""
    m = Y >= 0
    z = out[..., :len(CLASSES) - 1].float()[m]
    y = Y[m]
    ks = torch.arange(1, len(CLASSES), device=Y.device)
    tgt = (y[:, None] >= ks[None, :]).float()
    l = F.binary_cross_entropy_with_logits(z, tgt, reduction="none").sum(-1)
    ww = w[y]
    return (l * ww).sum() / ww.sum()


def fit_net(cfg, train_seqs, device, seed, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sc = Scaler().fit([s for s in train_seqs if s["ver"] == "k1o0"] or train_seqs)
    data = [(sc.transform(s["X"]), s["y"]) for s in train_seqs]
    c_in = data[0][0].shape[1]
    net = nets.build(cfg["model"], c_in, len(CLASSES), **cfg.get("net_kw", {})).to(device)
    cw = class_weights(train_seqs, cfg.get("cw_power", 1.0))
    cw[[1, 2]] *= cfg.get("cw23", 1.0)  # 2·3단계 가중 (기본 1 = 변화 없음)
    w = torch.as_tensor(cw, dtype=torch.float32, device=device)
    aux23 = cfg.get("aux23", 0.0)  # 2↔3 경계 보조 손실: 정답이 2·3인 행에서 logit3−logit2로 이진 분류
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 1e-2))
    bs, ep = cfg["batch"], cfg["epochs"]
    steps = ep * math.ceil(len(data) / bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
    use_amp = device.type == "cuda" and cfg.get("amp", True)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for e in range(ep):
        net.train()
        perm = rng.permutation(len(data))
        tot, nb = 0.0, 0
        for i in range(0, len(perm), bs):
            X, Y = _pad([data[j] for j in perm[i:i + bs]], device, cfg.get("noise", 0.0))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = net(X)
            if cfg.get("ordinal"):
                loss = _ord_loss(out, Y, w)
            else:
                loss = F.cross_entropy(out.float().reshape(-1, len(CLASSES)), Y.reshape(-1), weight=w, ignore_index=-100)
            if aux23:
                m23 = (Y == 1) | (Y == 2)
                if m23.any():
                    z = (out[..., 2] - out[..., 1]).float()[m23]
                    loss = loss + aux23 * F.binary_cross_entropy_with_logits(z, (Y[m23] == 2).float())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sch.step()
            tot, nb = tot + loss.item(), nb + 1
        if (e + 1) % max(1, ep // 5) == 0 or e == ep - 1:
            log(f"    epoch {e + 1:3d}/{ep}  loss {tot / nb:.4f}  ({time.time() - t0:.0f}s)")
    net.eval()
    return {"kind": "net", "cfg": cfg, "c_in": c_in, "scaler": sc.state(), "state": {k: v.cpu() for k, v in net.state_dict().items()}}


def _load_net(m, device):
    net = nets.build(m["cfg"]["model"], m["c_in"], len(CLASSES), **m["cfg"].get("net_kw", {})).to(device)
    net.load_state_dict(m["state"])
    net.eval()
    return net


@torch.no_grad()
def predict_net(m, seqs, device, _cache={}):
    key = id(m)
    if key not in _cache:
        _cache.clear()
        _cache[key] = (_load_net(m, device), Scaler.from_state(m["scaler"]))
    net, sc = _cache[key]
    out = []
    for s in seqs:  # 실험 하나씩 (패딩 없음) → 다른 실험·미래 행의 영향 없음
        X = torch.as_tensor(sc.transform(s["X"]), device=device)[None]
        o = net(X).float()[0]
        out.append((ord_probs(o) if m["cfg"].get("ordinal") else torch.softmax(o, -1)).cpu().numpy())
    return out


# ---------------------------------------------------------------- Rocket
def _row_subsample(n, stride, rng):
    return np.arange(int(rng.integers(0, stride)), n, stride) if stride > 1 else np.arange(n)


def fit_rocket(cfg, train_seqs, device, seed, log=print):
    rng = np.random.default_rng(seed)
    sc = Scaler().fit([s for s in train_seqs if s["ver"] == "k1o0"] or train_seqs)
    Zs = [sc.transform(s["X"]) for s in train_seqs]
    rk = CausalRocket(Zs[0].shape[1], dilations=cfg.get("dilations", (1, 2, 4, 8, 16, 32)),
                      window=cfg.get("window", 24), seed=seed, device=device)
    rk.fit_bias([Zs[i] for i in rng.permutation(len(Zs))[:200]], seed=seed)
    n_tot = sum(len(z) for z in Zs)
    stride = max(1, math.ceil(n_tot / cfg.get("max_rows", 300000)))
    Xs, ys = [], []
    t0 = time.time()
    for z, s in zip(Zs, train_seqs):
        idx = _row_subsample(len(z), stride, rng)
        Xs.append(rk.transform(z)[torch.as_tensor(idx, device=device)].to(torch.float16).cpu())
        ys.append(s["y"][idx] - 1)
    X = torch.cat(Xs).float()
    y = np.concatenate(ys)
    log(f"    rocket 피처 {X.shape[0]:,}행 × {X.shape[1]} (stride {stride}, {time.time() - t0:.0f}s)")
    head = LinearHead(len(CLASSES), epochs=cfg.get("epochs", 30), lr=cfg.get("lr", 1e-3), wd=cfg.get("wd", 1e-4),
                      hidden=cfg.get("hidden", 0), device=device, seed=seed)
    head.fit(X, y, class_weights(train_seqs, cfg.get("cw_power", 1.0)))
    log(f"    선형 분류기 학습 완료 ({time.time() - t0:.0f}s)")
    return {"kind": "rocket", "cfg": cfg, "scaler": sc.state(),
            "rocket": {"W": rk.W.cpu(), "bias": rk.bias, "q": rk.q, "dilations": rk.dilations, "window": rk.window},
            "head": {"net": {k: v.cpu() for k, v in head.net.state_dict().items()}, "mu": head.mu, "sd": head.sd,
                     "d": X.shape[1]}}


@torch.no_grad()
def predict_rocket(m, seqs, device):
    sc = Scaler.from_state(m["scaler"])
    r = m["rocket"]
    rk = CausalRocket.__new__(CausalRocket)
    rk.W, rk.bias, rk.q, rk.dilations, rk.window, rk.device = r["W"].to(device), r["bias"], r["q"], r["dilations"], r["window"], device
    rk.n_conv = len(rk.dilations) * rk.W.shape[1]
    cfg = m["cfg"]
    head = LinearHead(len(CLASSES), hidden=cfg.get("hidden", 0), device=device)
    d = m["head"]["d"]
    head.net = (torch.nn.Linear(d, len(CLASSES)) if not head.hidden else torch.nn.Sequential(
        torch.nn.Linear(d, head.hidden), torch.nn.GELU(), torch.nn.Dropout(0.2),
        torch.nn.Linear(head.hidden, len(CLASSES)))).to(device)
    head.net.load_state_dict(m["head"]["net"])
    head.net.eval()
    head.mu, head.sd = m["head"]["mu"], m["head"]["sd"]
    return [head.predict_proba(rk.transform(sc.transform(s["X"])).float()) for s in seqs]


def fit(cfg, train_seqs, device, seed, log=print):
    return (fit_rocket if cfg["model"] == "rocket" else fit_net)(cfg, train_seqs, device, seed, log)


def predict(m, seqs, device):
    return (predict_rocket if m["kind"] == "rocket" else predict_net)(m, seqs, device)


def predict_ensemble(models, seqs, device):
    """여러 모델(fold × seed) 확률 평균. 반환: 실험별 (L, 6) 배열 리스트."""
    acc = None
    for m in models:
        P = predict(m, seqs, device)
        acc = P if acc is None else [a + p for a, p in zip(acc, P)]
    return [a / len(models) for a in acc]
