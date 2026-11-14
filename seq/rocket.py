"""인과 MiniRocket 스타일 피처 + 선형 분류기 (커널은 고정 규칙/난수, 학습되는 것은 선형층뿐).

* 커널: 길이 9, 가중치 {-1, 2} (2가 3개) → C(9,3)=84개, dilation 여러 개 (MiniRocket과 같은 구성)
* 다변량: 커널마다 무작위 채널 1~3개의 합에 적용
* 인과: 왼쪽에만 (9-1)·d 패딩 → 행 t의 합성곱 값은 t 이전 관측만 사용
* 풀링(모두 과거 방향):
    - 최근 W행 PPV (bias 2개)        : 최근 구간에서 패턴이 얼마나 자주 나타났나
    - 시작부터 지금까지 PPV (bias 1개) : 누적 빈도
    - 시작부터 지금까지 최댓값         : 한 번이라도 강하게 나타났나 (단조 단계에 유리)
* bias는 fold 학습 실험의 합성곱 값 분위수로 정한다 (Test 정보 미사용)
"""
import itertools

import numpy as np
import torch
import torch.nn.functional as F

KLEN = 9
COMBOS = list(itertools.combinations(range(KLEN), 3))  # 84


class CausalRocket:
    def __init__(self, c_in, dilations=(1, 2, 4, 8, 16, 32), window=24, seed=0, device="cpu"):
        g = np.random.default_rng(seed)
        self.dilations, self.window, self.device = list(dilations), window, device
        n = len(COMBOS)
        W = np.zeros((len(self.dilations), n, c_in, KLEN), dtype=np.float32)
        for di in range(len(self.dilations)):
            for k, pos in enumerate(COMBOS):
                ker = -np.ones(KLEN, dtype=np.float32)
                ker[list(pos)] = 2.0
                chans = g.choice(c_in, size=int(g.integers(1, 4)), replace=False)
                W[di, k, chans] = ker
        self.W = torch.tensor(W, device=device)
        self.n_conv = len(self.dilations) * n
        self.bias = None  # (n_conv, 3) : 최근 PPV 2개 + 누적 PPV 1개
        self.q = np.sort(g.uniform(0.15, 0.85, size=(self.n_conv, 3)), axis=1)

    @torch.no_grad()
    def _conv(self, Z):  # Z: (L, C) → (L, n_conv)
        x = torch.as_tensor(Z, device=self.device).T[None]  # (1, C, L)
        outs = []
        for di, d in enumerate(self.dilations):
            outs.append(F.conv1d(F.pad(x, ((KLEN - 1) * d, 0)), self.W[di], dilation=d)[0])
        return torch.cat(outs).T  # (L, n_conv)

    def fit_bias(self, Zs, n_sample=20000, seed=0):
        g = np.random.default_rng(seed)
        per = max(1, n_sample // max(1, len(Zs)))
        vals = []
        for Z in Zs:
            c = self._conv(Z)
            idx = torch.as_tensor(g.integers(0, len(Z), size=min(per, len(Z))), device=self.device)
            vals.append(c[idx])
        V = torch.cat(vals).cpu().numpy()
        self.bias = np.stack([np.quantile(V[:, j], self.q[j]) for j in range(self.n_conv)]).astype(np.float32)
        return self

    @torch.no_grad()
    def transform(self, Z):
        """Z (L, C) → (L, n_feat). 모든 출력은 행 t까지의 값만 사용."""
        c = self._conv(Z)  # (L, n)
        b = torch.as_tensor(self.bias, device=self.device)
        L = c.shape[0]
        cnt = torch.arange(1, L + 1, device=self.device, dtype=torch.float32)[:, None]
        feats = []
        for j in range(2):  # 최근 W행 PPV
            pos = (c > b[:, j]).float()
            cs = torch.cumsum(pos, 0)
            lag = torch.cat([torch.zeros(self.window, cs.shape[1], device=self.device), cs])[:L]
            feats.append((cs - lag) / torch.clamp(cnt, max=self.window))
        pos = (c > b[:, 2]).float()
        feats.append(torch.cumsum(pos, 0) / cnt)          # 누적 PPV
        feats.append(torch.cummax(c, 0).values)           # 누적 최댓값
        feats.append(torch.as_tensor(Z, device=self.device))  # 현재 입력 채널 그대로
        return torch.cat(feats, 1)


class LinearHead:
    """표준화 + 다항 로지스틱 회귀 (class-balanced, L2). 미니배치 Adam."""

    def __init__(self, n_cls, epochs=30, lr=1e-3, wd=1e-4, hidden=0, device="cpu", seed=0):
        self.n_cls, self.epochs, self.lr, self.wd, self.hidden, self.device, self.seed = n_cls, epochs, lr, wd, hidden, device, seed

    def fit(self, X, y, cls_w):
        torch.manual_seed(self.seed)
        X = torch.as_tensor(X, dtype=torch.float32)
        self.mu, self.sd = X.mean(0), X.std(0).clamp_min(1e-6)
        d = X.shape[1]
        self.net = (torch.nn.Linear(d, self.n_cls) if not self.hidden else torch.nn.Sequential(
            torch.nn.Linear(d, self.hidden), torch.nn.GELU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(self.hidden, self.n_cls))).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        yt = torch.as_tensor(y, dtype=torch.long)
        wt = torch.as_tensor(cls_w, dtype=torch.float32, device=self.device)
        n, bs = len(X), 2048
        steps = self.epochs * ((n + bs - 1) // bs)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
        self.net.train()
        for ep in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                xb = ((X[idx] - self.mu) / self.sd).clamp(-10, 10).to(self.device)
                loss = F.cross_entropy(self.net(xb), yt[idx].to(self.device), weight=wt)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sch.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        z = ((X - self.mu.to(self.device)) / self.sd.to(self.device)).clamp(-10, 10)
        return torch.softmax(self.net(z), -1).cpu().numpy()
