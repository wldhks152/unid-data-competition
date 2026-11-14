"""인과 시퀀스 신경망 (가중치는 무작위 초기화에서 처음부터 학습).

* CausalGRU: 단방향 GRU → 행 t의 출력은 0..t 입력만 사용
* CausalTCN: 왼쪽에만 패딩한 dilated Conv1d + 시점별 LayerNorm (시간축 정규화 없음)
둘 다 은닉 상태의 '시간 누적 최댓값'(torch.cummax, 과거만)을 헤드에 함께 넣는다.
label이 실험 안에서 단조 증가하므로, 한 번 나타난 사건 신호를 오래 기억하는 데 도움이 된다.
배치는 오른쪽 패딩만 쓰므로 패딩이 실제 시점의 출력에 영향을 주지 않는다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalGRU(nn.Module):
    def __init__(self, c_in, n_cls, hid=128, layers=2, drop=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(c_in, hid), nn.GELU(), nn.Dropout(drop))
        self.rnn = nn.GRU(hid, hid, num_layers=layers, batch_first=True, dropout=drop if layers > 1 else 0)
        self.head = nn.Sequential(nn.LayerNorm(2 * hid), nn.Dropout(drop), nn.Linear(2 * hid, n_cls))

    def forward(self, x):  # x: (B, L, C)
        h, _ = self.rnn(self.inp(x))
        hm = torch.cummax(h, dim=1).values
        return self.head(torch.cat([h, hm], -1))


class _Block(nn.Module):
    def __init__(self, ch, k, d, drop):
        super().__init__()
        self.pad = (k - 1) * d
        self.c1 = nn.Conv1d(ch, ch, k, dilation=d)
        self.c2 = nn.Conv1d(ch, ch, k, dilation=d)
        self.n1, self.n2 = nn.LayerNorm(ch), nn.LayerNorm(ch)
        self.drop = nn.Dropout(drop)

    def _conv(self, conv, norm, x):  # x: (B, C, L) — 왼쪽 패딩만 → 인과
        y = conv(F.pad(x, (self.pad, 0)))
        return F.gelu(norm(y.transpose(1, 2)).transpose(1, 2))

    def forward(self, x):
        y = self.drop(self._conv(self.c1, self.n1, x))
        y = self.drop(self._conv(self.c2, self.n2, y))
        return x + y


class CausalTCN(nn.Module):
    def __init__(self, c_in, n_cls, ch=96, k=3, dilations=(1, 2, 4, 8, 16, 32, 64), drop=0.15):
        super().__init__()
        self.inp = nn.Conv1d(c_in, ch, 1)
        self.blocks = nn.ModuleList([_Block(ch, k, d, drop) for d in dilations])
        self.head = nn.Sequential(nn.LayerNorm(2 * ch), nn.Dropout(drop), nn.Linear(2 * ch, n_cls))

    def forward(self, x):  # x: (B, L, C)
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        h = h.transpose(1, 2)
        hm = torch.cummax(h, dim=1).values
        return self.head(torch.cat([h, hm], -1))


def build(name, c_in, n_cls, **kw):
    return {"gru": CausalGRU, "tcn": CausalTCN}[name](c_in, n_cls, **kw)
