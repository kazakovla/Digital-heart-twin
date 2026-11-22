"""
Эмулятор HeartSimSage (HSS) для реальных бивентрикулярных КЭ-сеток.

Архитектура следует разделу 3 НИР, но адаптирована к масштабу данных,
доступному в настоящем эксперименте (11 геометрий, ~7.5 тыс. узлов каждая):

  1. Кодирование узловых признаков:
       - волокна (f0, s0) — 6 компонент,
       - поля Лапласа-Дирихле — 5 компонент (позиционное кодирование),
       - нормированные координаты — 3 компоненты.
  2. Стратифицированные окрестности (K соседей из 5 страт по расстоянию):
       геодезически-взвешенное внимание вместо равномерного усреднения.
  3. Глобальные условия (материал beta, давления, признаки формы) подаются
       через FiLM-модуляцию на каждом слое сообщений.
  Отличия от [15] (сознательные, из-за малого объёма данных): несколько слоёв
  сообщений с FiLM вместо одного слоя и отдельного глобального кодировщика,
  один декодер на 3 компоненты вместо трёх; координаты и смещения --
  в анатомических осях сердца.
  4. Декодер предсказывает 3 компоненты смещения в узле.

Отличие от учебной реализации heartsim_sage.py: там сетка и признаки
генерировались случайно (np.random.randn), здесь всё считается на реальной
геометрии из Zenodo 4590294.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

N_STRATA = 5


@dataclass
class HSSConfig:
    """Гиперпараметры эмулятора."""

    d_node: int = 14          # 6 (волокна) + 5 (Лаплас) + 3 (координаты)
    d_global: int = 0         # заполняется по факту (beta + давления + форма)
    d_hidden: int = 128
    n_layers: int = 4
    n_heads: int = 4
    k_neighbors: int = 16
    dropout: float = 0.0


class FiLM(nn.Module):
    """Модуляция скрытых признаков глобальным контекстом (масштаб и сдвиг)."""

    def __init__(self, d_global: int, d_hidden: int):
        super().__init__()
        self.net = nn.Linear(d_global, 2 * d_hidden)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(g).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta


class StratifiedAttention(nn.Module):
    """
    Слой передачи сообщений по стратифицированным окрестностям.

    Для каждого узла i задан фиксированный набор соседей j (K штук,
    распределённых по 5 стратам расстояния). Вес сообщения определяется
    многоголовым вниманием, в логит которого добавлено геометрическое
    смещение: относительный вектор r_ij, его длина и номер страты.
    """

    def __init__(self, d_hidden: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_hidden % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_hidden // n_heads

        self.q = nn.Linear(d_hidden, d_hidden)
        self.k = nn.Linear(d_hidden, d_hidden)
        self.v = nn.Linear(d_hidden, d_hidden)
        self.out = nn.Linear(d_hidden, d_hidden)

        # геометрический кодировщик ребра: [r_ij (3), |r_ij|, one-hot страты]
        self.edge_enc = nn.Sequential(
            nn.Linear(4 + N_STRATA, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.edge_bias = nn.Linear(d_hidden, n_heads)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,          # (N, d)
        rel: torch.Tensor,        # (N, K, 3) относительные координаты соседей
        strata: torch.Tensor,     # (N, K) номер страты
    ) -> torch.Tensor:
        n, k, _ = rel.shape
        nb_idx = self.nb_idx                    # (N, K), выставляется извне

        dist = rel.norm(dim=-1, keepdim=True)   # (N, K, 1)
        st_oh = F.one_hot(strata, N_STRATA).to(rel.dtype)
        e = self.edge_enc(torch.cat([rel, dist, st_oh], dim=-1))   # (N,K,d)

        q = self.q(h).view(n, self.n_heads, self.d_head)
        kk = self.k(h)[nb_idx].view(n, k, self.n_heads, self.d_head)
        vv = self.v(h)[nb_idx].view(n, k, self.n_heads, self.d_head)
        vv = vv + e.view(n, k, self.n_heads, self.d_head)

        logits = torch.einsum("nhd,nkhd->nkh", q, kk) / self.d_head**0.5
        logits = logits + self.edge_bias(e)                        # (N,K,H)
        attn = self.drop(torch.softmax(logits, dim=1))
        self.last_attn = attn.detach()          # (N, K, H) -- для анализа по стратам

        msg = torch.einsum("nkh,nkhd->nhd", attn, vv).reshape(n, -1)
        return self.out(msg)


class HSSBlock(nn.Module):
    """Блок: внимание по окрестности + FiLM + MLP, всё с остаточной связью."""

    def __init__(self, cfg: HSSConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_hidden)
        self.attn = StratifiedAttention(cfg.d_hidden, cfg.n_heads, cfg.dropout)
        self.film = FiLM(cfg.d_global, cfg.d_hidden)
        self.norm2 = nn.LayerNorm(cfg.d_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_hidden, 2 * cfg.d_hidden),
            nn.SiLU(),
            nn.Linear(2 * cfg.d_hidden, cfg.d_hidden),
        )

    def forward(self, h, rel, strata, nb_idx, g):
        self.attn.nb_idx = nb_idx
        h = h + self.attn(self.norm1(h), rel, strata)
        h = self.film(h, g)
        h = h + self.mlp(self.norm2(h))
        return h


class HeartSimSage(nn.Module):
    """Полная модель: кодировщик узлов -> блоки сообщений -> декодер смещений."""

    def __init__(self, cfg: HSSConfig):
        super().__init__()
        self.cfg = cfg
        self.node_enc = nn.Sequential(
            nn.Linear(cfg.d_node, cfg.d_hidden),
            nn.SiLU(),
            nn.Linear(cfg.d_hidden, cfg.d_hidden),
        )
        self.global_enc = nn.Sequential(
            nn.Linear(cfg.d_global, cfg.d_hidden),
            nn.SiLU(),
            nn.Linear(cfg.d_hidden, cfg.d_global),
        )
        self.blocks = nn.ModuleList(HSSBlock(cfg) for _ in range(cfg.n_layers))
        self.norm_out = nn.LayerNorm(cfg.d_hidden)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.d_hidden, cfg.d_hidden),
            nn.SiLU(),
            nn.Linear(cfg.d_hidden, 3),
        )

    def forward(
        self,
        node_feats: torch.Tensor,    # (N, d_node)
        rel: torch.Tensor,           # (N, K, 3)
        strata: torch.Tensor,        # (N, K)
        nb_idx: torch.Tensor,        # (N, K)
        g: torch.Tensor,             # (d_global,)
    ) -> torch.Tensor:
        h = self.node_enc(node_feats)
        g = g + self.global_enc(g)
        for blk in self.blocks:
            h = blk(h, rel, strata, nb_idx, g)
        return self.decoder(self.norm_out(h))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
