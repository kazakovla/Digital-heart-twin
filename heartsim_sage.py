"""
HeartSimSage — GNN-эмулятор биомеханики сердца (учебная реализация)
=====================================================================
Программная реализация архитектуры, разобранной в отчёте по практике
(разделы 2.3–3.1), по первоисточнику:

    [15] Shi L., Chen Y., Vedula V. HeartSimSage: Attention-Enhanced Graph
         Neural Networks for Accelerating Cardiac Mechanics Modeling.
         arXiv:2504.18968, 2025.

ВАЖНО. Все числовые значения в комментариях (val loss, ошибки, времена,
веса внимания) — это результаты статьи [15], а не результаты этого кода.
Собственный эксперимент на реальных сетках — в каталоге src/ (см. README).

Версия 2 (исправления по замечаниям руководителя практики):
  1. solve_laplace_dirichlet: прежний итерационный цикл расходился
     (в числитель попадало собственное значение узла). Теперь задача решается
     точно — разреженной прямой факторизацией графового лапласиана.
  2. build_stratified_neighborhoods: пороги страт считались по 199 ближайшим
     соседям (дальняя страта ~6.7 мм вместо ~30 мм). Теперь Q_p% — перцентили
     ВСЕХ попарных расстояний между узлами, как в [15].
  3. AttentionMessagePassing: запрос (Query) теперь формируется из
     представления узла-получателя i, ключи/значения — из рёбер (i, k).
     Ранее каждое ребро «сравнивалось само с собой» (Q_k · K_k).
  4. Обучение на подмножестве N_S узлов действительно экономит вычисления:
     передача сообщений и декодирование выполняются только для N_S узлов.
  5. personalize_beta оптимизирует log β (раньше шаг Adam ≈0.01 при β ~ 10^3
     практически не менял параметры) и не допускает β < 0.
  6. Удалены неиспользуемые зависимости (torch_geometric.DataLoader,
     fenics-dolfinx); демонстрация работает на реальной КЭ-сетке из data/geom.

Зависимости:
    pip install torch numpy scipy
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# ГИПЕРПАРАМЕТРЫ (оптимальные значения по [15])
# ---------------------------------------------------------------------------

D_EMB = 128      # размерность векторных представлений
N_HEADS = 8      # число голов внимания
N_NB = 12        # число соседей в передаче сообщений
N_S = 300        # число узлов-подмножества на образец за эпоху
N_STRAT = 5      # число страт

# Границы страт — перцентили ВСЕХ попарных расстояний между узлами [15]:
# SC1 ≤ Q0.2%;  SC2: Q0.2–1%;  SC3: Q1–2%;  SC4: Q2–5%;  SC5: Q5–10%
STRAT_PERCENTILES = (0.2, 1.0, 2.0, 5.0, 10.0)   # в процентах
STRAT_SIZES = (20, 30, 30, 10, 10)               # кандидатов из каждой страты


# ===========================================================================
# ЧАСТЬ 1. ПОСТРОЕНИЕ ГРАФА ИЗ КЭ-СЕТКИ (раздел 2.3 отчёта)
# ===========================================================================

def mesh_to_graph(node_coords: np.ndarray,
                  elements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Тетраэдрическая КЭ-сетка -> неориентированный граф.

    Каждый тетраэдр даёт 6 рёбер (C(4,2)); общие рёбра соседних элементов
    учитываются один раз. Число уникальных рёбер ≈ 1.2–1.5 на тетраэдр.

    Возвращает
        edge_index : (2, M) — пары [src, dst] в обоих направлениях
        edge_attr  : (M, 3) — X_dst − X_src
    """
    pairs = np.concatenate([elements[:, [a, b]]
                            for a in range(4) for b in range(a + 1, 4)])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]          # вырожденные элементы
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    edge_attr = node_coords[dst] - node_coords[src]
    return np.stack([src, dst]).astype(np.int64), edge_attr


# ===========================================================================
# ЧАСТЬ 2. ПОЗИЦИОННОЕ КОДИРОВАНИЕ ЛАПЛАСА–ДИРИХЛЕ (раздел 2.3 отчёта)
# ===========================================================================

def solve_laplace_dirichlet(node_coords: np.ndarray,
                            adjacency: np.ndarray,
                            boundary_one: np.ndarray,
                            boundary_zero: np.ndarray,
                            weighted: bool = True) -> np.ndarray:
    """
    Задача Лапласа–Дирихле на графе сетки:

        L Φ = 0 во внутренних узлах,  Φ = 1 на Γa,  Φ = 0 на Γb,

    на остальной границе — естественное условие нулевого потока.
    L — графовый лапласиан с весами 1/|X_i − X_j| (weighted=True) либо
    единичными весами. Решение точное: L_ff Φ_f = −L_fc Φ_c.

    Для точной МКЭ-дискретизации (котангенсные веса) используйте
    src/fem_core.compute_laplace_fields — она применяется в эксперименте.

    Узлы, попавшие одновременно в Γa и Γb, относятся к Γa.
    """
    n = node_coords.shape[0]
    src, dst = adjacency
    if weighted:
        w = 1.0 / np.maximum(np.linalg.norm(node_coords[src] - node_coords[dst],
                                            axis=1), 1e-12)
    else:
        w = np.ones(len(src))
    W = sp.csr_matrix((w, (src, dst)), shape=(n, n))
    L = sp.diags(np.asarray(W.sum(axis=1)).ravel()) - W

    one = np.unique(boundary_one)
    zero = np.setdiff1d(np.unique(boundary_zero), one)
    fixed = np.zeros(n, dtype=bool)
    fixed[one] = fixed[zero] = True
    phi = np.zeros(n)
    phi[one] = 1.0

    free = ~fixed
    rhs = -(L[free][:, fixed] @ phi[fixed])
    A = L[free][:, free].tocsc()
    # изолированные узлы (нулевая строка) оставляем со значением 0
    deg = np.asarray(abs(A).sum(axis=1)).ravel()
    ok = deg > 0
    free_idx = np.where(free)[0]
    if np.all(ok):
        phi[free] = spla.spsolve(A, rhs)
    else:
        A2 = A[ok][:, ok]
        phi[free_idx[ok]] = spla.spsolve(A2.tocsc(), rhs[ok])
    return np.clip(phi, 0.0, 1.0).astype(np.float32)


def compute_all_laplace_fields(node_coords: np.ndarray,
                               adjacency: np.ndarray,
                               boundaries: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Пять полей Лапласа–Дирихле [15]:
        φ_AB    : база = 1, верхушка = 0         (апикобазальная координата)
        φ_EP    : эпикард = 1, эндокард ЛЖ+ПЖ = 0 (трансмуральная)
        φ_LV-RV : эндокард ЛЖ = 1, ПЖ = 0         (межжелудочковая)
        φ_LV-EP : эндокард ЛЖ = 1, эпикард = 0
        φ_RV-EP : эндокард ПЖ = 1, эпикард = 0

    boundaries: словарь {'base','apex','epi','endo_lv','endo_rv'} -> индексы.
    Возвращает (N, 5).
    """
    b = boundaries
    configs = [
        (b["base"], b["apex"]),
        (b["epi"], np.concatenate([b["endo_lv"], b["endo_rv"]])),
        (b["endo_lv"], b["endo_rv"]),
        (b["endo_lv"], b["epi"]),
        (b["endo_rv"], b["epi"]),
    ]
    return np.stack([solve_laplace_dirichlet(node_coords, adjacency, one, zero)
                     for one, zero in configs], axis=-1)


# ===========================================================================
# ЧАСТЬ 3. СТРАТИФИКАЦИОННЫЙ ВЫБОР ОКРЕСТНОСТЕЙ (раздел 2.4 отчёта)
# ===========================================================================

def pairwise_distance_percentiles(node_coords: np.ndarray,
                                  percentiles=STRAT_PERCENTILES,
                                  n_sample: int = 2000,
                                  rng: Optional[np.random.Generator] = None
                                  ) -> np.ndarray:
    """
    Оценка перцентилей Q_p% распределения ВСЕХ попарных расстояний.
    Точный расчёт требует N^2/2 пар (~1.5·10^8 при N = 17 000), поэтому
    берутся расстояния от n_sample случайных узлов до всех узлов
    (несмещённая оценка того же распределения).
    """
    rng = rng or np.random.default_rng(0)
    n = len(node_coords)
    idx = rng.choice(n, size=min(n, n_sample), replace=False)
    chunks = []
    for part in np.array_split(idx, max(1, len(idx) // 200)):
        d = np.linalg.norm(node_coords[part][:, None, :] - node_coords[None],
                           axis=-1)
        chunks.append(d[d > 0])
    return np.percentile(np.concatenate(chunks), percentiles)


def build_stratified_neighborhoods(node_coords: np.ndarray,
                                   n_nb: int = N_NB,
                                   seed: int = 0,
                                   return_strata: bool = False):
    """
    Стратификационные окрестности по [15] (раздел 2.4 отчёта).

    1. Пороги Q0.2%, Q1%, Q2%, Q5%, Q10% — перцентили всех попарных
       расстояний между узлами сетки (одни для всей геометрии).
    2. Для узла i кандидаты страты SCm — узлы на расстоянии (Q_{m-1}, Q_m];
       из каждой страты случайно берётся до STRAT_SIZES[m] кандидатов.
    3. Из объединения кандидатов (до 100) случайно выбираются n_nb соседей.

    Возвращает (N, n_nb) индексы соседей; при return_strata=True —
    также (N, n_nb) номера страт (нужны для анализа весов внимания).
    """
    rng = np.random.default_rng(seed)
    n = len(node_coords)
    thr = pairwise_distance_percentiles(node_coords, rng=rng)
    tree = cKDTree(node_coords)
    lower = np.concatenate([[0.0], thr[:-1]])

    nb = np.zeros((n, n_nb), dtype=np.int64)
    nb_strat = np.zeros((n, n_nb), dtype=np.int64)
    for i in range(n):
        cand = np.asarray(tree.query_ball_point(node_coords[i], thr[-1]))
        cand = cand[cand != i]
        d = np.linalg.norm(node_coords[cand] - node_coords[i], axis=1)
        pick, pick_s = [], []
        for s in range(N_STRAT):
            c = cand[(d > lower[s]) & (d <= thr[s])]
            if len(c):
                take = min(STRAT_SIZES[s], len(c))
                pick.append(rng.choice(c, size=take, replace=False))
                pick_s.append(np.full(take, s))
        if not pick:                       # изолированный узел
            _, idx = tree.query(node_coords[i], k=n_nb + 1)
            nb[i] = idx[1:]
            continue
        pool, pool_s = np.concatenate(pick), np.concatenate(pick_s)
        sel = rng.choice(len(pool), size=n_nb, replace=len(pool) < n_nb)
        nb[i], nb_strat[i] = pool[sel], pool_s[sel]
    return (nb, nb_strat) if return_strata else nb


# ===========================================================================
# ЧАСТЬ 4. АРХИТЕКТУРА HEARTSIM SAGE — 5 МОДУЛЕЙ (раздел 3.1 отчёта)
# ===========================================================================

def _mlp(d_in: int, d_out: int, d_hidden: int = D_EMB, n_hidden: int = 1):
    layers, d = [], d_in
    for _ in range(n_hidden):
        layers += [nn.Linear(d, d_hidden), nn.ReLU()]
        d = d_hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class NodeEncoder(nn.Module):
    """
    Модуль 1. Раздельное кодирование ориентаций волокон (f0, s0 — 6 чисел)
    и полей Лапласа (5 чисел) с последующим суммированием:
        n_i = ENC_f(a_i) + ENC_l(Φ_i)
    """

    def __init__(self, fiber_dim: int = 6, laplace_dim: int = 5,
                 d_emb: int = D_EMB):
        super().__init__()
        self.enc_fiber = _mlp(fiber_dim, d_emb, d_emb)
        self.enc_laplace = _mlp(laplace_dim, d_emb, d_emb)

    def forward(self, fiber_feats, laplace_feats):
        return self.enc_fiber(fiber_feats) + self.enc_laplace(laplace_feats)


class EdgeEncoder(nn.Module):
    """Рёберный вектор ê_ik = (n_i, n_k, Φ_k − Φ_i, X_k − X_i) -> e_ik."""

    def __init__(self, d_emb: int = D_EMB, laplace_dim: int = 5):
        super().__init__()
        self.mlp = _mlp(2 * d_emb + laplace_dim + 3, d_emb, d_emb)

    def forward(self, ni, nk, delta_phi, delta_x):
        return self.mlp(torch.cat([ni, nk, delta_phi, delta_x], dim=-1))


class AttentionMessagePassing(nn.Module):
    """
    Модуль 3. Анизотропная передача сообщений с многоголовым вниманием.

        q_i   = W_Q n_i                  (запрос — от узла-получателя)
        k_ik  = W_K e_ik,  v_ik = W_V e_ik
        α_ik  = softmax_k( q_i · k_ik / √d_head )   (по каждой голове)
        n'_i  = n_i + W_O Σ_k α_ik v_ik              (остаточное соединение)

    Веса α_ik зависят и от узла i, и от ребра (i, k) — это и есть
    анизотропная схема в терминах раздела 2.2 отчёта.

    target_idx позволяет вычислять сообщения только для подмножества узлов
    (обучение на N_S узлах): соседи при этом берутся из полного графа.
    """

    def __init__(self, d_emb: int = D_EMB, n_heads: int = N_HEADS,
                 laplace_dim: int = 5):
        super().__init__()
        assert d_emb % n_heads == 0, "D_EMB должна делиться на N_HEADS"
        self.h, self.dh, self.d = n_heads, d_emb // n_heads, d_emb
        self.edge_encoder = EdgeEncoder(d_emb, laplace_dim)
        self.W_Q = nn.Linear(d_emb, d_emb, bias=False)
        self.W_K = nn.Linear(d_emb, d_emb, bias=False)
        self.W_V = nn.Linear(d_emb, d_emb, bias=False)
        self.W_O = nn.Linear(d_emb, d_emb)
        self.last_alpha: Optional[torch.Tensor] = None   # для анализа весов

    def forward(self, node_emb, neighbor_idx, laplace_feats, node_coords,
                target_idx: Optional[torch.Tensor] = None):
        if target_idx is None:
            target_idx = torch.arange(node_emb.shape[0],
                                      device=node_emb.device)
        nb = neighbor_idx[target_idx]                         # (T, K)
        t, k = nb.shape
        ni = node_emb[target_idx]                             # (T, D)
        nk = node_emb[nb]                                     # (T, K, D)
        dphi = laplace_feats[nb] - laplace_feats[target_idx].unsqueeze(1)
        dx = node_coords[nb] - node_coords[target_idx].unsqueeze(1)
        e = self.edge_encoder(ni.unsqueeze(1).expand(-1, k, -1), nk, dphi, dx)

        q = self.W_Q(ni).view(t, self.h, self.dh)             # (T, H, dh)
        kk = self.W_K(e).view(t, k, self.h, self.dh)          # (T, K, H, dh)
        vv = self.W_V(e).view(t, k, self.h, self.dh)
        scores = torch.einsum("thd,tkhd->tkh", q, kk) / math.sqrt(self.dh)
        alpha = torch.softmax(scores, dim=1)                  # по соседям
        self.last_alpha = alpha.detach()
        msg = torch.einsum("tkh,tkhd->thd", alpha, vv).reshape(t, self.d)
        return ni + self.W_O(msg)


class GlobalEncoder(nn.Module):
    """
    Модуль 4. Параметры материала β (8), давления (2) и признаки формы (40)
    -> глобальный вектор g. Параметры a, a_f, a_s (диапазон в несколько
    порядков) подаются в логарифмической шкале.
    """

    LOG_IDX = (0, 2, 4)

    def __init__(self, beta_dim: int = 8, pressure_dim: int = 2,
                 shape_dim: int = 40, d_emb: int = D_EMB):
        super().__init__()
        self.mlp = _mlp(beta_dim + pressure_dim + shape_dim, d_emb, d_emb, 2)

    def forward(self, beta, pressure, shape_feats):
        beta = beta.clone()
        li = list(self.LOG_IDX)
        beta[:, li] = torch.log(beta[:, li].clamp(min=1e-8))
        return self.mlp(torch.cat([beta, pressure, shape_feats], dim=-1))


class DisplacementDecoder(nn.Module):
    """
    Модуль 5. Три независимых MLP-декодера для u_x, u_y, u_z.

    Замечание: оси x, y, z — оси системы координат изображения, а не
    направления волокон, поэтому раздельные декодеры не «учитывают
    анизотропию миокарда»; анизотропия учитывается признаками волокон
    (модуль 1). Раздельные декодеры лишь дают каждой компоненте свою
    подсеть (так сделано в [15]).
    """

    def __init__(self, d_emb: int = D_EMB):
        super().__init__()
        self.dec = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * d_emb, d_emb), nn.ReLU(),
                          nn.Linear(d_emb, d_emb // 2), nn.ReLU(),
                          nn.Linear(d_emb // 2, 1)) for _ in range(3))

    def forward(self, node_emb, global_emb):
        x = torch.cat([node_emb, global_emb.expand(node_emb.shape[0], -1)], -1)
        return torch.cat([d(x) for d in self.dec], dim=-1)


class HeartSimSage(nn.Module):
    """
    HeartSimSage: û_i = HSS_θ(ĥ_i, {ĥ_k}_{k∈N_i}; ĝ).

    Полностью индуктивная модель: нет обучаемых вложений конкретных узлов,
    поэтому применима к сеткам новых пациентов с любым числом узлов.
    """

    def __init__(self, fiber_dim=6, laplace_dim=5, beta_dim=8,
                 pressure_dim=2, shape_dim=40, d_emb=D_EMB, n_heads=N_HEADS):
        super().__init__()
        self.node_encoder = NodeEncoder(fiber_dim, laplace_dim, d_emb)
        # Модуль 2 (окрестности) — build_stratified_neighborhoods, вне сети
        self.message_passing = AttentionMessagePassing(d_emb, n_heads,
                                                       laplace_dim)
        self.global_encoder = GlobalEncoder(beta_dim, pressure_dim,
                                            shape_dim, d_emb)
        self.decoder = DisplacementDecoder(d_emb)

    def forward(self, fiber_feats, laplace_feats, node_coords, neighbor_idx,
                beta, pressure, shape_feats,
                target_idx: Optional[torch.Tensor] = None):
        """
        Возвращает смещения (N, 3) либо (|target_idx|, 3), если задано
        подмножество узлов.
        """
        node_emb = self.node_encoder(fiber_feats, laplace_feats)
        node_emb = self.message_passing(node_emb, neighbor_idx, laplace_feats,
                                        node_coords, target_idx)
        g = self.global_encoder(beta, pressure, shape_feats)
        return self.decoder(node_emb, g)


# ===========================================================================
# ЧАСТЬ 5. ОБУЧЕНИЕ НА ПОДМНОЖЕСТВАХ УЗЛОВ (раздел 3.1 отчёта)
# ===========================================================================

MODEL_INPUTS = ("fiber_feats", "laplace_feats", "node_coords", "neighbor_idx",
                "beta", "pressure", "shape_feats")


class HeartSimSageTrainer:
    """
    На каждой эпохе для каждого образца случайно выбирается N_S узлов;
    сообщения и декодирование вычисляются ТОЛЬКО для них, поэтому затраты
    на передачу сообщений снижаются в N / N_S раз (≈57 раз при N≈17 000).
    По [15]: val loss 0.0091 при N_S = 300 и при N_S = 1000; 0.0099 при N_S = 10.

    Образец — словарь с ключами MODEL_INPUTS и 'displacements' (N, 3).
    """

    def __init__(self, model: HeartSimSage, lr: float = 1e-4,
                 n_subset: int = N_S, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.n_subset = n_subset
        self.optim = torch.optim.Adam(model.parameters(), lr=lr)

    def _to(self, sample):
        return {k: v.to(self.device) for k, v in sample.items()}

    def train_epoch(self, samples: Iterable[Dict[str, torch.Tensor]]) -> float:
        self.model.train()
        losses = []
        for s in samples:
            s = self._to(s)
            n = s["fiber_feats"].shape[0]
            sub = torch.randperm(n, device=self.device)[:self.n_subset]
            pred = self.model(**{k: s[k] for k in MODEL_INPUTS},
                              target_idx=sub)
            loss = F.mse_loss(pred, s["displacements"][sub])
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    @torch.no_grad()
    def evaluate(self, samples) -> Dict[str, float]:
        """Нормированная ошибка |û − u| / max|u| по осям (метрика [15])."""
        self.model.eval()
        errs = []
        for s in samples:
            s = self._to(s)
            pred = self.model(**{k: s[k] for k in MODEL_INPUTS})
            target = s["displacements"]
            scale = target.abs().max(dim=0).values.clamp(min=1e-8)
            errs.append(((pred - target).abs() / scale).mean(dim=0).cpu().numpy())
        e = np.mean(errs, axis=0)
        return {"Err_ux": float(e[0]), "Err_uy": float(e[1]),
                "Err_uz": float(e[2]), "Err_mean": float(e.mean())}

    def train(self, train_samples, val_samples, n_epochs: int = 3000,
              log_every: int = 100):
        for epoch in range(1, n_epochs + 1):
            loss = self.train_epoch(train_samples)
            if epoch % log_every == 0 or epoch == 1:
                m = self.evaluate(val_samples)
                print(f"эпоха {epoch:4d} | loss {loss:.6f} | "
                      f"Err_mean {m['Err_mean'] * 100:.3f}%")


# ===========================================================================
# ЧАСТЬ 6. КОНВЕЙЕР ЦИФРОВОГО ДВОЙНИКА
# ===========================================================================

class CardiacDigitalTwinPipeline:
    """
    1. Геометрия из КТ/МРТ -> тетраэдральная сетка
    2. Предобработка: граф, 5 полей Лапласа, окрестности, признаки формы
    3. Предсказание HSS
    4. Персонализация β градиентным спуском через дифференцируемый HSS
    5. Контрольный расчёт КЭА

    Ориентиры времени по [15]: КЭА 112 ± 35 с (CPU), HSS 0.0084 с (GPU).
    Оценка «100 итераций персонализации ≈ 0.84 с против ≈ 3.1 ч» — это
    пересчёт 100 × 0.0084 с и 100 × 112 с, без учёта обратного прохода;
    в [15] такой замер не приводится.
    """

    def __init__(self, hss_model: HeartSimSage, device: str = "cpu"):
        self.model = hss_model.eval().to(device)
        self.device = device

    def _t(self, x, unsq=False):
        t = torch.as_tensor(np.asarray(x), dtype=torch.float32,
                            device=self.device)
        return t.unsqueeze(0) if unsq else t

    def preprocess_geometry(self, node_coords, elements, boundaries,
                            shape_feats: Optional[np.ndarray] = None):
        edge_index, _ = mesh_to_graph(node_coords, elements)
        laplace = compute_all_laplace_fields(node_coords, edge_index,
                                             boundaries)
        nb = build_stratified_neighborhoods(node_coords)
        if shape_feats is None:
            # 40 признаков формы в [15] получают по модам статистической
            # модели формы; здесь — нулевая заглушка (см. src/graph_features
            # для реальных признаков формы, используемых в эксперименте).
            shape_feats = np.zeros(40, dtype=np.float32)
        return {"laplace_feats": laplace, "neighbor_idx": nb,
                "shape_feats": shape_feats, "node_coords": node_coords}

    def _inputs(self, pre, fiber_feats, pressure):
        return dict(fiber_feats=self._t(fiber_feats),
                    laplace_feats=self._t(pre["laplace_feats"]),
                    node_coords=self._t(pre["node_coords"]),
                    neighbor_idx=torch.as_tensor(pre["neighbor_idx"],
                                                 device=self.device),
                    pressure=self._t(pressure, True),
                    shape_feats=self._t(pre["shape_feats"], True))

    @torch.no_grad()
    def predict(self, pre, fiber_feats, beta, pressure) -> np.ndarray:
        x = self._inputs(pre, fiber_feats, pressure)
        return self.model(beta=self._t(beta, True), **x).cpu().numpy()

    def personalize_beta(self, pre, fiber_feats, pressure, observed_disp,
                         beta_init: Optional[np.ndarray] = None,
                         n_iter: int = 100, lr: float = 0.05,
                         fit_idx: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
                         ) -> np.ndarray:
        """
        β* = argmin ‖HSS(β) − u_obs‖². Оптимизируется log β (β > 0 всегда,
        параметры разных порядков обновляются соизмеримо). Параметры,
        не входящие в fit_idx (по умолчанию a_fs, b_fs), фиксированы.
        """
        if beta_init is None:
            beta_init = np.array([500., 10., 5000., 10., 500., 10., 1., 1.],
                                 dtype=np.float32)
        log_b = torch.log(self._t(np.maximum(beta_init, 1e-6)))
        mask = torch.zeros_like(log_b)
        mask[list(fit_idx)] = 1.0
        delta = nn.Parameter(torch.zeros_like(log_b))
        opt = torch.optim.Adam([delta], lr=lr)
        x = self._inputs(pre, fiber_feats, pressure)
        target = self._t(observed_disp)
        for it in range(n_iter):
            beta = torch.exp(log_b + delta * mask).unsqueeze(0)
            loss = F.mse_loss(self.model(beta=beta, **x), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (it + 1) % 20 == 0:
                print(f"      итерация {it + 1:3d}: loss = {loss.item():.6f}")
        return torch.exp(log_b + delta.detach() * mask).cpu().numpy()


# ===========================================================================
# ДЕМОНСТРАЦИЯ НА РЕАЛЬНОЙ КЭ-СЕТКЕ
# ===========================================================================

def load_real_geometry(path: str):
    """
    Загрузка бивентрикулярной сетки из data/geom/geom_XX.npz
    (Rodero et al., 2021, Zenodo 4590294; подготовка — src/prepare_meshes.py).
    Возвращает координаты (мм), элементы, волокна (узловые), границы.
    """
    d = np.load(path)
    X, E = d["coords"].astype(np.float64), d["elements"].astype(np.int64)
    faces, ftype = d["faces"], d["face_type"]
    nodes = lambda code: np.unique(faces[ftype == code])   # noqa: E731
    epi = nodes(0)
    z = d["uvc_z"]
    apex = epi[z[epi] <= np.percentile(z[epi], 2.0)]
    boundaries = {"epi": epi, "endo_lv": nodes(1), "endo_rv": nodes(2),
                  "base": nodes(3), "apex": apex}
    # волокна заданы поэлементно -> усреднение по инцидентным тетраэдрам
    fib = np.zeros((len(X), 6))
    np.add.at(fib, E.ravel(),
              np.repeat(np.hstack([d["f0"], d["s0"]]), 4, axis=0))
    for a in (slice(0, 3), slice(3, 6)):
        fib[:, a] /= np.maximum(np.linalg.norm(fib[:, a], axis=1,
                                               keepdims=True), 1e-12)
    return X, E, fib.astype(np.float32), boundaries


if __name__ == "__main__":
    torch.manual_seed(42)
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "data", "geom", "geom_01.npz")

    print("=" * 64)
    print("  HeartSimSage — демонстрация компонентов на реальной сетке")
    print("=" * 64)
    X, E, fib, bnd = load_real_geometry(path)
    print(f"\nГеометрия: {os.path.basename(path)} | узлов {len(X)} | "
          f"тетраэдров {len(E)}")

    print("\n[1] Граф из КЭ-сетки")
    ei, _ = mesh_to_graph(X, E)
    print(f"    уникальных рёбер: {ei.shape[1] // 2} "
          f"({ei.shape[1] // 2 / len(E):.2f} на тетраэдр)")

    print("\n[2] Поля Лапласа–Дирихле")
    lap = compute_all_laplace_fields(X, ei, bnd)
    print(f"    форма {lap.shape}, диапазон [{lap.min():.3f}, {lap.max():.3f}]")
    corr = np.corrcoef(lap[:, 0], np.load(path)["uvc_z"])[0, 1]
    print(f"    корреляция φ_AB с апикобазальной координатой UVC: {corr:.3f}")

    print("\n[3] Стратифицированные окрестности")
    nb, st = build_stratified_neighborhoods(X, return_strata=True)
    thr = pairwise_distance_percentiles(X)
    print("    пороги Q0.2/1/2/5/10%, мм: " + ", ".join(f"{t:.1f}" for t in thr))
    print("    доля соседей по стратам SC1..SC5: "
          + ", ".join(f"{np.mean(st == s):.2f}" for s in range(N_STRAT)))

    print("\n[4] Модель и прямой проход")
    model = HeartSimSage()
    print(f"    параметров: {sum(p.numel() for p in model.parameters()):,}")
    Xn = (X - X.mean(0)) / np.abs(X - X.mean(0)).max()
    beta = torch.tensor([[500., 10., 5000., 10., 500., 10., 1., 1.]])
    inputs = dict(fiber_feats=torch.tensor(fib), laplace_feats=torch.tensor(lap),
                  node_coords=torch.tensor(Xn, dtype=torch.float32),
                  neighbor_idx=torch.tensor(nb), beta=beta,
                  pressure=torch.tensor([[10., 4.]]),
                  shape_feats=torch.zeros(1, 40))
    with torch.no_grad():
        u_all = model(**inputs)
        sub = torch.randperm(len(X))[:N_S]
        u_sub = model(**inputs, target_idx=sub)
    print(f"    все узлы: {tuple(u_all.shape)}; подмножество N_S: "
          f"{tuple(u_sub.shape)}; совпадение {torch.allclose(u_all[sub], u_sub, atol=1e-5)}")

    print("\n[5] Один шаг обучения на подмножестве (данные-заглушка)")
    trainer = HeartSimSageTrainer(model, device="cpu")
    sample = {**inputs, "displacements": torch.zeros(len(X), 3)}
    print(f"    loss = {trainer.train_epoch([sample]):.6f}")

    print("\nДемонстрация завершена. Обучение на реальных КЭ-решениях: "
          "src/train_hss.py")
