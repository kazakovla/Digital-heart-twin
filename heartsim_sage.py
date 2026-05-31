"""
HeartSimSage — GNN-эмулятор биомеханики сердца
=================================================
Программная реализация архитектуры, описанной в НИР:
«Аппарат связи временного ряда с графовой структурой,
 применяемый к цифровому двойнику сердца»

Источник: Shi L., Chen Y., Vedula V.
          HeartSimSage: Attention-Enhanced Graph Neural Networks
          for Accelerating Cardiac Mechanics Modeling
          arXiv:2504.18968, 2025.

Зависимости:
    pip install torch torch-geometric numpy scipy fenics-dolfinx
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from scipy.spatial import cKDTree
from typing import Optional, Tuple, List, Dict

# ---------------------------------------------------------------------------
# КОНСТАНТЫ / ГИПЕРПАРАМЕТРЫ
# ---------------------------------------------------------------------------

D_EMB   = 128    # Размерность векторных представлений (embeddings)
N_HEADS = 8      # Число голов в механизме внимания (Multi-Head Attention)
N_NB    = 12     # Число соседей, выбираемых при передаче сообщений (MP)
N_S     = 300    # Число узлов-подмножества, используемых на одну эпоху
N_STRAT = 5      # Число перцентильных страт при формировании окрестностей

# Перцентильные границы страт (из формулы 2.4 в НИР):
# SC1 ≤ 0.2%;  SC2: 0.2–1%;  SC3: 1–2%;  SC4: 2–5%;  SC5: 5–10%
STRAT_PERCENTILES = [0.002, 0.01, 0.02, 0.05, 0.10]

# Число соседей, выбираемых из каждой страты (суммарно > N_NB, затем clip)
STRAT_SIZES = [20, 30, 30, 10, 10]  # n_SC1..n_SC5


# ===========================================================================
# ЧАСТЬ 1.  ПОСТРОЕНИЕ ГРАФА ИЗ КЭ-СЕТКИ
# (Раздел 2.2 НИР)
# ===========================================================================

def mesh_to_graph(node_coords: np.ndarray,
                  elements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Преобразует тетраэдрическую КЭ-сетку в граф (список рёбер).

    КЭ-сетка порождает граф «органично»: каждый тетраэдральный элемент
    определяет 6 пар узлов, совместно участвующих в передаче механических
    усилий (рёбра тетраэдра → рёбра графа).

    Параметры
    ----------
    node_coords : (N, 3) — координаты X, Y, Z узлов сетки
    elements    : (E, 4) — индексы узлов для каждого тетраэдра

    Возвращает
    ----------
    edge_index : (2, M) — индексы пар [src, dst] (симметричный граф)
    edge_attr  : (M, 3) — вектор-смещение между узлами ребра (Xj − Xi)
    """
    edge_set = set()
    for tet in elements:
        # Перебираем все 6 рёбер тетраэдра (C(4,2) = 6)
        for a in range(4):
            for b in range(a + 1, 4):
                i, j = int(tet[a]), int(tet[b])
                edge_set.add((i, j))
                edge_set.add((j, i))   # граф ненаправленный → два направления

    src_list, dst_list = zip(*sorted(edge_set))
    src = np.array(src_list, dtype=np.int64)
    dst = np.array(dst_list, dtype=np.int64)

    # Атрибут ребра — вектор смещения между узлами (используется в MP-слое)
    edge_attr = node_coords[dst] - node_coords[src]  # (M, 3)

    return np.stack([src, dst], axis=0), edge_attr


# ===========================================================================
# ЧАСТЬ 2.  ПОЗИЦИОННОЕ КОДИРОВАНИЕ ЛАПЛАСА–ДИРИХЛЕ
# (Раздел 2.2, формула 2.3 НИР)
# ===========================================================================

def solve_laplace_dirichlet(node_coords: np.ndarray,
                            adjacency: np.ndarray,
                            boundary_one: np.ndarray,
                            boundary_zero: np.ndarray) -> np.ndarray:
    """
    Решает задачу Лапласа–Дирихле методом конечных разностей на графе:

        ∇²Φ = 0   в Ω
        Φ = 1     на Γa  (boundary_one)
        Φ = 0     на Γb  (boundary_zero)
        ∇Φ·n = 0  на Γn  (остальные граничные узлы)

    Пять таких задач (φ_AB, φ_EP, φ_LV-RV, φ_LV-EP, φ_RV-EP) дают
    5-мерный вектор позиционного кодирования для каждого узла (Табл. 2.1).

    Преимущества перед бинарным one-hot кодированием:
      - гладкость поля по всему объёму миокарда;
      - инвариантность к нумерации узлов;
      - явная метрическая информация о расстоянии от границ.
    Замена Лапласа бинарным кодированием ухудшает val-loss на 20.9%.

    Параметры
    ----------
    node_coords  : (N, 3) — координаты узлов
    adjacency    : (2, M) — рёбра графа (edge_index)
    boundary_one : (K1,)  — индексы узлов с условием Φ = 1 (Γa)
    boundary_zero: (K0,)  — индексы узлов с условием Φ = 0 (Γb)

    Возвращает
    ----------
    phi : (N,) — значения скалярного поля Φ в каждом узле
    """
    N = node_coords.shape[0]
    phi = np.zeros(N, dtype=np.float64)

    # Применяем граничные условия Дирихле
    phi[boundary_one]  = 1.0
    phi[boundary_zero] = 0.0

    # Маска «свободных» узлов (не принадлежат Γa и Γb)
    fixed = np.zeros(N, dtype=bool)
    fixed[boundary_one]  = True
    fixed[boundary_zero] = True
    free_mask = ~fixed

    # Итеративное решение (метод Якоби):
    # Φ_i ← среднее значение среди соседей (аппроксимация Лапласиана на графе)
    src, dst = adjacency[0], adjacency[1]
    for _ in range(1000):          # число итераций → сходимость
        phi_new = phi.copy()
        # Для каждого свободного узла i: Φ_i = среднее(Φ_j для j ∈ N(i))
        np.add.at(phi_new, dst, phi[src])
        count = np.bincount(dst, minlength=N).astype(np.float64)
        count = np.where(count == 0, 1.0, count)
        phi_new /= count
        phi_new[boundary_one]  = 1.0   # восстанавливаем граничные условия
        phi_new[boundary_zero] = 0.0
        phi_new[~free_mask] = phi_new[~free_mask]
        # Проверка сходимости
        if np.max(np.abs(phi_new[free_mask] - phi[free_mask])) < 1e-6:
            phi = phi_new
            break
        phi = phi_new

    return phi.astype(np.float32)


def compute_all_laplace_fields(node_coords: np.ndarray,
                               adjacency: np.ndarray,
                               boundaries: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Вычисляет все 5 полей Лапласа–Дирихле из Таблицы 2.1 НИР:
        φ_AB    — апикобазальное положение (base=1, apex=0)
        φ_EP    — трансмуральная позиция  (epi=1, endo=0)
        φ_LV-RV — разделение ЛЖ/ПЖ      (endo_lv=1, endo_rv=0)
        φ_LV-EP — перегородка (сторона ЛЖ)
        φ_RV-EP — перегородка (сторона ПЖ)

    Параметры
    ----------
    boundaries : словарь с ключами 'base', 'apex', 'epi', 'endo_lv', 'endo_rv'
                 и значениями — массивами индексов граничных узлов

    Возвращает
    ----------
    laplace_features : (N, 5) — 5-мерное позиционное кодирование каждого узла
    """
    configs = [
        # (имя,   Γa=1,         Γb=0)
        ("AB",    "base",       "apex"),
        ("EP",    "epi",        "endo_lv+endo_rv"),
        ("LV_RV", "endo_lv",    "endo_rv"),
        ("LV_EP", "endo_lv",    "epi"),
        ("RV_EP", "endo_rv",    "epi"),
    ]

    fields = []
    for name, gamma_a_key, gamma_b_key in configs:
        # Объединяем индексы, если граница составная (напр. endo_lv + endo_rv)
        if "+" in gamma_a_key:
            a = np.concatenate([boundaries[k] for k in gamma_a_key.split("+")])
        else:
            a = boundaries[gamma_a_key]

        if "+" in gamma_b_key:
            b = np.concatenate([boundaries[k] for k in gamma_b_key.split("+")])
        else:
            b = boundaries[gamma_b_key]

        phi = solve_laplace_dirichlet(node_coords, adjacency, a, b)
        fields.append(phi)

    return np.stack(fields, axis=-1)  # (N, 5)


# ===========================================================================
# ЧАСТЬ 3.  СТРАТИФИКАЦИОННЫЙ ВЫБОР ОКРЕСТНОСТЕЙ
# (Раздел 2.3, формула 2.4 НИР)
# ===========================================================================

def build_stratified_neighborhoods(node_coords: np.ndarray,
                                   n_nb: int = N_NB
                                   ) -> np.ndarray:
    """
    Формирует стратификационные окрестности для каждого узла (формула 2.4).

    Физический принцип: механическое взаимодействие убывает с расстоянием,
    поэтому ближние узлы важнее, но дальнодействующие связи (SC4, SC5)
    также необходимы — это подтверждается ненулевыми весами внимания.

    Стратегия GraphSAGE [Hamilton et al., 2017]:
    для каждого узла i формируются 5 страт по перцентилям расстояний:
        SC1: d ≤ Q0.2%   (20 узлов — ближайшие соседи)
        SC2: Q0.2%–Q1%   (30 узлов)
        SC3: Q1%–Q2%     (30 узлов)
        SC4: Q2%–Q5%     (10 узлов)
        SC5: Q5%–Q10%    (10 узлов — дальние)
    Итого: 100 кандидатов → случайно отбирается n_nb = 12.

    Сравнение стратегий (из Табл. 4.2 НИР):
        случайная окрестность   : val_loss 0.0165 (+81.3%)
        ближнесоседняя (kNN)    : val_loss 0.0119 (+30.8%)
        стратификационная (★)   : val_loss 0.0091 (базовый)

    Параметры
    ----------
    node_coords : (N, 3) — координаты узлов
    n_nb        : число соседей, отбираемых для MP (оптимально 12)

    Возвращает
    ----------
    neighbor_index : (N, n_nb) — индексы соседей для каждого узла
    """
    N = node_coords.shape[0]
    tree = cKDTree(node_coords)

    # Вычисляем попарные расстояния для определения глобальных перцентилей
    # (берём выборку для экономии памяти при больших N)
    sample_size = min(N, 500)
    sample_idx  = np.random.choice(N, sample_size, replace=False)
    all_dists   = []
    for i in sample_idx:
        dists, _ = tree.query(node_coords[i:i+1], k=min(200, N))
        all_dists.append(dists[0, 1:])   # исключаем сам узел (dist=0)
    all_dists = np.concatenate(all_dists)

    # Глобальные перцентильные пороги (одинаковые для всех узлов)
    pct_thresholds = np.percentile(all_dists, [p * 100 for p in STRAT_PERCENTILES])

    neighbor_index = np.zeros((N, n_nb), dtype=np.int64)

    for i in range(N):
        # Запрашиваем максимальное количество кандидатов (SC5 = до 10% от N)
        k_max = min(int(N * STRAT_PERCENTILES[-1]) + 1, N)
        dists, idxs = tree.query(node_coords[i:i+1], k=k_max)
        dists, idxs = dists[0, 1:], idxs[0, 1:]   # убираем сам узел

        # Делим кандидатов по стратам
        selected = []
        prev_thr = 0.0
        for strat_idx, (thr, n_strat) in enumerate(
                zip(pct_thresholds, STRAT_SIZES)):
            mask = (dists > prev_thr) & (dists <= thr)
            strat_candidates = idxs[mask]
            if len(strat_candidates) > 0:
                # Случайно выбираем из страты (не более n_strat узлов)
                chosen = np.random.choice(
                    strat_candidates,
                    size=min(n_strat, len(strat_candidates)),
                    replace=False
                )
                selected.append(chosen)
            prev_thr = thr

        if len(selected) == 0:
            # Запасной вариант: ближайшие n_nb узлов
            neighbor_index[i] = idxs[:n_nb]
            continue

        selected = np.concatenate(selected)

        # Финальная случайная выборка n_nb узлов из объединения страт
        if len(selected) >= n_nb:
            chosen = np.random.choice(selected, size=n_nb, replace=False)
        else:
            # Дополняем повторами, если кандидатов недостаточно
            chosen = np.random.choice(selected, size=n_nb, replace=True)

        neighbor_index[i] = chosen

    return neighbor_index


# ===========================================================================
# ЧАСТЬ 4.  АРХИТЕКТУРА HEARTSIM SAGE — 5 МОДУЛЕЙ
# (Раздел 3.2 НИР)
# ===========================================================================

class NodeEncoder(nn.Module):
    """
    Модуль 1: Кодирование узлов (Node Encoding).

    Признаки волокон ai = (f0i, s0i) и значения Лапласа Φi кодируются
    РАЗДЕЛЬНЫМИ MLP в векторы размерности D_EMB, которые суммируются:

        ni = ENCNf(ai) + ENCNl(Φi)

    Раздельное кодирование позволяет независимо представлять:
      - механическую анизотропию (ориентации волокон f0, s0)
      - геометрическое положение (5 значений полей Лапласа)
    Это физически обоснованное разделение улучшает интерпретируемость.
    """

    def __init__(self, fiber_dim: int = 6, laplace_dim: int = 5,
                 d_emb: int = D_EMB):
        super().__init__()
        # MLP для ориентаций волокон (f0 и s0 — два 3D-вектора = 6 чисел)
        self.enc_fiber = nn.Sequential(
            nn.Linear(fiber_dim, d_emb),
            nn.ReLU(),
            nn.Linear(d_emb, d_emb)
        )
        # MLP для 5 полей Лапласа–Дирихле
        self.enc_laplace = nn.Sequential(
            nn.Linear(laplace_dim, d_emb),
            nn.ReLU(),
            nn.Linear(d_emb, d_emb)
        )

    def forward(self, fiber_feats: torch.Tensor,
                laplace_feats: torch.Tensor) -> torch.Tensor:
        """
        Параметры
        ----------
        fiber_feats  : (N, 6)  — ориентации волокон (f0x,f0y,f0z,s0x,s0y,s0z)
        laplace_feats: (N, 5)  — значения 5 полей Лапласа

        Возвращает
        ----------
        node_emb : (N, D_EMB) — суммарное представление узлов
        """
        emb_fiber   = self.enc_fiber(fiber_feats)     # (N, D_EMB)
        emb_laplace = self.enc_laplace(laplace_feats) # (N, D_EMB)
        return emb_fiber + emb_laplace                 # поэлементная сумма


class EdgeEncoder(nn.Module):
    """
    Вспомогательный MLP для кодирования рёберных признаков (формула 3.2).

    Рёберный вектор: ê_{i,k} = (n_i, n_k, Φ_k − Φ_i, X̂_k − X̂_i)
    После кодирования: e_{i,k} = ENC_E(ê_{i,k})
    """

    def __init__(self, d_emb: int = D_EMB):
        super().__init__()
        # Вход: 2*D_EMB (узловые представления i и k) + 5 (ΔΦ) + 3 (ΔX)
        in_dim = 2 * d_emb + 5 + 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_emb),
            nn.ReLU(),
            nn.Linear(d_emb, d_emb)
        )

    def forward(self, ni: torch.Tensor, nk: torch.Tensor,
                delta_phi: torch.Tensor,
                delta_x: torch.Tensor) -> torch.Tensor:
        edge_input = torch.cat([ni, nk, delta_phi, delta_x], dim=-1)
        return self.mlp(edge_input)


class AttentionMessagePassing(nn.Module):
    """
    Модуль 3: Передача сообщений с вниманием (Message Passing).

    Реализует анизотропный механизм Multi-Head Attention (формула 3.3):

        α_{i,k} = softmax(e^T_{i,k} W_Q W_K e_{i,k} / √d_emb)
        e'_{i,k} = α_{i,k} · W_V e_{i,k}

    Агрегирование с остаточным соединением (ResNet [He et al., 2016]):
        n'_i = (1/n_NB) Σ_k e'_{i,k}  +  n_i

    Анизотропный MP (с вниманием) превосходит изотропный (без внимания):
    замена на MLP-агрегатор ухудшает val_loss на 56% (Табл. 4.2 НИР).

    Физический смысл весов: профиль α монотонно убывает по стратам
    расстояния (0.159 → 0.082 → 0.064 → 0.027 → 0.022), что согласуется
    с теорией упругости (убывание взаимодействия с расстоянием).
    """

    def __init__(self, d_emb: int = D_EMB, n_heads: int = N_HEADS):
        super().__init__()
        assert d_emb % n_heads == 0, "D_EMB должна делиться на N_HEADS"
        self.n_heads  = n_heads
        self.d_head   = d_emb // n_heads
        self.d_emb    = d_emb

        self.edge_encoder = EdgeEncoder(d_emb)

        # Проекционные матрицы для Q, K, V (по одной на каждую голову)
        self.W_Q = nn.Linear(d_emb, d_emb, bias=False)
        self.W_K = nn.Linear(d_emb, d_emb, bias=False)
        self.W_V = nn.Linear(d_emb, d_emb, bias=False)
        self.out_proj = nn.Linear(d_emb, d_emb)

    def forward(self, node_emb: torch.Tensor,
                neighbor_idx: torch.Tensor,
                laplace_feats: torch.Tensor,
                node_coords: torch.Tensor) -> torch.Tensor:
        """
        Параметры
        ----------
        node_emb     : (N, D_EMB)      — текущие представления узлов
        neighbor_idx : (N, N_NB)       — индексы соседей (стратиф. выборка)
        laplace_feats: (N, 5)          — поля Лапласа (для ΔΦ)
        node_coords  : (N, 3)          — координаты (для ΔX)

        Возвращает
        ----------
        node_emb_new : (N, D_EMB)      — обновлённые представления + резидуал
        """
        N, n_nb = neighbor_idx.shape

        # --- Формируем рёберные векторы для всех пар (i, k) ---
        ni = node_emb.unsqueeze(1).expand(-1, n_nb, -1)            # (N, K, D)
        nk = node_emb[neighbor_idx]                                 # (N, K, D)

        delta_phi = (laplace_feats[neighbor_idx]
                     - laplace_feats.unsqueeze(1))                  # (N, K, 5)
        delta_x   = (node_coords[neighbor_idx]
                     - node_coords.unsqueeze(1))                    # (N, K, 3)

        # Кодируем рёбра: ê_{i,k} → e_{i,k}  (формула 3.2)
        ni_flat = ni.reshape(N * n_nb, -1)
        nk_flat = nk.reshape(N * n_nb, -1)
        dp_flat = delta_phi.reshape(N * n_nb, -1)
        dx_flat = delta_x.reshape(N * n_nb, -1)
        edge_emb = self.edge_encoder(ni_flat, nk_flat,
                                     dp_flat, dx_flat)             # (N*K, D)
        edge_emb = edge_emb.reshape(N, n_nb, self.d_emb)          # (N, K, D)

        # --- Multi-Head Attention (формула 3.3) ---
        # Разбиваем на головы: (N, K, H, d_head)
        def split_heads(x):
            return x.reshape(N, n_nb, self.n_heads, self.d_head
                             ).permute(0, 2, 1, 3)   # (N, H, K, d_head)

        Q = split_heads(self.W_Q(edge_emb))
        K = split_heads(self.W_K(edge_emb))
        V = split_heads(self.W_V(edge_emb))

        # Скалярное произведение с масштабированием
        scale  = math.sqrt(self.d_head)
        scores = (Q * K).sum(-1) / scale                # (N, H, K)
        alpha  = F.softmax(scores, dim=-1)               # веса внимания α_{i,k}

        # Взвешенная сумма
        out = (alpha.unsqueeze(-1) * V).sum(dim=2)       # (N, H, d_head)
        out = out.reshape(N, self.d_emb)                 # (N, D_EMB)
        out = self.out_proj(out)

        # Остаточное соединение (ResNet): предотвращает размывание признаков
        return out + node_emb


class GlobalEncoder(nn.Module):
    """
    Модуль 4: Глобальное кодирование (Global Encoding).

    Параметры конститутивной модели β, давление p и форма ξ кодируются
    единым MLP-кодировщиком ENC_G в глобальный вектор g' ∈ ℝ^D_EMB.

    Параметры a, af, as обрабатываются в ЛОГАРИФМИЧЕСКОМ масштабе,
    так как их диапазон охватывает несколько порядков величин:
        a  ∈ [10², 10⁴] дин/см²
        af ∈ [10³, 10⁶]
        as ∈ [10², 10⁵]
    Остальные параметры (b, bf, bs) ∈ [1, 50] — в линейном масштабе.
    """

    def __init__(self, beta_dim: int = 8, pressure_dim: int = 2,
                 shape_dim: int = 40, d_emb: int = D_EMB):
        super().__init__()
        # beta (8) + pressure (2) + shape features (40) = 50
        in_dim = beta_dim + pressure_dim + shape_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_emb),
            nn.ReLU(),
            nn.Linear(d_emb, d_emb),
            nn.ReLU(),
            nn.Linear(d_emb, d_emb)
        )
        # Индексы параметров β, требующих логарифмирования:
        # β = {a, b, af, bf, as, bs, afs, bfs}
        # Логарифм нужен для a (idx=0), af (idx=2), as (idx=4)
        self.log_indices = [0, 2, 4]

    def forward(self, beta: torch.Tensor,
                pressure: torch.Tensor,
                shape_feats: torch.Tensor) -> torch.Tensor:
        """
        Параметры
        ----------
        beta        : (B, 8)  — параметры модели Холзапфеля–Огдена {a,b,af,...}
        pressure    : (B, 2)  — давление [p_lv, p_rv] в мм рт. ст.
        shape_feats : (B, 40) — 40 признаков формы желудочков

        Возвращает
        ----------
        g : (B, D_EMB) — глобальный вектор контекста
        """
        # Применяем log к параметрам с широким диапазоном значений
        beta = beta.clone()
        beta[:, self.log_indices] = torch.log(
            beta[:, self.log_indices].clamp(min=1e-8))

        x = torch.cat([beta, pressure, shape_feats], dim=-1)
        return self.mlp(x)


class DisplacementDecoder(nn.Module):
    """
    Модуль 5: Декодирование смещений (Decoding).

    Три НЕЗАВИСИМЫХ MLP-декодера предсказывают компоненты смещения
    по каждой координатной оси:

        DECDx: (n'_i, g') → û_x,i
        DECDy: (n'_i, g') → û_y,i
        DECDz: (n'_i, g') → û_z,i

    Раздельные декодеры важны при анизотропии деформаций миокарда:
    смещения вдоль длинной оси ЛЖ (z) могут существенно превышать
    поперечные смещения (x, y). Независимая оптимизация каждой
    компоненты позволяет модели адаптироваться к этой асимметрии.
    """

    def __init__(self, d_emb: int = D_EMB):
        super().__init__()
        in_dim = 2 * d_emb   # конкатенация (n'_i, g')

        def make_decoder():
            return nn.Sequential(
                nn.Linear(in_dim, d_emb),
                nn.ReLU(),
                nn.Linear(d_emb, d_emb // 2),
                nn.ReLU(),
                nn.Linear(d_emb // 2, 1)    # скалярный выход
            )

        self.dec_x = make_decoder()  # компонента смещения X
        self.dec_y = make_decoder()  # компонента смещения Y
        self.dec_z = make_decoder()  # компонента смещения Z

    def forward(self, node_emb: torch.Tensor,
                global_emb: torch.Tensor) -> torch.Tensor:
        """
        Параметры
        ----------
        node_emb   : (N, D_EMB)  — обновлённые представления узлов n'_i
        global_emb : (1, D_EMB)  — глобальный вектор g' (бродкастируется)

        Возвращает
        ----------
        disp : (N, 3) — предсказанные смещения [û_x, û_y, û_z]
        """
        # Конкатенируем локальный и глобальный контекст для каждого узла
        combined = torch.cat(
            [node_emb, global_emb.expand(node_emb.shape[0], -1)], dim=-1)

        ux = self.dec_x(combined)   # (N, 1)
        uy = self.dec_y(combined)   # (N, 1)
        uz = self.dec_z(combined)   # (N, 1)

        return torch.cat([ux, uy, uz], dim=-1)  # (N, 3)


# ===========================================================================
# ЧАСТЬ 5.  ПОЛНАЯ МОДЕЛЬ HEARTSIM SAGE
# (Раздел 3.1 НИР, формула 3.1)
# ===========================================================================

class HeartSimSage(nn.Module):
    """
    HeartSimSage (HSS) — GNN-эмулятор биомеханики сердца.

    Предсказывает поля пассивных смещений û_i в каждом узле i
    бивентрикулярной КЭ-сетки (формула 3.1):

        û_i = HSS_θ(ĥ_i, {ĥ_k}_{k ∈ N^NB_i}; ĝ)

    Ключевые свойства:
      - Работает с произвольными 3D-геометриями (переменная топология)
      - Полная индуктивность: без обучаемых узловых вложений →
        перенос на новых пациентов без переобучения
      - Ускорение ≈13 000× относительно КЭА (GPU vs CPU-КЭА)
      - Средняя ошибка смещений: 0.13 ± 0.12%

    Параметры архитектуры (оптимальные, Табл. 4.2 НИР):
        D_EMB = 128, N_HEADS = 8, N_NB = 12, N_S = 300
    """

    def __init__(self,
                 fiber_dim: int = 6,
                 laplace_dim: int = 5,
                 beta_dim: int = 8,
                 pressure_dim: int = 2,
                 shape_dim: int = 40,
                 d_emb: int = D_EMB,
                 n_heads: int = N_HEADS):
        super().__init__()

        # Модуль 1: Кодирование узлов
        self.node_encoder = NodeEncoder(fiber_dim, laplace_dim, d_emb)

        # Модуль 2: Стратификационные связи — реализованы вне модели
        #           (функция build_stratified_neighborhoods)

        # Модуль 3: Передача сообщений с многоголовым вниманием
        self.message_passing = AttentionMessagePassing(d_emb, n_heads)

        # Модуль 4: Глобальное кодирование
        self.global_encoder = GlobalEncoder(beta_dim, pressure_dim,
                                            shape_dim, d_emb)

        # Модуль 5: Декодирование смещений (три независимых MLP)
        self.decoder = DisplacementDecoder(d_emb)

    def forward(self,
                fiber_feats: torch.Tensor,
                laplace_feats: torch.Tensor,
                node_coords: torch.Tensor,
                neighbor_idx: torch.Tensor,
                beta: torch.Tensor,
                pressure: torch.Tensor,
                shape_feats: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход HeartSimSage.

        Параметры
        ----------
        fiber_feats  : (N, 6)  — ориентации волокон f0, s0
        laplace_feats: (N, 5)  — поля Лапласа–Дирихле
        node_coords  : (N, 3)  — координаты узлов (нормированные)
        neighbor_idx : (N, K)  — стратификационные соседи
        beta         : (1, 8)  — параметры модели HO {a, b, af, bf, as, bs, afs, bfs}
        pressure     : (1, 2)  — [p_lv, p_rv] в мм рт. ст.
        shape_feats  : (1, 40) — признаки формы желудочков

        Возвращает
        ----------
        displacements : (N, 3) — нормированные смещения [û_x, û_y, û_z]
        """
        # ── Модуль 1: Кодирование узлов ──────────────────────────────────
        node_emb = self.node_encoder(fiber_feats, laplace_feats)  # (N, D_EMB)

        # ── Модуль 3: Передача сообщений с вниманием ─────────────────────
        node_emb = self.message_passing(
            node_emb, neighbor_idx, laplace_feats, node_coords)   # (N, D_EMB)

        # ── Модуль 4: Глобальное кодирование ─────────────────────────────
        global_emb = self.global_encoder(beta, pressure, shape_feats)  # (1, D_EMB)

        # ── Модуль 5: Декодирование смещений ─────────────────────────────
        displacements = self.decoder(node_emb, global_emb)  # (N, 3)

        return displacements


# ===========================================================================
# ЧАСТЬ 6.  ОБУЧЕНИЕ С ВЫБОРКОЙ ПОДМНОЖЕСТВ УЗЛОВ
# (Раздел 3.3 НИР)
# ===========================================================================

class HeartSimSageTrainer:
    """
    Стратегия обучения HeartSimSage на подмножествах узлов.

    На каждой эпохе для каждого образца случайно отбирается N_S = 300 узлов
    из ~17 000 (≈1.7%). Это сокращает число MP-операций в ~57 раз
    при НЕИЗМЕННОМ качестве:

        val_loss(N_S=300) = 0.0091 = val_loss(N_S=1000)
        val_loss(N_S=10)  = 0.0099  (+8% — ухудшение)

    Случайная смена подмножества на каждой эпохе обеспечивает:
      1. Неявную регуляризацию (аналог Dropout)
      2. Полное покрытие объёма миокарда за несколько эпох

    Оптимальные гиперпараметры (Табл. 4.2 НИР):
        η = 1e-4,  n_epoch = 3000,  n_B = 20
    """

    def __init__(self,
                 model: HeartSimSage,
                 lr: float = 1e-4,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model  = model.to(device)
        self.device = device
        self.optim  = torch.optim.Adam(model.parameters(), lr=lr)

    def _subset_loss(self,
                     pred: torch.Tensor,
                     target: torch.Tensor,
                     subset_idx: torch.Tensor) -> torch.Tensor:
        """
        L2-потеря на подмножестве N_S узлов (нормированные смещения).

        Функция потерь из раздела 3.1 НИР:
            L = (1 / (N_S · 3)) Σ_{i ∈ S} ‖û_i − u_i‖²
        """
        pred_sub   = pred[subset_idx]     # (N_S, 3)
        target_sub = target[subset_idx]   # (N_S, 3)
        return F.mse_loss(pred_sub, target_sub)

    def train_epoch(self, dataloader) -> float:
        """Одна эпоха обучения; возвращает среднее значение потери."""
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            N = batch["fiber_feats"].shape[0]

            # ── Случайный выбор N_S = 300 узлов ─────────────────────────
            # Именно СЛУЧАЙНАЯ смена подмножества на каждой итерации
            # обеспечивает неявную регуляризацию и полное покрытие.
            subset_idx = torch.randperm(N, device=self.device)[:N_S]

            # ── Прямой проход ─────────────────────────────────────────────
            pred = self.model(
                fiber_feats   = batch["fiber_feats"],
                laplace_feats = batch["laplace_feats"],
                node_coords   = batch["node_coords"],
                neighbor_idx  = batch["neighbor_idx"],
                beta          = batch["beta"],
                pressure      = batch["pressure"],
                shape_feats   = batch["shape_feats"]
            )

            # ── Потеря только на подмножестве узлов ──────────────────────
            loss = self._subset_loss(pred, batch["displacements"], subset_idx)

            # ── Обратное распространение ──────────────────────────────────
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        """
        Оценка модели на всех узлах (без подмножества).
        Вычисляет нормированные ошибки по осям x, y, z и объёмам ЛЖ/ПЖ
        (аналог Табл. 4.1 НИР).
        """
        self.model.eval()
        err_x, err_y, err_z = [], [], []

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            pred   = self.model(**{k: batch[k] for k in [
                "fiber_feats", "laplace_feats", "node_coords",
                "neighbor_idx", "beta", "pressure", "shape_feats"]})
            target = batch["displacements"]

            # Нормированная ошибка: |û_i - u_i| / max(|u_i|)
            scale = target.abs().max(dim=0).values.clamp(min=1e-8)
            rel_err = (pred - target).abs() / scale

            err_x.append(rel_err[:, 0].mean().item())
            err_y.append(rel_err[:, 1].mean().item())
            err_z.append(rel_err[:, 2].mean().item())

        return {
            "Err_ux": np.mean(err_x),   # цель: 0.15% (Табл. 4.1)
            "Err_uy": np.mean(err_y),   # цель: 0.11%
            "Err_uz": np.mean(err_z),   # цель: 0.10%
            "Err_mean": np.mean(err_x + err_y + err_z),  # цель: 0.13%
        }

    def train(self, train_loader, val_loader, n_epochs: int = 3000):
        """
        Полный цикл обучения.
        Оптимальное число эпох: 3 000 (из раздела 4.1 НИР).
        """
        print(f"Обучение на {self.device} | "
              f"N_S={N_S} узлов/эпоху | lr=1e-4")
        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            if epoch % 100 == 0:
                metrics = self.evaluate(val_loader)
                print(f"Эпоха {epoch:4d} | loss={train_loss:.6f} | "
                      f"Err_mean={metrics['Err_mean']*100:.3f}%")


# ===========================================================================
# ЧАСТЬ 7.  КОНВЕЙЕР ЦИФРОВОГО ДВОЙНИКА (Digital Twin Pipeline)
# (Раздел 5.1, Таблица 5.1 НИР)
# ===========================================================================

class CardiacDigitalTwinPipeline:
    """
    Конвейер персонализированного цифрового двойника сердца.

    Этапы (из Табл. 5.1 НИР):
      1. Сегментация геометрии из КТ/МРТ → тетраэдрическая сетка
      2. Предобработка графа: 7 задач Лапласа + волокна + 40 признаков формы
      3. Начальное предсказание HSS (≈0.01 с)
      4. Оптимизация параметров β (100 итераций ≈ 0.84 с на GPU)
      5. Финальная валидация КЭА + интеграция в DT (≈112 с)

    Ключевая метрика ускорения:
        КЭА (CPU):   112 ± 35 с / решение
        HSS (GPU):     0.0084 с / решение
        Коэффициент: ≈ 13 300×
    """

    def __init__(self, hss_model: HeartSimSage,
                 device: str = "cpu"):
        self.model  = hss_model.eval().to(device)
        self.device = device

    def preprocess_geometry(self,
                             node_coords: np.ndarray,
                             elements: np.ndarray,
                             boundaries: Dict[str, np.ndarray]
                             ) -> Dict[str, np.ndarray]:
        """
        Этап 2: Предобработка графа (выполняется однократно для пациента).

        Включает:
          - Построение графа из КЭ-сетки
          - Решение 7 задач Лапласа–Дирихле (5 позиционных + 2 для волокон)
          - Стратификационные окрестности
          - Вычисление 40 признаков формы (здесь заглушка)
        """
        print("  [2] Предобработка графа...")

        # Граф из тетраэдральной сетки
        edge_index, edge_attr = mesh_to_graph(node_coords, elements)

        # 5 полей Лапласа–Дирихле (позиционное кодирование)
        laplace_feats = compute_all_laplace_fields(
            node_coords, edge_index, boundaries)
        print(f"      Поля Лапласа вычислены: {laplace_feats.shape}")

        # Стратификационные окрестности (один раз на геометрию)
        neighbor_idx = build_stratified_neighborhoods(node_coords)
        print(f"      Стратиф. окрестности: {neighbor_idx.shape}")

        # Признаки формы: 40 чисел (моды формы, объёмы, кривизна и т.д.)
        # Здесь заглушка — в реальном конвейере вычисляются через SimVascular
        shape_feats = np.zeros(40, dtype=np.float32)

        return {
            "edge_index":    edge_index,
            "laplace_feats": laplace_feats,
            "neighbor_idx":  neighbor_idx,
            "shape_feats":   shape_feats,
            "node_coords":   node_coords,
        }

    @torch.no_grad()
    def predict(self,
                preprocessed: Dict[str, np.ndarray],
                fiber_feats: np.ndarray,
                beta: np.ndarray,
                pressure: np.ndarray) -> np.ndarray:
        """
        Этап 3: Быстрое предсказание поля смещений (≈ 0.0084 с на GPU).

        Параметры
        ----------
        fiber_feats : (N, 6)  — ориентации волокон (f0, s0)
        beta        : (8,)    — параметры HO {a, b, af, bf, as, bs, afs, bfs}
        pressure    : (2,)    — [p_lv, p_rv] в мм рт. ст.

        Возвращает
        ----------
        displacements : (N, 3) — предсказанные смещения в мм
        """
        def to_tensor(x, unsqueeze=False):
            t = torch.tensor(x, dtype=torch.float32, device=self.device)
            return t.unsqueeze(0) if unsqueeze else t

        displacements = self.model(
            fiber_feats   = to_tensor(fiber_feats),
            laplace_feats = to_tensor(preprocessed["laplace_feats"]),
            node_coords   = to_tensor(preprocessed["node_coords"]),
            neighbor_idx  = torch.tensor(preprocessed["neighbor_idx"],
                                         device=self.device),
            beta          = to_tensor(beta, unsqueeze=True),
            pressure      = to_tensor(pressure, unsqueeze=True),
            shape_feats   = to_tensor(preprocessed["shape_feats"],
                                       unsqueeze=True),
        )
        return displacements.cpu().numpy()

    def personalize_beta(self,
                         preprocessed: Dict[str, np.ndarray],
                         fiber_feats: np.ndarray,
                         pressure: np.ndarray,
                         mri_displacements: np.ndarray,
                         beta_init: Optional[np.ndarray] = None,
                         n_iter: int = 100) -> np.ndarray:
        """
        Этап 4: Оптимизация параметров β модели Холзапфеля–Огдена.

        Дифференцируемость HSS позволяет использовать градиентный спуск
        для минимизации невязки предсказаний vs МРТ-смещений:

            β* = argmin_β ‖HSS(β) − u_МРТ‖²

        Ускорение:
            100 итераций с HSS:  ≈ 0.84 с (GPU)
            100 итераций с КЭА:  ≈ 3.1 ч  (CPU)
            Коэффициент: ≈ 13 000×

        Параметры
        ----------
        mri_displacements : (N, 3) — истинные смещения из МРТ
        beta_init         : (8,)   — начальное значение β (если None → единицы)
        n_iter            : число итераций оптимизации
        """
        print(f"  [4] Оптимизация β ({n_iter} итераций)...")

        if beta_init is None:
            # Стартовые значения: середины диапазонов из раздела 4.1 НИР
            beta_init = np.array([500., 10., 5000., 10., 500., 10., 0., 1.],
                                  dtype=np.float32)

        # β — обучаемые параметры (с градиентом)
        beta_param = nn.Parameter(
            torch.tensor(beta_init, dtype=torch.float32, device=self.device))
        optim = torch.optim.Adam([beta_param], lr=1e-2)

        target = torch.tensor(mri_displacements, dtype=torch.float32,
                               device=self.device)

        def to_t(x, unsq=False):
            t = torch.tensor(x, dtype=torch.float32, device=self.device)
            return t.unsqueeze(0) if unsq else t

        for it in range(n_iter):
            optim.zero_grad()
            pred = self.model(
                fiber_feats   = to_t(fiber_feats),
                laplace_feats = to_t(preprocessed["laplace_feats"]),
                node_coords   = to_t(preprocessed["node_coords"]),
                neighbor_idx  = torch.tensor(preprocessed["neighbor_idx"],
                                             device=self.device),
                beta          = beta_param.unsqueeze(0),
                pressure      = to_t(pressure, unsq=True),
                shape_feats   = to_t(preprocessed["shape_feats"], unsq=True),
            )
            loss = F.mse_loss(pred, target)
            loss.backward()
            optim.step()

            if (it + 1) % 20 == 0:
                print(f"      iter {it+1:3d}: loss = {loss.item():.6f}")

        return beta_param.detach().cpu().numpy()


# ===========================================================================
# ТОЧКА ВХОДА — ДЕМОНСТРАЦИЯ
# ===========================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("  HeartSimSage — демонстрация работы компонентов")
    print("=" * 60)

    # ── Синтетическая сетка для тестирования ─────────────────────────────
    N_NODES = 500    # в реальности: 15 864 – 19 873 узлов
    N_ELEMS = 2000   # в реальности: 86 596 – 105 395 тетраэдров

    coords   = np.random.randn(N_NODES, 3).astype(np.float32)
    elements = np.random.randint(0, N_NODES, (N_ELEMS, 4))

    # Границы для задач Лапласа (случайные индексы для демонстрации)
    boundaries = {
        "base":    np.random.choice(N_NODES, 30, replace=False),
        "apex":    np.random.choice(N_NODES, 20, replace=False),
        "epi":     np.random.choice(N_NODES, 80, replace=False),
        "endo_lv": np.random.choice(N_NODES, 40, replace=False),
        "endo_rv": np.random.choice(N_NODES, 30, replace=False),
    }

    # ── Шаг 1: Граф из КЭ-сетки ──────────────────────────────────────────
    print("\n[1] Построение графа из КЭ-сетки...")
    edge_index, edge_attr = mesh_to_graph(coords, elements)
    print(f"    Узлов: {N_NODES} | Рёбер: {edge_index.shape[1]}")

    # ── Шаг 2: Поля Лапласа–Дирихле ──────────────────────────────────────
    print("\n[2] Вычисление полей Лапласа–Дирихле...")
    laplace_feats = compute_all_laplace_fields(coords, edge_index, boundaries)
    print(f"    Позиционное кодирование: {laplace_feats.shape} "
          f"(min={laplace_feats.min():.3f}, max={laplace_feats.max():.3f})")

    # ── Шаг 3: Стратификационные окрестности ─────────────────────────────
    print("\n[3] Стратификационный выбор окрестностей...")
    neighbor_idx = build_stratified_neighborhoods(coords, n_nb=N_NB)
    print(f"    Матрица соседей: {neighbor_idx.shape}")

    # ── Шаг 4: Создание и прямой проход модели ───────────────────────────
    print("\n[4] Создание HeartSimSage и прямой проход...")
    model = HeartSimSage()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Параметров модели: {total_params:,}")

    fiber_feats  = torch.randn(N_NODES, 6)
    laplace_t    = torch.tensor(laplace_feats)
    coords_t     = torch.tensor(coords)
    neighbor_t   = torch.tensor(neighbor_idx)
    beta_t       = torch.tensor([[500., 10., 5000., 10., 500., 10., 0., 1.]])
    pressure_t   = torch.tensor([[10., 6.]])   # p_lv=10, p_rv=6 мм рт.ст.
    shape_t      = torch.zeros(1, 40)

    with torch.no_grad():
        displacements = model(fiber_feats, laplace_t, coords_t,
                              neighbor_t, beta_t, pressure_t, shape_t)

    print(f"    Предсказанные смещения: {displacements.shape} "
          f"(mean={displacements.abs().mean():.4f})")

    # ── Шаг 5: Конвейер цифрового двойника ───────────────────────────────
    print("\n[5] Конвейер цифрового двойника сердца...")
    pipeline = CardiacDigitalTwinPipeline(model)

    preprocessed = pipeline.preprocess_geometry(coords, elements, boundaries)
    fiber_np     = np.random.randn(N_NODES, 6).astype(np.float32)
    beta_np      = np.array([500., 10., 5000., 10., 500., 10., 0., 1.],
                             dtype=np.float32)
    pressure_np  = np.array([10., 6.], dtype=np.float32)

    pred_disp = pipeline.predict(preprocessed, fiber_np, beta_np, pressure_np)
    print(f"    Предсказание DT: {pred_disp.shape}")

    print("\n" + "=" * 60)
    print("  Демонстрация завершена успешно.")
    print("=" * 60)
