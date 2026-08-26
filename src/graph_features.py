"""
Шаг 3. Признаки графа для эмулятора (раздел 2.2-2.3 НИР) на РЕАЛЬНОЙ сетке.

Содержит:
  1. Поля Лапласа-Дирихле Phi (5 полей) -- решаются МКЭ на той же
     тетраэдральной сетке, что и задача механики;
  2. Стратифицированные окрестности (формула 2.4);
  3. Признаки формы желудочков (глобальный дескриптор xi).

Поля Лапласа-Дирихле
--------------------
Каждое поле phi_j -- решение задачи

    Delta phi_j = 0 в Omega_0,   phi_j = 1 на Г_A,  phi_j = 0 на Г_B,

где (Г_A, Г_B) -- пары анатомических границ. Такие поля дают гладкую
координатизацию стенки, инвариантную к сетке и к жёстким движениям, и
служат "геодезическими" признаками положения узла:

    phi_1: эндокард ЛЖ -> эпикард        (трансмуральная координата ЛЖ)
    phi_2: эндокард ПЖ -> эпикард        (трансмуральная координата ПЖ)
    phi_3: база -> верхушка              (продольная координата)
    phi_4: эндокард ЛЖ -> эндокард ПЖ    (межжелудочковая координата)
    phi_5: (эндокард ЛЖ + ПЖ) -> база    (смешанная)

Дискретизация -- те же линейные тетраэдры P1; матрица жёсткости
K_ab = int grad N_a . grad N_b dV собирается из градиентов формы,
уже вычисленных в fem_core.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fem_core import BASE, ENDO_LV, ENDO_RV, EPI, tet_shape_gradients

N_LAPLACE = 5

# пары (метки границы со значением 1, метки границы со значением 0)
LAPLACE_PAIRS = (
    ((ENDO_LV,), (EPI,)),
    ((ENDO_RV,), (EPI,)),
    ((BASE,), None),                 # None -> верхушка (наиболее удалённый узел)
    ((ENDO_LV,), (ENDO_RV,)),
    ((ENDO_LV, ENDO_RV), (BASE,)),
)


def stiffness_matrix(coords: np.ndarray, elements: np.ndarray):
    """Матрица Лапласа (жёсткости) для скалярного поля на P1-тетраэдрах."""
    gradN, vol = tet_shape_gradients(coords, elements)
    ke = np.einsum("eai,ebi,e->eab", gradN, gradN, vol)     # (E,4,4)
    rows = np.repeat(elements, 4, axis=1).ravel()
    cols = np.tile(elements, (1, 4)).ravel()
    n = len(coords)
    return sp.csr_matrix((ke.ravel(), (rows, cols)), shape=(n, n)), vol


def solve_dirichlet(K: sp.csr_matrix, fixed: np.ndarray,
                    values: np.ndarray) -> np.ndarray:
    """
    Решает Delta phi = 0 с условиями Дирихле phi[fixed] = values.

    Реализуется исключением закреплённых степеней свободы:
        K_ff phi_f = -K_fc phi_c.
    """
    n = K.shape[0]
    phi = np.zeros(n)
    phi[fixed] = values
    free = np.ones(n, dtype=bool)
    free[fixed] = False
    Kff = K[free][:, free].tocsc()
    rhs = -K[free][:, fixed] @ values
    phi[free] = spla.spsolve(Kff, rhs)
    return phi


def compute_laplace_fields(coords: np.ndarray, elements: np.ndarray,
                           faces: np.ndarray, face_type: np.ndarray
                           ) -> np.ndarray:
    """
    Возвращает (N, 5) значения полей Лапласа-Дирихле.
    """
    K, _ = stiffness_matrix(coords, elements)
    n = len(coords)

    def nodes_of(codes) -> np.ndarray:
        m = np.isin(face_type, codes)
        return np.unique(faces[m].ravel())

    base_nodes = nodes_of((BASE,))
    # "верхушка": узел, максимально удалённый от плоскости основания
    base_centre = coords[base_nodes].mean(axis=0)
    d_from_base = np.linalg.norm(coords - base_centre, axis=1)
    apex = np.argsort(d_from_base)[-max(1, n // 500):]      # небольшая шапочка

    fields = np.zeros((n, N_LAPLACE))
    for j, (hi_codes, lo_codes) in enumerate(LAPLACE_PAIRS):
        hi = nodes_of(hi_codes)
        lo = apex if lo_codes is None else nodes_of(lo_codes)
        # узлы, попавшие в оба множества, относим к "1"
        lo = np.setdiff1d(lo, hi)
        fixed = np.concatenate([hi, lo])
        values = np.concatenate([np.ones(len(hi)), np.zeros(len(lo))])
        fields[:, j] = np.clip(solve_dirichlet(K, fixed, values), 0.0, 1.0)
    return fields


# ---------------------------------------------------------------------------
# стратифицированные окрестности (формула 2.4 НИР)
# ---------------------------------------------------------------------------
STRAT_PERCENTILES = (0.002, 0.01, 0.02, 0.05, 0.10)
STRAT_SIZES = (20, 30, 30, 10, 10)


def build_stratified_neighborhoods(coords: np.ndarray, n_nb: int = 12,
                                   seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Для каждого узла отбирает n_nb соседей из 5 перцентильных страт
    расстояния (SC1..SC5). Возвращает (индексы соседей, номер страты).

    Номер страты возвращается отдельно: он нужен для анализа профиля
    весов внимания по стратам (Табл. 4.3 НИР).
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    n = len(coords)
    tree = cKDTree(coords)

    k_max = min(int(n * STRAT_PERCENTILES[-1]) + 1, n)
    dists, idxs = tree.query(coords, k=k_max, workers=-1)
    dists, idxs = dists[:, 1:], idxs[:, 1:]          # исключаем сам узел

    # глобальные пороги по перцентилям расстояний
    sample = rng.choice(n, size=min(n, 2000), replace=False)
    thr = np.percentile(dists[sample].ravel(),
                        [100.0 * p / STRAT_PERCENTILES[-1] for p in STRAT_PERCENTILES])
    thr = np.minimum.accumulate(thr[::-1])[::-1]     # монотонность на всякий случай
    thr = np.sort(thr)

    nb = np.zeros((n, n_nb), dtype=np.int64)
    nb_strat = np.zeros((n, n_nb), dtype=np.int8)

    for i in range(n):
        d_i, id_i = dists[i], idxs[i]
        pick_idx, pick_strat = [], []
        prev = 0.0
        for s, (t, ns) in enumerate(zip(thr, STRAT_SIZES)):
            m = (d_i > prev) & (d_i <= t)
            cand = id_i[m]
            if len(cand):
                take = min(ns, len(cand))
                sel = rng.choice(cand, size=take, replace=False)
                pick_idx.append(sel)
                pick_strat.append(np.full(take, s, dtype=np.int8))
            prev = t
        if not pick_idx:
            nb[i] = id_i[:n_nb]
            continue
        cand = np.concatenate(pick_idx)
        cst = np.concatenate(pick_strat)
        if len(cand) >= n_nb:
            sel = rng.choice(len(cand), size=n_nb, replace=False)
        else:
            sel = rng.choice(len(cand), size=n_nb, replace=True)
        nb[i] = cand[sel]
        nb_strat[i] = cst[sel]

    return nb, nb_strat


def build_knn_neighborhoods(coords: np.ndarray, n_nb: int = 12) -> np.ndarray:
    """Ближнесоседняя (kNN) стратегия -- для сравнения в ablation."""
    from scipy.spatial import cKDTree

    _, idx = cKDTree(coords).query(coords, k=n_nb + 1, workers=-1)
    return idx[:, 1:]


def build_random_neighborhoods(coords: np.ndarray, n_nb: int = 12,
                               seed: int = 0) -> np.ndarray:
    """Случайная стратегия -- для сравнения в ablation."""
    rng = np.random.default_rng(seed)
    n = len(coords)
    return rng.integers(0, n, size=(n, n_nb))


# ---------------------------------------------------------------------------
# признаки формы (глобальный дескриптор xi)
# ---------------------------------------------------------------------------
def shape_features(coords: np.ndarray, elements: np.ndarray,
                   faces: np.ndarray, face_type: np.ndarray,
                   laplace: np.ndarray) -> np.ndarray:
    """
    Глобальный дескриптор геометрии (инвариантный к сдвигу и повороту).

    Состав (20 чисел):
      * объём миокарда, площади эндокарда ЛЖ/ПЖ, эпикарда, основания;
      * длина по продольной оси, три главных полуоси (SVD облака точек);
      * отношения главных осей (2), сферичность;
      * средняя и стандартная толщина стенки ЛЖ и ПЖ, оценённая по
        трансмуральным полям Лапласа: h ~ 1 / |grad phi|;
      * перцентили толщины (p10, p50, p90) для ЛЖ.
    """
    from fem_core import face_areas_normals

    gradN, vol = tet_shape_gradients(coords, elements)
    V = float(vol.sum())
    area, _ = face_areas_normals(coords, faces)

    a_endo_lv = float(area[face_type == ENDO_LV].sum())
    a_endo_rv = float(area[face_type == ENDO_RV].sum())
    a_epi = float(area[face_type == EPI].sum())
    a_base = float(area[face_type == BASE].sum())

    centred = coords - coords.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False) / np.sqrt(len(coords))
    long_axis = float(np.ptp(centred @ (np.linalg.svd(centred,
                                                     full_matrices=False)[2][0])))
    sphericity = float((36.0 * np.pi * V ** 2) ** (1.0 / 3.0)
                       / max(a_epi + a_endo_lv + a_endo_rv, 1e-9))

    # толщина стенки по трансмуральному полю: |grad phi| ~ 1/h.
    # Учитываются только элементы, реально лежащие внутри трансмурального
    # "коридора" (0.05 < phi < 0.95): вне него поле почти постоянно, градиент
    # вырождается и оценка 1/|grad phi| теряет смысл.
    def thickness(field_col: int) -> np.ndarray:
        phi_e = laplace[elements, field_col]
        g = np.einsum("ea,eai->ei", phi_e, gradN)
        gm = np.linalg.norm(g, axis=1)
        inside = (phi_e.mean(axis=1) > 0.05) & (phi_e.mean(axis=1) < 0.95)
        keep = inside & (gm > 1e-3)
        if not np.any(keep):
            return np.array([0.0])
        h = 1.0 / gm[keep]
        return h[h < np.percentile(h, 99)]      # отсечение единичных выбросов

    h_lv = thickness(0)
    h_rv = thickness(1)

    feats = [
        V / 1e4, a_endo_lv / 1e3, a_endo_rv / 1e3, a_epi / 1e3, a_base / 1e3,
        long_axis / 10.0, sv[0] / 10.0, sv[1] / 10.0, sv[2] / 10.0,
        sv[1] / max(sv[0], 1e-9), sv[2] / max(sv[0], 1e-9), sphericity,
        h_lv.mean() / 10.0, h_lv.std() / 10.0,
        h_rv.mean() / 10.0, h_rv.std() / 10.0,
        np.percentile(h_lv, 10) / 10.0, np.percentile(h_lv, 50) / 10.0,
        np.percentile(h_lv, 90) / 10.0,
        float(len(coords)) / 1e4,
    ]
    return np.asarray(feats, dtype=np.float32)
