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

    phi_AB, phi_EP, phi_LV-RV, phi_LV-EP, phi_RV-EP -- как в отчёте и [15]
    (состав и порядок см. compute_laplace_fields).

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
                           faces: np.ndarray, face_type: np.ndarray,
                           uvc_z: np.ndarray) -> np.ndarray:
    """
    Пять полей Лапласа-Дирихле в том же составе и порядке, что в отчёте
    и в [15] (решаются МКЭ, см. fem_core.compute_laplace_fields):

        0: Ф_AB    база = 1, верхушка = 0
        1: Ф_EP    эпикард = 1, эндокард ЛЖ+ПЖ = 0
        2: Ф_LV-RV эндокард ЛЖ = 1, эндокард ПЖ = 0
        3: Ф_LV-EP эндокард ЛЖ = 1, эпикард = 0
        4: Ф_RV-EP эндокард ПЖ = 1, эпикард = 0

    ИСПРАВЛЕНИЕ: в первой версии здесь был другой набор полей, чем в отчёте,
    а верхушка определялась как самая удалённая от основания точка; теперь
    верхушка -- эпикардиальные узлы с минимальной апикобазальной UVC.
    """
    from fem_core import compute_laplace_fields as _fem_fields
    return _fem_fields(coords, elements, faces, face_type, uvc_z)


# ---------------------------------------------------------------------------
# стратифицированные окрестности (формула 2.4 НИР)
# ---------------------------------------------------------------------------
STRAT_PERCENTILES = (0.2, 1.0, 2.0, 5.0, 10.0)    # Q_p%, в процентах
STRAT_SIZES = (20, 30, 30, 10, 10)


def pairwise_distance_percentiles(coords: np.ndarray, seed: int = 0,
                                  n_sample: int = 2000) -> np.ndarray:
    """
    Перцентили Q0.2%, Q1%, Q2%, Q5%, Q10% распределения ВСЕХ попарных
    расстояний между узлами (определение [15]). Оценка по расстояниям от
    n_sample случайных узлов до всех узлов.
    """
    rng = np.random.default_rng(seed)
    n = len(coords)
    idx = rng.choice(n, size=min(n, n_sample), replace=False)
    parts = []
    for chunk in np.array_split(idx, max(1, len(idx) // 200)):
        d = np.linalg.norm(coords[chunk][:, None] - coords[None], axis=-1)
        parts.append(d[d > 0])
    return np.percentile(np.concatenate(parts), STRAT_PERCENTILES)


def build_stratified_neighborhoods(coords: np.ndarray, n_nb: int = 12,
                                   seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Для каждого узла отбирает n_nb соседей из 5 страт расстояния SC1..SC5
    с границами Q0.2%, Q1%, Q2%, Q5%, Q10% всех попарных расстояний [15].
    Возвращает (индексы соседей, номер страты).

    ИСПРАВЛЕНИЕ: в первой версии пороги брались по расстояниям до 10%
    ближайших узлов, т.е. страты определялись иначе, чем в [15] и в отчёте.
    Перцентили всех попарных расстояний почти не зависят от плотности узлов,
    поэтому одни и те же страты получаются и на прореженном графе, и на
    полной сетке -- это позволяет применять обученную модель к полной сетке.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    n = len(coords)
    thr = pairwise_distance_percentiles(coords, seed)
    lower = np.concatenate([[0.0], thr[:-1]])
    tree = cKDTree(coords)
    balls = tree.query_ball_point(coords, thr[-1], workers=-1)

    nb = np.zeros((n, n_nb), dtype=np.int64)
    nb_strat = np.zeros((n, n_nb), dtype=np.int8)
    for i in range(n):
        cand = np.asarray(balls[i])
        cand = cand[cand != i]
        d = np.linalg.norm(coords[cand] - coords[i], axis=1)
        pick, pick_s = [], []
        for s in range(5):
            c = cand[(d > lower[s]) & (d <= thr[s])]
            if len(c):
                take = min(STRAT_SIZES[s], len(c))
                pick.append(rng.choice(c, size=take, replace=False))
                pick_s.append(np.full(take, s, dtype=np.int8))
        if not pick:
            _, idx = tree.query(coords[i], k=n_nb + 1)
            nb[i] = idx[1:]
            continue
        pool, pool_s = np.concatenate(pick), np.concatenate(pick_s)
        sel = rng.choice(len(pool), size=n_nb, replace=len(pool) < n_nb)
        nb[i], nb_strat[i] = pool[sel], pool_s[sel]
    return nb, nb_strat


def anatomical_frame(coords: np.ndarray, uvc_z: np.ndarray,
                     is_lv_node: np.ndarray) -> np.ndarray:
    """
    Поворот в анатомическую систему координат сердца (строки -- оси):
      e3 -- длинная ось (верхушка -> база), направление роста UVC Z;
      e1 -- направление центр ЛЖ -> центр ПЖ, ортогонально e3;
      e2 -- дополняет до правой тройки.
    Убирает зависимость признаков от положения пациента в томографе.
    """
    x = coords - coords.mean(axis=0)
    a = np.column_stack([np.ones(len(x)), x])
    g = np.linalg.lstsq(a, uvc_z, rcond=None)[0][1:]
    e3 = g / np.linalg.norm(g)
    d = x[~is_lv_node].mean(axis=0) - x[is_lv_node].mean(axis=0)
    d -= (d @ e3) * e3
    e1 = d / np.linalg.norm(d)
    return np.stack([e1, np.cross(e3, e1), e3])


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

    h_lv = thickness(3)   # Ф_LV-EP: трансмуральная координата ЛЖ
    h_rv = thickness(4)   # Ф_RV-EP: трансмуральная координата ПЖ

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
