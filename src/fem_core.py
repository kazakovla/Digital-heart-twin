"""
Базовые конечно-элементные операции на тетраэдральной сетке P1
и поля Лапласа-Дирихле (позиционное кодирование узлов, раздел 2.2 НИР).

Все функции векторизованы по элементам (numpy), сборка -- разреженная (scipy).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# типы граничных граней
EPI, ENDO_LV, ENDO_RV, BASE = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# геометрия элементов
# ---------------------------------------------------------------------------
def tet_shape_gradients(coords: np.ndarray, elements: np.ndarray):
    """
    Градиенты линейных базисных функций и объёмы тетраэдров.

    Возвращает
        gradN : (E, 4, 3)  dN_a/dX_J -- постоянны внутри элемента
        vol   : (E,)       объёмы элементов отсчётной конфигурации
    """
    p = coords[elements]                       # (E,4,3)
    J = np.stack(
        [p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=1
    )                                          # (E,3,3): dX/dxi
    detJ = np.linalg.det(J)
    vol = detJ / 6.0
    Jinv = np.linalg.inv(J)                    # (E,3,3): Jinv[K,m] = dxi_m/dX_K
    # dN/dxi для (N0..N3) = (1-x-y-z, x, y, z)
    dNdxi = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    gradN = np.einsum("am,ekm->eak", dNdxi, Jinv)   # (E,4,3): dN_a/dX_k
    return gradN, vol


def face_areas_normals(coords: np.ndarray, faces: np.ndarray):
    """Площади и единичные внешние нормали граничных треугольников."""
    p0, p1, p2 = coords[faces[:, 0]], coords[faces[:, 1]], coords[faces[:, 2]]
    cr = np.cross(p1 - p0, p2 - p0)
    twoA = np.linalg.norm(cr, axis=1)
    area = 0.5 * twoA
    normal = cr / np.maximum(twoA, 1e-30)[:, None]
    return area, normal


def boundary_node_sets(faces: np.ndarray, face_type: np.ndarray, n_nodes: int):
    """Множества узлов каждой границы (по граням соответствующего типа)."""
    sets = {}
    for name, code in (("epi", EPI), ("endo_lv", ENDO_LV),
                       ("endo_rv", ENDO_RV), ("base", BASE)):
        mask = np.zeros(n_nodes, dtype=bool)
        mask[np.unique(faces[face_type == code])] = True
        sets[name] = mask
    return sets


# ---------------------------------------------------------------------------
# оператор Лапласа
# ---------------------------------------------------------------------------
def laplace_matrix(coords: np.ndarray, elements: np.ndarray) -> sp.csr_matrix:
    """Матрица жёсткости оператора -div(grad) для P1-тетраэдров."""
    gradN, vol = tet_shape_gradients(coords, elements)
    ke = np.einsum("eai,ebi,e->eab", gradN, gradN, np.abs(vol))   # (E,4,4)
    rows = np.repeat(elements, 4, axis=1).ravel()
    cols = np.tile(elements, (1, 4)).ravel()
    n = len(coords)
    return sp.csr_matrix((ke.ravel(), (rows, cols)), shape=(n, n))


def solve_laplace_dirichlet(K: sp.csr_matrix, nodes_one: np.ndarray,
                            nodes_zero: np.ndarray) -> np.ndarray:
    """
    Решение задачи Лапласа с условиями Дирихле Ф=1 на nodes_one, Ф=0 на nodes_zero
    и условием Неймана (нулевой поток) на остальной границе.
    """
    n = K.shape[0]
    fixed = nodes_one | nodes_zero
    free = ~fixed
    phi = np.zeros(n)
    phi[nodes_one] = 1.0
    rhs = -K[:, fixed] @ phi[fixed]
    A = K[free][:, free].tocsc()
    phi[free] = spla.spsolve(A, rhs[free])
    return np.clip(phi, 0.0, 1.0)


def compute_laplace_fields(coords: np.ndarray, elements: np.ndarray,
                           faces: np.ndarray, face_type: np.ndarray,
                           uvc_z: np.ndarray) -> np.ndarray:
    """
    Пять полей Лапласа-Дирихле (Таблица 2.1 НИР):
        Ф_AB     : база = 1, верхушка = 0
        Ф_EP     : эпикард = 1, эндокард ЛЖ+ПЖ = 0
        Ф_LV-RV  : эндокард ЛЖ = 1, эндокард ПЖ = 0
        Ф_LV-EP  : эндокард ЛЖ = 1, эпикард = 0
        Ф_RV-EP  : эндокард ПЖ = 1, эпикард = 0

    Возвращает массив (N, 5).
    """
    n = len(coords)
    S = boundary_node_sets(faces, face_type, n)
    K = laplace_matrix(coords, elements)

    # верхушка: эпикардиальные узлы с наименьшей апикобазальной координатой UVC
    epi = S["epi"]
    z_epi = uvc_z[epi]
    thr = np.percentile(z_epi, 2.0)
    apex = epi & (uvc_z <= thr)

    fields = [
        solve_laplace_dirichlet(K, S["base"], apex),
        solve_laplace_dirichlet(K, epi, S["endo_lv"] | S["endo_rv"]),
        solve_laplace_dirichlet(K, S["endo_lv"], S["endo_rv"]),
        solve_laplace_dirichlet(K, S["endo_lv"], epi),
        solve_laplace_dirichlet(K, S["endo_rv"], epi),
    ]
    return np.stack(fields, axis=1)


# ---------------------------------------------------------------------------
# объём полости (для метрик ErrV_LV / ErrV_RV)
# ---------------------------------------------------------------------------
def cavity_volume(coords: np.ndarray, faces: np.ndarray) -> float:
    """
    Объём, ограниченный набором граней (в мм^3), по теореме о дивергенции.
    Набор граней эндокарда не замкнут (открыт со стороны клапанов), поэтому
    величина используется как относительная мера (сравнение до/после нагрузки),
    что достаточно для оценки ошибки предсказания изменения объёма.
    """
    p0, p1, p2 = coords[faces[:, 0]], coords[faces[:, 1]], coords[faces[:, 2]]
    return float(np.abs(np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum()) / 6.0)
