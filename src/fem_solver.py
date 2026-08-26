"""
Нелинейный конечно-элементный решатель пассивной механики бивентрикулярного
миокарда (эталон "КЭА" для обучения и валидации эмулятора HSS).

Постановка (конечные деформации, полностью лагранжева формулировка):
    div_X P = 0  в  Omega_0,
    P = dW/dF,
    F = I + Grad u,   J = det F.

Материал -- почти несжимаемый трансверсально-изотропный неогуковский с двумя
армирующими членами по направлениям волокна f0 и слоя s0 (упрощение модели
Холзапфеля-Огдена: экспоненциальные члены заменены квадратичными, сохранены
структура анизотропии и характер армирования при растяжении):

    W = mu/2 (I1 - 3) - mu ln J + lam/2 (ln J)^2
        + kf/2 <I4f - 1>^2 + ks/2 <I4s - 1>^2 ,
    I4f = |F f0|^2,  I4s = |F s0|^2,  <x> = max(x, 0).

Граничные условия (раздел 2.1 НИР):
    * эндокард ЛЖ / ПЖ -- давление p_lv / p_rv (нагрузка на отсчётную
      конфигурацию, t0 = -p N);
    * эпикард и база -- условия Робена (перикардиальные пружины):
      t0 = -k u,  k_epi, k_base.

Численная схема: инкрементальное нагружение (n_steps шагов по давлению),
на каждом шаге -- метод Ньютона с аналитическим касательным оператором
и возвратом по шагу (backtracking) при росте невязки.
Линейные системы решаются разреженным LU-разложением.

Тангенс проверяется конечными разностями в src/verify_solver.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fem_core import (BASE, ENDO_LV, ENDO_RV, EPI, face_areas_normals,
                      tet_shape_gradients)

MMHG_TO_KPA = 0.1333224

# Прямой разреженный решатель: MKL PARDISO (примерно в 7 раз быстрее SuperLU
# на этих системах), с откатом на scipy при отсутствии пакета.
try:
    import pypardiso as _pardiso

    def _sparse_solve(A, b):
        return _pardiso.spsolve(A.tocsr(), b)
except ImportError:  # pragma: no cover
    def _sparse_solve(A, b):
        return spla.spsolve(A.tocsc(), b)


@dataclass
class Material:
    """Параметры материала (кПа). lam задаётся через отношение к mu."""
    mu: float = 2.0
    kf: float = 10.0
    ks: float = 4.0
    lam_ratio: float = 25.0

    @property
    def lam(self) -> float:
        return self.lam_ratio * self.mu

    def as_vector(self) -> np.ndarray:
        return np.array([self.mu, self.kf, self.ks, self.lam_ratio], dtype=np.float64)


@dataclass
class Loading:
    """Нагрузка: давления (мм рт. ст.) и жёсткости пружин Робена (кПа/мм)."""
    p_lv: float = 10.0
    p_rv: float = 4.0
    k_epi: float = 0.2
    k_base: float = 2.0

    def as_vector(self) -> np.ndarray:
        return np.array([self.p_lv, self.p_rv, self.k_epi, self.k_base],
                        dtype=np.float64)


@dataclass
class Geometry:
    coords: np.ndarray
    elements: np.ndarray
    f0: np.ndarray
    s0: np.ndarray
    faces: np.ndarray
    face_type: np.ndarray
    gradN: np.ndarray = field(init=False)
    vol: np.ndarray = field(init=False)

    def __post_init__(self):
        self.gradN, self.vol = tet_shape_gradients(self.coords, self.elements)
        if np.any(self.vol < 0):          # приведение к положительной ориентации
            bad = self.vol < 0
            self.elements[bad] = self.elements[bad][:, [0, 1, 3, 2]]
            self.gradN, self.vol = tet_shape_gradients(self.coords, self.elements)


# ---------------------------------------------------------------------------
# конститутивные соотношения
# ---------------------------------------------------------------------------
def _energy_stress_tangent(F: np.ndarray, f0: np.ndarray, s0: np.ndarray,
                           mat: Material, want_tangent: bool = True):
    """
    Плотность энергии W, первый тензор Пиолы-Кирхгофа P и касательный
    оператор A = dP/dF для каждого элемента.

    F  : (E,3,3),  f0/s0 : (E,3)
    """
    mu, lam = mat.mu, mat.lam
    E = F.shape[0]
    I = np.eye(3)

    J = np.linalg.det(F)
    Finv = np.linalg.inv(F)
    FinvT = np.transpose(Finv, (0, 2, 1))
    lnJ = np.log(np.maximum(J, 1e-12))
    I1 = np.einsum("eij,eij->e", F, F)

    W = 0.5 * mu * (I1 - 3.0) - mu * lnJ + 0.5 * lam * lnJ ** 2
    P = mu * (F - FinvT) + lam * lnJ[:, None, None] * FinvT

    Ff = np.einsum("eij,ej->ei", F, f0)
    Fs = np.einsum("eij,ej->ei", F, s0)
    I4f = np.einsum("ei,ei->e", Ff, Ff)
    I4s = np.einsum("ei,ei->e", Fs, Fs)
    df = np.maximum(I4f - 1.0, 0.0)
    ds = np.maximum(I4s - 1.0, 0.0)

    W += 0.5 * mat.kf * df ** 2 + 0.5 * mat.ks * ds ** 2
    P += 2.0 * mat.kf * df[:, None, None] * np.einsum("ei,ej->eij", Ff, f0)
    P += 2.0 * mat.ks * ds[:, None, None] * np.einsum("ei,ej->eij", Fs, s0)

    if not want_tangent:
        return W, P, None

    # A_iJkL
    A = mu * np.einsum("ik,JL->iJkL", I, I)[None, ...].repeat(E, axis=0)
    A += mu * np.einsum("eiL,ekJ->eiJkL", FinvT, FinvT)
    A += lam * np.einsum("ekL,eiJ->eiJkL", FinvT, FinvT)
    A -= (lam * lnJ)[:, None, None, None, None] * np.einsum(
        "eiL,ekJ->eiJkL", FinvT, FinvT)

    act_f = (I4f > 1.0).astype(np.float64)
    A += (4.0 * mat.kf * act_f)[:, None, None, None, None] * np.einsum(
        "ei,eJ,ek,eL->eiJkL", Ff, f0, Ff, f0)
    A += (2.0 * mat.kf * df)[:, None, None, None, None] * np.einsum(
        "ik,eJ,eL->eiJkL", I, f0, f0)

    act_s = (I4s > 1.0).astype(np.float64)
    A += (4.0 * mat.ks * act_s)[:, None, None, None, None] * np.einsum(
        "ei,eJ,ek,eL->eiJkL", Fs, s0, Fs, s0)
    A += (2.0 * mat.ks * ds)[:, None, None, None, None] * np.einsum(
        "ik,eJ,eL->eiJkL", I, s0, s0)

    return W, P, A


# ---------------------------------------------------------------------------
# внешние силы и пружины Робена
# ---------------------------------------------------------------------------
def external_forces(geo: Geometry, load: Loading):
    """
    Возвращает
        f_p    : (N,3) узловые силы от давления при полном уровне нагрузки
        k_diag : (N,)  диагональ пружин Робена (кПа * мм)
    """
    n = len(geo.coords)
    area, normal = face_areas_normals(geo.coords, geo.faces)
    f_p = np.zeros((n, 3))
    k_diag = np.zeros(n)

    for code, p in ((ENDO_LV, load.p_lv), (ENDO_RV, load.p_rv)):
        m = geo.face_type == code
        contrib = (-p * MMHG_TO_KPA) * (area[m] / 3.0)[:, None] * normal[m]
        for j in range(3):
            np.add.at(f_p, geo.faces[m][:, j], contrib)

    for code, k in ((EPI, load.k_epi), (BASE, load.k_base)):
        m = geo.face_type == code
        contrib = k * area[m] / 3.0
        for j in range(3):
            np.add.at(k_diag, geo.faces[m][:, j], contrib)

    return f_p, k_diag


# ---------------------------------------------------------------------------
# внутренние силы и глобальная касательная матрица
# ---------------------------------------------------------------------------
def _assembly_index(elements: np.ndarray):
    """Индексы строк/столбцов для сборки блоков 12x12."""
    dofs = (elements[:, :, None] * 3 + np.arange(3)[None, None, :]).reshape(-1, 12)
    rows = np.repeat(dofs, 12, axis=1).ravel()
    cols = np.tile(dofs, (1, 12)).ravel()
    return dofs, rows, cols


def internal_force(geo: Geometry, u: np.ndarray, mat: Material,
                   want_tangent: bool, idx=None):
    """Внутренние силы (N,3), полная энергия и (опционально) матрица касательных."""
    F = np.eye(3)[None, ...] + np.einsum("eai,eaJ->eiJ", u[geo.elements], geo.gradN)
    W, P, A = _energy_stress_tangent(F, geo.f0, geo.s0, mat, want_tangent)

    vol = geo.vol
    energy = float(np.sum(W * vol))
    fe = np.einsum("eiJ,eaJ,e->eai", P, geo.gradN, vol)      # (E,4,3)
    f_int = np.zeros_like(u)
    np.add.at(f_int, geo.elements.ravel(), fe.reshape(-1, 3))

    if not want_tangent:
        return energy, f_int, None

    ke = np.einsum("eiJkL,eaJ,ebL,e->eaibk", A, geo.gradN, geo.gradN, vol)
    ke = ke.reshape(-1, 12, 12)
    _, rows, cols = idx
    n_dof = u.size
    K = sp.csr_matrix((ke.ravel(), (rows, cols)), shape=(n_dof, n_dof))
    return energy, f_int, K


# ---------------------------------------------------------------------------
# решатель
# ---------------------------------------------------------------------------
def solve(geo: Geometry, mat: Material, load: Loading,
          n_steps: int = 8, tol: float = 1e-7, max_newton: int = 25,
          verbose: bool = False):
    """
    Решает задачу равновесия инкрементальным нагружением + Ньютоном.

    Возвращает (u, info): u -- (N,3) смещения в мм.
    """
    n = len(geo.coords)
    u = np.zeros((n, 3))
    f_p_full, k_diag = external_forces(geo, load)
    idx = _assembly_index(geo.elements)
    K_rob = sp.diags(np.repeat(k_diag, 3))
    n_iter_total = 0

    for step in range(1, n_steps + 1):
        lam_load = step / n_steps
        f_ext = lam_load * f_p_full
        converged = False
        for it in range(max_newton):
            energy, f_int, K = internal_force(geo, u, mat, True, idx)
            R = (f_int + (k_diag[:, None] * u) - f_ext).ravel()
            rnorm = np.linalg.norm(R)
            scale = max(np.linalg.norm(f_ext), 1e-12)
            if verbose:
                print(f"   шаг {step} итер {it}: |R|/|F| = {rnorm / scale:.3e}")
            if rnorm / scale < tol:
                converged = True
                break
            du = _sparse_solve(K + K_rob, -R).reshape(n, 3)
            # возврат по шагу при отсутствии убывания невязки
            alpha = 1.0
            for _ in range(12):
                u_try = u + alpha * du
                _, f_try, _ = internal_force(geo, u_try, mat, False)
                R_try = (f_try + (k_diag[:, None] * u_try) - f_ext).ravel()
                if np.linalg.norm(R_try) < rnorm or alpha < 1e-3:
                    break
                alpha *= 0.5
            u = u + alpha * du
            n_iter_total += 1
        if not converged:
            energy, f_int, _ = internal_force(geo, u, mat, False)
            R = (f_int + (k_diag[:, None] * u) - f_ext).ravel()
            if np.linalg.norm(R) / max(np.linalg.norm(f_ext), 1e-12) > 1e-4:
                return u, {"ok": False, "iters": n_iter_total, "step": step}

    F = np.eye(3)[None, ...] + np.einsum("eai,eaJ->eiJ", u[geo.elements], geo.gradN)
    info = {
        "ok": True,
        "iters": n_iter_total,
        "u_max": float(np.linalg.norm(u, axis=1).max()),
        "J_min": float(np.linalg.det(F).min()),
        "J_max": float(np.linalg.det(F).max()),
    }
    return u, info
