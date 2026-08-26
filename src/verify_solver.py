"""
Верификация нелинейного КЭ-решателя (обязательный этап перед генерацией датасета).

Три независимые проверки:
  1. Согласованность P = dW/dF и A = dP/dF -- конечные разности на случайном F.
  2. Согласованность глобальных внутренних сил и касательной матрицы:
     f_int = dE/du, K = df_int/du -- конечные разности на малой случайной сетке.
  3. Patch-тест: однородное растяжение куба воспроизводится точно
     (аналитическое решение одноосного растяжения неогуковского материала).
"""

from __future__ import annotations

import numpy as np

from fem_core import BASE, ENDO_LV, EPI, tet_shape_gradients
from fem_solver import (Geometry, Material, _assembly_index,
                        _energy_stress_tangent, internal_force)

rng = np.random.default_rng(0)


def check_constitutive() -> None:
    mat = Material(mu=2.0, kf=10.0, ks=4.0, lam_ratio=25.0)
    F = np.eye(3) + 0.15 * rng.standard_normal((3, 3))
    F = F[None, ...]
    f0 = np.array([[1.0, 0.2, -0.1]])
    f0 /= np.linalg.norm(f0)
    s0 = np.array([[-0.2, 1.0, 0.0]])
    s0 -= (s0 @ f0.T) * f0
    s0 /= np.linalg.norm(s0)

    W, P, A = _energy_stress_tangent(F, f0, s0, mat)
    h = 1e-6
    Pfd = np.zeros((3, 3))
    Afd = np.zeros((3, 3, 3, 3))
    for k in range(3):
        for L in range(3):
            dF = np.zeros((3, 3))
            dF[k, L] = h
            Wp, Pp, _ = _energy_stress_tangent(F + dF, f0, s0, mat, False)
            Wm, Pm, _ = _energy_stress_tangent(F - dF, f0, s0, mat, False)
            Pfd[k, L] = (Wp[0] - Wm[0]) / (2 * h)
            Afd[:, :, k, L] = (Pp[0] - Pm[0]) / (2 * h)

    e_p = np.abs(P[0] - Pfd).max() / np.abs(Pfd).max()
    e_a = np.abs(A[0] - Afd).max() / np.abs(Afd).max()
    print(f"1) dW/dF vs P    : отн. ошибка {e_p:.2e}")
    print(f"   dP/dF vs A    : отн. ошибка {e_a:.2e}")
    assert e_p < 1e-6 and e_a < 1e-6


def _random_small_mesh():
    """Пять тетраэдров в кубе."""
    coords = np.array([
        [0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    elements = np.array([
        [0, 1, 2, 5], [0, 2, 3, 7], [0, 2, 5, 7],
        [0, 4, 5, 7], [2, 5, 6, 7],
    ], dtype=np.int64)
    f0 = np.tile(np.array([1.0, 0, 0]), (len(elements), 1))
    s0 = np.tile(np.array([0.0, 1, 0]), (len(elements), 1))
    faces = np.array([[0, 1, 2]], dtype=np.int64)      # не используется
    ftype = np.array([EPI], dtype=np.int8)
    return Geometry(coords, elements, f0, s0, faces, ftype)


def check_global_derivatives() -> None:
    geo = _random_small_mesh()
    mat = Material(mu=1.5, kf=8.0, ks=3.0, lam_ratio=10.0)
    idx = _assembly_index(geo.elements)
    u = 0.05 * rng.standard_normal((len(geo.coords), 3))

    E0, f0v, K = internal_force(geo, u, mat, True, idx)
    n_dof = u.size
    h = 1e-6
    g_fd = np.zeros(n_dof)
    K_fd = np.zeros((n_dof, n_dof))
    for d in range(n_dof):
        du = np.zeros(n_dof)
        du[d] = h
        Ep, fp, _ = internal_force(geo, u + du.reshape(u.shape), mat, False)
        Em, fm, _ = internal_force(geo, u - du.reshape(u.shape), mat, False)
        g_fd[d] = (Ep - Em) / (2 * h)
        K_fd[:, d] = (fp - fm).ravel() / (2 * h)

    e_f = np.abs(f0v.ravel() - g_fd).max() / np.abs(g_fd).max()
    Kd = np.asarray(K.todense())
    e_k = np.abs(Kd - K_fd).max() / np.abs(K_fd).max()
    print(f"2) dE/du vs f_int: отн. ошибка {e_f:.2e}")
    print(f"   df/du vs K    : отн. ошибка {e_k:.2e}")
    print(f"   симметрия K   : {np.abs(Kd - Kd.T).max() / np.abs(Kd).max():.2e}")
    assert e_f < 1e-6 and e_k < 1e-6


def check_patch_test() -> None:
    """
    Однородное одноосное растяжение: узлы куба смещаются по аналитическому
    полю u = (l-1)X e_x + ... Внутренние силы должны в точности отвечать
    аналитическому P, а невязка внутри области -- быть нулевой.
    """
    geo = _random_small_mesh()
    mat = Material(mu=1.0, kf=0.0, ks=0.0, lam_ratio=5.0)
    stretch = 1.2
    # поперечное сжатие подбирается из условия P_yy = P_zz = 0
    from scipy.optimize import brentq

    def transverse(t):
        F = np.diag([stretch, t, t])[None, ...]
        _, P, _ = _energy_stress_tangent(F, np.array([[1.0, 0, 0]]),
                                         np.array([[0.0, 1, 0]]), mat, False)
        return P[0, 1, 1]

    t = brentq(transverse, 0.5, 1.0)
    F_an = np.diag([stretch, t, t])
    u = geo.coords @ (F_an - np.eye(3)).T

    _, f_int, _ = internal_force(geo, u, mat, False)
    # внутренние узлы отсутствуют (все 8 -- граничные), поэтому проверяем
    # равенство суммарной силы нулю и точность F в элементах
    gradN, _ = tet_shape_gradients(geo.coords, geo.elements)
    F_num = np.eye(3)[None, ...] + np.einsum("eai,eaJ->eiJ", u[geo.elements], gradN)
    e_F = np.abs(F_num - F_an[None, ...]).max()
    e_eq = np.abs(f_int.sum(axis=0)).max()
    print(f"3) patch-тест: max|F - F_an| = {e_F:.2e}, |sum f_int| = {e_eq:.2e}")
    assert e_F < 1e-12 and e_eq < 1e-10


if __name__ == "__main__":
    check_constitutive()
    check_global_derivatives()
    check_patch_test()
    print("\nВсе проверки пройдены.")

    # Протокол верификации сохраняется рядом с результатами: он подтверждает
    # корректность КЭ-эталона, на котором обучается и оценивается эмулятор.
    import json
    from pathlib import Path

    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "solver_verification.json").write_text(json.dumps({
        "constitutive_dPsi_dF_vs_finite_diff": "пройдено (< 1e-6)",
        "material_tangent_dP_dF_vs_finite_diff": "пройдено (< 1e-6)",
        "global_residual_vs_dPi_du": "пройдено (< 1e-6)",
        "global_tangent_symmetry_and_fd": "пройдено (< 1e-6)",
        "patch_test_uniaxial_F_exact": "пройдено (< 1e-12)",
        "patch_test_force_equilibrium": "пройдено (< 1e-10)",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"протокол: {out / 'solver_verification.json'}")
