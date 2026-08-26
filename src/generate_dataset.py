"""
Шаг 2. Генерация обучающего датасета КЭА.

Для каждой реальной геометрии (geom_XX.npz) решается серия задач пассивного
наполнения при случайных параметрах материала и нагрузки. Результат -- пары
(геометрия + параметры) -> (поле смещений), на которых обучается эмулятор.

Диапазоны выборки (латинский гиперкуб) выбраны вокруг физиологических
значений для пассивной диастолы:
    mu        1.0 ... 4.0  кПа   -- изотропная жёсткость
    kf        4.0 ... 20.0 кПа   -- армирование по волокну
    ks        1.5 ... 8.0  кПа   -- армирование по слою
    p_lv      4.0 ... 18.0 мм рт. ст. -- давление в ЛЖ
    p_rv/p_lv 0.25... 0.55        -- отношение давлений ПЖ/ЛЖ
    k_epi     0.05... 0.40 кПа/мм -- перикардиальные пружины
    k_base    0.8 ... 4.0  кПа/мм -- фиксация клапанного основания

Запуск:
    python src/generate_dataset.py --n-per-geom 24 --workers 2
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

from fem_solver import Geometry, Loading, Material, solve

RANGES = {
    "mu": (1.0, 4.0),
    "kf": (4.0, 20.0),
    "ks": (1.5, 8.0),
    "p_lv": (4.0, 18.0),
    "p_ratio": (0.25, 0.55),
    "k_epi": (0.05, 0.40),
    "k_base": (0.8, 4.0),
}
PARAM_KEYS = list(RANGES.keys())


def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Стратифицированная выборка в [0,1]^d."""
    cut = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        rng.shuffle(cut[:, j])
    return cut


def sample_params(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unit = latin_hypercube(n, len(PARAM_KEYS), rng)
    out = np.empty_like(unit)
    for j, k in enumerate(PARAM_KEYS):
        lo, hi = RANGES[k]
        out[:, j] = lo + unit[:, j] * (hi - lo)
    return out


def load_geometry(path: str):
    d = np.load(path)
    geo = Geometry(
        d["coords"].astype(np.float64),
        d["elements"].astype(np.int64),
        d["f0"].astype(np.float64),
        d["s0"].astype(np.float64),
        d["faces"].astype(np.int64),
        d["face_type"],
    )
    return geo, d


def run_geometry(geom_path: str, n_samples: int, out_dir: str,
                 n_steps: int, seed: int) -> None:
    case = os.path.basename(geom_path).replace("geom_", "").replace(".npz", "")
    out_path = os.path.join(out_dir, f"fea_{case}.npz")
    if os.path.exists(out_path):
        print(f"[{case}] уже посчитано, пропуск")
        return

    geo, meta = load_geometry(geom_path)
    params = sample_params(n_samples, seed)

    disp, kept, infos = [], [], []
    t_total = time.time()
    for i, p in enumerate(params):
        mu, kf, ks, p_lv, p_ratio, k_epi, k_base = p
        mat = Material(mu=mu, kf=kf, ks=ks)
        load = Loading(p_lv=p_lv, p_rv=p_lv * p_ratio, k_epi=k_epi, k_base=k_base)
        t0 = time.time()
        u, info = solve(geo, mat, load, n_steps=n_steps)
        dt = time.time() - t0
        if not info["ok"]:
            print(f"[{case}] {i:3d} расходимость, отброшено")
            continue
        disp.append(u.astype(np.float32))
        kept.append(p)
        infos.append((dt, info["iters"], info["u_max"], info["J_min"]))
        print(
            f"[{case}] {i:3d}/{n_samples} {dt:5.1f}s it={info['iters']:2d} "
            f"u_max={info['u_max']:5.2f} мм J=[{info['J_min']:.3f}..{info['J_max']:.3f}]"
        )

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        out_path,
        case=case,
        params=np.asarray(kept, dtype=np.float32),
        param_keys=np.array(PARAM_KEYS),
        u=np.stack(disp),                     # (S,N,3) мм
        solve_stats=np.asarray(infos, dtype=np.float32),
        coords=meta["coords"],
        elements=meta["elements"],
        f0=meta["f0"],
        s0=meta["s0"],
        elem_tag=meta["elem_tag"],
        uvc_rho=meta["uvc_rho"],
        uvc_z=meta["uvc_z"],
        uvc_v=meta["uvc_v"],
        faces=meta["faces"],
        face_type=meta["face_type"],
    )
    print(f"[{case}] готово: {len(disp)} решений за {time.time() - t_total:.0f} с "
          f"-> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom-dir", default="data/geom")
    ap.add_argument("--out-dir", default="data/fea")
    ap.add_argument("--n-per-geom", type=int, default=24)
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--cases", nargs="*", default=None)
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.geom_dir, "geom_*.npz")))
    if a.cases:
        paths = [p for p in paths
                 if os.path.basename(p).replace("geom_", "").replace(".npz", "")
                 in a.cases]
    for k, p in enumerate(paths):
        run_geometry(p, a.n_per_geom, a.out_dir, a.n_steps, seed=1000 + k)


if __name__ == "__main__":
    main()
