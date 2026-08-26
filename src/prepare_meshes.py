"""
Шаг 1. Подготовка реальных бивентрикулярных КЭ-сеток.

Источник данных: открытый датасет
    Rodero C. et al. "Virtual cohort of adult healthy four-chamber hearts
    for hemodynamic and electrophysiological studies" (2021),
    Zenodo, record 4590294, CC-BY-4.0.
    https://zenodo.org/records/4590294

Каждый случай -- тетраэдральная сетка целого сердца (~0.5 млн узлов,
~2.5 млн элементов), построенная по КТ реального пациента, с полями:
    ID       (cell)  -- метка анатомической области (1=ЛЖ, 2=ПЖ, 3=ЛП, 4=ПП, ...)
    fibres   (cell)  -- направление волокна f0 (правило Bayer/UVC)
    sheets   (cell)  -- направление слоя s0
    RHO/PHI/Z/V (point) -- универсальные вентрикулярные координаты (UVC)

Что делает скрипт:
  1. читает case.vtk;
  2. выделяет бивентрикулярный миокард (ID in {1,2});
  3. извлекает его замкнутую поверхность и равномерно её перестраивает
     (ACVD-кластеризация) -- огрубление до вычислительно посильного размера;
  4. заново тетраэдризует объём (TetGen) -> сетка ~7-8 тыс. узлов,
     ~28 тыс. элементов (порядок, использованный в HeartSimSage: 15-20 тыс.
     узлов; здесь мельче из-за ограничений CPU-среды);
  5. переносит на новую сетку РЕАЛЬНЫЕ поля исходного датасета:
     f0, s0 (по ближайшему элементу), RHO/Z/V (по ближайшему узлу),
     метку ЛЖ/ПЖ;
  6. классифицирует граничные грани: Гendo-lv, Гendo-rv, Гepi, Гbase;
  7. сохраняет geom_XX.npz.

Запуск:
    python src/prepare_meshes.py --cases 01 02 03 --raw-dir /tmp/heart
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tarfile
import warnings

import numpy as np

warnings.filterwarnings("ignore")

ZENODO_URL = "https://zenodo.org/api/records/4590294/files/{case}.tar.gz/content"

TAG_LV = 1  # миокард левого желудочка
TAG_RV = 2  # миокард правого желудочка

N_CLUSTERS = 6000  # число вершин огрублённой поверхности (-> ~7.5k узлов объёма)


# ---------------------------------------------------------------------------
# загрузка исходного файла
# ---------------------------------------------------------------------------
def download_case(case: str, raw_dir: str) -> str:
    vtk_path = os.path.join(raw_dir, f"{case}.vtk")
    if os.path.exists(vtk_path):
        return vtk_path
    os.makedirs(raw_dir, exist_ok=True)
    tgz = os.path.join(raw_dir, f"{case}.tar.gz")
    print(f"[{case}] загрузка с Zenodo ...")
    subprocess.run(
        ["curl", "-sL", "--max-time", "1800", "-o", tgz, ZENODO_URL.format(case=case)],
        check=True,
    )
    with tarfile.open(tgz) as tf:
        tf.extractall(raw_dir)
    os.remove(tgz)
    return vtk_path


# ---------------------------------------------------------------------------
# геометрия: границы тетраэдральной сетки
# ---------------------------------------------------------------------------
def boundary_faces(elements: np.ndarray) -> np.ndarray:
    """Грани, встречающиеся ровно один раз, -- граница области."""
    f = np.concatenate(
        [
            elements[:, [0, 2, 1]],
            elements[:, [0, 1, 3]],
            elements[:, [0, 3, 2]],
            elements[:, [1, 2, 3]],
        ],
        axis=0,
    )
    key = np.sort(f, axis=1)
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    return f[idx[cnt == 1]]


def outward_orient(coords: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Ориентирует граничные треугольники наружу (по знаку объёма)."""
    p0, p1, p2 = coords[faces[:, 0]], coords[faces[:, 1]], coords[faces[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    c = (p0 + p1 + p2) / 3.0
    centre = coords.mean(axis=0)
    flip = np.einsum("ij,ij->i", n, c - centre) < 0
    out = faces.copy()
    out[flip] = out[flip][:, [0, 2, 1]]
    return out


# ---------------------------------------------------------------------------
# основной конвейер для одного случая
# ---------------------------------------------------------------------------
def prepare_case(case: str, raw_dir: str, out_dir: str,
                 n_clusters: int = N_CLUSTERS) -> str:
    import pyacvd
    import pyvista as pv
    import tetgen
    from scipy.spatial import cKDTree

    vtk_path = download_case(case, raw_dir)
    print(f"[{case}] чтение {vtk_path}")
    mesh = pv.read(vtk_path)

    # --- 1. бивентрикулярный миокард -------------------------------------
    biv = mesh.threshold(value=(TAG_LV, TAG_RV), scalars="ID")
    print(f"[{case}] исходный миокард: {biv.n_points} узлов, {biv.n_cells} элементов")

    src_pts = np.asarray(biv.points, dtype=np.float64)
    src_rho = np.asarray(biv.point_data["RHO.dat"], dtype=np.float64)
    src_z = np.asarray(biv.point_data["Z.dat"], dtype=np.float64)
    src_v = np.asarray(biv.point_data["V.dat"], dtype=np.float64)

    cell_centres = np.asarray(biv.cell_centers().points, dtype=np.float64)
    src_f0 = np.asarray(biv.cell_data["fibres"], dtype=np.float64)
    src_s0 = np.asarray(biv.cell_data["sheets"], dtype=np.float64)
    src_tag = np.asarray(biv.cell_data["ID"], dtype=np.int32)

    # --- 2. огрубление поверхности ---------------------------------------
    surf = biv.extract_surface(algorithm="dataset_surface").triangulate().clean()
    surf = surf.smooth(n_iter=30, relaxation_factor=0.1)  # снятие ступенек воксельной сегментации
    clus = pyacvd.Clustering(surf)
    clus.cluster(n_clusters)
    remesh = clus.create_mesh().clean().triangulate()

    # --- 3. новая объёмная сетка -----------------------------------------
    tg = tetgen.TetGen(remesh)
    tg.tetrahedralize(order=1, mindihedral=18, minratio=1.3)
    coords = np.asarray(tg.node, dtype=np.float64)
    elements = np.asarray(tg.elem, dtype=np.int64)
    print(f"[{case}] огрублённая сетка: {len(coords)} узлов, {len(elements)} элементов")

    # --- 4. перенос реальных полей ---------------------------------------
    tree_pts = cKDTree(src_pts)
    _, ip = tree_pts.query(coords, k=1)
    rho, zc, vc = src_rho[ip], src_z[ip], src_v[ip]

    tree_cells = cKDTree(cell_centres)
    new_centres = coords[elements].mean(axis=1)
    _, ic = tree_cells.query(new_centres, k=1)
    f0 = src_f0[ic]
    s0 = src_s0[ic]
    tag = src_tag[ic]

    f0 /= np.linalg.norm(f0, axis=1, keepdims=True) + 1e-12
    s0 -= np.einsum("ij,ij->i", s0, f0)[:, None] * f0  # ортогонализация Грама-Шмидта
    s0 /= np.linalg.norm(s0, axis=1, keepdims=True) + 1e-12

    # узловая метка ЛЖ/ПЖ: голосование элементов
    node_lv = np.zeros(len(coords))
    node_cnt = np.zeros(len(coords))
    np.add.at(node_lv, elements.ravel(), np.repeat(tag == TAG_LV, 4).astype(float))
    np.add.at(node_cnt, elements.ravel(), 1.0)
    is_lv_node = (node_lv / np.maximum(node_cnt, 1)) > 0.5

    # --- 5. классификация границы ----------------------------------------
    faces = outward_orient(coords, boundary_faces(elements))
    f_rho = rho[faces].mean(axis=1)
    f_z = zc[faces].mean(axis=1)
    f_v = vc[faces].mean(axis=1)
    f_lv = is_lv_node[faces].mean(axis=1)

    # плоскость клапанного основания: Z (UVC) ~ 1 у базы, 0 у верхушки
    is_base = f_z > 0.94
    is_endo = (~is_base) & (f_rho < 0.5)
    is_epi = (~is_base) & (~is_endo)
    # ЛЖ/ПЖ эндокард: UVC V = -1 (ЛЖ) / +1 (ПЖ), подстраховка меткой области
    endo_lv = is_endo & ((f_v < 0) | (f_lv > 0.5))
    endo_rv = is_endo & ~endo_lv

    face_type = np.zeros(len(faces), dtype=np.int8)  # 0=epi,1=endo_lv,2=endo_rv,3=base
    face_type[endo_lv] = 1
    face_type[endo_rv] = 2
    face_type[is_base] = 3

    print(
        f"[{case}] границы: epi={int(is_epi.sum())} endoLV={int(endo_lv.sum())} "
        f"endoRV={int(endo_rv.sum())} base={int(is_base.sum())}"
    )

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"geom_{case}.npz")
    np.savez_compressed(
        out,
        case=case,
        coords=coords.astype(np.float32),      # мм
        elements=elements.astype(np.int32),
        f0=f0.astype(np.float32),
        s0=s0.astype(np.float32),
        elem_tag=tag.astype(np.int8),          # 1=ЛЖ, 2=ПЖ
        uvc_rho=rho.astype(np.float32),
        uvc_z=zc.astype(np.float32),
        uvc_v=vc.astype(np.float32),
        is_lv_node=is_lv_node,
        faces=faces.astype(np.int32),
        face_type=face_type,
        n_src_points=biv.n_points,
        n_src_cells=biv.n_cells,
    )
    print(f"[{case}] сохранено -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["01", "02", "03", "04", "05"])
    ap.add_argument("--raw-dir", default="/tmp/heart")
    ap.add_argument("--out-dir", default="data/geom")
    ap.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    a = ap.parse_args()
    for c in a.cases:
        try:
            prepare_case(c, a.raw_dir, a.out_dir, a.n_clusters)
        except Exception as exc:  # noqa: BLE001
            print(f"[{c}] ОШИБКА: {exc}")


if __name__ == "__main__":
    main()
