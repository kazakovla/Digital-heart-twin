"""
Рис. 2.1 — визуализация РЕАЛЬНОЙ конечно-элементной сетки бивентрикулярного
миокарда, построенной по КТ пациента (Zenodo 4590294, Rodero et al., 2021).

Заменяет прежнюю иллюстрацию, где рисовалось случайное облако точек со
случайными рёбрами: морфологически то изображение не соответствовало миокарду
и не могло служить иллюстрацией графа КЭ-сетки.

Панели:
  (а) поверхность миокарда, камеры ЛЖ / ПЖ разными цветами;
  (б) истинное плоское сечение (четырёхкамерная проекция) — видна толщина
      стенок и полости желудочков;
  (в) направления мышечных волокон f0 на эпикарде;
  (г) фрагмент графа сетки в толще стенки ЛЖ: узлы и рёбра тетраэдров.

Ориентация задаётся не через SVD облака точек, а по универсальным
желудочковым координатам (UVC) исходного датасета: длинная ось сердца — это
направление роста продольной координаты Z (от верхушки к базе), а вектор
"центр ЛЖ -> центр ПЖ" укладывается в плоскость изображения, чтобы обе
камеры были видны одновременно.

Запуск:
    python fig_mesh.py --case 01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "figure.dpi": 200,
})

CM_LV = LinearSegmentedColormap.from_list("lv", ["#5b1a13", "#c0392b", "#f0a89f"])
CM_RV = LinearSegmentedColormap.from_list("rv", ["#123b56", "#2874a6", "#a2c8e3"])
C_EDGE = "#4d5f6d"
C_FIBER = "#14456b"


def orient(coords: np.ndarray, uvc_z: np.ndarray,
           is_lv_node: np.ndarray) -> np.ndarray:
    """
    Матрица поворота в анатомические оси.

    e3 — длинная ось (верхушка -> база), найдена как направление наибольшего
         роста UVC-координаты Z методом наименьших квадратов;
    e1 — направление ЛЖ -> ПЖ в плоскости, ортогональной e3;
    e2 — дополняет до правой тройки (ось обзора).

    Рисуем в плоскости (e1, e3): это стандартная четырёхкамерная проекция,
    где оба желудочка видны рядом.
    """
    ctr = coords.mean(axis=0)
    x = coords - ctr
    # grad Z по МНК: Z ~ a + g.x  =>  e3 = g / |g|
    a = np.column_stack([np.ones(len(x)), x])
    g = np.linalg.lstsq(a, uvc_z, rcond=None)[0][1:]
    e3 = g / np.linalg.norm(g)

    lv_c = x[is_lv_node].mean(axis=0)
    rv_c = x[~is_lv_node].mean(axis=0)
    d = rv_c - lv_c
    d -= (d @ e3) * e3                      # проекция на плоскость, |_ e3
    e1 = d / np.linalg.norm(d)
    e2 = np.cross(e3, e1)
    return np.stack([e1, e2, e3])


def shade(coords: np.ndarray, tris: np.ndarray,
          light=np.array([0.35, -0.85, 0.4])) -> np.ndarray:
    """Ламбертово затенение по нормали грани — придаёт поверхности объём."""
    p = coords[tris]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return 0.3 + 0.7 * np.abs(n @ (light / np.linalg.norm(light)))


def paint(ax, coords, tris, values, cmap, vmin=0.25, vmax=1.0):
    """Отрисовка поверхности как полигонов с сортировкой по глубине (ось e2)."""
    order = np.argsort(-coords[tris][:, :, 1].mean(axis=1))
    tris, values = tris[order], values[order]
    polys = coords[tris][:, :, [0, 2]]
    norm = plt.Normalize(vmin, vmax)
    ax.add_collection(PolyCollection(polys, facecolors=cmap(norm(values)),
                                     edgecolors="none", antialiaseds=True))


def slice_polygons(coords: np.ndarray, elements: np.ndarray, y0: float):
    """
    Истинное сечение тетраэдральной сетки плоскостью e2 = y0.

    Для каждого пересечённого тетраэдра находятся точки пересечения его рёбер
    с плоскостью (3 или 4 точки), которые упорядочиваются по углу и образуют
    выпуклый полигон. Это даёт чистый срез миокарда без наложения граней,
    в отличие от простого отбрасывания половины элементов.
    """
    s = coords[:, 1] - y0
    sv = s[elements]
    crossing = (sv > 0).any(axis=1) & (sv < 0).any(axis=1)
    el = elements[crossing]

    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    polys, owners = [], []
    for ei, e in enumerate(el):
        pts = []
        for a, b in edges:
            sa, sb = s[e[a]], s[e[b]]
            if sa * sb < 0:
                t = sa / (sa - sb)
                pts.append(coords[e[a]] + t * (coords[e[b]] - coords[e[a]]))
        if len(pts) < 3:
            continue
        p = np.array(pts)[:, [0, 2]]
        c = p.mean(axis=0)
        p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
        polys.append(p)
        owners.append(ei)
    return polys, np.asarray(owners), crossing


def frame(ax, coords, pad=0.03):
    """Одинаковые границы и равный масштаб осей для всех панелей."""
    x, z = coords[:, 0], coords[:, 2]
    px, pz = pad * np.ptp(x), pad * np.ptp(z)
    ax.set_xlim(x.min() - px, x.max() + px)
    ax.set_ylim(z.min() - pz, z.max() + pz)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="01")
    # Пути привязаны к расположению файла, а не к рабочему каталогу.
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--geom-dir", default=str(root / "data" / "geom"))
    ap.add_argument("--out", default=str(root / "figures" / "fig_2_1_mesh.png"))
    args = ap.parse_args()

    d = np.load(Path(args.geom_dir) / f"geom_{args.case}.npz")
    coords = d["coords"].astype(np.float64)
    elements = d["elements"].astype(np.int64)
    faces = d["faces"].astype(np.int64)
    face_type = d["face_type"]
    is_lv_node = d["is_lv_node"]
    is_lv_el = d["elem_tag"] == 1

    R = orient(coords, d["uvc_z"].astype(np.float64), is_lv_node)
    coords = (coords - coords.mean(axis=0)) @ R.T

    # волокна заданы поэлементно -> переносим в узлы для отрисовки
    f0_el = d["f0"].astype(np.float64) @ R.T
    f0 = np.zeros_like(coords)
    np.add.at(f0, elements.ravel(), np.repeat(f0_el, 4, axis=0))
    f0 /= np.maximum(np.linalg.norm(f0, axis=1, keepdims=True), 1e-12)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(11.4, 3.6))

    # ---- (а) поверхность миокарда ----------------------------------------
    ax = axes[0]
    surf = faces
    face_lv = is_lv_node[surf].mean(axis=1) > 0.5
    sh = shade(coords, surf)
    for mask, cm in ((face_lv, CM_LV), (~face_lv, CM_RV)):
        if mask.any():
            paint(ax, coords, surf[mask], sh[mask], cm)
    frame(ax, coords)
    ax.set_title("(а) поверхность миокарда:\nЛЖ (красный), ПЖ (синий)",
                 fontsize=8.5)

    # ---- (б) истинное плоское сечение ------------------------------------
    ax = axes[1]
    y0 = float(np.median(coords[:, 1]))
    polys, owners, crossing = slice_polygons(coords, elements, y0)
    lv_cut = is_lv_el[crossing][owners]
    for mask, cm in ((lv_cut, CM_LV), (~lv_cut, CM_RV)):
        sel = [p for p, m in zip(polys, mask) if m]
        if sel:
            ax.add_collection(PolyCollection(
                sel, facecolors=cm(0.55), edgecolors="#ffffff",
                linewidths=0.12, antialiaseds=True))
    frame(ax, coords)
    ax.set_title("(б) сечение: толщина стенок\nи полости желудочков",
                 fontsize=8.5)

    # ---- (в) направления волокон ------------------------------------------
    ax = axes[2]
    epi = faces[face_type == 0]
    paint(ax, coords, epi, shade(coords, epi),
          LinearSegmentedColormap.from_list("g", ["#8d9aa4", "#d8dfe4"]))
    epi_nodes = np.unique(epi)
    # рисуем волокна только на видимой (передней) половине эпикарда
    epi_nodes = epi_nodes[coords[epi_nodes, 1] < y0]
    rng = np.random.default_rng(0)
    sel = rng.choice(epi_nodes, size=min(900, len(epi_nodes)), replace=False)
    p = coords[sel][:, [0, 2]]
    v = f0[sel][:, [0, 2]]
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    L = 2.6
    ax.add_collection(LineCollection(
        np.stack([p - v * L / 2, p + v * L / 2], axis=1),
        colors=C_FIBER, linewidths=0.55, alpha=0.9))
    frame(ax, coords)
    ax.set_title("(в) направления мышечных\nволокон $f_0$ на эпикарде",
                 fontsize=8.5)

    # ---- (г) фрагмент графа сетки в толще стенки ЛЖ -----------------------
    # Берётся ТОНКИЙ слой элементов (толщиной порядка одного элемента) в
    # окрестности точки на стенке ЛЖ. Проецирование трёхмерного шара дало бы
    # наложение рёбер и визуально неотличимое от случайного облако точек;
    # тонкий слой сохраняет читаемую структуру тетраэдральной сетки.
    ax = axes[3]
    ctr_el = coords[elements].mean(axis=1)
    zq = np.quantile(coords[:, 2], [0.35, 0.6])
    cand = np.where(is_lv_node & (coords[:, 2] > zq[0]) & (coords[:, 2] < zq[1])
                    & (np.abs(coords[:, 1] - y0) < 3.0))[0]
    seed = int(cand[np.argmin(coords[cand, 0])])
    p0 = coords[seed]

    # характерный размер элемента -> толщина слоя
    h = np.cbrt(np.abs(np.linalg.det(
        (coords[elements[:, 1:]] - coords[elements[:, :1]]).transpose(0, 1, 2)
    )) / 6.0).mean()
    win, slab = 16.0, 1.6 * h

    in_slab = (np.abs(ctr_el[:, 1] - p0[1]) < slab) & (
        np.linalg.norm(ctr_el[:, [0, 2]] - p0[[0, 2]], axis=1) < win)
    el_loc = elements[in_slab]

    # грани элементов слоя — светлая заливка, чтобы был виден сам "объём"
    tri = np.concatenate([el_loc[:, [0, 1, 2]], el_loc[:, [0, 1, 3]],
                          el_loc[:, [0, 2, 3]], el_loc[:, [1, 2, 3]]])
    ax.add_collection(PolyCollection(coords[tri][:, :, [0, 2]],
                                     facecolors="#dfe6ea", edgecolors="none",
                                     alpha=0.5, zorder=1))
    pairs = np.unique(np.sort(np.concatenate(
        [el_loc[:, [0, 1]], el_loc[:, [0, 2]], el_loc[:, [0, 3]],
         el_loc[:, [1, 2]], el_loc[:, [1, 3]], el_loc[:, [2, 3]]]),
        axis=1), axis=0)
    ax.add_collection(LineCollection(coords[pairs][:, :, [0, 2]],
                                     colors=C_EDGE, linewidths=0.5, alpha=0.9,
                                     zorder=2))
    nodes = np.unique(el_loc)
    ax.scatter(coords[nodes, 0], coords[nodes, 2], s=5.0, c="#1b2631",
               zorder=3, linewidths=0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"(г) слой КЭ-сетки в стенке ЛЖ:\n"
                 f"{len(nodes)} узлов, {len(pairs)} рёбер, "
                 f"{len(el_loc)} тетраэдров", fontsize=8.5)

    fig.suptitle(
        f"Реальная КЭ-сетка бивентрикулярного миокарда по КТ пациента "
        f"(Zenodo 4590294, случай {args.case}): "
        f"{len(coords)} узлов, {len(elements)} тетраэдров",
        fontsize=9.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, bbox_inches="tight")
    print(f"сохранено: {out}")


if __name__ == "__main__":
    main()
