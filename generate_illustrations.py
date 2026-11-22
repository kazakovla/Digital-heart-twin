"""
Иллюстрации к отчёту по практике (версия 2, по замечаниям руководителя).

Все рисунки строятся на РЕАЛЬНОЙ КЭ-сетке бивентрикулярного миокарда
(Rodero et al., 2021, Zenodo 4590294; подготовка -- src/prepare_meshes.py).
Прежняя версия рисовала случайное облако точек (np.random.randn) и не
запускалась из-за импорта несуществующего пакета digital_heart_twin.

Рисунки:
  laplace_fields.png          -- 5 полей Лапласа–Дирихле в сечении сердца
  stratified_neighborhoods.png -- страты SC1..SC5 и выбранные соседи узла
  attention_profile.png       -- веса внимания по стратам: [15] и своя модель
  pipeline_timing.png         -- время КЭА и эмулятора: [15] и свой эксперимент
  architecture_diagram.png    -- схема пяти модулей HeartSimSage
Рис. 2.1 (реальная сетка) строит src/fig_mesh.py, результаты эксперимента --
src/fig_results.py.

Запуск (из корня репозитория):
    python generate_illustrations.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from heartsim_sage import (STRAT_PERCENTILES, build_stratified_neighborhoods,
                           compute_all_laplace_fields, load_real_geometry,
                           mesh_to_graph, pairwise_distance_percentiles)

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "figures")
GEOM = os.path.join(ROOT, "data", "geom", "geom_01.npz")
RESULTS = os.path.join(ROOT, "results", "hss_metrics.json")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
SC_COLORS = ["#1b4f72", "#2e86c1", "#48c9b0", "#f5b041", "#cb4335"]

# Средние веса внимания по стратам, [15], табл. 6 (N_NB = 12)
ATTN_PAPER = [0.159, 0.082, 0.064, 0.027, 0.022]


def anatomical(coords, d):
    """Поворот в анатомические оси: e1 ЛЖ->ПЖ, e3 -- длинная ось."""
    x = coords - coords.mean(0)
    g = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), d["uvc_z"],
                        rcond=None)[0][1:]
    e3 = g / np.linalg.norm(g)
    lv = d["is_lv_node"]
    v = x[~lv].mean(0) - x[lv].mean(0)
    v -= (v @ e3) * e3
    e1 = v / np.linalg.norm(v)
    return x @ np.stack([e1, np.cross(e3, e1), e3]).T


def figure_laplace_fields(xa, lap):
    labels = ["Φ_AB\nбаза = 1, верхушка = 0",
              "Φ_EP\nэпикард = 1, эндокард = 0",
              "Φ_LV-RV\nэндокард ЛЖ = 1, ПЖ = 0",
              "Φ_LV-EP\nэндокард ЛЖ = 1, эпикард = 0",
              "Φ_RV-EP\nэндокард ПЖ = 1, эпикард = 0"]
    slab = np.abs(xa[:, 1] - np.median(xa[:, 1])) < 4.0
    fig, axs = plt.subplots(1, 5, figsize=(17, 4.6))
    for i, ax in enumerate(axs):
        sc = ax.scatter(xa[slab, 0], xa[slab, 2], c=lap[slab, i], cmap="viridis",
                        s=7, vmin=0, vmax=1, linewidths=0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(labels[i], fontsize=9)
    fig.colorbar(sc, ax=axs, fraction=0.015, pad=0.01)
    fig.suptitle("Поля Лапласа–Дирихле на реальной КЭ-сетке "
                 "(сечение в четырёхкамерной проекции, слой ±4 мм)", fontsize=11)
    fig.savefig(os.path.join(FIG_DIR, "laplace_fields.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def figure_stratified_neighborhoods(coords, xa):
    thr = pairwise_distance_percentiles(coords)
    nb, st = build_stratified_neighborhoods(coords, n_nb=12, return_strata=True)
    # узел в середине боковой стенки ЛЖ
    mid = np.abs(xa[:, 2] - np.median(xa[:, 2])) < 5
    node = int(np.where(mid)[0][np.argmin(xa[mid, 0])])
    d = np.linalg.norm(coords - coords[node], axis=1)
    lower = np.concatenate([[0.0], thr[:-1]])

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2),
                            gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    slab = np.abs(xa[:, 1] - xa[node, 1]) < 4.0
    ax.scatter(xa[slab, 0], xa[slab, 2], s=3, c="#d5d8dc", linewidths=0)
    for s in range(5):
        m = slab & (d > lower[s]) & (d <= thr[s])
        ax.scatter(xa[m, 0], xa[m, 2], s=6, c=SC_COLORS[s], linewidths=0,
                   label=f"SC{s + 1}: {lower[s]:.0f}–{thr[s]:.0f} мм")
    ax.scatter(xa[nb[node], 0], xa[nb[node], 2], s=70, facecolors="none",
               edgecolors="k", linewidths=1.2, label="выбранные соседи (12)")
    ax.scatter(*xa[node, [0, 2]], s=120, marker="*", c="k", label="узел i")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title("Страты расстояний вокруг узла стенки ЛЖ\n"
                 "(сечение ±4 мм; соседи вне сечения не показаны)", fontsize=10)

    ax = axs[1]
    share = [np.mean(st == s) for s in range(5)]
    bars = ax.bar([f"SC{s + 1}" for s in range(5)], share, color=SC_COLORS)
    for b, v in zip(bars, share):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                ha="center", fontsize=9)
    ax.set_ylabel("Доля соседей по всем узлам сетки")
    ax.set_title("Распределение выбранных соседей по стратам\n"
                 "(пороги — Q0.2/1/2/5/10% всех попарных расстояний)",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "stratified_neighborhoods.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    return thr


def figure_attention_profile(res):
    labels = [f"SC{s + 1}" for s in range(5)]
    paper = np.array(ATTN_PAPER) * 12          # отношение к равномерному 1/K
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - 0.2, paper, 0.4, color="#7f8c8d", label="HeartSimSage [15], K = 12")
    if res is not None:
        k = res["config"]["k_neighbors"]
        own = np.array(res["attention_by_stratum"]) * k
        ax.bar(x + 0.2, own, 0.4, color="#c0392b",
               label=f"собственная модель (тест, полные сетки), K = {k}")
        for xi, v in zip(x, own):
            ax.text(xi + 0.2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x, paper):
        ax.text(xi - 0.2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.text(4.45, 1.03, "равномерный вес 1/K", ha="right", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Средний вес внимания × K")
    ax.set_xlabel("Страта расстояний (SC1 — ближайшие)")
    ax.set_title("Профиль весов внимания по стратам", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "attention_profile.png"), dpi=200)
    plt.close(fig)


def figure_pipeline_timing(res):
    names = ["КЭА\n[15], CPU", "HSS\n[15], GPU", "HSS\n[15], CPU"]
    times = [112.0, 0.0084, 0.6]
    colors = ["#2471a3", "#27ae60", "#27ae60"]
    if res is not None:
        names += ["КЭА\nсвой, CPU", "HSS\nсвой, CPU"]
        times += [res["fea_solve_s"], res["hss_inference_full_mesh_s"]]
        colors += ["#5dade2", "#82e0aa"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.bar(names, times, color=colors)
    ax.set_yscale("log")
    for b, t in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, t * 1.25, f"{t:.3g} с",
                ha="center", fontsize=9)
    ax.set_ylabel("Время одного решения, с (лог. шкала)")
    ax.set_title("Время прямого расчёта: КЭА и эмулятор HeartSimSage", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pipeline_timing.png"), dpi=200)
    plt.close(fig)


def figure_architecture_diagram():
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.axis("off")
    boxes = [
        (0.02, 0.62, 0.22, 0.24, "1. Кодирование узлов\n(волокна f0, s0 +\nполя Лапласа Φ)"),
        (0.30, 0.62, 0.22, 0.24, "2. Стратифицированные\nокрестности\n(SC1…SC5)"),
        (0.58, 0.62, 0.24, 0.24, "3. Передача сообщений\nс многоголовым\nвниманием"),
        (0.30, 0.12, 0.22, 0.24, "4. Глобальное\nкодирование\n(β, давления, форма)"),
        (0.58, 0.12, 0.24, 0.24, "5. Декодирование\nсмещений\nû_x, û_y, û_z"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                    lw=1.3, ec="#2c3e50", fc="#eaf2f8"))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9.5)
    arrows = [((0.24, 0.74), (0.30, 0.74)), ((0.52, 0.74), (0.58, 0.74)),
              ((0.70, 0.62), (0.70, 0.36)), ((0.52, 0.24), (0.58, 0.24))]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#2c3e50"))
    ax.text(0.5, 0.97, "Архитектура HeartSimSage (по [15])", ha="center",
            fontsize=12, weight="bold")
    fig.savefig(os.path.join(FIG_DIR, "architecture_diagram.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    coords, elements, _, bnd = load_real_geometry(GEOM)
    d = np.load(GEOM)
    xa = anatomical(coords, d)
    ei, _ = mesh_to_graph(coords, elements)
    lap = compute_all_laplace_fields(coords, ei, bnd)
    res = json.load(open(RESULTS, encoding="utf-8")) if os.path.exists(RESULTS) else None
    if res is not None and "attention_by_stratum" not in res:
        res = None          # результаты старой версии эксперимента
    figure_laplace_fields(xa, lap)
    thr = figure_stratified_neighborhoods(coords, xa)
    figure_attention_profile(res)
    figure_pipeline_timing(res)
    figure_architecture_diagram()
    print("Пороги страт Q" + "/".join(f"{p:g}" for p in STRAT_PERCENTILES)
          + "%, мм: " + ", ".join(f"{t:.1f}" for t in thr))
    print(f"Рисунки сохранены в {FIG_DIR}")


if __name__ == "__main__":
    main()
