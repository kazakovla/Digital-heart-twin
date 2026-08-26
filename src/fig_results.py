"""
Рисунки к главе 4: результаты собственного эксперимента на реальных
бивентрикулярных сетках.

Рис. 4.1 — сопоставление поля смещений КЭА и предсказания HSS на тестовом
           сердце, которое не участвовало в обучении: истина, предсказание,
           поле абсолютной погрешности (одна цветовая шкала для первых двух
           панелей, иначе сравнение визуально некорректно).
Рис. 4.2 — сводка точности: диаграмма рассеяния "истина-предсказание",
           распределение поузловой погрешности, кривая обучения и
           столбиковая диаграмма времени решения (КЭА против HSS).

Запуск:
    python fig_results.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap

from fig_mesh import frame, orient, slice_polygons
from graph_features import (build_stratified_neighborhoods,
                            compute_laplace_fields, shape_features)
from hss_model import HSSConfig, HeartSimSage
from train_hss import TEST_CASES, load_geometry_cache, make_globals

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "figure.dpi": 200,
})

C_FEA = "#c0392b"
C_HSS = "#1a5276"
C_GREY = "#5d6d7e"
CM_U = LinearSegmentedColormap.from_list(
    "u", ["#1a3d5c", "#3d8fb5", "#e8dfc8", "#d98b57", "#a63a2a"])
CM_ERR = LinearSegmentedColormap.from_list(
    "e", ["#f5f5f0", "#e8c39e", "#c0392b"])


def load_model(results_dir: Path):
    ck = torch.load(results_dir / "hss_model.pt", weights_only=False)
    cfg = HSSConfig(**ck["cfg"])
    model = HeartSimSage(cfg)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, float(ck["u_scale"])


def predict(model, cache, params, u_scale):
    g = make_globals(torch.from_numpy(params), cache["shape"])
    with torch.no_grad():
        out = model(cache["node_feats"], cache["rel"], cache["strata"],
                    cache["nb_idx"], g)
    return out.numpy() * u_scale


def draw_slice(ax, coords, elements, values, cmap, vmin, vmax, y0):
    """Заливка истинного сечения сетки значениями узлового поля."""
    polys, owners, crossing = slice_polygons(coords, elements, y0)
    el = elements[crossing][owners]
    val = values[el].mean(axis=1)
    norm = plt.Normalize(vmin, vmax)
    ax.add_collection(PolyCollection(polys, facecolors=cmap(norm(val)),
                                     edgecolors="none", antialiaseds=True))
    frame(ax, coords)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def main() -> None:
    ap = argparse.ArgumentParser()
    # Пути привязаны к расположению файла, а не к рабочему каталогу.
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--geom-dir", default=str(root / "data" / "geom"))
    ap.add_argument("--fea-dir", default=str(root / "data" / "fea"))
    ap.add_argument("--results-dir", default=str(root / "results"))
    ap.add_argument("--fig-dir", default=str(root / "figures"))
    args = ap.parse_args()

    geom_dir, fea_dir = Path(args.geom_dir), Path(args.fea_dir)
    res_dir, fig_dir = Path(args.results_dir), Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((res_dir / "hss_metrics.json").read_text("utf-8"))
    model, u_scale = load_model(res_dir)
    k = summary["config"]["k_neighbors"]
    seed = summary["config"]["seed"]
    n_nodes = summary["config"].get("n_nodes", 0)

    # ---- собираем предсказания по всем тестовым образцам ------------------
    rows, per_node_err, pairs = [], [], []
    demo = None
    for case in TEST_CASES:
        path = fea_dir / f"fea_{case}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        cache = load_geometry_cache(geom_dir, case, k, seed, n_nodes)
        ids = cache["node_ids"]          # узлы, на которых работает эмулятор
        for i in range(len(d["params"])):
            u_true = d["u"][i][ids].astype(np.float64)
            u_pred = predict(model, cache, d["params"][i], u_scale)
            err = np.linalg.norm(u_pred - u_true, axis=1)
            rows.append({"case": case,
                         "u_max": float(np.linalg.norm(u_true, axis=1).max()),
                         "mae": float(np.abs(u_pred - u_true).mean()),
                         "l2_rel": float(np.linalg.norm(u_pred - u_true)
                                         / np.linalg.norm(u_true))})
            per_node_err.append(err)
            sel = np.random.default_rng(i).choice(len(u_true), 400, False)
            pairs.append((u_true[sel].ravel(), u_pred[sel].ravel()))
            # для рис. 4.1 берём образец с медианной амплитудой
            if demo is None or abs(rows[-1]["u_max"] - 5.0) < demo["d"]:
                demo = {"d": abs(rows[-1]["u_max"] - 5.0), "case": case,
                        "u_true": u_true, "u_pred": u_pred, "ids": ids,
                        "params": d["params"][i], "elements": d["elements"],
                        "coords": d["coords"], "uvc_z": d["uvc_z"],
                        "tag": d["elem_tag"]}

    per_node_err = np.concatenate(per_node_err)

    # =================== Рис. 4.1 — поля смещений =========================
    coords = demo["coords"].astype(np.float64)
    elements = demo["elements"].astype(np.int64)
    is_lv_node = np.zeros(len(coords), bool)
    lv_el = demo["tag"] == 1
    is_lv_node[elements[lv_el].ravel()] = True

    R = orient(coords, demo["uvc_z"].astype(np.float64), is_lv_node)
    cr = (coords - coords.mean(axis=0)) @ R.T
    y0 = float(np.median(cr[:, 1]))

    # Эмулятор выдаёт значения в прореженном наборе узлов, а сечение строится
    # по полной сетке, поэтому поля переносятся на все узлы интерполяцией по
    # ближайшему узлу графа (для заливки полигонов этого достаточно).
    from scipy.spatial import cKDTree

    ids = demo["ids"]
    near = cKDTree(coords[ids]).query(coords)[1]

    mt_g = np.linalg.norm(demo["u_true"], axis=1)
    mp_g = np.linalg.norm(demo["u_pred"], axis=1)
    mt, mp = mt_g[near], mp_g[near]
    me = np.abs(mp - mt)
    vmax = float(np.percentile(np.concatenate([mt_g, mp_g]), 99.5))

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    sm1 = draw_slice(axes[0], cr, elements, mt, CM_U, 0, vmax, y0)
    axes[0].set_title("(а) КЭА (эталон)", fontsize=9)
    draw_slice(axes[1], cr, elements, mp, CM_U, 0, vmax, y0)
    axes[1].set_title("(б) HSS (предсказание)", fontsize=9)
    sm2 = draw_slice(axes[2], cr, elements, me, CM_ERR, 0,
                     float(np.percentile(me, 99)), y0)
    axes[2].set_title("(в) абсолютная погрешность", fontsize=9)

    cb1 = fig.colorbar(sm1, ax=axes[:2], fraction=0.035, pad=0.02)
    cb1.set_label("$|u|$, мм", fontsize=8.5)
    cb1.ax.tick_params(labelsize=8)
    cb2 = fig.colorbar(sm2, ax=axes[2], fraction=0.07, pad=0.02)
    cb2.set_label("$|u_{HSS} - u_{КЭА}|$, мм", fontsize=8.5)
    cb2.ax.tick_params(labelsize=8)

    p = demo["params"]
    fig.suptitle(
        f"Поле смещений на тестовом сердце (случай {demo['case']}, "
        f"не участвовало в обучении): $p_{{ЛЖ}}$ = {p[3]:.1f} мм рт.ст., "
        f"$\\mu$ = {p[0]:.2f} кПа; макс. смещение {mt.max():.2f} мм",
        fontsize=9, y=1.02)
    out = fig_dir / "fig_4_1_displacement.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"сохранено: {out}")

    # =================== Рис. 4.2 — сводка точности =======================
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.2))

    # (а) истина против предсказания
    ax = axes[0]
    ut = np.concatenate([a for a, _ in pairs])
    up = np.concatenate([b for _, b in pairs])
    ax.scatter(ut, up, s=1.2, c=C_HSS, alpha=0.18, linewidths=0)
    lim = [min(ut.min(), up.min()), max(ut.max(), up.max())]
    ax.plot(lim, lim, color=C_FEA, lw=1.0, ls="--", label="идеальное совпадение")
    r2 = 1 - ((up - ut) ** 2).sum() / ((ut - ut.mean()) ** 2).sum()
    ax.set_xlabel("КЭА, мм", fontsize=8.5)
    ax.set_ylabel("HSS, мм", fontsize=8.5)
    ax.set_title(f"(а) покомпонентные смещения\n$R^2$ = {r2:.4f}", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, lw=0.4)

    # (б) распределение поузловой погрешности
    ax = axes[1]
    ax.hist(per_node_err, bins=70, color=C_HSS, alpha=0.85)
    q = np.percentile(per_node_err, [50, 95])
    for v, ls, lab in ((q[0], "-", "медиана"), (q[1], "--", "95-й перцентиль")):
        ax.axvline(v, color=C_FEA, lw=1.0, ls=ls,
                   label=f"{lab}: {v:.3f} мм")
    ax.set_xlabel("$|u_{HSS} - u_{КЭА}|$, мм", fontsize=8.5)
    ax.set_ylabel("число узлов", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_title("(б) распределение\nпоузловой погрешности", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, lw=0.4)

    # (в) кривая обучения
    ax = axes[2]
    hist = summary["history"]
    ep = [h["epoch"] for h in hist]
    ax.plot(ep, [h["train_loss"] for h in hist], color=C_GREY, lw=1.1,
            label="потеря (обучение)")
    ax.set_xlabel("эпоха", fontsize=8.5)
    ax.set_ylabel("потеря", fontsize=8.5)
    ax.set_yscale("log")
    ax2 = ax.twinx()
    vl = [h.get("err_l2_rel", np.nan) * 100 for h in hist]
    ax2.plot(ep, vl, color=C_FEA, lw=1.1, label="отн. $L_2$ (валидация)")
    ax2.set_ylabel("отн. $L_2$-ошибка, %", fontsize=8.5, color=C_FEA)
    ax2.tick_params(labelsize=8, colors=C_FEA)
    ax.tick_params(labelsize=8)
    ax.set_title("(в) сходимость обучения", fontsize=9)
    ax.grid(alpha=0.25, lw=0.4)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc="upper right")

    # (г) время решения
    ax = axes[3]
    t_fea = summary["fea_solve_s"]
    t_hss = summary["hss_inference_s"]
    ax.bar(["КЭА\n(CPU)", "HSS\n(CPU)"], [t_fea, t_hss],
           color=[C_FEA, C_HSS], width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("время решения, с", fontsize=8.5)
    for i, v in enumerate([t_fea, t_hss]):
        ax.text(i, v * 1.35, f"{v:.3g} с", ha="center", fontsize=8)
    # запас сверху, иначе подпись столбца КЭА наезжает на заголовок панели
    ax.set_ylim(t_hss / 3.0, t_fea * 4.0)
    ax.set_title(f"(г) стоимость решения:\nускорение "
                 f"{t_fea / t_hss:.0f}×", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, lw=0.4, axis="y")

    m = summary["metrics"]["test"]
    fig.suptitle(
        f"Точность эмулятора HSS на невиданных сердцах "
        f"(тест: случаи {', '.join(TEST_CASES)}): "
        f"средняя абс. погрешность {m['mae_mm']:.3f} мм, "
        f"отн. $L_2$-ошибка {m['err_l2_rel'] * 100:.2f} %",
        fontsize=9.5, y=1.03)
    fig.tight_layout()
    out = fig_dir / "fig_4_2_accuracy.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"сохранено: {out}")

    # сводная таблица по тестовым образцам
    tab = {"per_sample": rows,
           "median_node_err_mm": float(np.median(per_node_err)),
           "p95_node_err_mm": float(np.percentile(per_node_err, 95)),
           "r2": float(r2)}
    (res_dir / "test_report.json").write_text(
        json.dumps(tab, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"сохранено: {res_dir / 'test_report.json'}")


if __name__ == "__main__":
    main()
