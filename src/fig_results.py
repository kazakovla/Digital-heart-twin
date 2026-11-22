"""
Рисунки по результатам собственного эксперимента (раздел 3.3 отчёта).

Читает results/hss_metrics.json и results/test_predictions.npz, которые
сохраняет train_hss.py (предсказания на ПОЛНЫХ сетках тестовых сердец).

  figures/fig_3_2_displacement.png -- поле |u|: КЭА, эмулятор, ошибка
                                      (тестовое сердце, не участвовавшее
                                      в обучении; сечение в 4-камерной проекции)
  figures/fig_3_3_accuracy.png     -- (а) предсказание vs КЭА по компонентам,
                                      (б) кривые обучения,
                                      (в) ошибка в зависимости от давления в ЛЖ

Запуск (из каталога src/):
    python fig_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from graph_features import anatomical_frame

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    res = json.loads((ROOT / "results" / "hss_metrics.json").read_text("utf-8"))
    pr = np.load(ROOT / "results" / "test_predictions.npz", allow_pickle=True)
    cases, params, preds = pr["cases"], pr["params"], pr["u_pred"]

    fea = {c: np.load(ROOT / "data" / "fea" / f"fea_{c}.npz") for c in set(cases)}
    keys = list(fea[cases[0]]["param_keys"])
    i_plv = keys.index("p_lv")

    # ---- образцы: истина и предсказание ----------------------------------
    rows, true_all, pred_all = [], [], []
    counter = {}
    for c, p, up in zip(cases, params, preds):
        j = counter.get(c, 0)
        counter[c] = j + 1
        ut = fea[c]["u"][j].astype(np.float64)
        assert np.allclose(fea[c]["params"][j], p)
        up = np.asarray(up, dtype=np.float64)
        rows.append({"case": c, "j": j, "p_lv": float(p[i_plv]),
                     "l2": np.linalg.norm(up - ut) / np.linalg.norm(ut),
                     "umax": np.linalg.norm(ut, axis=1).max()})
        true_all.append(ut)
        pred_all.append(up)

    # =============== Рис. 3.2: поле смещений на тестовом сердце ===========
    l2 = np.array([r["l2"] for r in rows])
    k = int(np.argsort(l2)[len(l2) // 2])             # медианная ошибка
    r = rows[k]
    g = np.load(ROOT / "data" / "geom" / f"geom_{r['case']}.npz")
    X = g["coords"].astype(np.float64)
    R = anatomical_frame(X, g["uvc_z"].astype(np.float64), g["is_lv_node"])
    xa = (X - X.mean(0)) @ R.T
    ut, up = true_all[k], pred_all[k]
    mt, mp = np.linalg.norm(ut, axis=1), np.linalg.norm(up, axis=1)
    err = np.linalg.norm(up - ut, axis=1)
    slab = np.abs(xa[:, 1] - np.median(xa[:, 1])) < 4.0
    vmax = np.percentile(mt, 99)

    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.6))
    for ax, val, title, cm, vm in (
            (axs[0], mt, "КЭА (эталон): |u|, мм", "viridis", vmax),
            (axs[1], mp, "Эмулятор HSS: |û|, мм", "viridis", vmax),
            (axs[2], err, "Ошибка |û − u|, мм", "magma", np.percentile(err, 99))):
        sc = ax.scatter(xa[slab, 0], xa[slab, 2], c=val[slab], cmap=cm, s=9,
                        vmin=0, vmax=vm, linewidths=0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)
        fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle(
        f"Тестовое сердце {r['case']} (не участвовало в обучении), "
        f"p_ЛЖ = {r['p_lv']:.1f} мм рт. ст.; все {len(X)} узлов сетки, "
        f"сечение ±4 мм; отн. ошибка L2 = {r['l2'] * 100:.1f} %", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(ROOT / "figures" / "fig_3_2_displacement.png", dpi=200)
    plt.close(fig)

    # =============== Рис. 3.3: точность и обучение ========================
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
    rng = np.random.default_rng(0)
    ax = axs[0]
    names = ["e1 (ЛЖ→ПЖ)", "e2", "e3 (длинная ось)"]
    cols = ["#c0392b", "#27ae60", "#2471a3"]
    lim = 0
    for t, p, c in zip(true_all, pred_all, cases):
        gg = np.load(ROOT / "data" / "geom" / f"geom_{c}.npz")
        Rc = anatomical_frame(gg["coords"].astype(np.float64),
                              gg["uvc_z"].astype(np.float64), gg["is_lv_node"])
        sel = rng.choice(len(t), 150, replace=False)
        ta, pa = t[sel] @ Rc.T, p[sel] @ Rc.T
        for d in range(3):
            ax.scatter(ta[:, d], pa[:, d], s=2, c=cols[d], alpha=0.35,
                       linewidths=0)
        lim = max(lim, np.abs(ta).max())
    for d in range(3):
        ax.scatter([], [], s=12, c=cols[d], label=names[d])
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax.set_xlabel("КЭА, мм")
    ax.set_ylabel("эмулятор, мм")
    ax.set_title(f"(а) компоненты смещений, тест\nR² = "
                 f"{res['metrics']['test_full']['r2']:.3f}")
    ax.legend(fontsize=7.5, frameon=False, markerscale=1.5)
    ax.set_aspect("equal")

    ax = axs[1]
    h = res["history"]
    ax.plot([e["epoch"] for e in h], [e["train_loss"] for e in h], "o-", ms=3,
            c="#7f8c8d", label="функция потерь (обучение)")
    ax2 = ax.twinx()
    ax2.plot([e["epoch"] for e in h], [e["err_l2_rel"] * 100 for e in h], "s-",
             ms=3, c="#c0392b", label="отн. ошибка L2 (валидация), %")
    ax.set_xlabel("эпоха")
    ax.set_ylabel("Smooth L1")
    ax2.set_ylabel("L2, %", color="#c0392b")
    ax.set_title("(б) обучение")
    l1, n1 = ax.get_legend_handles_labels()
    l2_, n2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2_, n1 + n2, fontsize=7.5, frameon=False)

    ax = axs[2]
    for c, m in zip(sorted(set(cases)), ("o", "s")):
        rr = [x for x in rows if x["case"] == c]
        ax.scatter([x["p_lv"] for x in rr], [x["l2"] * 100 for x in rr],
                   marker=m, s=22, label=f"сердце {c}")
    ax.set_xlabel("давление в ЛЖ, мм рт. ст.")
    ax.set_ylabel("отн. ошибка L2, %")
    ax.set_title("(в) ошибка по тестовым образцам")
    ax.legend(fontsize=8, frameon=False)
    for a in axs:
        a.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig_3_3_accuracy.png", dpi=200)
    plt.close(fig)
    print("сохранено: fig_3_2_displacement.png, fig_3_3_accuracy.png")


if __name__ == "__main__":
    main()
