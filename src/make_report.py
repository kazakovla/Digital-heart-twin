"""
Формирует сводку эксперимента results/ОТЧЁТ.md (числа для раздела 3.3 отчёта).

Все значения берутся из результатов обучения (results/hss_metrics.json) и из
самих наборов данных, поэтому отчёт нельзя рассинхронизировать с расчётом.

Запуск:
    python make_report.py
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def fmt(x: float, n: int = 2) -> str:
    return f"{x:.{n}f}".replace("nan", "—")


def main() -> None:
    ap = argparse.ArgumentParser()
    # Пути привязаны к расположению файла, а не к рабочему каталогу.
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--geom-dir", default=str(root / "data" / "geom"))
    ap.add_argument("--fea-dir", default=str(root / "data" / "fea"))
    ap.add_argument("--results-dir", default=str(root / "results"))
    args = ap.parse_args()

    res = Path(args.results_dir)
    s = json.loads((res / "hss_metrics.json").read_text("utf-8"))

    # ---- статистика по геометриям -----------------------------------------
    g_rows = []
    for p in sorted(glob.glob(f"{args.geom_dir}/geom_*.npz")):
        d = np.load(p)
        c, e = d["coords"], d["elements"]
        # Объём тетраэдра: |det[v1-v0, v2-v0, v3-v0]| / 6. Матрица рёбер
        # собирается как (число элементов, 3, 3), поэтому вершина v0 вычитается
        # с сохранением оси элементов.
        v = np.abs(np.linalg.det(c[e[:, 1:]] - c[e[:, 0]][:, None, :])) / 6.0
        g_rows.append({"case": Path(p).stem[-2:], "n": len(c), "e": len(e),
                       "vol": v.sum() / 1000.0})

    # ---- статистика по решениям КЭА ---------------------------------------
    f_rows = []
    for p in sorted(glob.glob(f"{args.fea_dir}/fea_*.npz")):
        d = np.load(p)
        mag = np.linalg.norm(d["u"], axis=2)
        st = d["solve_stats"]
        f_rows.append({"case": Path(p).stem[-2:], "n": len(d["params"]),
                       "u_mean": mag.mean(), "u_max": mag.max(),
                       "t": st[:, 0].mean(), "it": st[:, 1].mean(),
                       "dv_lv": st[:, 4].mean() * 100, "dv_rv": st[:, 5].mean() * 100})

    m = s["metrics"]
    cfg = s["config"]
    lines: list[str] = []
    add = lines.append

    add("# Отчёт по эксперименту на реальных КЭ-сетках\n")
    add("Сформирован автоматически из `results/hss_metrics.json`.\n")

    add("## 1. Подготовленные геометрии\n")
    add("| Случай | Узлов | Тетраэдров | Объём миокарда, мл |")
    add("|---|---|---|---|")
    for r in g_rows:
        add(f"| {r['case']} | {r['n']} | {r['e']} | {fmt(r['vol'], 0)} |")
    add(f"\nВсего геометрий: **{len(g_rows)}**, "
        f"в среднем {np.mean([r['n'] for r in g_rows]):.0f} узлов и "
        f"{np.mean([r['e'] for r in g_rows]):.0f} тетраэдров.\n")

    add("## 2. Набор конечно-элементных решений\n")
    add("| Случай | Решений | Среднее \\|u\\|, мм | Макс. \\|u\\|, мм | "
        "Время КЭА, с | Итераций Ньютона | ΔV ЛЖ, % | ΔV ПЖ, % |")
    add("|---|---|---|---|---|---|---|---|")
    for r in f_rows:
        add(f"| {r['case']} | {r['n']} | {fmt(r['u_mean'])} | "
            f"{fmt(r['u_max'])} | {fmt(r['t'], 1)} | {fmt(r['it'], 1)} | "
            f"{fmt(r['dv_lv'], 1)} | {fmt(r['dv_rv'], 1)} |")
    tot = sum(r["n"] for r in f_rows)
    add(f"\nВсего решений: **{tot}** "
        f"(обучение {s['n_samples']['train']}, валидация "
        f"{s['n_samples']['val']}, тест {s['n_samples']['test']}).\n")

    add("## 3. Разбиение по пациентам\n")
    add(f"- обучение: {', '.join(s['split']['train'])}")
    add(f"- валидация: {', '.join(s['split']['val'])}")
    add(f"- тест: {', '.join(s['split']['test'])}\n")
    add("Тестовые сердца не встречаются в обучении ни в одном образце.\n")

    add("## 4. Конфигурация эмулятора\n")
    add(f"- скрытая размерность: {cfg['hidden']}")
    add(f"- слоёв стратифицированного внимания: {cfg['layers']}")
    add(f"- соседей в окрестности (K): {cfg['k_neighbors']}")
    add(f"- узлов в графе на сердце: {cfg['n_nodes']}")
    add(f"- обучаемых параметров: {s['n_params']:,}".replace(",", " "))
    add(f"- эпох: {cfg['epochs']}, время обучения "
        f"{s['train_time_s'] / 60:.1f} мин (CPU)\n")

    add("## 5. Точность\n")
    add("| Выборка | MAE, мм | Ср. ошибка в узле, мм | Отн. погрешность $L_2$, % "
        "| Err_mean, % | Макс. ошибка в узле, мм |")
    add("|---|---|---|---|---|---|")
    for name, key in (("Обучение (прореж. граф)", "train"),
                      ("Валидация (прореж. граф)", "val"),
                      ("Тест, прореж. граф", "test_subgraph"),
                      ("Тест, полные сетки", "test_full")):
        r = m[key]
        add(f"| {name} | {fmt(r['mae_mm'], 3)} | {fmt(r['node_err_mm'], 3)} | "
            f"{fmt(r['err_l2_rel'] * 100)} | {fmt(r['err_mean'] * 100)} | "
            f"{fmt(r['max_err_mm'], 2)} |")
    t = m["test_full"]
    add(f"\n- коэффициент детерминации (тест, полные сетки): **$R^2$ = {fmt(t['r2'], 3)}**")
    add(f"- поузловая ошибка: медиана **{fmt(t['node_err_median_mm'], 3)} мм**, "
        f"95-й перцентиль **{fmt(t['node_err_p95_mm'], 3)} мм** при среднем |u| = "
        f"{fmt(t['u_mean_mm'])} мм\n")
    add("- средний вес внимания по стратам SC1..SC5 (× K): " + ", ".join(
        fmt(a * cfg["k_neighbors"]) for a in s["attention_by_stratum"]) + "\n")

    add("## 6. Время решения (CPU, одна и та же полная сетка)\n")
    add(f"- КЭА: **{fmt(s['fea_solve_s'], 1)} с** на задачу")
    add(f"- эмулятор HSS: **{fmt(s['hss_inference_full_mesh_s'] * 1000, 0)} мс** "
        f"на задачу ({s['n_nodes_test_mesh']} узлов)")
    add(f"- ускорение: **{fmt(s['speedup_cpu'], 0)}x**\n")

    out = res / "ОТЧЁТ.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"сохранено: {out}")
    print("\n".join(lines[-12:]))


if __name__ == "__main__":
    main()
