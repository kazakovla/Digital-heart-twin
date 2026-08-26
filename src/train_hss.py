"""
Обучение эмулятора HeartSimSage на датасете КЭ-решений, полученных
на реальных бивентрикулярных сетках (Zenodo 4590294).

Ключевой момент методики: разбиение выполняется ПО ГЕОМЕТРИЯМ, а не по
образцам. Обучающие и тестовые сердца не пересекаются, поэтому тестовая
ошибка измеряет способность модели переноситься на анатомию, которую она
никогда не видела (индуктивное обобщение из раздела 3.2 НИР). Разбиение по
случайным образцам дало бы оптимистично смещённую оценку, так как модель
видела бы ту же сетку при других давлениях.

Запуск:
    python train_hss.py --epochs 400
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from graph_features import (build_stratified_neighborhoods,
                            compute_laplace_fields, shape_features)
from hss_model import HSSConfig, HeartSimSage

# Разбиение по геометриям (пациентам), фиксировано для воспроизводимости
TRAIN_CASES = ["01", "02", "03", "04", "05", "06", "07"]
VAL_CASES = ["08", "09"]
TEST_CASES = ["10", "11"]


def load_geometry_cache(geom_dir: Path, case: str, k_neighbors: int, seed: int,
                        n_nodes: int = 0):
    """
    Готовит все величины, зависящие только от геометрии (считаются один раз
    на сердце): признаки узлов, окрестности, признаки формы.

    Параметр n_nodes прореживает граф: эмулятор предсказывает поле смещений
    в фиксированном подмножестве узлов сетки, а стратифицированные окрестности
    строятся внутри этого подмножества. Поля Лапласа и признаки формы при этом
    считаются на ПОЛНОЙ сетке (они требуют конечно-элементной дискретизации),
    и лишь затем сужаются на выбранные узлы. Прореживание нужно потому, что
    рецептивное поле четырёхслойной модели покрывает почти весь граф, и
    подвыборка узлов внутри forward не даёт экономии.
    """
    d = np.load(geom_dir / f"geom_{case}.npz")
    coords = d["coords"].astype(np.float64)
    elements = d["elements"].astype(np.int64)
    faces = d["faces"].astype(np.int64)
    face_type = d["face_type"]

    laplace = compute_laplace_fields(coords, elements, faces, face_type)
    shape = shape_features(coords, elements, faces, face_type, laplace)

    # Волокна (f0, s0) заданы ПОЭЛЕМЕНТНО, а модель работает с узловыми
    # признаками; переносим их в узлы усреднением по инцидентным тетраэдрам.
    # Это делается ДО прореживания, пока индексация elements ещё валидна.
    def to_nodes(vec_el: np.ndarray) -> np.ndarray:
        acc = np.zeros_like(coords)
        np.add.at(acc, elements.ravel(), np.repeat(vec_el, 4, axis=0))
        return acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)

    f0_n = to_nodes(d["f0"].astype(np.float64))
    s0_n = to_nodes(d["s0"].astype(np.float64))

    # Прореживание: сохраняются все узлы поверхности (эндокард и эпикард важны
    # для оценки объёмов камер), внутренние узлы добираются случайно.
    if n_nodes and n_nodes < len(coords):
        rng = np.random.default_rng(seed)
        on_surf = np.zeros(len(coords), dtype=bool)
        on_surf[np.unique(faces)] = True
        surf_ids = np.where(on_surf)[0]
        inner_ids = np.where(~on_surf)[0]
        if len(surf_ids) >= n_nodes:
            keep = rng.choice(surf_ids, n_nodes, replace=False)
        else:
            keep = np.concatenate(
                [surf_ids, rng.choice(inner_ids, n_nodes - len(surf_ids),
                                      replace=False)])
        node_ids = np.sort(keep)
        coords = coords[node_ids]
        laplace = laplace[node_ids]
        f0_n, s0_n = f0_n[node_ids], s0_n[node_ids]
    else:
        node_ids = np.arange(len(coords))

    nb_idx, strata = build_stratified_neighborhoods(coords, k_neighbors, seed)

    # Координаты центрируются и приводятся к безразмерному виду делением на
    # характерный размер сердца, иначе абсолютные миллиметры доминируют.
    ctr = coords.mean(axis=0)
    scale = np.linalg.norm(coords - ctr, axis=1).max()
    coords_n = (coords - ctr) / scale

    node_feats = np.concatenate(
        [f0_n.astype(np.float32),
         s0_n.astype(np.float32),
         laplace.astype(np.float32),
         coords_n.astype(np.float32)],
        axis=1,
    )

    rel = (coords_n[nb_idx] - coords_n[:, None, :]).astype(np.float32)

    return {
        "node_feats": torch.from_numpy(node_feats),
        "rel": torch.from_numpy(rel),
        "strata": torch.from_numpy(strata.astype(np.int64)),
        "nb_idx": torch.from_numpy(nb_idx.astype(np.int64)),
        "shape": torch.from_numpy(shape.astype(np.float32)),
        "scale": float(scale),
        "n_nodes": len(coords),
        "node_ids": node_ids,          # индексы в полной сетке
    }


def load_samples(fea_dir: Path, cases: list[str], cache: dict):
    """
    Загружает КЭ-решения указанных геометрий. Целевые смещения сужаются на
    подмножество узлов, на котором работает эмулятор (см. load_geometry_cache).
    """
    out = []
    for case in cases:
        path = fea_dir / f"fea_{case}.npz"
        if not path.exists() or case not in cache:
            continue
        d = np.load(path)
        params, disp = d["params"], d["u"]
        ids = cache[case]["node_ids"]
        for i in range(len(params)):
            out.append({"case": case,
                        "params": params[i].astype(np.float32),
                        "u": disp[i][ids].astype(np.float32)})
    return out


def make_globals(params: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """
    Вектор глобальных условий: параметры материала и нагрузки (в лог-шкале,
    так как задаются логарифмически равномерно) плюс признаки формы.
    """
    return torch.cat([torch.log(params.clamp(min=1e-6)), params, shape])


def relative_errors(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    Метрики из Табл. 4.1 НИР: покомпонентная ошибка, нормированная на
    максимальную амплитуду истинного смещения, плюс среднее по норме.
    """
    scale = np.maximum(np.abs(target).max(axis=0), 1e-8)
    rel = np.abs(pred - target) / scale
    err_norm = (np.linalg.norm(pred - target, axis=1).mean()
                / max(np.linalg.norm(target, axis=1).mean(), 1e-12))
    return {
        "err_ux": float(rel[:, 0].mean()),
        "err_uy": float(rel[:, 1].mean()),
        "err_uz": float(rel[:, 2].mean()),
        "err_mean": float(rel.mean()),
        "err_l2_rel": float(err_norm),
        "mae_mm": float(np.abs(pred - target).mean()),
        "max_err_mm": float(np.abs(pred - target).max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    # Пути по умолчанию привязаны к расположению файла, а не к рабочему
    # каталогу, чтобы скрипт запускался и из корня проекта, и из src/.
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--geom-dir", default=str(root / "data" / "geom"))
    ap.add_argument("--fea-dir", default=str(root / "data" / "fea"))
    ap.add_argument("--out-dir", default=str(root / "results"))
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--k-neighbors", type=int, default=16)
    ap.add_argument("--n-subset", type=int, default=1024,
                    help="узлов на шаг оптимизации (стратегия из разд. 3.3)")
    ap.add_argument("--n-nodes", type=int, default=2500,
                    help="размер прореженного графа на геометрию (0 — вся сетка)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # На малом числе ядер лишние потоки только конкурируют за кеш.
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    geom_dir, fea_dir = Path(args.geom_dir), Path(args.fea_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cases = TRAIN_CASES + VAL_CASES + TEST_CASES
    print("Подготовка геометрических кэшей...")
    cache = {}
    for case in all_cases:
        if (geom_dir / f"geom_{case}.npz").exists():
            t0 = time.time()
            cache[case] = load_geometry_cache(geom_dir, case,
                                              args.k_neighbors, args.seed,
                                              args.n_nodes)
            print(f"  {case}: {cache[case]['n_nodes']} узлов "
                  f"({time.time() - t0:.1f} с)")

    train = load_samples(fea_dir, TRAIN_CASES, cache)
    val = load_samples(fea_dir, VAL_CASES, cache)
    test = load_samples(fea_dir, TEST_CASES, cache)
    print(f"Образцов: обучение {len(train)}, валидация {len(val)}, "
          f"тест {len(test)}")
    if not train or not test:
        raise SystemExit("Датасет пуст — сначала запустите generate_dataset.py")

    # Нормировка целевых смещений: общая шкала по обучающей выборке, иначе
    # образцы с высоким давлением доминируют в функции потерь.
    u_scale = float(np.concatenate([s["u"] for s in train]).std())
    print(f"Масштаб смещений (обучение): {u_scale:.3f} мм")

    d_global = len(make_globals(torch.from_numpy(train[0]["params"]),
                                cache[train[0]["case"]]["shape"]))
    cfg = HSSConfig(d_node=cache[all_cases[0]]["node_feats"].shape[1],
                    d_global=d_global, d_hidden=args.hidden,
                    n_layers=args.layers, k_neighbors=args.k_neighbors)
    model = HeartSimSage(cfg)
    print(f"Параметров модели: {model.n_params() / 1e3:.1f} тыс.")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train),
        pct_start=0.1)

    def forward(sample):
        c = cache[sample["case"]]
        g = make_globals(torch.from_numpy(sample["params"]), c["shape"])
        return model(c["node_feats"], c["rel"], c["strata"], c["nb_idx"], g)

    @torch.no_grad()
    def evaluate(samples):
        model.eval()
        rows = []
        for s in samples:
            pred = forward(s).numpy() * u_scale
            rows.append(relative_errors(pred, s["u"]))
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    history = []
    best_val, best_state = float("inf"), None
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(train)
        total = 0.0
        for s in train:
            pred = forward(s)
            target = torch.from_numpy(s["u"]) / u_scale
            idx = torch.randperm(pred.shape[0])[:args.n_subset]
            loss = torch.nn.functional.smooth_l1_loss(
                pred[idx], target[idx], beta=0.1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        train_loss = total / len(train)

        if epoch % 10 == 0 or epoch == 1:
            vm = evaluate(val) if val else {}
            history.append({"epoch": epoch, "train_loss": train_loss, **vm})
            if vm and vm["err_l2_rel"] < best_val:
                best_val = vm["err_l2_rel"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"эпоха {epoch:4d} | loss {train_loss:.5f} | "
                  f"val L2 {vm.get('err_l2_rel', float('nan')) * 100:5.2f}% | "
                  f"val MAE {vm.get('mae_mm', float('nan')):.3f} мм | "
                  f"{time.time() - t_start:.0f} с")

    if best_state is not None:
        model.load_state_dict(best_state)

    train_time = time.time() - t_start
    metrics = {"train": evaluate(train), "val": evaluate(val) if val else {},
               "test": evaluate(test)}

    # Замер времени инференса против КЭА
    s = test[0]
    with torch.no_grad():
        forward(s)
        t0 = time.time()
        for _ in range(20):
            forward(s)
        t_hss = (time.time() - t0) / 20

    # Среднее время конечно-элементного решения на тестовых сердцах: замерено
    # при генерации набора (столбец 0 в solve_stats) тем же процессом на тех же
    # ядрах, поэтому сравнение с временем инференса корректно.
    fea_times = []
    for case in TEST_CASES:
        p = fea_dir / f"fea_{case}.npz"
        if p.exists():
            fea_times.append(np.load(p)["solve_stats"][:, 0])
    t_fea = float(np.concatenate(fea_times).mean()) if fea_times else float("nan")

    summary = {
        "config": vars(args),
        "n_params": model.n_params(),
        "u_scale_mm": u_scale,
        "fea_solve_s": t_fea,
        "speedup": t_fea / t_hss if t_hss > 0 else float("nan"),
        "split": {"train": TRAIN_CASES, "val": VAL_CASES, "test": TEST_CASES},
        "n_samples": {"train": len(train), "val": len(val), "test": len(test)},
        "metrics": metrics,
        "train_time_s": train_time,
        "hss_inference_s": t_hss,
        "history": history,
    }
    (out_dir / "hss_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save({"cfg": cfg.__dict__, "state": model.state_dict(),
                "u_scale": u_scale}, out_dir / "hss_model.pt")

    print("\n=== Итоговые метрики (тест — невиданные сердца) ===")
    for k, v in metrics["test"].items():
        print(f"  {k}: {v:.5f}")
    print(f"Инференс HSS: {t_hss * 1000:.2f} мс/решение")
    print(f"КЭА (те же ядра): {t_fea:.1f} с/решение — "
          f"ускорение {t_fea / t_hss:.0f}x")


if __name__ == "__main__":
    main()
