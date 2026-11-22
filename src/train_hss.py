"""
Обучение эмулятора HeartSimSage на КЭ-решениях для реальных бивентрикулярных
сеток (Rodero et al., 2021, Zenodo 4590294).

Методика:
  * разбиение ПО ГЕОМЕТРИЯМ (пациентам): тестовые сердца не встречаются
    в обучении -- тест измеряет перенос на новую анатомию;
  * признаки и смещения выражены в АНАТОМИЧЕСКОЙ системе координат сердца
    (длинная ось, направление ЛЖ->ПЖ), а не в координатах томографа;
  * при обучении граф прореживается до n_nodes узлов, выбранных РАВНОМЕРНО
    по всему объёму (поверхность и толща стенки);
  * итоговая проверка -- на ПОЛНЫХ сетках тестовых сердец (все узлы), т.е.
    модель предсказывает всё поле смещений, и ускорение сравнивается
    с КЭА на той же сетке;
  * профиль весов внимания по стратам SC1..SC5 считается по обученной
    модели (а не переносится из статьи).

Исправления относительно первой версии (по замечаниям руководителя):
  1. данные КЭА пересчитаны после исправления ориентации нормалей эндокарда;
  2. в прореженный граф раньше попадали только узлы поверхности;
  3. ускорение раньше сравнивало КЭА на полной сетке с эмулятором на 24% узлов;
  4. абсолютные координаты томографа заменены анатомическими.

Запуск (из каталога src/):
    python train_hss.py --epochs 80
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from graph_features import (anatomical_frame, build_stratified_neighborhoods,
                            compute_laplace_fields, shape_features)
from hss_model import HSSConfig, HeartSimSage

TRAIN_CASES = ["01", "02", "03", "04", "05", "06", "07"]
VAL_CASES = ["08", "09"]
TEST_CASES = ["10", "11"]

COORD_SCALE_MM = 60.0     # общий масштаб для всех сердец (сохраняет размер)


def load_geometry_cache(geom_dir: Path, case: str, k_neighbors: int, seed: int,
                        n_nodes: int = 0) -> dict:
    """
    Всё, что зависит только от геометрии: признаки узлов, окрестности,
    признаки формы, поворот в анатомические оси. n_nodes = 0 -- полная сетка.
    """
    d = np.load(geom_dir / f"geom_{case}.npz")
    coords = d["coords"].astype(np.float64)
    elements = d["elements"].astype(np.int64)
    faces = d["faces"].astype(np.int64)
    face_type = d["face_type"]
    uvc_z = d["uvc_z"].astype(np.float64)

    laplace = compute_laplace_fields(coords, elements, faces, face_type, uvc_z)
    shape = shape_features(coords, elements, faces, face_type, laplace)
    R = anatomical_frame(coords, uvc_z, d["is_lv_node"])

    def to_nodes(vec_el: np.ndarray) -> np.ndarray:
        acc = np.zeros_like(coords)
        np.add.at(acc, elements.ravel(), np.repeat(vec_el, 4, axis=0))
        return acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)

    f0 = to_nodes(d["f0"].astype(np.float64)) @ R.T
    s0 = to_nodes(d["s0"].astype(np.float64)) @ R.T
    x = (coords - coords.mean(axis=0)) @ R.T

    if n_nodes and n_nodes < len(coords):
        rng = np.random.default_rng(seed + int(case))
        node_ids = np.sort(rng.choice(len(coords), n_nodes, replace=False))
    else:
        node_ids = np.arange(len(coords))
    x, laplace, f0, s0 = x[node_ids], laplace[node_ids], f0[node_ids], s0[node_ids]

    nb_idx, strata = build_stratified_neighborhoods(x, k_neighbors, seed)
    xn = x / COORD_SCALE_MM
    node_feats = np.concatenate([f0, s0, laplace, xn], axis=1).astype(np.float32)
    rel = (xn[nb_idx] - xn[:, None, :]).astype(np.float32)

    return {
        "node_feats": torch.from_numpy(node_feats),
        "rel": torch.from_numpy(rel),
        "strata": torch.from_numpy(strata.astype(np.int64)),
        "nb_idx": torch.from_numpy(nb_idx),
        "shape": torch.from_numpy(shape.astype(np.float32)),
        "R": R,
        "n_nodes": len(node_ids),
        "node_ids": node_ids,
    }


def load_samples(fea_dir: Path, cases: list[str], cache: dict) -> list[dict]:
    """КЭ-решения; смещения переводятся в анатомические оси и сужаются на узлы графа."""
    out = []
    for case in cases:
        path = fea_dir / f"fea_{case}.npz"
        if not path.exists() or case not in cache:
            continue
        d = np.load(path)
        c = cache[case]
        for i in range(len(d["params"])):
            u = d["u"][i][c["node_ids"]].astype(np.float64) @ c["R"].T
            out.append({"case": case, "params": d["params"][i].astype(np.float32),
                        "u": u.astype(np.float32)})
    return out


def make_globals(params: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """Параметры материала и нагрузки (в лог- и линейной шкале) + признаки формы."""
    return torch.cat([torch.log(params.clamp(min=1e-6)), params, shape])


def relative_errors(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    err_ux/uy/uz, err_mean -- |û−u| / max|u| по компоненте (метрика [15]);
    err_l2_rel -- ‖û−u‖ / ‖u‖ по всему полю; mae_mm, max_err_mm -- в мм;
    компоненты -- в анатомических осях (e1: ЛЖ->ПЖ, e2, e3: длинная ось).
    """
    scale = np.maximum(np.abs(target).max(axis=0), 1e-8)
    rel = np.abs(pred - target) / scale
    return {
        "err_ux": float(rel[:, 0].mean()), "err_uy": float(rel[:, 1].mean()),
        "err_uz": float(rel[:, 2].mean()), "err_mean": float(rel.mean()),
        "err_l2_rel": float(np.linalg.norm(pred - target) / np.linalg.norm(target)),
        "mae_mm": float(np.abs(pred - target).mean()),
        "node_err_mm": float(np.linalg.norm(pred - target, axis=1).mean()),
        "max_err_mm": float(np.linalg.norm(pred - target, axis=1).max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--geom-dir", default=str(root / "data" / "geom"))
    ap.add_argument("--fea-dir", default=str(root / "data" / "fea"))
    ap.add_argument("--out-dir", default=str(root / "results"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--k-neighbors", type=int, default=16)
    ap.add_argument("--n-subset", type=int, default=1000,
                    help="узлов в функции потерь на шаг (разд. 3.1 отчёта)")
    ap.add_argument("--n-nodes", type=int, default=2500,
                    help="размер прореженного графа при обучении")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    geom_dir, fea_dir, out_dir = (Path(args.geom_dir), Path(args.fea_dir),
                                  Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Подготовка геометрий (прореженные графы для обучения)...")
    cache = {}
    for case in TRAIN_CASES + VAL_CASES + TEST_CASES:
        if (geom_dir / f"geom_{case}.npz").exists():
            cache[case] = load_geometry_cache(geom_dir, case, args.k_neighbors,
                                              args.seed, args.n_nodes)
    train = load_samples(fea_dir, TRAIN_CASES, cache)
    val = load_samples(fea_dir, VAL_CASES, cache)
    print(f"Образцов: обучение {len(train)}, валидация {len(val)}")

    u_scale = float(np.concatenate([s["u"] for s in train]).std())
    d_global = len(make_globals(torch.from_numpy(train[0]["params"]),
                                cache[TRAIN_CASES[0]]["shape"]))
    cfg = HSSConfig(d_node=cache[TRAIN_CASES[0]]["node_feats"].shape[1],
                    d_global=d_global, d_hidden=args.hidden,
                    n_layers=args.layers, k_neighbors=args.k_neighbors)
    model = HeartSimSage(cfg)
    print(f"Параметров модели: {model.n_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train), pct_start=0.1)

    def forward(c, params):
        g = make_globals(torch.from_numpy(params), c["shape"])
        return model(c["node_feats"], c["rel"], c["strata"], c["nb_idx"], g)

    @torch.no_grad()
    def evaluate(samples, caches):
        model.eval()
        rows = [relative_errors(forward(caches[s["case"]], s["params"]).numpy()
                                * u_scale, s["u"]) for s in samples]
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    history, best_val, best_state = [], float("inf"), None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(train)
        total = 0.0
        for s in train:
            pred = forward(cache[s["case"]], s["params"])
            target = torch.from_numpy(s["u"]) / u_scale
            idx = torch.randperm(pred.shape[0])[:args.n_subset]
            loss = torch.nn.functional.smooth_l1_loss(pred[idx], target[idx],
                                                      beta=0.1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        if epoch % 5 == 0 or epoch == 1:
            vm = evaluate(val, cache)
            history.append({"epoch": epoch, "train_loss": total / len(train), **vm})
            if vm["err_l2_rel"] < best_val:
                best_val = vm["err_l2_rel"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"эпоха {epoch:3d} | loss {total / len(train):.5f} | "
                  f"val L2 {vm['err_l2_rel'] * 100:5.2f}% | "
                  f"val MAE {vm['mae_mm']:.3f} мм | {time.time() - t0:.0f} с",
                  flush=True)
    model.load_state_dict(best_state)
    train_time = time.time() - t0

    metrics = {"train": evaluate(train, cache), "val": evaluate(val, cache),
               "test_subgraph": evaluate(load_samples(fea_dir, TEST_CASES, cache),
                                         cache)}

    # ---- проверка на ПОЛНЫХ сетках тестовых сердец --------------------------
    print("Проверка на полных сетках тестовых сердец...")
    full = {c: load_geometry_cache(geom_dir, c, args.k_neighbors, args.seed, 0)
            for c in TEST_CASES}
    test_full = load_samples(fea_dir, TEST_CASES, full)
    metrics["test_full"] = evaluate(test_full, full)

    preds, per_sample = [], []
    model.eval()
    with torch.no_grad():
        for s in test_full:
            c = full[s["case"]]
            p = forward(c, s["params"]).numpy() * u_scale
            # обратно в координаты сетки (мм) для визуализации
            preds.append((p @ c["R"]).astype(np.float32))
            per_sample.append({"case": s["case"], **relative_errors(p, s["u"]),
                               "u_max": float(np.linalg.norm(s["u"], axis=1).max())})
    u_true = np.concatenate([s["u"] for s in test_full])
    u_pred = np.concatenate([p @ full[s["case"]]["R"].T
                             for p, s in zip(preds, test_full)])
    ss_res = ((u_pred - u_true) ** 2).sum()
    ss_tot = ((u_true - u_true.mean(axis=0)) ** 2).sum()
    node_err = np.linalg.norm(u_pred - u_true, axis=1)
    metrics["test_full"].update({
        "r2": float(1 - ss_res / ss_tot),
        "node_err_median_mm": float(np.median(node_err)),
        "node_err_p95_mm": float(np.percentile(node_err, 95)),
        "u_mean_mm": float(np.linalg.norm(u_true, axis=1).mean()),
    })

    # ---- профиль внимания по стратам (обученная модель, полные сетки) ------
    sums, cnts = np.zeros(5), np.zeros(5)
    with torch.no_grad():
        for s in test_full[::4]:
            c = full[s["case"]]
            forward(c, s["params"])
            st = c["strata"].numpy()
            for blk in model.blocks:
                a = blk.attn.last_attn.mean(dim=-1).numpy()      # (N, K)
                for k in range(5):
                    m = st == k
                    sums[k] += a[m].sum()
                    cnts[k] += m.sum()
    attn_profile = (sums / np.maximum(cnts, 1)).tolist()

    # ---- время: КЭА и эмулятор на одной и той же полной сетке, CPU ----------
    c = full[TEST_CASES[0]]
    with torch.no_grad():
        forward(c, test_full[0]["params"])
        t1 = time.time()
        for _ in range(10):
            forward(c, test_full[0]["params"])
        t_hss = (time.time() - t1) / 10
    t_fea = float(np.concatenate([np.load(fea_dir / f"fea_{cs}.npz")["solve_stats"][:, 0]
                                  for cs in TEST_CASES]).mean())

    summary = {
        "config": vars(args), "n_params": model.n_params(), "u_scale_mm": u_scale,
        "split": {"train": TRAIN_CASES, "val": VAL_CASES, "test": TEST_CASES},
        "n_samples": {"train": len(train), "val": len(val), "test": len(test_full)},
        "metrics": metrics, "attention_by_stratum": attn_profile,
        "train_time_s": train_time, "hss_inference_full_mesh_s": t_hss,
        "fea_solve_s": t_fea, "speedup_cpu": t_fea / t_hss,
        "n_nodes_test_mesh": int(c["n_nodes"]), "history": history,
    }
    (out_dir / "hss_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "test_report.json").write_text(
        json.dumps({"per_sample": per_sample}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    np.savez_compressed(out_dir / "test_predictions.npz",
                        cases=np.array([s["case"] for s in test_full]),
                        params=np.stack([s["params"] for s in test_full]),
                        u_pred=np.array(preds + [None], dtype=object)[:-1])
    torch.save({"cfg": cfg.__dict__, "state": model.state_dict(),
                "u_scale": u_scale}, out_dir / "hss_model.pt")

    m = metrics["test_full"]
    print("\n=== Тест: новые сердца, полные сетки ===")
    print(f"  L2 {m['err_l2_rel'] * 100:.2f}% | MAE {m['mae_mm']:.3f} мм | "
          f"R2 {m['r2']:.3f} | Err_mean {m['err_mean'] * 100:.2f}%")
    print(f"  внимание по стратам: {np.round(attn_profile, 3)}")
    print(f"  КЭА {t_fea:.1f} с, эмулятор {t_hss * 1000:.0f} мс "
          f"({c['n_nodes']} узлов, CPU) -> ускорение {t_fea / t_hss:.0f}x")


if __name__ == "__main__":
    main()
