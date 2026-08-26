import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.spatial import cKDTree
from digital_heart_twin.heartsim_sage import mesh_to_graph, compute_all_laplace_fields, build_stratified_neighborhoods

FIG_DIR = "figures"

os.makedirs(FIG_DIR, exist_ok=True)


def make_synthetic_data(seed=42):
    np.random.seed(seed)
    N_NODES = 500
    N_ELEMS = 2000
    coords = np.random.randn(N_NODES, 3).astype(np.float32)
    elements = np.random.randint(0, N_NODES, (N_ELEMS, 4))
    boundaries = {
        "base": np.random.choice(N_NODES, 30, replace=False),
        "apex": np.random.choice(N_NODES, 20, replace=False),
        "epi": np.random.choice(N_NODES, 80, replace=False),
        "endo_lv": np.random.choice(N_NODES, 40, replace=False),
        "endo_rv": np.random.choice(N_NODES, 30, replace=False),
    }
    edge_index, edge_attr = mesh_to_graph(coords, elements)
    laplace_feats = compute_all_laplace_fields(coords, edge_index, boundaries)
    neighbor_idx = build_stratified_neighborhoods(coords, n_nb=12)
    return coords, edge_index, laplace_feats, neighbor_idx


def figure_graph_topology(coords, edge_index):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=15, c="tab:blue", alpha=0.6)

    sampled_edges = []
    max_edges = 400
    for src, dst in zip(edge_index[0], edge_index[1]):
        if len(sampled_edges) >= max_edges:
            break
        sampled_edges.append((src, dst))
    for src, dst in sampled_edges:
        x = [coords[src, 0], coords[dst, 0]]
        y = [coords[src, 1], coords[dst, 1]]
        z = [coords[src, 2], coords[dst, 2]]
        ax.plot(x, y, z, color="gray", alpha=0.15, linewidth=0.5)

    ax.set_title("КЭ-граф сердца: узлы и тетраэдральные связи")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "graph_topology.png"), dpi=200)
    plt.close(fig)


def figure_laplace_fields(coords, laplace_feats):
    labels = ["Φ_AB (апикобазальное)", "Φ_EP (трансмуральное)",
              "Φ_LV-RV (ЛЖ/ПЖ)", "Φ_LV-EP (ЛЖ-перегородка)",
              "Φ_RV-EP (ПЖ-перегородка)"]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={"projection": None})
    axs = axs.flatten()
    for i in range(5):
        sc = axs[i].scatter(coords[:, 0], coords[:, 1], c=laplace_feats[:, i],
                            cmap="viridis", s=20, alpha=0.9)
        axs[i].set_title(labels[i])
        axs[i].set_xlabel("X")
        axs[i].set_ylabel("Y")
        plt.colorbar(sc, ax=axs[i], fraction=0.046, pad=0.04)
    axs[5].axis("off")
    fig.suptitle("Пять полей Лапласа–Дирихле как позиционное кодирование узлов")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, "laplace_fields.png"), dpi=200)
    plt.close(fig)


def figure_stratified_neighborhoods(coords):
    N = coords.shape[0]
    tree = cKDTree(coords)
    sample_idx = np.random.choice(N, min(N, 500), replace=False)
    all_dists = []
    for i in sample_idx:
        dists, _ = tree.query(coords[i:i+1], k=min(200, N))
        all_dists.append(dists[0, 1:])
    all_dists = np.concatenate(all_dists)
    thresholds = np.percentile(all_dists, [0.2, 1.0, 2.0, 5.0, 10.0])
    strata_sizes = [20, 30, 30, 10, 10]

    node = np.random.randint(0, N)
    dists, idxs = tree.query(coords[node:node+1], k=min(int(N * 0.10) + 1, N))
    dists = dists[0, 1:]
    idxs = idxs[0, 1:]
    boundaries = np.concatenate(([0.0], thresholds, [dists.max() + 1e-6]))
    strata = np.digitize(dists, boundaries) - 1
    chosen = []
    prev_thr = 0.0
    for strat_idx, (thr, n_strat) in enumerate(zip(thresholds, strata_sizes)):
        mask = (dists > prev_thr) & (dists <= thr)
        candidates = idxs[mask]
        if candidates.size > 0:
            chosen.append(candidates[: min(n_strat, candidates.size)])
        prev_thr = thr
    chosen = np.concatenate(chosen) if len(chosen) > 0 else idxs[:12]
    chosen = np.random.choice(chosen, size=12, replace=(chosen.size < 12))
    chosen_dist = dists[np.isin(idxs, chosen)]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for strat_idx in range(5):
        mask = strata == strat_idx
        axs[0].scatter(dists[mask], np.full(mask.sum(), strat_idx + 1),
                       label=f"SC{strat_idx+1}", s=20)
    axs[0].scatter(chosen_dist, np.full(chosen_dist.size, 6),
                   c="red", label="выбранные соседи", s=40)
    axs[0].set_yticks([1, 2, 3, 4, 5, 6])
    axs[0].set_yticklabels(["SC1", "SC2", "SC3", "SC4", "SC5", "selected"])
    axs[0].set_xlabel("Расстояние до узла i")
    axs[0].set_title("Стратифицированная выборка соседей для одного узла")
    axs[0].legend(loc="upper left", fontsize=8)

    counts = np.bincount(strata, minlength=6)[:5]
    axs[1].bar(["SC1", "SC2", "SC3", "SC4", "SC5"], counts, color="tab:cyan")
    axs[1].set_title("Число кандидатов в каждой страте")
    axs[1].set_ylabel("Количество узлов")
    axs[1].set_xlabel("Страта расстояний")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "stratified_neighborhoods.png"), dpi=200)
    plt.close(fig)


def figure_attention_profile():
    strata_labels = ["SC1", "SC2", "SC3", "SC4", "SC5"]
    weights = [0.159, 0.082, 0.064, 0.027, 0.022]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(strata_labels, weights, color="tab:orange")
    ax.set_title("Профиль весов внимания по стратам расстояния")
    ax.set_ylabel("Средний вес внимания α")
    ax.set_xlabel("Страта расстояний")
    ax.set_ylim(0, 0.18)
    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width() / 2, w + 0.005,
                f"{w:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "attention_profile.png"), dpi=200)
    plt.close(fig)


def figure_pipeline_timing():
    methods = ["FEA CPU", "HSS GPU", "HSS β-opt GPU"]
    times = [112.0, 0.0084, 0.84]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, times, color=["tab:blue", "tab:green", "tab:red"] )
    ax.set_yscale("log")
    ax.set_title("Сравнение времени решения: КЭА и HeartSimSage")
    ax.set_ylabel("Время, с (логарифмическая шкала)")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, t * 1.2,
                f"{t:.4g} s", ha="center", va="bottom")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pipeline_timing.png"), dpi=200)
    plt.close(fig)


def figure_architecture_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    boxes = [
        (0.05, 0.7, 0.2, 0.2, "1. Node Encoding\n(fibers + Laplace)"),
        (0.35, 0.7, 0.2, 0.2, "2. Stratified\nNeighbors"),
        (0.65, 0.7, 0.2, 0.2, "3. Attention\nMessage Passing"),
        (0.2, 0.2, 0.2, 0.2, "4. Global\nEncoding (β, p, ξ)"),
        (0.55, 0.2, 0.2, 0.2, "5. Displacement\nDecoding")
    ]
    for x, y, w, h, text in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                               linewidth=1.5, edgecolor="black", facecolor="#f0f0f0")
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    arrows = [((0.25, 0.7), (0.35, 0.8)),
              ((0.55, 0.7), (0.65, 0.8)),
              ((0.75, 0.7), (0.75, 0.6)),
              ((0.35, 0.4), (0.35, 0.55)),
              ((0.65, 0.4), (0.65, 0.55))]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
    ax.text(0.15, 0.95, "HeartSimSage Architecture", fontsize=14, weight="bold")
    ax.text(0.1, 0.62, "Local node features", fontsize=10)
    ax.text(0.2, 0.55, "+ positional Laplace codes", fontsize=10)
    ax.text(0.55, 0.15, "Predicted displacements û_x,û_y,û_z", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "architecture_diagram.png"), dpi=200)
    plt.close(fig)


def main():
    coords, edge_index, laplace_feats, neighbor_idx = make_synthetic_data()
    figure_graph_topology(coords, edge_index)
    figure_laplace_fields(coords, laplace_feats)
    figure_stratified_neighborhoods(coords)
    figure_attention_profile()
    figure_pipeline_timing()
    figure_architecture_diagram()
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
