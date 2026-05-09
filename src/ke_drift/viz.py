from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import ensure_dir, read_json_records, slugify, write_csv
from .internal import load_hidden_states


def _setup_chinese_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def discover_run_dirs(root: str | Path) -> list[Path]:
    root_path = Path(root)
    return sorted(path for path in root_path.rglob("config.json") if path.parent.is_dir())


def collect_checkpoint_rows(run_config_path: Path) -> list[dict[str, Any]]:
    run_dir = run_config_path.parent
    config = json.loads(run_config_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for checkpoint_dir in sorted(run_dir.glob("checkpoint_*")):
        metrics_path = checkpoint_dir / "metrics.json"
        summary_path = checkpoint_dir / "internal_summary.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        row = {
            "run_dir": str(run_dir),
            "method": config.get("method", ""),
            "model": config.get("model", ""),
            "dataset": config.get("dataset", ""),
            "checkpoint": int(metrics.get("checkpoint", checkpoint_dir.name.split("_")[-1])),
            "reliability": metrics.get("dimensions", {}).get("reliability", 0.0),
            "generalization": metrics.get("dimensions", {}).get("generalization", 0.0),
            "locality": metrics.get("dimensions", {}).get("locality", 0.0),
            "portability": metrics.get("dimensions", {}).get("portability", 0.0),
            "average": metrics.get("dimensions", {}).get("average", 0.0),
            "hidden_mean_cosine_distance": summary.get("hidden_mean_cosine_distance", 0.0),
            "weight_mean_relative_l2": summary.get("weight_mean_relative_l2", 0.0),
        }
        rows.append(row)
    return rows


def aggregate_metrics(runs_root: str | Path, output_dir: str | Path) -> pd.DataFrame:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for config_path in discover_run_dirs(runs_root):
        rows.extend(collect_checkpoint_rows(config_path))
    df = pd.DataFrame(rows)
    out = ensure_dir(output_dir)
    if not df.empty:
        df.to_csv(out / "aggregate_metrics.csv", index=False)
    return df


def plot_metric_curves(df: pd.DataFrame, output_dir: str | Path) -> None:
    import pandas as pd  # noqa: F401

    if df.empty:
        return
    import matplotlib.pyplot as plt
    import seaborn as sns

    _setup_chinese_matplotlib()
    out = ensure_dir(output_dir)
    metrics = ["reliability", "generalization", "locality", "portability", "average"]
    long_df = df.melt(
        id_vars=["method", "checkpoint"],
        value_vars=metrics,
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=long_df, x="checkpoint", y="score", hue="method", style="metric", markers=True)
    plt.ylim(-0.02, 1.02)
    plt.title("自回归行为指标随连续编辑次数变化")
    plt.tight_layout()
    plt.savefig(out / "behavior_metrics.png", dpi=220)
    plt.close()


def plot_internal_curves(df: pd.DataFrame, output_dir: str | Path) -> None:
    import pandas as pd  # noqa: F401

    if df.empty:
        return
    import matplotlib.pyplot as plt
    import seaborn as sns

    _setup_chinese_matplotlib()
    out = ensure_dir(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(data=df, x="checkpoint", y="hidden_mean_cosine_distance", hue="method", marker="o", ax=axes[0])
    axes[0].set_title("平均 hidden-state 漂移")
    sns.lineplot(data=df, x="checkpoint", y="weight_mean_relative_l2", hue="method", marker="o", ax=axes[1])
    axes[1].set_title("平均参数 relative L2")
    plt.tight_layout()
    plt.savefig(out / "internal_summary.png", dpi=220)
    plt.close()


def plot_weight_heatmaps(run_config_path: Path, output_dir: str | Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    _setup_chinese_matplotlib()
    run_dir = run_config_path.parent
    out = ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []
    for checkpoint_dir in sorted(run_dir.glob("checkpoint_*")):
        csv_path = checkpoint_dir / "weight_delta.csv"
        if csv_path.exists() and csv_path.stat().st_size > 0:
            df = pd.read_csv(csv_path)
            if not df.empty:
                grouped = df.groupby(["checkpoint", "layer"], as_index=False)["relative_l2"].mean()
                rows.extend(grouped.to_dict("records"))
    if not rows:
        return
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="layer", columns="checkpoint", values="relative_l2", aggfunc="mean").fillna(0.0)
    plt.figure(figsize=(8, max(4, len(pivot) * 0.2)))
    sns.heatmap(pivot, cmap="magma", cbar_kws={"label": "relative L2"})
    plt.title(f"参数扰动热力图：{run_dir.name}")
    plt.tight_layout()
    plt.savefig(out / f"{slugify(run_dir.name)}_weight_heatmap.png", dpi=220)
    plt.close()


def plot_hidden_layer_curves(run_config_path: Path, output_dir: str | Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    _setup_chinese_matplotlib()
    run_dir = run_config_path.parent
    out = ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []
    for checkpoint_dir in sorted(run_dir.glob("checkpoint_*")):
        csv_path = checkpoint_dir / "hidden_drift.csv"
        if csv_path.exists() and csv_path.stat().st_size > 0:
            rows.extend(pd.read_csv(csv_path).to_dict("records"))
    if not rows:
        return
    df = pd.DataFrame(rows)
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="layer", y="mean_cosine_distance", hue="checkpoint", palette="viridis")
    plt.title(f"层级 hidden-state 漂移：{run_dir.name}")
    plt.tight_layout()
    plt.savefig(out / f"{slugify(run_dir.name)}_hidden_layer_drift.png", dpi=220)
    plt.close()


def _projection(values: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(values)
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30, max(5, values.shape[0] // 10))
        return TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca", learning_rate="auto").fit_transform(values)
    if method == "umap":
        import umap

        return umap.UMAP(n_components=2, random_state=seed).fit_transform(values)
    raise ValueError(method)


def plot_hidden_projections(
    run_config_path: Path,
    output_dir: str | Path,
    tsne_seeds: list[int] | None = None,
    sample_per_checkpoint: int = 200,
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    _setup_chinese_matplotlib()
    run_dir = run_config_path.parent
    out = ensure_dir(output_dir)
    arrays: list[np.ndarray] = []
    labels: list[str] = []
    categories: list[str] = []
    rng = np.random.default_rng(42)
    for checkpoint_dir in sorted(run_dir.glob("checkpoint_*")):
        hidden_path = checkpoint_dir / "hidden_states.npz"
        if not hidden_path.exists():
            continue
        payload = load_hidden_states(hidden_path)
        embeddings = payload["embeddings"]
        if embeddings.size == 0:
            continue
        last_layer = embeddings[-1]
        n = last_layer.shape[0]
        indices = np.arange(n)
        if n > sample_per_checkpoint:
            indices = rng.choice(indices, size=sample_per_checkpoint, replace=False)
        arrays.append(last_layer[indices])
        labels.extend([checkpoint_dir.name.replace("checkpoint_", "")] * len(indices))
        categories.extend([str(x) for x in payload["categories"][indices]])

    if not arrays:
        return
    values = np.concatenate(arrays, axis=0)
    label_arr = np.array(labels)
    cat_arr = np.array(categories)
    seeds = tsne_seeds or [42, 43, 44]
    projection_specs = [("pca", seeds[0])] + [("tsne", seed) for seed in seeds]
    try:
        import umap  # noqa: F401

        projection_specs.append(("umap", seeds[0]))
    except Exception:
        pass

    for method, seed in projection_specs:
        points = _projection(values, method, seed)
        df = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "checkpoint": label_arr, "category": cat_arr})
        plt.figure(figsize=(7, 6))
        sns.scatterplot(data=df, x="x", y="y", hue="checkpoint", style="category", s=22, alpha=0.75)
        plt.title(f"{method.upper()} hidden-state 投影：{run_dir.name}")
        plt.tight_layout()
        suffix = f"{method}_seed{seed}" if method == "tsne" else method
        plt.savefig(out / f"{slugify(run_dir.name)}_{suffix}.png", dpi=220)
        plt.close()


def plot_token_trajectories(run_config_path: Path, output_dir: str | Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    _setup_chinese_matplotlib()
    run_dir = run_config_path.parent
    out = ensure_dir(output_dir)
    for checkpoint_dir in sorted(run_dir.glob("checkpoint_*")):
        path = checkpoint_dir / "token_trajectory.jsonl"
        if not path.exists() or path.stat().st_size == 0:
            continue
        rows = read_json_records(path)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df = df[df["token_label"].isin(["target_new", "ground_truth"])]
        if df.empty:
            continue
        plt.figure(figsize=(9, 5))
        sns.lineplot(data=df, x="layer", y="probability", hue="token_label", style="edit_id")
        plt.yscale("log")
        plt.title(f"目标 token 概率轨迹：{run_dir.name} {checkpoint_dir.name}")
        plt.tight_layout()
        plt.savefig(out / f"{slugify(run_dir.name)}_{checkpoint_dir.name}_token_trajectory.png", dpi=220)
        plt.close()


def write_correlations(df: pd.DataFrame, output_dir: str | Path) -> None:
    import pandas as pd  # noqa: F401

    if df.empty:
        return
    out = ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []
    for method, group in df.groupby("method"):
        cols = ["average", "hidden_mean_cosine_distance", "weight_mean_relative_l2"]
        valid = group[cols].dropna()
        if len(valid) < 2:
            continue
        corr = valid.corr(method="spearman")
        rows.append(
            {
                "method": method,
                "spearman_average_vs_hidden_drift": corr.loc["average", "hidden_mean_cosine_distance"],
                "spearman_average_vs_weight_delta": corr.loc["average", "weight_mean_relative_l2"],
            }
        )
    write_csv(out / "correlations.csv", rows)
