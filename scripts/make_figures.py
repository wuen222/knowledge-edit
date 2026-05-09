from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ke_drift.viz import (
    aggregate_metrics,
    discover_run_dirs,
    plot_hidden_layer_curves,
    plot_hidden_projections,
    plot_internal_curves,
    plot_metric_curves,
    plot_token_trajectories,
    plot_weight_heatmaps,
    write_correlations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据实验输出生成静态诊断图表。", add_help=False)
    parser._optionals.title = "选项"
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出。")
    parser.add_argument("--runs", default="/root/autodl-tmp/runs")
    parser.add_argument("--out", default="/root/autodl-tmp/artifacts")
    parser.add_argument("--tsne-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--sample-per-checkpoint", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    df = aggregate_metrics(args.runs, out)
    plot_metric_curves(df, out)
    plot_internal_curves(df, out)
    write_correlations(df, out)

    for config_path in discover_run_dirs(args.runs):
        run_out = out / config_path.parent.name
        plot_weight_heatmaps(config_path, run_out)
        plot_hidden_layer_curves(config_path, run_out)
        plot_hidden_projections(
            config_path,
            run_out,
            tsne_seeds=args.tsne_seeds,
            sample_per_checkpoint=args.sample_per_checkpoint,
        )
        plot_token_trajectories(config_path, run_out)
    print(f"图表已写入：{out}")


if __name__ == "__main__":
    main()
