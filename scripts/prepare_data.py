from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ke_drift.data import default_input_path, prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备知识编辑序列和固定 probe 集合。", add_help=False)
    parser._optionals.title = "选项"
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出。")
    parser.add_argument("--dataset", default="zsre", choices=["zsre", "counterfact"])
    parser.add_argument("--input", default=None, help="原始 ZsRE/CounterFact JSON 或 JSONL 文件。")
    parser.add_argument("--data-root", default="/root/autodl-tmp/data")
    parser.add_argument("--out", default=None, help="输出目录。默认写入 /root/autodl-tmp/data/prepared/<dataset>。")
    parser.add_argument("--n-edits", type=int, default=100)
    parser.add_argument("--n-controls", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input) if args.input else default_input_path(args.dataset, args.data_root)
    output_dir = Path(args.out) if args.out else Path(args.data_root) / "prepared" / args.dataset
    manifest = prepare_dataset(
        input_path=input_path,
        dataset=args.dataset,
        output_dir=output_dir,
        n_edits=args.n_edits,
        seed=args.seed,
        n_controls=args.n_controls,
    )
    print(f"数据准备完成：{manifest['n_edits']} 条编辑、{manifest['n_probes']} 条 probe，输出目录：{output_dir}")


if __name__ == "__main__":
    main()
