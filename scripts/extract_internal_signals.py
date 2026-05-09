from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ke_drift.generation_eval import load_transformers_model
from ke_drift.internal import (
    collect_hidden_states,
    compare_hidden_states,
    load_hidden_states,
    save_hidden_states,
)
from ke_drift.io_utils import read_json_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="独立的 hidden-state 抽取与比较脚本。", add_help=False)
    parser._optionals.title = "选项"
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出。")
    subparsers = parser.add_subparsers(title="命令", dest="command", required=True)

    hidden = subparsers.add_parser("hidden", add_help=False, help="抽取 hidden states。")
    hidden._optionals.title = "选项"
    hidden.add_argument("-h", "--help", action="help", help="显示帮助信息并退出。")
    hidden.add_argument("--model", required=True)
    hidden.add_argument("--probes", default="/root/autodl-tmp/data/prepared/zsre/probes.jsonl")
    hidden.add_argument("--out", required=True)
    hidden.add_argument("--cache-dir", default="/root/autodl-tmp/models")
    hidden.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32", "auto"])
    hidden.add_argument("--device-map", default="auto")
    hidden.add_argument("--batch-size", type=int, default=4)
    hidden.add_argument("--max-input-length", type=int, default=256)
    hidden.add_argument("--probe-limit", type=int, default=200)

    compare = subparsers.add_parser("compare-hidden", add_help=False, help="比较两个 hidden-state 文件。")
    compare._optionals.title = "选项"
    compare.add_argument("-h", "--help", action="help", help="显示帮助信息并退出。")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--current", required=True)
    compare.add_argument("--checkpoint", type=int, required=True)
    compare.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "hidden":
        probes = read_json_records(args.probes)
        model, tokenizer = load_transformers_model(
            args.model,
            dtype=args.dtype,
            device_map=args.device_map,
            cache_dir=args.cache_dir,
        )
        payload = collect_hidden_states(
            model=model,
            tokenizer=tokenizer,
            probes=probes,
            max_prompts=args.probe_limit,
            batch_size=args.batch_size,
            max_input_length=args.max_input_length,
        )
        save_hidden_states(Path(args.out), payload)
        print(f"hidden states 已保存到：{args.out}")
    elif args.command == "compare-hidden":
        baseline = load_hidden_states(args.baseline)
        current = load_hidden_states(args.current)
        rows = compare_hidden_states(baseline, current, checkpoint=args.checkpoint, output_csv=args.out)
        print(f"已写入 {len(rows)} 行层级漂移结果：{args.out}")


if __name__ == "__main__":
    main()
