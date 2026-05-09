from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ke_drift.generation_eval import evaluate_model_on_probes, load_transformers_model
from ke_drift.io_utils import read_json_records
from ke_drift.scr import build_memory, build_scr_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="独立的自回归生成评估脚本。", add_help=False)
    parser._optionals.title = "选项"
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出。")
    parser.add_argument("--model", required=True)
    parser.add_argument("--probes", default="/root/autodl-tmp/data/prepared/zsre/probes.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--cache-dir", default="/root/autodl-tmp/models")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32", "auto"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scr-memory", default=None, help="可选的 edits.jsonl，用于 SCR-lite 检索提示。")
    parser.add_argument("--scr-memory-limit", type=int, default=100)
    parser.add_argument("--scr-top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probes = read_json_records(args.probes)
    if args.limit:
        probes = probes[: args.limit]
    model, tokenizer = load_transformers_model(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        cache_dir=args.cache_dir,
    )

    transform = None
    if args.scr_memory:
        memory = build_memory(read_json_records(args.scr_memory), limit=args.scr_memory_limit)

        def transform(prompt: str, _probe: dict) -> str:
            return build_scr_prompt(prompt, memory, top_k=args.scr_top_k)

    metrics = evaluate_model_on_probes(
        model=model,
        tokenizer=tokenizer,
        probes=probes,
        output_dir=Path(args.out),
        checkpoint=args.checkpoint,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        prompt_transform=transform,
    )
    print("评估完成，核心指标：")
    print(metrics["dimensions"])


if __name__ == "__main__":
    main()
