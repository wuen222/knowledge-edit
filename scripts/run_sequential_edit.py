from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ke_drift.easyedit_runner import apply_single_edit, get_editor_model_tokenizer, load_easyedit_editor
from ke_drift.generation_eval import evaluate_model_on_probes, load_transformers_model
from ke_drift.internal import (
    capture_weight_snapshot,
    collect_hidden_states,
    collect_token_trajectory,
    compare_hidden_states,
    compute_weight_deltas,
    save_hidden_states,
    summarize_internal,
    write_token_trajectory,
)
from ke_drift.io_utils import ensure_dir, read_json_records, set_seed, slugify, write_csv, write_json, write_jsonl
from ke_drift.scr import build_memory, build_scr_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行连续知识编辑诊断实验。", add_help=False)
    parser._optionals.title = "选项"
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出。")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset", default="zsre")
    parser.add_argument("--method", required=True, help="知识编辑方法，例如 FT-L、ROME、MEMIT、PMET、AlphaEdit、SCR-LITE。")
    parser.add_argument("--prepared-dir", default="/root/autodl-tmp/data/prepared/zsre")
    parser.add_argument("--runs-root", default="/root/autodl-tmp/runs")
    parser.add_argument("--models-cache", default="/root/autodl-tmp/models")
    parser.add_argument("--easyedit-root", default="/root/EasyEdit")
    parser.add_argument("--hparams", default=None)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 1, 10, 50, 100])
    parser.add_argument("--max-edits", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32", "auto"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-batch-size", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--probe-limit", type=int, default=200)
    parser.add_argument("--weight-name-filter", default="model.layers.")
    parser.add_argument("--token-trajectory-count", type=int, default=3)
    parser.add_argument("--scr-top-k", type=int, default=3)
    return parser.parse_args()


def checkpoint_dir(run_dir: Path, checkpoint: int) -> Path:
    return ensure_dir(run_dir / f"checkpoint_{checkpoint:03d}")


def evaluate_checkpoint(
    model,
    tokenizer,
    probes: list[dict],
    edits: list[dict],
    run_dir: Path,
    checkpoint: int,
    args: argparse.Namespace,
    baseline_hidden: dict,
    baseline_weights: dict | None,
    use_scr: bool,
) -> dict:
    out_dir = checkpoint_dir(run_dir, checkpoint)
    prompt_transform = None
    if use_scr:
        memory = build_memory(edits, limit=checkpoint)

        def prompt_transform(prompt: str, _probe: dict) -> str:
            return build_scr_prompt(prompt, memory, top_k=args.scr_top_k)

    metrics = evaluate_model_on_probes(
        model=model,
        tokenizer=tokenizer,
        probes=probes,
        output_dir=out_dir,
        checkpoint=checkpoint,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        prompt_transform=prompt_transform,
    )

    current_hidden = collect_hidden_states(
        model=model,
        tokenizer=tokenizer,
        probes=probes,
        max_prompts=args.probe_limit,
        batch_size=args.hidden_batch_size,
        max_input_length=args.max_input_length,
    )
    save_hidden_states(out_dir / "hidden_states.npz", current_hidden)
    hidden_rows = compare_hidden_states(
        baseline_hidden,
        current_hidden,
        checkpoint=checkpoint,
        output_csv=out_dir / "hidden_drift.csv",
    )

    if baseline_weights is not None:
        weight_rows = compute_weight_deltas(
            baseline=baseline_weights,
            model=model,
            checkpoint=checkpoint,
            name_filter=args.weight_name_filter,
            output_csv=out_dir / "weight_delta.csv",
        )
    else:
        weight_rows = []
        write_csv(out_dir / "weight_delta.csv", weight_rows, fieldnames=["checkpoint", "name", "layer", "delta_l2", "base_l2", "relative_l2", "cosine_distance", "numel"])

    token_rows: list[dict] = []
    for edit in edits[: args.token_trajectory_count]:
        token_rows.extend(
            collect_token_trajectory(
                model=model,
                tokenizer=tokenizer,
                edit=edit,
                checkpoint=checkpoint,
                max_input_length=args.max_input_length,
            )
        )
    write_token_trajectory(out_dir / "token_trajectory.jsonl", token_rows)

    summary = summarize_internal(
        checkpoint=checkpoint,
        hidden_rows=hidden_rows,
        weight_rows=weight_rows,
        output_path=out_dir / "internal_summary.json",
    )
    return {"metrics": metrics, "internal": summary}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    method = args.method.upper()
    prepared_dir = Path(args.prepared_dir)
    edits = read_json_records(prepared_dir / "edits.jsonl")[: args.max_edits]
    probes = read_json_records(prepared_dir / "probes.jsonl")
    if args.probe_limit:
        probes = probes[: args.probe_limit]
    checkpoints = sorted(set(cp for cp in args.checkpoints if cp <= args.max_edits))
    if 0 not in checkpoints:
        checkpoints = [0] + checkpoints

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(
        Path(args.runs_root)
        / args.dataset
        / slugify(args.model)
        / slugify(method)
        / run_id
    )
    write_json(
        run_dir / "config.json",
        {
            "model": args.model,
            "dataset": args.dataset,
            "method": method,
            "seed": args.seed,
            "checkpoints": checkpoints,
            "max_edits": args.max_edits,
            "prepared_dir": str(prepared_dir),
            "easyedit_root": args.easyedit_root,
            "hparams": args.hparams,
        },
    )
    write_jsonl(run_dir / "edit_stream.jsonl", edits)

    use_scr = method in {"SCR", "SCR-LITE", "SCR_LITE"}
    if use_scr:
        model, tokenizer = load_transformers_model(
            args.model,
            dtype=args.dtype,
            device_map=args.device_map,
            cache_dir=args.models_cache,
        )
        baseline_weights = None
    else:
        editor = load_easyedit_editor(
            method=method,
            model_name=args.model,
            easyedit_root=args.easyedit_root,
            hparams_path=args.hparams,
        )
        model, tokenizer = get_editor_model_tokenizer(editor)
        baseline_weights = capture_weight_snapshot(model, name_filter=args.weight_name_filter)

    baseline_hidden = collect_hidden_states(
        model=model,
        tokenizer=tokenizer,
        probes=probes,
        max_prompts=args.probe_limit,
        batch_size=args.hidden_batch_size,
        max_input_length=args.max_input_length,
    )

    aggregate_rows: list[dict] = []
    if 0 in checkpoints:
        result = evaluate_checkpoint(
            model=model,
            tokenizer=tokenizer,
            probes=probes,
            edits=edits,
            run_dir=run_dir,
            checkpoint=0,
            args=args,
            baseline_hidden=baseline_hidden,
            baseline_weights=baseline_weights,
            use_scr=use_scr,
        )
        aggregate_rows.append({"checkpoint": 0, **result["metrics"]["dimensions"], **result["internal"]})

    if use_scr:
        for checkpoint in checkpoints:
            if checkpoint == 0:
                continue
            result = evaluate_checkpoint(
                model=model,
                tokenizer=tokenizer,
                probes=probes,
                edits=edits,
                run_dir=run_dir,
                checkpoint=checkpoint,
                args=args,
                baseline_hidden=baseline_hidden,
                baseline_weights=baseline_weights,
                use_scr=True,
            )
            aggregate_rows.append({"checkpoint": checkpoint, **result["metrics"]["dimensions"], **result["internal"]})
    else:
        checkpoints_after_zero = [cp for cp in checkpoints if cp > 0]
        if checkpoints_after_zero:
            for edit_idx, edit in enumerate(edits, start=1):
                model, edit_info = apply_single_edit(editor, edit)
                if edit_idx in checkpoints_after_zero:
                    write_json(checkpoint_dir(run_dir, edit_idx) / "last_edit_info.json", edit_info)
                    result = evaluate_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        probes=probes,
                        edits=edits,
                        run_dir=run_dir,
                        checkpoint=edit_idx,
                        args=args,
                        baseline_hidden=baseline_hidden,
                        baseline_weights=baseline_weights,
                        use_scr=False,
                    )
                    aggregate_rows.append({"checkpoint": edit_idx, **result["metrics"]["dimensions"], **result["internal"]})
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                if edit_idx >= max(checkpoints_after_zero):
                    break

    write_csv(run_dir / "aggregate_metrics.csv", aggregate_rows)
    print(f"运行完成，结果目录：{run_dir}")


if __name__ == "__main__":
    main()
