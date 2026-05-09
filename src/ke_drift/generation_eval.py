from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .io_utils import ensure_dir, write_json, write_jsonl


PromptTransform = Callable[[str, dict[str, Any]], str]


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_matches(output: str, target: str) -> bool:
    if not target:
        return False
    norm_output = normalize_answer(output)
    targets = [target] if isinstance(target, str) else list(target)
    return any(normalize_answer(item) in norm_output for item in targets if normalize_answer(item))


def repetition_score(text: str, n: int = 3) -> float:
    tokens = normalize_answer(text).split()
    if len(tokens) < n * 2:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    if not counts:
        return 0.0
    return max(counts.values()) / len(ngrams)


def failure_tags(output: str, target: str, prompt: str) -> list[str]:
    tags: list[str] = []
    stripped = output.strip()
    if not stripped:
        tags.append("empty")
    if target and not answer_matches(stripped, target):
        tags.append("answer_miss")
    if repetition_score(stripped) >= 0.25:
        tags.append("repetition")
    if len(stripped.split()) <= 2 and target and not answer_matches(stripped, target):
        tags.append("too_short")
    norm_prompt = normalize_answer(prompt)
    norm_output = normalize_answer(stripped)
    if norm_prompt and len(norm_prompt) > 30 and norm_prompt in norm_output:
        tags.append("prompt_copy")
    return tags


def load_transformers_model(model_name: str, dtype: str = "float16", device_map: str = "auto", cache_dir: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=dtype_map.get(dtype, torch.float16),
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _model_device(model: Any):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def generate_texts(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int = 32,
    batch_size: int = 4,
    max_input_length: int = 256,
) -> list[str]:
    import torch
    from tqdm import tqdm

    outputs: list[str] = []
    device = _model_device(model)
    for start in tqdm(range(0, len(prompts), batch_size), desc="生成", leave=False):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for row_idx, sequence in enumerate(generated):
            prompt_len = int(encoded["attention_mask"][row_idx].sum().item())
            new_tokens = sequence[prompt_len:]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs


def score_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in predictions:
        by_category[item["category"]].append(item)

    category_scores: dict[str, dict[str, float]] = {}
    for category, rows in by_category.items():
        scored = [row for row in rows if row.get("target")]
        if scored:
            acc = sum(1 for row in scored if row["match"]) / len(scored)
        else:
            acc = 0.0
        category_scores[category] = {
            "accuracy": acc,
            "n": float(len(rows)),
            "n_scored": float(len(scored)),
            "repetition_rate": sum(1 for row in rows if "repetition" in row.get("failure_tags", [])) / max(len(rows), 1),
        }

    dimension_map = {
        "reliability": "edit",
        "generalization": "rephrase",
        "locality": "locality",
        "portability": "portability",
    }
    dimensions = {
        name: category_scores.get(category, {"accuracy": 0.0, "n": 0.0, "n_scored": 0.0})["accuracy"]
        for name, category in dimension_map.items()
    }
    scored_dimensions = [
        score
        for name, score in dimensions.items()
        if category_scores.get(dimension_map[name], {"n_scored": 0.0})["n_scored"] > 0
    ]
    dimensions["average"] = sum(scored_dimensions) / len(scored_dimensions) if scored_dimensions else 0.0

    failure_counts: Counter[str] = Counter(tag for row in predictions for tag in row.get("failure_tags", []))
    return {
        "dimensions": dimensions,
        "categories": category_scores,
        "failure_counts": dict(failure_counts),
        "n_predictions": len(predictions),
    }


def evaluate_model_on_probes(
    model: Any,
    tokenizer: Any,
    probes: list[dict[str, Any]],
    output_dir: str | Path,
    checkpoint: int,
    max_new_tokens: int = 32,
    batch_size: int = 4,
    max_input_length: int = 256,
    prompt_transform: PromptTransform | None = None,
) -> dict[str, Any]:
    out = ensure_dir(output_dir)
    eval_prompts = [
        prompt_transform(item["prompt"], item) if prompt_transform else item["prompt"]
        for item in probes
    ]
    generations = generate_texts(
        model,
        tokenizer,
        eval_prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        max_input_length=max_input_length,
    )
    predictions: list[dict[str, Any]] = []
    for probe, eval_prompt, output in zip(probes, eval_prompts, generations):
        target = str(probe.get("target") or "")
        predictions.append(
            {
                "checkpoint": checkpoint,
                "probe_id": probe.get("probe_id", ""),
                "category": probe.get("category", ""),
                "edit_id": probe.get("edit_id", ""),
                "prompt": probe.get("prompt", ""),
                "eval_prompt": eval_prompt,
                "target": target,
                "output": output,
                "match": answer_matches(output, target) if target else False,
                "failure_tags": failure_tags(output, target, str(probe.get("prompt", ""))),
            }
        )

    metrics = score_predictions(predictions)
    metrics["checkpoint"] = checkpoint
    write_jsonl(out / "predictions.jsonl", predictions)
    write_json(out / "metrics.json", metrics)
    return metrics
