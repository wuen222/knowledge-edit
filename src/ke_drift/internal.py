from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import ensure_dir, write_csv, write_json


def _model_device(model: Any):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def _layer_from_name(name: str) -> int:
    patterns = (r"layers\.(\d+)", r"h\.(\d+)", r"block\.(\d+)", r"transformer\.(\d+)")
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return -1


def _cosine_distance_np(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x_norm = np.linalg.norm(x, axis=-1)
    y_norm = np.linalg.norm(y, axis=-1)
    dot = np.sum(x * y, axis=-1)
    return 1.0 - dot / np.maximum(x_norm * y_norm, eps)


def linear_cka(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xty = x.T @ y
    xtx = x.T @ x
    yty = y.T @ y
    numerator = float(np.sum(xty * xty))
    denominator = math.sqrt(float(np.sum(xtx * xtx)) * float(np.sum(yty * yty)))
    return numerator / max(denominator, eps)


def collect_hidden_states(
    model: Any,
    tokenizer: Any,
    probes: list[dict[str, Any]],
    max_prompts: int = 200,
    batch_size: int = 4,
    max_input_length: int = 256,
) -> dict[str, Any]:
    import torch
    from tqdm import tqdm

    selected = probes[:max_prompts]
    prompts = [str(item["prompt"]) for item in selected]
    prompt_ids = [str(item.get("probe_id", idx)) for idx, item in enumerate(selected)]
    categories = [str(item.get("category", "")) for item in selected]
    device = _model_device(model)
    collected: list[np.ndarray] | None = None

    model.eval()
    for start in tqdm(range(0, len(prompts), batch_size), desc="抽取hidden", leave=False):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        last_indices = encoded["attention_mask"].sum(dim=1) - 1
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states[1:] if len(outputs.hidden_states) > 1 else outputs.hidden_states
        batch_layers: list[np.ndarray] = []
        for layer_hidden in hidden_states:
            rows = layer_hidden[torch.arange(layer_hidden.shape[0], device=layer_hidden.device), last_indices]
            batch_layers.append(rows.detach().float().cpu().numpy())
        if collected is None:
            collected = [arr for arr in batch_layers]
        else:
            for layer_idx, arr in enumerate(batch_layers):
                collected[layer_idx] = np.concatenate([collected[layer_idx], arr], axis=0)

    if collected is None:
        embeddings = np.zeros((0, 0, 0), dtype=np.float32)
    else:
        embeddings = np.stack(collected, axis=0).astype(np.float32)

    return {
        "embeddings": embeddings,
        "prompt_ids": np.array(prompt_ids),
        "categories": np.array(categories),
        "prompts": np.array(prompts),
    }


def save_hidden_states(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    np.savez_compressed(
        target,
        embeddings=payload["embeddings"],
        prompt_ids=payload["prompt_ids"],
        categories=payload["categories"],
        prompts=payload["prompts"],
    )


def load_hidden_states(path: str | Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {
        "embeddings": data["embeddings"],
        "prompt_ids": data["prompt_ids"],
        "categories": data["categories"],
        "prompts": data["prompts"],
    }


def compare_hidden_states(
    baseline: dict[str, Any],
    current: dict[str, Any],
    checkpoint: int,
    output_csv: str | Path | None = None,
) -> list[dict[str, Any]]:
    base = baseline["embeddings"]
    cur = current["embeddings"]
    if base.shape != cur.shape:
        raise ValueError(f"hidden state 形状不一致：baseline={base.shape}, current={cur.shape}")
    rows: list[dict[str, Any]] = []
    for layer_idx in range(base.shape[0]):
        distances = _cosine_distance_np(base[layer_idx], cur[layer_idx])
        rows.append(
            {
                "checkpoint": checkpoint,
                "layer": layer_idx,
                "mean_cosine_distance": float(np.mean(distances)) if distances.size else 0.0,
                "median_cosine_distance": float(np.median(distances)) if distances.size else 0.0,
                "p95_cosine_distance": float(np.percentile(distances, 95)) if distances.size else 0.0,
                "linear_cka": linear_cka(base[layer_idx], cur[layer_idx]) if base.shape[1] >= 2 else 1.0,
                "n_prompts": int(base.shape[1]),
            }
        )
    if output_csv is not None:
        write_csv(output_csv, rows)
    return rows


def capture_weight_snapshot(model: Any, name_filter: str = "model.layers.") -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name, param in model.named_parameters():
        if name_filter and name_filter not in name:
            continue
        if not param.requires_grad and param.ndim < 2:
            continue
        snapshot[name] = param.detach().cpu().clone()
    return snapshot


def compute_weight_deltas(
    baseline: dict[str, Any],
    model: Any,
    checkpoint: int,
    name_filter: str = "model.layers.",
    output_csv: str | Path | None = None,
) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    for name, param in model.named_parameters():
        if name not in baseline:
            continue
        if name_filter and name_filter not in name:
            continue
        base = baseline[name].float()
        cur = param.detach().float().cpu()
        diff = cur - base
        diff_norm = float(torch.linalg.vector_norm(diff).item())
        base_norm = float(torch.linalg.vector_norm(base).item())
        cur_norm = float(torch.linalg.vector_norm(cur).item())
        dot = float(torch.sum(base.flatten() * cur.flatten()).item())
        denom = max(base_norm * cur_norm, 1e-12)
        rows.append(
            {
                "checkpoint": checkpoint,
                "name": name,
                "layer": _layer_from_name(name),
                "delta_l2": diff_norm,
                "base_l2": base_norm,
                "relative_l2": diff_norm / max(base_norm, 1e-12),
                "cosine_distance": 1.0 - dot / denom,
                "numel": int(param.numel()),
            }
        )
    if output_csv is not None:
        write_csv(output_csv, rows)
    return rows


def summarize_internal(
    checkpoint: int,
    hidden_rows: list[dict[str, Any]],
    weight_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> dict[str, Any]:
    hidden_mean = float(np.mean([row["mean_cosine_distance"] for row in hidden_rows])) if hidden_rows else 0.0
    hidden_max = float(np.max([row["mean_cosine_distance"] for row in hidden_rows])) if hidden_rows else 0.0
    weight_mean = float(np.mean([row["relative_l2"] for row in weight_rows])) if weight_rows else 0.0
    weight_max = float(np.max([row["relative_l2"] for row in weight_rows])) if weight_rows else 0.0
    summary = {
        "checkpoint": checkpoint,
        "hidden_mean_cosine_distance": hidden_mean,
        "hidden_max_layer_mean_cosine_distance": hidden_max,
        "weight_mean_relative_l2": weight_mean,
        "weight_max_relative_l2": weight_max,
    }
    write_json(output_path, summary)
    return summary


def collect_token_trajectory(
    model: Any,
    tokenizer: Any,
    edit: dict[str, Any],
    checkpoint: int,
    top_k: int = 5,
    max_input_length: int = 256,
) -> list[dict[str, Any]]:
    import torch

    prompt = str(edit.get("prompt") or "")
    target_new = str(edit.get("target_new") or "")
    ground_truth = str(edit.get("ground_truth") or "")
    if not prompt:
        return []
    device = _model_device(model)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    last_idx = encoded["attention_mask"].sum(dim=1) - 1
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)

    hidden_states = outputs.hidden_states[1:] if len(outputs.hidden_states) > 1 else outputs.hidden_states
    lm_head = getattr(model, "lm_head", None) or getattr(getattr(model, "model", None), "lm_head", None)
    norm = getattr(getattr(model, "model", None), "norm", None)
    if lm_head is None:
        return []

    track_ids: dict[int, str] = {}
    for label, text in (("target_new", target_new), ("ground_truth", ground_truth)):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            track_ids[int(ids[0])] = label

    rows: list[dict[str, Any]] = []
    for layer_idx, layer_hidden in enumerate(hidden_states):
        vector = layer_hidden[0, last_idx.item(), :].unsqueeze(0)
        if norm is not None:
            vector = norm(vector)
        logits = lm_head(vector)
        probs = torch.softmax(logits.float(), dim=-1)[0]
        top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.numel()))
        ids_for_layer = dict(track_ids)
        for token_id, prob in zip(top_ids.tolist(), top_probs.tolist()):
            ids_for_layer.setdefault(int(token_id), "top")
        for token_id, label in ids_for_layer.items():
            token = tokenizer.decode([token_id])
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "edit_id": edit.get("edit_id", ""),
                    "layer": layer_idx,
                    "token_id": token_id,
                    "token": token,
                    "token_label": label,
                    "probability": float(probs[token_id].item()),
                }
            )
    return rows


def write_token_trajectory(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
