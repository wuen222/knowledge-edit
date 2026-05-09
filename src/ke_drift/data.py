from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, read_json_records, write_json, write_jsonl


@dataclass
class EditRecord:
    edit_id: str
    dataset: str
    subject: str
    prompt: str
    target_new: str
    ground_truth: str = ""
    rephrase_prompts: list[str] = field(default_factory=list)
    locality_prompts: list[dict[str, str]] = field(default_factory=list)
    portability_prompts: list[dict[str, str]] = field(default_factory=list)
    raw_index: int = -1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


GENERAL_PROBES = [
    "Answer briefly: what is 17 plus 28?",
    "Answer briefly: name the capital city of France.",
    "Answer briefly: if all roses are flowers and some flowers fade, what can be inferred?",
    "Answer briefly: translate 'knowledge' into Spanish.",
    "Answer briefly: write one synonym for 'reliable'.",
]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("str", "text", "answer", "target", "value", "name"):
            if key in value:
                return _text(value[key])
        return ""
    if isinstance(value, list):
        for item in value:
            item_text = _text(item)
            if item_text:
                return item_text
        return ""
    return str(value).strip()


def _first_record_text(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in record:
            value = _text(record[key])
            if value:
                return value
    return ""


def _collect_prompts(value: Any) -> list[str]:
    prompts: list[str] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            prompt = _first_record_text(item, ("prompt", "src", "question", "input", "text"))
        else:
            prompt = _text(item)
        if prompt:
            prompts.append(prompt)
    return prompts


def _collect_prompt_targets(value: Any, default_target: str = "") -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    items = value.values() if isinstance(value, dict) and "prompt" not in value else _as_list(value)
    for item in items:
        if isinstance(item, dict):
            prompt = _first_record_text(item, ("prompt", "src", "question", "input", "text"))
            target = _first_record_text(item, ("target", "target_new", "answer", "ground_truth", "label", "output"))
            if not target:
                target = default_target
        else:
            prompt = _text(item)
            target = default_target
        if prompt:
            pairs.append({"prompt": prompt, "target": target})
    return pairs


def normalize_record(record: dict[str, Any], index: int, dataset: str) -> EditRecord | None:
    rewrite = record.get("requested_rewrite")
    subject = _first_record_text(record, ("subject", "entity", "head", "name"))
    prompt = _first_record_text(record, ("prompt", "src", "question", "input", "edit_prompt"))
    target_new = _first_record_text(record, ("target_new", "alt", "answer", "target", "edit_target"))
    ground_truth = _first_record_text(record, ("ground_truth", "target_true", "answers", "original_answer", "pred"))

    if isinstance(rewrite, dict):
        subject = subject or _text(rewrite.get("subject"))
        target_new = target_new or _text(rewrite.get("target_new"))
        ground_truth = ground_truth or _text(rewrite.get("target_true"))
        template = _text(rewrite.get("prompt"))
        if template:
            prompt = template.format(subject) if "{}" in template else template

    if not prompt or not target_new:
        return None

    rephrase_prompts: list[str] = []
    for key in ("rephrase", "rephrases", "rephrase_prompts", "paraphrase_prompts"):
        rephrase_prompts.extend(_collect_prompts(record.get(key)))

    locality_prompts: list[dict[str, str]] = []
    for key in ("locality", "loc", "neighborhood", "neighborhood_prompts"):
        locality_prompts.extend(_collect_prompt_targets(record.get(key), default_target=ground_truth))

    portability_prompts: list[dict[str, str]] = []
    for key in ("portability", "portability_prompts", "subject_replace", "inverse_relation", "one_hop"):
        portability_prompts.extend(_collect_prompt_targets(record.get(key), default_target=target_new))

    return EditRecord(
        edit_id=f"{dataset}-{index:06d}",
        dataset=dataset,
        subject=subject,
        prompt=prompt,
        target_new=target_new,
        ground_truth=ground_truth,
        rephrase_prompts=dedupe_keep_order(rephrase_prompts),
        locality_prompts=dedupe_prompt_targets(locality_prompts),
        portability_prompts=dedupe_prompt_targets(portability_prompts),
        raw_index=index,
    )


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def dedupe_prompt_targets(values: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, str]] = []
    for item in values:
        prompt = item.get("prompt", "").strip()
        target = item.get("target", "").strip()
        key = (prompt, target)
        if prompt and key not in seen:
            seen.add(key)
            result.append({"prompt": prompt, "target": target})
    return result


def default_input_path(dataset: str, data_root: str | Path) -> Path:
    root = Path(data_root)
    candidates = {
        "zsre": [
            root / "editing-data" / "zsre" / "zsre_mend_eval.json",
            root / "zsre" / "zsre_mend_eval.json",
            root / "zsre_mend_eval.json",
        ],
        "counterfact": [
            root / "editing-data" / "counterfact" / "counterfact-edit.json",
            root / "counterfact" / "counterfact-edit.json",
            root / "counterfact-edit.json",
        ],
    }
    for candidate in candidates.get(dataset.lower(), []):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"没有找到 {dataset} 的默认原始数据文件。请传入 --input，或把数据放到 {root} 下。"
    )


def load_normalized_records(path: str | Path, dataset: str) -> list[EditRecord]:
    raw_records = read_json_records(path)
    normalized: list[EditRecord] = []
    for idx, record in enumerate(raw_records):
        item = normalize_record(record, idx, dataset)
        if item is not None:
            normalized.append(item)
    if not normalized:
        raise ValueError(f"在 {path} 中没有找到可用的编辑样本。")
    return normalized


def build_probe_set(edits: list[EditRecord], controls: list[EditRecord], general_prompts: list[str] | None = None) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for edit in edits:
        probes.append(
            {
                "probe_id": f"{edit.edit_id}:edit",
                "category": "edit",
                "edit_id": edit.edit_id,
                "prompt": edit.prompt,
                "target": edit.target_new,
            }
        )
        for idx, prompt in enumerate(edit.rephrase_prompts):
            probes.append(
                {
                    "probe_id": f"{edit.edit_id}:rephrase:{idx}",
                    "category": "rephrase",
                    "edit_id": edit.edit_id,
                    "prompt": prompt,
                    "target": edit.target_new,
                }
            )
        for idx, pair in enumerate(edit.locality_prompts):
            probes.append(
                {
                    "probe_id": f"{edit.edit_id}:locality:{idx}",
                    "category": "locality",
                    "edit_id": edit.edit_id,
                    "prompt": pair["prompt"],
                    "target": pair.get("target", ""),
                }
            )
        for idx, pair in enumerate(edit.portability_prompts):
            probes.append(
                {
                    "probe_id": f"{edit.edit_id}:portability:{idx}",
                    "category": "portability",
                    "edit_id": edit.edit_id,
                    "prompt": pair["prompt"],
                    "target": pair.get("target", edit.target_new),
                }
            )

    for idx, control in enumerate(controls):
        target = control.ground_truth or control.target_new
        probes.append(
            {
                "probe_id": f"control:{idx}",
                "category": "locality",
                "edit_id": "",
                "prompt": control.prompt,
                "target": target,
            }
        )

    for idx, prompt in enumerate(general_prompts or GENERAL_PROBES):
        probes.append(
            {
                "probe_id": f"general:{idx}",
                "category": "general",
                "edit_id": "",
                "prompt": prompt,
                "target": "",
            }
        )

    return probes


def prepare_dataset(
    input_path: str | Path,
    dataset: str,
    output_dir: str | Path,
    n_edits: int,
    seed: int,
    n_controls: int = 100,
) -> dict[str, Any]:
    records = load_normalized_records(input_path, dataset)
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    edits = shuffled[:n_edits]
    controls = shuffled[n_edits : n_edits + n_controls]
    if len(edits) < n_edits:
        raise ValueError(f"请求抽取 {n_edits} 条编辑，但只找到 {len(edits)} 条可用样本。")

    probes = build_probe_set(edits, controls)
    out = ensure_dir(output_dir)
    write_jsonl(out / "edits.jsonl", [item.to_dict() for item in edits])
    write_jsonl(out / "probes.jsonl", probes)
    manifest = {
        "dataset": dataset,
        "input_path": str(input_path),
        "seed": seed,
        "n_raw_records": len(records),
        "n_edits": len(edits),
        "n_controls": len(controls),
        "n_probes": len(probes),
    }
    write_json(out / "manifest.json", manifest)
    return manifest
