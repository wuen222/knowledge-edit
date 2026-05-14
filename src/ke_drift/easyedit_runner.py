from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


HPARAM_CLASS_BY_METHOD = {
    "ROME": ("ROMEHyperParams",),
    "R-ROME": ("R_ROMEHyperParams", "RROMEHyperParams", "ROMEHyperParams"),
    "MEMIT": ("MEMITHyperParams",),
    "PMET": ("PMETHyperParams",),
    "ALPHAEDIT": ("AlphaEditHyperParams",),
    "ALPHA-EDIT": ("AlphaEditHyperParams",),
    "FT": ("FTHyperParams",),
    "FT-L": ("FTHyperParams", "FTLHyperParams"),
}

HPARAM_DIR_CANDIDATES = {
    "FT-L": ("FT", "FT-L"),
    "ALPHAEDIT": ("AlphaEdit", "ALPHAEDIT", "Alpha-Edit", "ALPHA-EDIT"),
    "ALPHA-EDIT": ("AlphaEdit", "ALPHAEDIT", "Alpha-Edit", "ALPHA-EDIT"),
}


def normalize_method(method: str) -> str:
    return method.strip().upper()


def add_easyedit_to_path(easyedit_root: str | Path | None) -> None:
    if not easyedit_root:
        return
    root = Path(easyedit_root)
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _normalize_dir_name(value: str) -> str:
    return value.replace("-", "").replace("_", "").lower()


def find_hparams_dir(root: Path, method_key: str) -> Path:
    hparams_root = root / "hparams"
    candidate_names = HPARAM_DIR_CANDIDATES.get(method_key, (method_key,))
    for name in candidate_names:
        candidate = hparams_root / name
        if candidate.exists():
            return candidate

    if hparams_root.exists():
        normalized_candidates = {_normalize_dir_name(name) for name in candidate_names}
        for child in hparams_root.iterdir():
            if child.is_dir() and _normalize_dir_name(child.name) in normalized_candidates:
                return child

    expected = hparams_root / candidate_names[0]
    raise FileNotFoundError(
        f"没有找到 EasyEdit hparams 目录：{expected}。"
        f"已尝试目录名：{', '.join(candidate_names)}。"
    )


def find_hparams_file(easyedit_root: str | Path, method: str, model_name: str) -> Path:
    root = Path(easyedit_root)
    method_key = normalize_method(method)
    method_dir = find_hparams_dir(root, method_key)

    candidates = list(method_dir.rglob("*.yaml")) + list(method_dir.rglob("*.yml"))
    if not candidates:
        raise FileNotFoundError(f"在 {method_dir} 下没有找到 YAML hparams 文件。")

    model_l = model_name.lower()
    aliases = []
    if "llama-3" in model_l or "llama3" in model_l:
        aliases.extend(["llama3", "llama-3", "llama_3", "llama"])
    if "llama" in model_l:
        aliases.append("llama")
    if "mistral" in model_l:
        aliases.append("mistral")
    if "qwen" in model_l:
        aliases.append("qwen")
    if "8b" in model_l:
        aliases.append("8b")
    if "7b" in model_l:
        aliases.append("7b")

    def score(path: Path) -> tuple[int, int]:
        text = str(path).lower()
        alias_score = sum(1 for alias in aliases if alias in text)
        instruct_score = 1 if "instruct" in text or "chat" in text else 0
        return alias_score + instruct_score, -len(text)

    best = max(candidates, key=score)
    if score(best)[0] <= 0:
        raise FileNotFoundError(
            f"无法为 model={model_name} method={method} 自动推断 hparams。"
            f"请显式传入 --hparams。已搜索目录：{method_dir}。"
        )
    return best


def _resolve_hparams_class(method: str):
    import easyeditor

    method_key = normalize_method(method)
    candidates = HPARAM_CLASS_BY_METHOD.get(method_key, (f"{method_key}HyperParams",))
    for class_name in candidates:
        if hasattr(easyeditor, class_name):
            return getattr(easyeditor, class_name)
    available = [name for name in dir(easyeditor) if name.endswith("HyperParams")]
    raise AttributeError(
        f"没有找到 {method} 对应的 EasyEdit hparams 类。已尝试：{candidates}。"
        f"当前可用类包括：{available[:30]}"
    )


def load_easyedit_editor(
    method: str,
    model_name: str,
    easyedit_root: str | Path | None = None,
    hparams_path: str | Path | None = None,
) -> Any:
    add_easyedit_to_path(easyedit_root)
    from easyeditor import BaseEditor

    if hparams_path is None:
        if easyedit_root is None:
            raise ValueError("请传入 --hparams 或 --easyedit-root，以便定位 hparams。")
        hparams_path = find_hparams_file(easyedit_root, method, model_name)

    hparam_class = _resolve_hparams_class(method)
    hparams = hparam_class.from_hparams(str(hparams_path))
    for attr in ("model_name", "tokenizer_name"):
        if hasattr(hparams, attr):
            setattr(hparams, attr, model_name)
    return BaseEditor.from_hparams(hparams)


def get_editor_model_tokenizer(editor: Any) -> tuple[Any, Any]:
    model = getattr(editor, "model", None)
    if model is None:
        raise AttributeError("EasyEdit editor 没有暴露 .model。")
    tokenizer = getattr(editor, "tok", None) or getattr(editor, "tokenizer", None)
    if tokenizer is None:
        raise AttributeError("EasyEdit editor 没有暴露 .tok 或 .tokenizer。")
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def apply_single_edit(editor: Any, edit: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "prompts": [edit["prompt"]],
        "ground_truth": [edit.get("ground_truth") or None],
        "target_new": [edit["target_new"]],
        "sequential_edit": True,
    }
    if edit.get("subject"):
        kwargs["subject"] = [edit["subject"]]

    try:
        metrics, edited_model, extra = editor.edit(**kwargs)
    except TypeError:
        kwargs.pop("subject", None)
        metrics, edited_model, extra = editor.edit(**kwargs)

    if edited_model is not None and hasattr(editor, "model"):
        editor.model = edited_model
    return edited_model if edited_model is not None else editor.model, {"easyedit_metrics": metrics, "extra": str(type(extra))}
