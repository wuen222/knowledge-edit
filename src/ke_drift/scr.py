from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def build_memory(edits: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, str]]:
    memory: list[dict[str, str]] = []
    for edit in edits[:limit]:
        subject = str(edit.get("subject") or "").strip()
        prompt = str(edit.get("prompt") or "").strip()
        target = str(edit.get("target_new") or "").strip()
        if not target:
            continue
        fact = f"{subject} -> {target}" if subject else f"{prompt} -> {target}"
        memory.append({"edit_id": str(edit.get("edit_id", "")), "subject": subject, "prompt": prompt, "target": target, "fact": fact})
    return memory


def _score(query_tokens: Counter[str], item: dict[str, str]) -> float:
    doc_tokens = Counter(tokenize(" ".join([item.get("subject", ""), item.get("prompt", ""), item.get("target", "")])))
    if not query_tokens or not doc_tokens:
        return 0.0
    overlap = sum(min(query_tokens[tok], doc_tokens[tok]) for tok in query_tokens)
    norm = math.sqrt(sum(v * v for v in query_tokens.values())) * math.sqrt(sum(v * v for v in doc_tokens.values()))
    return overlap / norm if norm else 0.0


def retrieve_memory(prompt: str, memory: list[dict[str, str]], top_k: int = 3, min_score: float = 0.05) -> list[dict[str, str]]:
    query_tokens = Counter(tokenize(prompt))
    scored = [(_score(query_tokens, item), item) for item in memory]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for score, item in scored[:top_k] if score >= min_score]


def build_scr_prompt(prompt: str, memory: list[dict[str, str]], top_k: int = 3) -> str:
    retrieved = retrieve_memory(prompt, memory, top_k=top_k)
    if not retrieved:
        return prompt
    facts = "\n".join(f"- {item['fact']}" for item in retrieved)
    return (
        "Use the following updated facts only if they are relevant.\n"
        f"{facts}\n\n"
        f"Question: {prompt}\n"
        "Answer:"
    )

