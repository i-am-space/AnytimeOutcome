"""Workload manifests: chat / code / doc-grounded prompt families (proposal Sec. 9.2).

Sources (raw files under $SSD_DATASET_DIR/../raw):
  chat : Spec-Bench MT-bench categories (writing, roleplay, reasoning, math, extraction, stem, humanities)
  code : OpenAI HumanEval prompts
  doc  : Spec-Bench summarization + rag (document-grounded generation)

Prompts are rendered with the Qwen3 chat template (thinking disabled) and tokenized, so every
decoding policy sees identical token IDs. Tuning vs held-out splits are fixed by a seeded shuffle.
"""
import gzip
import json
import os
import random

CHAT_CATS = {"writing", "roleplay", "reasoning", "math", "extraction", "stem", "humanities"}
DOC_CATS = {"summarization", "rag"}


def _raw_dir():
    base = os.environ.get("SSD_DATASET_DIR", f"/scratch/{os.environ.get('USER')}/datasets/processed_datasets")
    return os.path.join(os.path.dirname(base.rstrip("/")), "raw")


def load_family(family: str):
    raw = _raw_dir()
    if family in ("chat", "doc"):
        cats = CHAT_CATS if family == "chat" else DOC_CATS
        rows = [json.loads(l) for l in open(os.path.join(raw, "spec_bench_question.jsonl"))]
        return [dict(id=f"specbench-{r['question_id']}", family=family, category=r["category"], text=r["turns"][0])
                for r in rows if r["category"] in cats]
    if family == "code":
        with gzip.open(os.path.join(raw, "HumanEval.jsonl.gz"), "rt") as f:
            rows = [json.loads(l) for l in f]
        return [dict(id=r["task_id"], family="code", category="humaneval",
                     text="Complete the following Python function.\n\n" + r["prompt"]) for r in rows]
    raise ValueError(family)


def get_prompts(families, n_per_family, split="heldout", seed=0, tokenizer=None, max_prompt_tokens=None):
    """Returns list of dicts with id/family/category/text and (if tokenizer) token_ids."""
    out = []
    for fam in families:
        items = load_family(fam)
        random.Random(seed).shuffle(items)
        half = len(items) // 2
        items = items[:half] if split == "tune" else items[half:]
        out.extend(items[:n_per_family])
    if tokenizer is not None:
        for it in out:
            msgs = [{"role": "user", "content": it["text"]}]
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if max_prompt_tokens:
                ids = ids[-max_prompt_tokens:]
            it["token_ids"] = ids
    return out
