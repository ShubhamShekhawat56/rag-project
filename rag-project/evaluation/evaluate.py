"""Minimal retrieval + answer evaluation against the 5-question gold set (+1 unanswerable).

Usage:  python -m evaluation.evaluate [--chunk-size 120] [--overlap 18]

The sample handbook is only ~700 tokens, so with production settings (1000-token chunks) it
is a single chunk and retrieval would be trivially perfect. By default this script therefore
re-indexes it with small 120-token chunks (15% overlap) to make retrieval non-trivial, in an
isolated namespace that is deleted afterwards.
"""
import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

from app import config, rag, vectorstore

log = logging.getLogger("rag.eval")
ROOT = Path(__file__).resolve().parent.parent


def contains_any(text, needles):
    t = (text or "").lower().replace("\u2019", "'")
    return any(n.lower() in t for n in (needles or []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=120)
    ap.add_argument("--overlap", type=int, default=18)
    args = ap.parse_args()

    if args.chunk_size <= 0:
        sys.exit("--chunk-size must be greater than 0")
    if args.overlap < 0 or args.overlap >= args.chunk_size:
        sys.exit("--overlap must be >= 0 and less than --chunk-size")

    if config.missing_keys():
        sys.exit(f"Missing env vars: {', '.join(config.missing_keys())}")

    vectorstore.ensure_collection()
    gold = json.loads((ROOT / "evaluation" / "gold_set.json").read_text("utf-8"))
    text = (ROOT / gold["document"]).read_text("utf-8")
    ns = "eval-" + uuid.uuid4().hex[:10]
    info = rag.ingest(text, "Northwind handbook", "sample", ns, args.chunk_size, args.overlap)
    print(f"Indexed {info['chunks']} chunks (size={args.chunk_size}, overlap={args.overlap})\n")

    rows, hits, prec_sum, ans_ok, n_ans = [], 0, 0.0, 0, 0
    refusal_ok = None
    try:
        for item in gold["questions"]:
            res = rag.answer(item["q"], ns)
            q_label = item["q"].replace("|", "\\|")
            if item["expect_any"] is None:  # unanswerable question -> must refuse
                refusal_ok = bool(res.get("no_answer"))
                rows.append((q_label, "n/a", "n/a", "refused" if refusal_ok else "HALLUCINATED"))
                continue
            n_ans += 1
            sources = res.get("sources") or []
            relevant = [contains_any(s.get("text", ""), item["expect_any"]) for s in sources]
            hit = any(relevant)
            hits += int(hit)
            prec_sum += sum(relevant) / max(len(relevant), 1)
            correct = (not res.get("no_answer")) and contains_any(res.get("answer", ""), item["expect_any"])
            ans_ok += int(correct)
            rows.append((
                q_label,
                "yes" if hit else "no",
                f"{sum(relevant)}/{len(relevant)}",
                "correct" if correct else "wrong",
            ))
    finally:
        try:
            vectorstore.clear_namespace(ns)
        except Exception as e:
            log.warning("Failed to clean up eval namespace %s: %s", ns, e)

    denom = max(n_ans, 1)
    refusal_label = "n/a" if refusal_ok is None else ("yes" if refusal_ok else "NO")
    lines = ["| Question | Evidence retrieved | Relevant in top-k | Answer |", "|---|---|---|---|"]
    lines += [f"| {q} | {h} | {p} | {a} |" for q, h, p, a in rows]
    summary = (
        f"\n**Recall@{config.RERANK_TOP_N} (hit rate):** {hits}/{n_ans}  \n"
        f"**Mean precision@{config.RERANK_TOP_N}:** {prec_sum / denom:.2f}  \n"
        f"**Answer success rate:** {ans_ok}/{n_ans}  \n"
        f"**Unanswerable question refused:** {refusal_label}\n"
    )
    out = "\n".join(lines) + "\n" + summary
    print(out)
    (ROOT / "evaluation" / "results.md").write_text(out, "utf-8")
    print("Saved to evaluation/results.md")


if __name__ == "__main__":
    main()