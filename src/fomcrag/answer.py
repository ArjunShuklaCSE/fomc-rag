"""Grounded answers with inline citations, abstention, and answer-quality evaluation.

    python -m fomcrag.answer "What did the FOMC decide in March 2020?"
    python -m fomcrag.answer --calibrate              # pick the abstention threshold on dev
    python -m fomcrag.answer --eval --split test      # correctness, faithfulness, abstention

Two layers of "I don't know":
  1. retrieval gate: if the cross-encoder's best score is below a threshold calibrated on dev, the LLM is never
     called (cheap, and a model can't hallucinate from context it never saw);
  2. the prompt: answer only from the numbered sources, otherwise reply exactly "I don't know."
"""

import argparse
import json
import re
import statistics
import time
from functools import lru_cache
from pathlib import Path

from .clean import page_of
from .evaluate import experiments, merge
from .gold import load_docs, load_gold
from .llm import backend_name, chat

RESULTS = Path("results")
THRESHOLD_FILE = RESULTS / "abstain_threshold.json"
IDK = "I don't know."
TOP_K = 5
SYSTEM = f"""You answer questions about Federal Open Market Committee (FOMC) documents using ONLY the numbered sources.
Rules:
- Every sentence must end with at least one citation like [1] or [2][3], naming the sources that support it.
- Use only facts stated in the sources. Do not use outside knowledge.
- If the sources do not contain the answer, reply exactly: {IDK}
- Be concise: 1-4 sentences."""
CITE = re.compile(r"\[(\d+)\]")


@lru_cache(maxsize=1)
def serving():
    from .retrieval import Retriever
    base, exps = experiments()
    spec = json.loads(Path("configs/experiments.json").read_text(encoding="utf-8"))
    cfg = merge(base, next(e for e in exps if e["name"] == spec["serving"]).get("set", {}))
    docs = load_docs()
    return Retriever(docs, cfg), docs


def threshold() -> float:
    return json.loads(THRESHOLD_FILE.read_text())["threshold"] if THRESHOLD_FILE.exists() else 0.0


def sources(question: str, k: int = TOP_K) -> tuple[list[dict], dict]:
    retriever, docs = serving()
    res = retriever.search(question, k=k)
    out = []
    for n, (i, s) in enumerate(zip(res["ids"], res["scores"]), 1):
        c = retriever.index.chunks[i]
        d = docs[c["doc_id"]]
        p0, p1 = page_of(d["page_starts"], c["start"]), page_of(d["page_starts"], max(c["start"], c["end"] - 1))
        out.append({"n": n, "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "title": d["title"], "date": d["date"],
                    "doc_type": d["doc_type"], "pages": [p0, p1], "url": f"{d['source_url']}#page={p0}",
                    "score": s, "start": c["start"], "end": c["end"], "text": retriever.index.text(c)})
    return out, res


def answer(question: str) -> dict:
    t0 = time.perf_counter()
    srcs, res = sources(question)
    top_score = srcs[0]["score"] if srcs else 0.0
    if res["rerank_scores"] is not None and top_score < threshold():
        return {"question": question, "answer": IDK, "abstained": "retrieval", "citations": [], "sources": srcs,
                "top_score": top_score, "ms": (time.perf_counter() - t0) * 1000, "llm": None}
    context = "\n\n".join(f"[{s['n']}] {s['title']}, p. {s['pages'][0]}"
                          f"{'-' + str(s['pages'][1]) if s['pages'][1] != s['pages'][0] else ''}\n{s['text']}" for s in srcs)
    text = chat(SYSTEM, f"Sources:\n\n{context}\n\nQuestion: {question}")
    cited = sorted({int(n) for n in CITE.findall(text) if 1 <= int(n) <= len(srcs)})
    abstained = "llm" if text.strip().lower().startswith("i don't know") else None
    return {"question": question, "answer": IDK if abstained else text, "abstained": abstained, "citations": cited,
            "sources": srcs, "top_score": top_score, "ms": (time.perf_counter() - t0) * 1000, "llm": backend_name()}


# ---------------------------------------------------------------- calibration and evaluation

def calibrate() -> dict:
    """Threshold on the top cross-encoder score that best separates answerable from unanswerable dev questions."""
    rows = [(sources(q["question"])[0][0]["score"], q["type"] == "unanswerable") for q in load_gold("dev")]
    best = max(((t, sum((s < t) == unans for s, unans in rows) / len(rows)) for t in sorted({s for s, _ in rows})),
               key=lambda x: x[1])
    ans = [s for s, u in rows if not u]
    out = {"threshold": best[0], "dev_accuracy": best[1],
           "false_abstain_on_answerable": sum(s < best[0] for s in ans) / len(ans),
           "n_unanswerable": sum(u for _, u in rows), "n_answerable": len(ans)}
    THRESHOLD_FILE.parent.mkdir(exist_ok=True)
    THRESHOLD_FILE.write_text(json.dumps(out, indent=1))
    return out


JUDGE_CORRECT = """You grade answers to questions about FOMC documents. Compare the ANSWER with the REFERENCE.
Reply with one word: CORRECT (states the reference's key facts, no contradiction), PARTIAL (some key facts, or \
correct but incomplete), or INCORRECT (wrong, contradicts the reference, or says it doesn't know)."""
JUDGE_SUPPORT = """You check whether a claim is supported by source passages. Reply with one word: SUPPORTED if every \
fact in the claim is stated in the passages, otherwise UNSUPPORTED."""


def _verdict(text: str, options: tuple[str, ...]) -> str:
    up = text.upper()
    return next((o for o in options if o in up.split()[:3] or up.startswith(o)), options[-1])


def split_claims(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if len(s.strip()) > 3]


def judge(q: dict, a: dict) -> dict:
    if a["abstained"]:
        return {"correct": "INCORRECT", "claims": 0, "supported": 0, "cited_claims": 0}
    correct = _verdict(chat(JUDGE_CORRECT, f"QUESTION: {q['question']}\nREFERENCE: {q['answer']}\nANSWER: {a['answer']}",
                            max_tokens=5), ("CORRECT", "PARTIAL", "INCORRECT"))
    by_n = {s["n"]: s for s in a["sources"]}
    claims = split_claims(a["answer"])
    supported = cited = 0
    for claim in claims:
        ns = [int(n) for n in CITE.findall(claim) if int(n) in by_n]
        if not ns:
            continue  # an uncited claim counts as unfaithful
        cited += 1
        passages = "\n\n".join(by_n[n]["text"] for n in ns)
        v = _verdict(chat(JUDGE_SUPPORT, f"PASSAGES:\n{passages}\n\nCLAIM: {CITE.sub('', claim).strip()}", max_tokens=5),
                     ("SUPPORTED", "UNSUPPORTED"))
        supported += v == "SUPPORTED"
    return {"correct": correct, "claims": len(claims), "supported": supported, "cited_claims": cited}


def evaluate(split: str) -> dict:
    rows = []
    for q in load_gold(split):
        a = answer(q["question"])
        j = judge(q, a) if q["type"] != "unanswerable" else {}
        rows.append({"qid": q["qid"], "type": q["type"], "question": q["question"], "reference": q["answer"],
                     "answer": a["answer"], "abstained": a["abstained"], "citations": a["citations"],
                     "top_score": a["top_score"], "ms": a["ms"], **j})
        print(f"{q['qid']} [{q['type']}] {j.get('correct', 'abstain-ok' if a['abstained'] else 'ANSWERED')}", flush=True)
    ans = [r for r in rows if r["type"] != "unanswerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    claims = sum(r["claims"] for r in ans)
    out = {
        "split": split, "llm": backend_name(), "threshold": threshold(), "n_answerable": len(ans), "n_unanswerable": len(una),
        "correct": sum(r["correct"] == "CORRECT" for r in ans) / len(ans),
        "correct_or_partial": sum(r["correct"] in ("CORRECT", "PARTIAL") for r in ans) / len(ans),
        "faithfulness": sum(r["supported"] for r in ans) / claims if claims else 0.0,
        "citation_coverage": sum(r["cited_claims"] for r in ans) / claims if claims else 0.0,
        "abstain_on_unanswerable": sum(bool(r["abstained"]) for r in una) / len(una),
        "false_abstain_on_answerable": sum(bool(r["abstained"]) for r in ans) / len(ans),
        "p50_ms": statistics.median(r["ms"] for r in rows),
        "rows": rows,
    }
    (RESULTS / f"answers_{split}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    if args.calibrate:
        print(json.dumps(calibrate(), indent=1))
    elif args.eval:
        r = evaluate(args.split)
        print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=1))
    else:
        a = answer(args.question)
        print(a["answer"])
        for s in a["sources"]:
            print(f"  [{s['n']}] {s['title']} p.{s['pages'][0]} score={s['score']:.3f} {s['url']}")


if __name__ == "__main__":
    main()
