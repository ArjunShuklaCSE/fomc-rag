"""Gold evaluation set: sample passages, anchor drafted questions to exact spans, split, review.

    python -m fomcrag.gold sample            # seeded passage sample for drafting questions
    python -m fomcrag.gold build             # drafts -> gold_v1.jsonl (quotes anchored to char spans, dev/test split)
    python -m fomcrag.gold review            # interactive accept / edit / reject pass over the gold set

A gold answer is (doc_id, start, end, quote) in the cleaned text, never a chunk id, so one gold set
scores every chunking strategy. The quote is kept so spans can be re-anchored if the cleaner changes.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .clean import CLEANER_VERSION

DOCS = Path("data/processed/docs.jsonl")
GOLD_DIR = Path("data/gold")
PASSAGES = GOLD_DIR / "passages_v1.jsonl"
DRAFTS = GOLD_DIR / "drafts_v1.jsonl"
GOLD = GOLD_DIR / "gold_v1.jsonl"
TYPES = ("exact", "paraphrase", "multi", "temporal", "unanswerable")
STRATA = {"minutes": 55, "presconf": 45, "transcript-modern": 30, "transcript-1980": 15}
SEED = 13
PAIRS = 30
PAIR_TOPICS = ("inflation", "labor market", "balance sheet", "financial conditions")


def load_docs() -> dict[str, dict]:
    return {d["doc_id"]: d for d in map(json.loads, DOCS.open(encoding="utf-8"))}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def paragraphs(text: str):
    """(start, end) of every paragraph in a cleaned document."""
    for m in re.finditer(r"[^\n]+", text):
        yield m.start(), m.end()


def is_candidate(p: str) -> bool:
    # skip headings, fragments, and attendance lists (they name dozens of staff, not content)
    return (not p.startswith("## ") and 250 <= len(p) <= 1800
            and p.count("Board of Governors") < 2 and p.count("Federal Reserve Bank") < 3)


def stratum(d: dict) -> str:
    return d["doc_type"] if d["doc_type"] != "transcript" else ("transcript-1980" if d["date"] < "2000" else "transcript-modern")


def sample(docs: dict[str, dict]) -> list[dict]:
    rng = random.Random(SEED)
    by = defaultdict(list)
    for d in docs.values():
        by[stratum(d)].append(d)
    out = []
    for s, n in STRATA.items():
        order = rng.sample(by[s], len(by[s]))
        for i in range(n):  # walk a shuffled doc list, wrapping around, so one long transcript can't dominate
            d = order[i % len(order)]
            cands = [(a, b) for a, b in paragraphs(d["text"]) if is_candidate(d["text"][a:b])]
            a, b = rng.choice(cands)
            out.append({"pid": f"p{len(out):03d}", "stratum": s, "doc_id": d["doc_id"], "date": d["date"],
                        "start": a, "end": b, "text": d["text"][a:b]})
    # multi-passage questions: same meeting, same topic, one passage from the minutes and one from the press conference
    dates = sorted({d["date"] for d in docs.values() if d["doc_type"] == "presconf"}
                   & {d["date"] for d in docs.values() if d["doc_type"] == "minutes"})
    for date in rng.sample(dates, PAIRS):
        topic = rng.choice(PAIR_TOPICS)
        for t in ("minutes", "presconf"):
            d = docs[f"{t}-{date}"]
            cands = [(a, b) for a, b in paragraphs(d["text"]) if is_candidate(d["text"][a:b]) and topic in d["text"][a:b]]
            if not cands:
                break
            a, b = rng.choice(cands)
            out.append({"pid": f"m{len(out):03d}", "stratum": f"pair-{topic}", "doc_id": d["doc_id"], "date": date,
                        "start": a, "end": b, "text": d["text"][a:b]})
    return out


def anchor(doc: dict, quote: str, near: int | None = None) -> tuple[int, int]:
    """Locate quote in the doc. Unique match, or the match nearest the sampled passage."""
    hits = [m.start() for m in re.finditer(re.escape(quote), doc["text"])]
    if not hits:
        raise ValueError(f"quote not found in {doc['doc_id']}: {quote[:80]!r}")
    if len(hits) > 1 and near is None:
        raise ValueError(f"quote occurs {len(hits)}x in {doc['doc_id']}, give a longer quote: {quote[:80]!r}")
    s = min(hits, key=lambda h: abs(h - near)) if near is not None else hits[0]
    return s, s + len(quote)


def build(docs: dict[str, dict]) -> list[dict]:
    passages = {p["pid"]: p for p in read_jsonl(PASSAGES)} if PASSAGES.exists() else {}
    rows, errors = [], []
    for q in read_jsonl(DRAFTS):
        assert q["type"] in TYPES, q
        answers = []
        for ev in q.get("evidence", []):
            try:
                near = passages[ev["pid"]]["start"] if ev.get("pid") in passages else None
                s, e = anchor(docs[ev["doc_id"]], ev["quote"], near)
                answers.append({"doc_id": ev["doc_id"], "start": s, "end": e, "quote": ev["quote"]})
            except (ValueError, KeyError) as err:
                errors.append(f"{q['qid']}: {err}")
        if (q["type"] == "unanswerable") != (not q.get("evidence")):
            errors.append(f"{q['qid']}: unanswerable questions have no evidence; all others need some")
        rows.append({"qid": q["qid"], "type": q["type"], "question": q["question"], "answer": q.get("answer", ""),
                     "answers": answers, "status": q.get("status", "drafted"), "cleaner_version": CLEANER_VERSION})
    if errors:
        raise SystemExit("gold build failed:\n  " + "\n  ".join(errors))
    # stratified 50/50 dev/test split, fixed seed: decided once, before any retrieval tuning
    rng = random.Random(SEED)
    for t in TYPES:
        group = [r for r in rows if r["type"] == t]
        rng.shuffle(group)
        for i, r in enumerate(group):
            r["split"] = "dev" if i % 2 == 0 else "test"
    return rows


def load_gold(split: str | None = None) -> list[dict]:
    rows = read_jsonl(GOLD)
    return [r for r in rows if split in (None, r["split"])]


def review(docs: dict[str, dict]) -> None:
    rows = read_jsonl(GOLD)
    for r in rows:
        if r["status"] == "human_reviewed":
            continue
        print("\n" + "=" * 80 + f"\n{r['qid']} [{r['type']}/{r['split']}]  {r['question']}\n  answer: {r['answer']}")
        for a in r["answers"]:
            t = docs[a["doc_id"]]["text"]
            print(f"  -- {a['doc_id']} --\n  ...{t[max(0, a['start'] - 300):a['start']]}>>>{a['quote']}<<<{t[a['end']:a['end'] + 300]}...")
        cmd = input("[a]ccept  [e]dit question  [r]eject  [s]kip  [q]uit > ").strip().lower()
        if cmd == "q":
            break
        if cmd == "a":
            r["status"] = "human_reviewed"
        elif cmd == "e":
            r["question"] = input("new question: ").strip() or r["question"]
            r["status"] = "human_reviewed"
        elif cmd == "r":
            r["status"] = "rejected"
        write_jsonl(GOLD, rows)  # save after every decision: review sessions are long


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "build", "review", "stats"])
    args = ap.parse_args()
    docs = load_docs()
    if args.cmd == "sample":
        write_jsonl(PASSAGES, sample(docs))
        print(f"wrote {PASSAGES}")
    elif args.cmd == "build":
        rows = build(docs)
        write_jsonl(GOLD, rows)
        print(f"wrote {GOLD}: {len(rows)} questions", dict(Counter((r["type"], r["split"]) for r in rows)))
    elif args.cmd == "review":
        review(docs)
    else:
        rows = read_jsonl(GOLD)
        print(dict(Counter(r["type"] for r in rows)), dict(Counter(r["split"] for r in rows)), dict(Counter(r["status"] for r in rows)))


if __name__ == "__main__":
    main()
