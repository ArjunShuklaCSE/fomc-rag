"""Retrieval evaluation against span-level gold labels.

    python -m fomcrag.evaluate --split dev                      # run every experiment in configs/experiments.json
    python -m fomcrag.evaluate --split dev --only E0,E5        # a subset
    python -m fomcrag.evaluate --split test --only E0,E9       # final numbers: test is touched only here
    python -m fomcrag.evaluate --compare dev E0 E5             # queries whose rank changed the most

Relevance: a chunk is relevant to a gold span if it contains >= 50% of the span's characters. Recall is
span-level (a multi-passage question needs every span), and nDCG credits each span once, so overlapping
chunks cannot inflate it. tokens@5 reports how much context the top 5 chunks cost, because bigger chunks
buy recall with context budget.
"""

import argparse
import copy
import csv
import json
import math
import random
import statistics
import subprocess
from pathlib import Path

from .chunking import tokenizer
from .gold import load_docs, load_gold

CONFIGS = Path("configs/experiments.json")
RESULTS = Path("results")
METRICS = ("recall@5", "recall@10", "mrr", "ndcg@10")
BOOTSTRAP, SEED = 2000, 7


def covers(span: dict, chunk: dict) -> bool:
    if span["doc_id"] != chunk["doc_id"]:
        return False
    overlap = min(span["end"], chunk["end"]) - max(span["start"], chunk["start"])
    return overlap >= 0.5 * (span["end"] - span["start"])


def score(answers: list[dict], ranked: list[dict], k: int = 10) -> dict:
    found, first, dcg, recall5 = set(), None, 0.0, 0.0
    for r, c in enumerate(ranked[:k]):
        hit = {j for j, a in enumerate(answers) if covers(a, c)}
        if hit and first is None:
            first = r + 1
        new = hit - found
        if new:
            dcg += 1 / math.log2(r + 2)
            found |= new
        if r == 4:
            recall5 = len(found) / len(answers)
    if len(ranked) < 5:
        recall5 = len(found) / len(answers)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(answers), k)))
    return {"recall@5": recall5, "recall@10": len(found) / len(answers), "mrr": 1 / first if first else 0.0,
            "ndcg@10": dcg / idcg, "first_rank": first}


def bootstrap(values: list[float], n: int = BOOTSTRAP) -> tuple[float, float]:
    rng = random.Random(SEED)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def merge(base: dict, override: dict) -> dict:
    cfg = copy.deepcopy(base)
    for key, value in override.items():
        cfg[key] = value  # chunker is replaced as a whole on purpose: one experiment = one explicit config
    return cfg


def experiments() -> tuple[dict, list[dict]]:
    spec = json.loads(CONFIGS.read_text(encoding="utf-8"))
    return spec["base"], spec["experiments"]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def run(name: str, cfg: dict, split: str, docs: dict) -> dict:
    from .retrieval import Retriever
    gold = [q for q in load_gold(split) if q["type"] != "unanswerable" and q["status"] != "rejected"]
    retriever = Retriever(docs, cfg)
    ix = retriever.index
    retriever.search("warm-up query about inflation")  # load models before timing
    per_query, latencies, tokens5 = [], [], []
    for q in gold:
        res = retriever.search(q["question"], k=10)
        ranked = [ix.chunks[i] for i in res["ids"]]
        s = score(q["answers"], ranked)
        latencies.append(res["timing"]["total_ms"])
        tokens5.append(sum(len(tokenizer().encode(ix.text(c), add_special_tokens=False).ids) for c in ranked[:5]))
        per_query.append({"qid": q["qid"], "type": q["type"], **s, "filtered": res["filtered"],
                          "top": [c["chunk_id"] for c in ranked[:10]]})
    metrics = {m: statistics.fmean(p[m] for p in per_query) for m in METRICS}
    ci = {m: bootstrap([p[m] for p in per_query]) for m in ("recall@10", "mrr")}
    by_type = {t: statistics.fmean(p["recall@10"] for p in per_query if p["type"] == t)
               for t in sorted({p["type"] for p in per_query})}
    lat = sorted(latencies)
    out = {"name": name, "split": split, "config": cfg, "index": ix.key, "n_chunks": len(ix.chunks),
           "n_queries": len(per_query), "metrics": metrics, "ci95": ci, "recall@10_by_type": by_type,
           "tokens@5": statistics.fmean(tokens5), "latency_ms": {"p50": lat[len(lat) // 2], "p95": lat[int(0.95 * len(lat)) - 1]},
           "commit": git_commit(), "per_query": per_query}
    path = RESULTS / "runs" / split / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def paired_delta(a: dict, b: dict, metric: str = "recall@10") -> tuple[float, float, float]:
    """Mean and 95% CI of (b - a) over the same queries."""
    pa = {p["qid"]: p[metric] for p in a["per_query"]}
    diffs = [p[metric] - pa[p["qid"]] for p in b["per_query"] if p["qid"] in pa]
    lo, hi = bootstrap(diffs)
    return statistics.fmean(diffs), lo, hi


def load_run(split: str, name: str) -> dict:
    return json.loads((RESULTS / "runs" / split / f"{name}.json").read_text(encoding="utf-8"))


def describe(cfg: dict) -> str:
    c = cfg["chunker"]
    parts = [f"{c['kind']}-{c['size']}" + (f"/ov{c['overlap']}" if c.get("overlap") else ""), cfg["method"]]
    if cfg["method"] == "hybrid":
        parts.append(f"w_dense={cfg.get('w_dense', 0.5)}")
    if cfg.get("header"):
        parts.append("+header")
    if cfg.get("rerank"):
        parts.append(f"+rerank@{cfg['rerank']}")
    if cfg.get("filters"):
        parts.append("+filter:" + "/".join(cfg["filters"]))
    if cfg.get("rewrite"):
        parts.append("+HyDE")
    return " ".join(parts)


def write_tables(split: str, baseline: str = "E0") -> None:
    runs = sorted((json.loads(p.read_text(encoding="utf-8")) for p in (RESULTS / "runs" / split).glob("*.json")),
                  key=lambda r: [int(x) if x.isdigit() else x for x in __import__("re").split(r"(\d+)", r["name"])])
    base = next((r for r in runs if r["name"] == baseline), None)
    rows = []
    lines = [f"# Retrieval results: {split} split ({runs[0]['n_queries'] if runs else 0} answerable questions)\n",
             "| Exp | Config | R@5 | R@10 (95% CI) | Δ R@10 vs E0 (95% CI) | MRR | nDCG@10 | tokens@5 | p50 ms | p95 ms |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        m, ci = r["metrics"], r["ci95"]["recall@10"]
        delta = ""
        if base and r["name"] != baseline:
            d, lo, hi = paired_delta(base, r)
            delta = f"{d:+.3f} ({lo:+.3f}, {hi:+.3f})"
        lines.append(f"| {r['name']} | {describe(r['config'])} | {m['recall@5']:.3f} | {m['recall@10']:.3f} "
                     f"({ci[0]:.2f}–{ci[1]:.2f}) | {delta} | {m['mrr']:.3f} | {m['ndcg@10']:.3f} | "
                     f"{r['tokens@5']:.0f} | {r['latency_ms']['p50']:.0f} | {r['latency_ms']['p95']:.0f} |")
        rows.append({"split": split, "name": r["name"], "config": describe(r["config"]), **{k: round(v, 4) for k, v in m.items()},
                     "recall@10_ci_lo": round(ci[0], 4), "recall@10_ci_hi": round(ci[1], 4), "tokens@5": round(r["tokens@5"]),
                     "p50_ms": round(r["latency_ms"]["p50"]), "p95_ms": round(r["latency_ms"]["p95"]),
                     "n_chunks": r["n_chunks"], "index": r["index"], "commit": r["commit"]})
    types = sorted({t for r in runs for t in r["recall@10_by_type"]})
    lines += ["", "Recall@10 by question type:", "", "| Exp | " + " | ".join(types) + " |", "|---|" + "---:|" * len(types)]
    for r in runs:
        lines.append(f"| {r['name']} | " + " | ".join(f"{r['recall@10_by_type'].get(t, 0):.3f}" for t in types) + " |")
    (RESULTS / f"{split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (RESULTS / f"{split}.csv").open("w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def compare(split: str, a: str, b: str, n: int = 5) -> None:
    ra, rb = load_run(split, a), load_run(split, b)
    gold = {q["qid"]: q for q in load_gold(split)}
    pa = {p["qid"]: p for p in ra["per_query"]}
    rank = lambda p: p["first_rank"] or 99
    changes = sorted(rb["per_query"], key=lambda p: rank(p) - rank(pa[p["qid"]]))
    for label, rows in (("improved", changes[:n]), ("hurt", changes[::-1][:n])):
        print(f"\n== {b} vs {a}: most {label}")
        for p in rows:
            print(f"  {p['qid']} [{p['type']}] rank {pa[p['qid']]['first_rank']} -> {p['first_rank']}: {gold[p['qid']]['question']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--only", help="comma-separated experiment names")
    ap.add_argument("--compare", nargs=3, metavar=("SPLIT", "A", "B"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    base, exps = experiments()
    wanted = set(args.only.split(",")) if args.only else None
    docs = load_docs()
    for e in exps:
        if wanted and e["name"] not in wanted:
            continue
        r = run(e["name"], merge(base, e.get("set", {})), args.split, docs)
        m = r["metrics"]
        print(f"{e['name']:>4} {describe(r['config']):<55} R@10={m['recall@10']:.3f} MRR={m['mrr']:.3f} "
              f"p50={r['latency_ms']['p50']:.0f}ms chunks={r['n_chunks']}", flush=True)
    write_tables(args.split)


if __name__ == "__main__":
    main()
