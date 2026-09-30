"""Phase 1 pipeline: discover -> download -> extract -> clean -> data/processed/docs.jsonl.

    uv run python -m fomcrag.ingest                 # full corpus
    uv run python -m fomcrag.ingest --limit 10      # smoke run
    uv run python -m fomcrag.ingest --refresh-manifest
"""

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from .clean import CLEANER_VERSION, clean_document
from .extract import extract_pdf
from .sources import build_manifest, download, load_manifest

DATA = Path("data")
MANIFEST = DATA / "manifest.jsonl"
RAW = DATA / "raw"
OUT = DATA / "processed"
MIN_DOC_CHARS = 500


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def process(m: dict):
    path = RAW / f"{m['doc_id']}.pdf"
    t0 = time.perf_counter()
    try:
        lines, n_pages, ocr_pages = extract_pdf(path)
        text, page_starts, stats = clean_document(lines, n_pages)
        if len(text) < MIN_DOC_CHARS:
            raise ValueError(f"only {len(text)} chars extracted")
    except Exception as e:
        return None, {"doc_id": m["doc_id"], "stage": "extract", "error": repr(e)}
    rec = {**m, "n_pages": n_pages, "n_chars": len(text), "ocr_pages": ocr_pages,
           "page_starts": page_starts, "pdf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "cleaner_version": CLEANER_VERSION, "text": text}
    return rec, {"stats": dict(stats), "seconds": time.perf_counter() - t0}


def summarize(docs: list[dict], extra: dict, failures: list[dict], wall: float) -> dict:
    by_type = defaultdict(list)
    for d in docs:
        by_type[d["doc_type"]].append(d)
    cleaning = Counter()
    for e in extra.values():
        cleaning.update(e["stats"])
    return {
        "docs": len(docs),
        "pages": sum(d["n_pages"] for d in docs),
        "chars": sum(d["n_chars"] for d in docs),
        "ocr_pages": sum(len(d["ocr_pages"]) for d in docs),
        "failures": len(failures),
        "failures_by_stage": dict(Counter(f["stage"] for f in failures)),
        "by_type": {t: {"docs": len(ds), "date_range": [ds[0]["date"], ds[-1]["date"]],
                        "pages": sum(d["n_pages"] for d in ds),
                        "avg_chars": round(statistics.mean(d["n_chars"] for d in ds)),
                        "median_chars": round(statistics.median(d["n_chars"] for d in ds))}
                    for t, ds in sorted(by_type.items())},
        "cleaning": dict(cleaning),
        "slowest_docs": sorted(((round(e["seconds"], 1), k) for k, e in extra.items()), reverse=True)[:5],
        "wall_seconds": round(wall, 1),
        "cleaner_version": CLEANER_VERSION,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-manifest", action="store_true")
    ap.add_argument("--limit", type=int, help="process only the first N docs per type (smoke test)")
    args = ap.parse_args()
    t0 = time.perf_counter()

    if args.refresh_manifest or not MANIFEST.exists():
        write_jsonl(MANIFEST, build_manifest())
    manifest = load_manifest(MANIFEST)
    if args.limit:
        per_type = defaultdict(list)
        for m in manifest:
            per_type[m["doc_type"]].append(m)
        manifest = [m for ms in per_type.values() for m in ms[: args.limit]]
    print(f"manifest: {len(manifest)} docs {dict(Counter(m['doc_type'] for m in manifest))}")

    failures = download(manifest, RAW)
    failed = {f["doc_id"] for f in failures}
    todo = [m for m in manifest if m["doc_id"] not in failed]
    print(f"downloaded; {len(failures)} download failures. extracting {len(todo)} docs...")

    docs, extra = [], {}
    # single process: the whole corpus extracts in about a minute, so a pool buys nothing
    for i, m in enumerate(todo, 1):
        rec, info = process(m)
        if rec is None:
            failures.append(info)
        else:
            docs.append(rec)
            extra[rec["doc_id"]] = info
        if i % 50 == 0:
            print(f"  {i}/{len(todo)}", flush=True)

    stats = summarize(docs, extra, failures, time.perf_counter() - t0)
    write_jsonl(OUT / "docs.jsonl", docs)
    write_jsonl(OUT / "failures.jsonl", failures)
    (OUT / "ingest_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
