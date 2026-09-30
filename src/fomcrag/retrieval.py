"""Indexes and the retrieval pipeline. Every stage is switched by a config dict, never by code changes.

    config = {
      "chunker":  {"kind": "fixed" | "structure" | "semantic", "size": 500, "overlap": 0},
      "header":   False,            # prepend "title | section" to each chunk before indexing
      "method":   "dense" | "bm25" | "hybrid",
      "w_dense":  0.5, "rrf_k": 60, # hybrid only: weighted Reciprocal Rank Fusion
      "rerank":   0,                # rerank this many fused candidates with the cross-encoder (0 = off)
      "filters":  [] | ["date"] | ["date", "type"],
      "rewrite":  False,            # LLM writes a hypothetical answer passage for dense search (HyDE)
    }

Dense search is exact (a matrix product over all chunks), not approximate: at ~30k chunks it takes a few
milliseconds, and it keeps ANN recall loss out of the experiments.
"""

import json
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .chunking import chunk_corpus, config_key
from .clean import CLEANER_VERSION
from .models import EMBEDDER, RERANKER, resolve

INDEX_DIR = Path("data/index")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # BGE v1.5 query instruction
N_CANDIDATES = 100
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
MONTH_DATE = re.compile(r"\b(" + "|".join(MONTHS) + r")\s+(?:\d{1,2},?\s+)?((?:19|20)\d{2})\b", re.I)
YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
TYPE_WORDS = {"presconf": re.compile(r"press conference", re.I), "minutes": re.compile(r"\bminutes\b", re.I),
              "transcript": re.compile(r"\btranscript|FOMC meeting|meeting transcript", re.I)}


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(resolve(EMBEDDER), device=device())
    return m.half() if device() == "cuda" else m


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder
    m = CrossEncoder(resolve(RERANKER), device=device(), max_length=512)
    if device() == "cuda":
        m.model.half()
    return m


def embed(texts: list[str], query: bool = False, batch_size: int = 64) -> np.ndarray:
    texts = [QUERY_PREFIX + t for t in texts] if query else texts
    return embedder().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=len(texts) > 5000).astype(np.float32)


def _stemmer():
    import Stemmer
    return Stemmer.Stemmer("english")


def bm25_tokens(texts: list[str]) -> list[list[str]]:
    import bm25s
    return bm25s.tokenize(texts, stopwords="en", stemmer=_stemmer(), return_ids=False, show_progress=False)


class Index:
    """Chunks + dense vectors + BM25 for one chunking config, cached on disk under a config hash."""

    def __init__(self, docs: dict[str, dict], chunker: dict, header: bool = False):
        import bm25s
        self.docs = docs
        self.key = config_key({"chunker": chunker, "header": header, "embedder": EMBEDDER, "cleaner": CLEANER_VERSION})
        path = INDEX_DIR / self.key
        if not (path / "dense.npy").exists():
            self._build(path, chunker, header)
        self.chunks = [json.loads(l) for l in (path / "chunks.jsonl").open(encoding="utf-8")]
        self.vecs = np.load(path / "dense.npy")
        self.bm25 = bm25s.BM25.load(str(path / "bm25"))
        self.dates = np.array([c["date"] for c in self.chunks])
        self.types = np.array([c["doc_type"] for c in self.chunks])

    def text(self, c: dict, header: bool = False) -> str:
        body = self.docs[c["doc_id"]]["text"][c["start"]:c["end"]]
        return f"{c['title']} | {c['section']}\n{body}" if header else body

    def _build(self, path: Path, chunker: dict, header: bool) -> None:
        import bm25s
        t0 = time.perf_counter()
        sent_embed = (lambda xs: embed(xs, batch_size=256)) if chunker["kind"] == "semantic" else None
        chunks = chunk_corpus(list(self.docs.values()), chunker, embed=sent_embed)
        texts = [self.text(c, header) for c in chunks]
        vecs = embed(texts)
        bm25 = bm25s.BM25()
        bm25.index(bm25s.tokenize(texts, stopwords="en", stemmer=_stemmer(), show_progress=False), show_progress=False)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        np.save(path / "dense.npy", vecs)
        bm25.save(str(path / "bm25"))
        (path / "build.json").write_text(json.dumps({"chunker": chunker, "header": header, "chunks": len(chunks),
                                                     "seconds": round(time.perf_counter() - t0, 1)}), encoding="utf-8")

    def dense_scores(self, qvec: np.ndarray) -> np.ndarray:
        return self.vecs @ qvec

    def bm25_scores(self, query: str) -> np.ndarray:
        toks = [t for t in bm25_tokens([query])[0] if t in self.bm25.vocab_dict]
        return self.bm25.get_scores(toks) if toks else np.zeros(len(self.chunks), dtype=np.float32)


def parse_filters(query: str, kinds: list[str]) -> dict:
    """Date range (month-level if the query names a month, else year-level) and doc types mentioned."""
    out = {}
    if "date" in kinds:
        months = {(y, MONTHS.index(m.lower()) + 1) for m, y in MONTH_DATE.findall(query)}
        years = set(YEAR.findall(query))
        if months:
            out["months"] = {f"{y}-{m:02d}" for y, m in months}
        elif years:
            out["years"] = years
    if "type" in kinds:
        types = {t for t, rx in TYPE_WORDS.items() if rx.search(query)}
        if len(types) == 1:  # a question naming two doc types usually needs both
            out["types"] = types
    return out


def filter_mask(index: Index, f: dict) -> np.ndarray | None:
    if not f:
        return None
    mask = np.ones(len(index.chunks), dtype=bool)
    if "months" in f:
        mask &= np.isin(np.char.ljust(index.dates.astype("<U7"), 7), list(f["months"]))
    if "years" in f:
        mask &= np.isin(index.dates.astype("<U4"), list(f["years"]))
    if "types" in f:
        mask &= np.isin(index.types, list(f["types"]))
    return mask if mask.sum() >= 10 else None  # a filter that leaves almost nothing is probably a misparse


def top(scores: np.ndarray, n: int, mask: np.ndarray | None) -> np.ndarray:
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    n = min(n, len(scores))
    idx = np.argpartition(-scores, n - 1)[:n]
    idx = idx[np.argsort(-scores[idx])]
    return idx[np.isfinite(scores[idx])]


def rrf(rankings: list[np.ndarray], weights: list[float], k: int) -> tuple[np.ndarray, np.ndarray]:
    fused: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for r, i in enumerate(ranking):
            fused[int(i)] = fused.get(int(i), 0.0) + w / (k + r + 1)
    order = sorted(fused, key=fused.get, reverse=True)
    return np.array(order), np.array([fused[i] for i in order])


class Retriever:
    def __init__(self, docs: dict[str, dict], cfg: dict):
        self.cfg = cfg
        self.index = Index(docs, cfg["chunker"], cfg.get("header", False))

    def search(self, query: str, k: int = 10) -> dict:
        cfg, ix, t = self.cfg, self.index, {}
        t0 = time.perf_counter()
        mask = filter_mask(ix, parse_filters(query, cfg.get("filters", [])))
        dense_query = query
        if cfg.get("rewrite"):
            from .llm import hyde
            dense_query = hyde(query)
            t["rewrite_ms"] = (time.perf_counter() - t0) * 1000
        method = cfg["method"]
        n = max(k, cfg.get("rerank", 0), N_CANDIDATES)
        rankings, weights = [], []
        if method in ("dense", "hybrid"):
            rankings.append(top(ix.dense_scores(embed([dense_query], query=True)[0]), n, mask))
            weights.append(cfg.get("w_dense", 0.5) if method == "hybrid" else 1.0)
        if method in ("bm25", "hybrid"):
            rankings.append(top(ix.bm25_scores(query), n, mask))
            weights.append(1 - cfg.get("w_dense", 0.5) if method == "hybrid" else 1.0)
        ids, scores = rrf(rankings, weights, cfg.get("rrf_k", 60)) if len(rankings) > 1 else (rankings[0], None)
        if scores is None:  # single retriever: keep its own rank order, report reciprocal-rank scores
            scores = 1.0 / (np.arange(len(ids)) + 1)
        t["retrieve_ms"] = (time.perf_counter() - t0) * 1000
        rerank_scores = None
        if cfg.get("rerank"):
            t1 = time.perf_counter()
            cand = ids[:cfg["rerank"]]
            logits = reranker().predict([(query, ix.text(ix.chunks[i])) for i in cand], batch_size=16,
                                        show_progress_bar=False)
            order = np.argsort(-logits)
            ids, rerank_scores = cand[order], 1 / (1 + np.exp(-logits[order]))  # sigmoid: 0..1 relevance
            scores = rerank_scores
            t["rerank_ms"] = (time.perf_counter() - t1) * 1000
        t["total_ms"] = (time.perf_counter() - t0) * 1000
        return {"ids": ids[:k].tolist(), "scores": np.asarray(scores[:k], dtype=float).tolist(),
                "rerank_scores": None if rerank_scores is None else rerank_scores[:k].tolist(),
                "filtered": mask is not None, "timing": t}
