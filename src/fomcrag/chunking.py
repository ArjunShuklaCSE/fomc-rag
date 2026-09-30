"""Chunking strategies. Every chunk is an exact (start, end) slice of the cleaned document text.

Sizes are counted in embedder tokens (BGE WordPiece), because the embedder truncates at 512 tokens:
a "500-token chunk" measured in words would silently lose its tail.

    fixed       size tokens, optional overlap                       (baseline)
    structure   pack whole paragraphs / speaker turns up to size, new chunk at every heading
    semantic    split where adjacent-sentence embedding similarity drops, capped at size
"""

import hashlib
import json
import re
from functools import lru_cache

import numpy as np

SENTENCE_END = re.compile(r"[.!?][\"”’)]?(\s+)(?=[A-Z“\"(\[])")  # group 1 = the gap between sentences


@lru_cache(maxsize=1)
def tokenizer():
    from tokenizers import Tokenizer

    from .models import EMBEDDER, resolve
    path = resolve(EMBEDDER)
    tok = Tokenizer.from_file(f"{path}/tokenizer.json") if not path.startswith("BAAI/") else Tokenizer.from_pretrained(path)
    tok.no_truncation()
    tok.no_padding()
    return tok


def token_spans(text: str) -> list[tuple[int, int]]:
    """Char span of every token (special tokens excluded)."""
    return [o for o in tokenizer().encode(text, add_special_tokens=False).offsets]


def fixed(text: str, size: int, overlap: int = 0) -> list[tuple[int, int]]:
    toks = token_spans(text)
    step = size - overlap
    out = []
    for i in range(0, len(toks), step):
        window = toks[i:i + size]
        out.append((window[0][0], window[-1][1]))
        if i + size >= len(toks):
            break
    return out


def paragraphs(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"[^\n]+", text)]


def _pack(units: list[tuple[int, int, int, bool]], size: int, text: str) -> list[tuple[int, int]]:
    """Greedy packing of (start, end, n_tokens, is_heading) units; a heading always opens a new chunk."""
    out, cur, cur_tokens = [], None, 0
    for s, e, n, heading in units:
        if n > size:  # one oversized unit (a long speaker turn): split it into fixed windows
            if cur:
                out.append(cur)
                cur, cur_tokens = None, 0
            out += [(s + a, s + b) for a, b in fixed(text[s:e], size)]
            continue
        if cur is None or heading or cur_tokens + n > size:
            if cur:
                out.append(cur)
            cur, cur_tokens = (s, e), n
        else:
            cur, cur_tokens = (cur[0], e), cur_tokens + n
    if cur:
        out.append(cur)
    return out


def structure(text: str, size: int) -> list[tuple[int, int]]:
    paras = paragraphs(text)
    counts = [len(e.ids) for e in tokenizer().encode_batch([text[s:e] for s, e in paras], add_special_tokens=False)]
    return _pack([(s, e, n, text[s:e].startswith("## ")) for (s, e), n in zip(paras, counts)], size, text)


def sentences(text: str) -> list[tuple[int, int]]:
    out = []
    for ps, pe in paragraphs(text):
        start = ps
        for m in SENTENCE_END.finditer(text, ps, pe):
            out.append((start, m.start(1)))
            start = m.end(1)
        out.append((start, pe))
    return out


def semantic(text: str, size: int, embed, percentile: float = 90) -> list[tuple[int, int]]:
    """Break between sentences whose embeddings are least similar (bottom 10% within the document)."""
    sents = sentences(text)
    if len(sents) < 2:
        return [(sents[0][0], sents[-1][1])] if sents else []
    vecs = embed([text[s:e] for s, e in sents])
    sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
    cut = np.percentile(sims, 100 - percentile)
    counts = [len(e.ids) for e in tokenizer().encode_batch([text[s:e] for s, e in sents], add_special_tokens=False)]
    # a unit is a run of sentences between breakpoints; _pack then enforces the size cap
    units, s0, n0 = [], sents[0][0], counts[0]
    for i in range(1, len(sents)):
        is_heading = text[sents[i][0]:sents[i][1]].startswith("## ")
        if sims[i - 1] <= cut or is_heading or n0 + counts[i] > size:
            units.append((s0, sents[i - 1][1], n0, True))
            s0, n0 = sents[i][0], 0
        n0 += counts[i]
    units.append((s0, sents[-1][1], n0, True))
    # each unit is its own chunk (forced break); _pack only splits a unit that is a single oversized sentence
    return _pack([(s, e, n, True) for s, e, n, _ in units], size, text)


def section_of(text: str, start: int) -> str:
    """Nearest heading at or before `start` (contextual header for the chunk)."""
    i = text.rfind("\n## ", 0, start + 1)
    if i < 0:
        return text[3:text.find("\n")] if text.startswith("## ") else ""
    return text[i + 4:text.find("\n", i + 1) if text.find("\n", i + 1) > 0 else len(text)]


def chunk_corpus(docs: list[dict], cfg: dict, embed=None) -> list[dict]:
    kind, size = cfg["kind"], cfg["size"]
    chunks = []
    for d in docs:
        t = d["text"]
        if kind == "fixed":
            spans = fixed(t, size, cfg.get("overlap", 0))
        elif kind == "structure":
            spans = structure(t, size)
        elif kind == "semantic":
            spans = semantic(t, size, embed)
        else:
            raise ValueError(kind)
        for s, e in spans:
            chunks.append({"chunk_id": f"{d['doc_id']}:{s}", "doc_id": d["doc_id"], "start": s, "end": e,
                           "doc_type": d["doc_type"], "date": d["date"], "title": d["title"],
                           "section": section_of(t, s)})
    return chunks


def config_key(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
