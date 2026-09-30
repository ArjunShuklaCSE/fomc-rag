"""Turn extracted PDF lines into one clean document string with page offsets.

Pure functions only (no PDF or I/O), so every rule here is unit-tested.

Output format:
  - paragraphs separated by a blank line
  - headings prefixed with "## "
  - speaker turns start their own paragraph ("CHAIR POWELL. ...")
  - page_starts[i] = char offset where PDF page i+1 begins (page_of() inverts it)
"""

import bisect
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

CLEANER_VERSION = "1"  # bump on any change that alters output text: gold-set spans depend on it


@dataclass
class Line:
    text: str
    page: int        # 1-based PDF page
    block: int       # layout block id within the page
    x0: float
    y0: float
    y1: float
    page_h: float
    bold: bool = False
    size: float = 0.0  # font size in pt; 0 = unknown (OCR)


SPEAKER = re.compile(
    r"^(?:(?:MR|MS|MRS|DR|CHAIR(?:MAN)?|VICE CHAIR(?:MAN)?|PRESIDENT|GOVERNOR)\.?\s+[A-Z][A-Z'’\-]+"
    r"|[A-Z][A-Z'’\-]+(?:\s+[A-Z][A-Z'’\-]+){1,3}"
    r"|SPEAKER\s*\(\?\)|QUESTION)\.(?:\s|$)"
)
PAGE_NO = re.compile(r"^(page\s+)?[-–—]?\s*\d{1,4}\s*[-–—]?(\s+of\s+\d{1,4})?$", re.I)
RULE = re.compile(r"^[_\-–—=.\s]{5,}$")
TERMINAL = tuple('.?!:;"”)')
WORD = re.compile(r"[a-z]+(?:-[a-z]+)*")
BAND = 0.1        # top/bottom fraction of a page where running headers/footers live
INDENT_PT = 12    # first-line indent that marks a new paragraph inside one layout block
SIZE_JUMP = 1.5   # font-size change (pt) that separates body text from footnotes
REPEAT_FRAC = 0.3 # margin line on >=30% of pages is furniture (odd/even headers alternate, so not 50%)


def normalize(s: str) -> str:
    # NFKC turns "¼" into "1⁄4" (fraction slash); map it to "/" so it matches the 1980s "1/4" style
    s = unicodedata.normalize("NFKC", s).replace("­", "").replace("⁄", "/")
    return re.sub(r"\s+", " ", s).strip()


def page_of(page_starts: list[int], offset: int) -> int:
    """1-based page containing char offset."""
    return max(bisect.bisect_right(page_starts, offset), 1)


def _in_band(l: Line) -> bool:
    return l.y1 < BAND * l.page_h or l.y0 > (1 - BAND) * l.page_h


def _key(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower())


def strip_furniture(lines: list[Line], n_pages: int) -> list[Line]:
    """Drop rules, margin page numbers, and margin lines that repeat (digits masked) across pages."""
    repeats = Counter(k for k, _ in {(_key(l.text), l.page) for l in lines if _in_band(l)})
    threshold = max(3, REPEAT_FRAC * n_pages)
    return [l for l in lines if not (
        RULE.match(l.text)
        or (_in_band(l) and (PAGE_NO.match(l.text) or repeats[_key(l.text)] >= threshold)))]


def _join(texts: list[str], vocab: Counter, stats: Counter) -> tuple[str, list[int]]:
    """Join wrapped lines into one paragraph, repairing line-break hyphenation. Returns text + line offsets."""
    s, offsets = "", []
    for t in texts:
        if not s:
            offsets.append(0)
            s = t
            continue
        sep = " "
        m = re.search(r"([A-Za-z]+)-$", s)
        if m and not s.endswith("--") and t[:1].islower():
            a, b = m.group(1).lower(), re.match(r"[a-z]+", t).group(0)
            # "longer-\nrun" stays hyphenated if this document writes "longer-run" elsewhere; "Sys-\ntem" -> "System"
            if vocab[f"{a}-{b}"] > vocab[a + b]:
                stats["hyphen_kept"] += 1
            else:
                s = s[:-1]
                stats["hyphen_joined"] += 1
            sep = ""
        elif s.endswith(("--", "—", "–")):
            sep = ""
        s += sep
        offsets.append(len(s))
        s += t
    return s, offsets


def clean_document(lines: list[Line], n_pages: int) -> tuple[str, list[int], Counter]:
    stats = Counter()
    for l in lines:
        l.text = normalize(l.text)
    lines = [l for l in lines if l.text]
    before = len(lines)
    lines = strip_furniture(lines, n_pages)
    stats["furniture_removed"] = before - len(lines)

    vocab = Counter(WORD.findall(" ".join(l.text for l in lines).lower()))
    block_x0 = {}
    for l in lines:
        block_x0[(l.page, l.block)] = min(l.x0, block_x0.get((l.page, l.block), l.x0))
    is_heading = lambda l: l.bold and len(l.text) <= 120 and not SPEAKER.match(l.text)

    def breaks(prev: Line, prev_h: bool, cur: Line, cur_h: bool) -> bool:
        if SPEAKER.match(cur.text) or prev_h != cur_h:
            return True
        if prev.size and cur.size and abs(prev.size - cur.size) >= SIZE_JUMP:
            return True
        same_block = (prev.page, prev.block) == (cur.page, cur.block)
        if cur_h:
            return not same_block
        mid_sentence = not prev.text.endswith(TERMINAL)
        if not same_block:  # new block, column, or page: continue only if the sentence was cut off
            return not mid_sentence
        return cur.x0 > block_x0[(cur.page, cur.block)] + INDENT_PT and not mid_sentence

    paragraphs: list[tuple[bool, list[Line]]] = []
    for l in lines:
        h = is_heading(l)
        if not paragraphs or breaks(paragraphs[-1][1][-1], paragraphs[-1][0], l, h):
            paragraphs.append((h, [l]))
        else:
            paragraphs[-1][1].append(l)

    out, pos = [], 0
    page_starts: list[int | None] = [None] * n_pages
    for h, para in paragraphs:
        if out:
            out.append("\n\n")
            pos += 2
        text, offsets = _join([l.text for l in para], vocab, stats)
        prefix = "## " if h else ""
        for l, off in zip(para, offsets):
            if page_starts[l.page - 1] is None:
                page_starts[l.page - 1] = pos + len(prefix) + off
        out.append(prefix + text)
        pos += len(prefix) + len(text)
        stats["headings" if h else "paragraphs"] += 1
        stats["speaker_turns"] += bool(SPEAKER.match(para[0].text))

    nxt = pos  # pages with no text start where the next page starts
    for i in reversed(range(n_pages)):
        if page_starts[i] is None:
            page_starts[i] = nxt
        nxt = page_starts[i]
    return "".join(out), page_starts, stats
