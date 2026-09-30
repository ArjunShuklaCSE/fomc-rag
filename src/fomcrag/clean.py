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

CLEANER_VERSION = "4"  # bump on any change that alters output text: gold-set spans depend on it


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
HEADING_PT = 1.5  # a short line this much larger than the body font is a heading
# a margin line repeated on >=10% of pages is furniture: odd/even headers alternate, and the minutes switch
# headers mid-document (SEP section), so "Minutes of the Meeting of ..." can cover only 5 of 28 pages
REPEAT_FRAC = 0.1


FRACTIONS = {"¼": "1/4", "½": "1/2", "¾": "3/4"}
NUMERIC = re.compile(r"[\d.,%/\-–]+")
CHART_MIN_TOKENS, CHART_NUMERIC = 8, 0.5  # axis labels of projection charts: "0.7- 0.8 0.9- 1.0 ..."


def normalize(s: str) -> str:
    # "4¾" must become "4-3/4" (the 1980s style), not NFKC's "43⁄4"; a bare "¼" becomes "1/4"
    s = re.sub(r"(\d?)([¼½¾])", lambda m: (m[1] + "-" if m[1] else "") + FRACTIONS[m[2]], s)
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
    repeats = Counter(k for k, _ in {(_key(l.text), l.page) for l in lines if _in_band(l) and not SPEAKER.match(l.text)})
    threshold = max(3, REPEAT_FRAC * n_pages)
    return [l for l in lines if not (
        RULE.match(l.text)
        or (_in_band(l) and (PAGE_NO.match(l.text) or repeats[_key(l.text)] >= threshold)))]


def vocab_of(texts) -> Counter:
    """Word counts that decide line-break hyphens. Ingest passes corpus-wide counts."""
    return Counter(WORD.findall(" ".join(texts).lower()))


def _is_chart(text: str) -> bool:
    toks = text.split()
    return len(toks) >= CHART_MIN_TOKENS and sum(bool(NUMERIC.fullmatch(t)) for t in toks) / len(toks) >= CHART_NUMERIC


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
            # "longer-\nrun" stays hyphenated if the corpus writes "longer-run" more than "longerrun"; "Sys-\ntem" -> "System"
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


def clean_document(lines: list[Line], n_pages: int, vocab: Counter | None = None) -> tuple[str, list[int], Counter]:
    stats = Counter()
    for l in lines:
        l.text = normalize(l.text)
    lines = [l for l in lines if l.text]
    before = len(lines)
    lines = strip_furniture(lines, n_pages)
    stats["furniture_removed"] = before - len(lines)

    vocab = vocab if vocab is not None else vocab_of(l.text for l in lines)
    block_x0 = {}
    for l in lines:
        block_x0[(l.page, l.block)] = min(l.x0, block_x0.get((l.page, l.block), l.x0))
    sizes = Counter()
    for l in lines:
        sizes[round(l.size)] += len(l.text)
    body = sizes.most_common(1)[0][0] if sizes else 0
    # older templates mark headings bold; the 2025+ minutes use a larger regular weight instead
    styled = lambda l: (len(l.text) <= 120 and not SPEAKER.match(l.text)
                        and (l.bold or (body and l.size >= body + HEADING_PT)))

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
        prev_h, prev = (paragraphs[-1][0], paragraphs[-1][1][-1]) if paragraphs else (False, None)
        # a lowercase start is mid-sentence, unless it wraps a heading ("Current Condi-" / "tions and the Outlook")
        wraps = prev_h and (prev.page, prev.block) == (l.page, l.block)
        h = styled(l) and (not l.text[0].islower() or wraps)
        if not paragraphs or breaks(paragraphs[-1][1][-1], paragraphs[-1][0], l, h):
            paragraphs.append((h, [l]))
        else:
            paragraphs[-1][1].append(l)

    out, pos = [], 0
    page_starts: list[int | None] = [None] * n_pages
    for h, para in paragraphs:
        text, offsets = _join([l.text for l in para], vocab, stats)
        if not h and _is_chart(text):  # chart axis labels carry no retrievable meaning
            stats["chart_paragraphs_dropped"] += 1
            continue
        if out:
            out.append("\n\n")
            pos += 2
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
