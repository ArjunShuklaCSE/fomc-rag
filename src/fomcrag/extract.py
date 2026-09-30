"""PDF -> Lines, using the embedded text layer and OCR only as a fallback.

Measured on the 1980 scans: the Fed's own text layer scores ~0.98 word-like tokens, while
RapidOCR on the same page scores lower and takes ~9 s/page. So OCR runs only when a page is
an image with a missing or garbage text layer.
"""

import re

import pymupdf

from .clean import Line

TOKEN = re.compile(r"[(\"'“‘]?([A-Za-z]+(['’\-][A-Za-z]+)*|[\d.,/%$\-]+)[)\"'”’.,;:?!]*")
MIN_CHARS, MIN_WORDLIKE = 100, 0.6
SUPERSCRIPT, BOLD = 1, 16
_ocr = None


def wordlike_ratio(text: str) -> float:
    toks = text.split()
    return sum(bool(TOKEN.fullmatch(t)) for t in toks) / max(len(toks), 1)


def needs_ocr(page: pymupdf.Page, text: str) -> bool:
    if not page.get_images():
        return False  # born-digital page: an empty text layer means a blank page
    return len(text.strip()) < MIN_CHARS or wordlike_ratio(text) < MIN_WORDLIKE


def text_lines(page: pymupdf.Page, pno: int) -> list[Line]:
    out = []
    h = page.rect.height
    for bno, b in enumerate(page.get_text("dict")["blocks"]):
        for l in b.get("lines", []):
            spans = [s for s in l["spans"] if not s["flags"] & SUPERSCRIPT]  # footnote markers: "Tetlow,21"
            text = "".join(s["text"] for s in spans)
            if not text.strip():
                continue
            visible = [s for s in spans if s["text"].strip()]
            bold = all(s["flags"] & BOLD or "bold" in s["font"].lower() for s in visible)
            x0, y0, _, y1 = l["bbox"]
            size = max(s["size"] for s in visible)
            out.append(Line(text, pno, bno, x0, y0, y1, h, bold, size))
    return out


def ocr_lines(page: pymupdf.Page, pno: int, dpi: int = 200) -> list[Line]:
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    scale = 72 / dpi
    boxes, _ = _ocr(page.get_pixmap(dpi=dpi).tobytes("png"))
    # OCR returns word groups; merge boxes whose vertical centers overlap into visual lines
    items = sorted(((b[0][1] + b[2][1]) / 2, b[0][0], b[2][1] - b[0][1], t) for b, t, _ in boxes or [])
    rows: list[list] = []
    for yc, x, hgt, t in items:
        if rows and abs(rows[-1][0][0] - yc) < hgt / 2:
            rows[-1].append((yc, x, hgt, t))
        else:
            rows.append([(yc, x, hgt, t)])
    out = []
    for row in rows:
        row.sort(key=lambda r: r[1])
        yc, hgt = row[0][0], row[0][2]
        out.append(Line(" ".join(r[3] for r in row), pno, 0, row[0][1] * scale,
                        (yc - hgt / 2) * scale, (yc + hgt / 2) * scale, page.rect.height))
    return out


def extract_pdf(path) -> tuple[list[Line], int, list[int]]:
    """Returns (lines, n_pages, ocr_pages)."""
    lines, ocr_pages = [], []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            if needs_ocr(page, page.get_text()):
                lines += ocr_lines(page, i)
                ocr_pages.append(i)
            else:
                lines += text_lines(page, i)
        return lines, len(doc), ocr_pages
