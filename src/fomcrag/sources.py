"""Discover and download FOMC documents from federalreserve.gov.

Corpus scope (see README for rationale):
  - minutes            2008-present (PDF era)
  - presconf           all press-conference transcripts, 2011-present
  - transcript         full meeting transcripts for 1980 (scanned typewriter era)
                       and 2016-2020 (transcripts are released with a 5-year lag)
"""

import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://www.federalreserve.gov"
UA = "fomc-rag/0.1 (retrieval research project)"
HISTORY_YEARS = [1980, *range(2008, 2021)]
TRANSCRIPT_YEARS = {1980, *range(2016, 2021)}

PATTERNS = {
    "minutes": re.compile(r'href="(/monetarypolicy/files/fomcminutes(\d{8})\.pdf)"'),
    "transcript": re.compile(r'href="(/monetarypolicy/files/FOMC(\d{8})meeting\.pdf)"'),
    # press conference links point to an HTML page; the transcript PDF lives in /mediacenter
    "presconf": re.compile(r'href="(/monetarypolicy/fomcpres+conf(\d{8})\.htm)"'),
}
TITLES = {"minutes": "FOMC Minutes", "transcript": "FOMC Meeting Transcript", "presconf": "Press Conference Transcript"}


def fetch(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)


def build_manifest() -> list[dict]:
    pages = [f"{BASE}/monetarypolicy/fomchistorical{y}.htm" for y in HISTORY_YEARS]
    pages.append(f"{BASE}/monetarypolicy/fomccalendars.htm")
    docs = {}
    for page in pages:
        html = fetch(page).decode("utf-8", "replace")
        for doc_type, pat in PATTERNS.items():
            for path, d in pat.findall(html):
                date = f"{d[:4]}-{d[4:6]}-{d[6:]}"
                if doc_type == "transcript" and int(d[:4]) not in TRANSCRIPT_YEARS:
                    continue
                if doc_type == "minutes" and int(d[:4]) < 2008:
                    continue
                url = f"{BASE}/mediacenter/files/FOMCpresconf{d}.pdf" if doc_type == "presconf" else BASE + path
                doc_id = f"{doc_type}-{date}"
                docs[doc_id] = {"doc_id": doc_id, "doc_type": doc_type, "date": date,
                                "title": f"{TITLES[doc_type]}, {date}", "source_url": url}
        time.sleep(0.3)
    return sorted(docs.values(), key=lambda m: (m["date"], m["doc_type"]))


def download(manifest: list[dict], raw_dir: Path, workers: int = 4) -> list[dict]:
    """Download missing PDFs. Returns failures. Writes via temp file so a crash never leaves a truncated PDF."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    def one(m):
        dest = raw_dir / f"{m['doc_id']}.pdf"
        if dest.exists():
            return None
        try:
            data = fetch(m["source_url"])
            if not data.startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
        except Exception as e:
            return {"doc_id": m["doc_id"], "stage": "download", "error": repr(e)}

    with ThreadPoolExecutor(workers) as ex:
        return [f for f in ex.map(one, manifest) if f]


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
