# fomc-rag

**Question answering over Federal Reserve policy documents from 1980 to 2026, built as a retrieval experiment: every design choice is measured against a hand-checked gold set, and nothing ships on intuition.**

[![ci](https://github.com/ArjunShuklaCSE/fomc-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/ArjunShuklaCSE/fomc-rag/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

> **Status:** Phase 1 of 7 (ingestion and cleaning) is complete. The results table below fills in as experiments land. Numbers that have not been measured are left blank rather than estimated.

Most RAG projects stop at "the chatbot answers questions." This one asks a narrower question and tries to answer it with evidence: **for this corpus, which retrieval choices actually move Recall@10, and what does each one cost in latency?** The chatbot is the last step, not the point.

---

## Contents

- [The corpus](#the-corpus)
- [Architecture](#architecture)
- [Ingestion: what makes this data hard](#ingestion-what-makes-this-data-hard)
- [Design decisions](#design-decisions)
- [Results](#results)
- [Roadmap](#roadmap)
- [Running it](#running-it)
- [Repository layout](#repository-layout)
- [Data and license](#data-and-license)

---

## The corpus

The Federal Open Market Committee (FOMC) sets U.S. interest rates. It publishes three kinds of documents about each meeting, and this project indexes all three:

| Type | Years | Docs | Pages | What it looks like |
|---|---|---:|---:|---|
| Minutes | {{minutes_years}} | {{minutes_docs}} | {{minutes_pages}} | Two-column PDF, bold section headings, footnoted attendance lists |
| Press conference transcripts | {{presconf_years}} | {{presconf_docs}} | {{presconf_pages}} | Chair's statement followed by reporter Q&A with speaker labels |
| Meeting transcripts | 1980, 2016–2020 | {{transcript_docs}} | {{transcript_pages}} | Verbatim discussion. The 1980 set is scanned typewriter pages. |
| **Total** | | **{{docs}}** | **{{pages}}** | {{chars_m}}M characters after cleaning |

Why this corpus and not a tidier one:

- **It is public domain.** These are works of the U.S. federal government (17 U.S.C. §105), so the live demo can show the source text it retrieved, and anyone can rebuild the index from the original URLs.
- **The formatting spans four eras.** A cleaner tuned on 2024 minutes breaks on 1980 scans, and a cleaner tuned on the scans mangles two-column layouts. Getting all of them right is most of the work.
- **The questions are specific.** "What was the target range after the March 15, 2020 meeting?" has a handful of correct passages, and so does "What did Chair Powell say about the debt ceiling on February 1, 2023?". A system either retrieves one of them or it doesn't, which keeps evaluation honest.
- **Dates carry meaning.** "What did the Committee say about inflation in 2021?" depends on metadata filtering, not just semantic similarity.
- **There is a 5-year disclosure lag.** Meeting transcripts are released five years late, so 2020 is the most recent full transcript. That's why transcripts stop there while minutes and press conferences run to the present.

## Architecture

Solid boxes exist today. Dashed boxes are the planned phases.

```mermaid
flowchart LR
    subgraph ingest["Phase 1: ingestion (done)"]
        A[federalreserve.gov<br/>year index pages] -->|crawl| B[manifest.jsonl<br/>versioned in git]
        B -->|download| C[(raw PDFs)]
        C --> D[extract<br/>PyMuPDF text layer]
        D -->|image page with bad text layer| E[OCR fallback<br/>RapidOCR]
        D --> F[clean<br/>furniture, hyphenation,<br/>speakers, headings]
        E --> F
        F --> G[(docs.jsonl<br/>text + page offsets)]
    end
    subgraph eval["Phase 2"]
        G -.-> H[gold set<br/>char-span labels]
    end
    subgraph retrieve["Phases 3-4"]
        G -.-> I[chunkers<br/>fixed / overlap /<br/>structure / semantic]
        I -.-> J[(Postgres<br/>pgvector + FTS)]
        J -.-> K[BM25 + dense<br/>RRF fusion]
        K -.-> L[cross-encoder<br/>rerank top 50]
    end
    subgraph answer["Phases 5-6"]
        L -.-> M[LLM answer with<br/>inline citations]
        M -.-> N[FastAPI + UI]
    end
    H -.->|Recall@k, MRR, nDCG| K
    classDef planned stroke-dasharray: 5 5
    class H,I,J,K,L,M,N planned
```

## Ingestion: what makes this data hard

Every rule in the cleaner exists because of a failure seen in the real documents. Each has a unit test in [`tests/test_clean.py`](tests/test_clean.py).

| Problem in the raw PDF | Example from the corpus | Fix |
|---|---|---|
| Running headers and footers inside the text stream | `Page 6 ___ Federal Open Market Committee`, `150 of 361`, `2/4-5/80`, `-41-`, `FINAL` | Lines in the top or bottom 10% of the page that repeat, with digits masked, on at least 30% of pages. The threshold isn't 50% because the minutes alternate odd and even page headers. |
| Line-break hyphenation | `Sys-` / `tem`, `rules_au-` / `thorizations` | Words are rejoined, except when the same document spells the hyphenated form mid-line (`longer-run`) more often than the joined form. The document's own vocabulary makes the decision. |
| Footnote markers glued to words | `Robert J. Tetlow,21 Advisers`, `9:00 a.m.1` | Superscript spans are dropped using PDF font flags. Footnote bodies are split out by font-size change. |
| Speaker turns buried in wrapped lines | `MR. ZEISEL.`, `CHAIR POWELL.`, `VICTORIA GUIDA.`, `SPEAKER(?).`, `QUESTION.` | Every speaker label starts a new paragraph, so structure-aware chunking can split on turns. |
| Headings that look like body text | `Staff Review of the Economic Situation` | A short, all-bold line becomes a `## ` heading. |
| Paragraphs split across pages and columns | `...be sure that we learn from the` / *(next page)* `...` | A new layout block or page continues the paragraph when the previous line ends mid-sentence. |
| Hundreds of whitespace-only lines per page | Redaction layout in the modern transcripts | Dropped after Unicode NFKC normalization. |
| Unicode fractions | `0 to ¼ percent` becomes `1⁄4` under NFKC | The fraction slash is mapped to `/`, matching how the 1980 documents write `5-1/2 percent`, so exact-number queries hit both eras. |
| Scanned pages | Every page of the 1980 transcripts is an image | See OCR below. |

**OCR is a fallback, and that decision was measured.** The 1980 scans already carry a text layer the Fed produced. On a sample transcript, 98% of its tokens (median) look like real words, and the worst page is at 91%. Running RapidOCR on that worst page took 8.9 seconds and produced *more* errors (`YeS`, `Ml` for `M1`, `IiI` for `III`). So OCR only runs on image pages where the text layer is missing or falls below 60% word-like tokens. In the full corpus that was {{ocr_pages}} pages.

Full-corpus statistics from the last run (`data/processed/ingest_stats.json`):

| Metric | Value |
|---|---:|
| Documents / pages | {{docs}} / {{pages}} |
| Failures (download / extract) | {{failures}} |
| Header and footer lines removed | {{furniture}} |
| Line-break hyphens rejoined / kept | {{hy_join}} / {{hy_keep}} |
| Speaker turns detected | {{speakers}} |
| Headings detected | {{headings}} |
| Pages sent to OCR | {{ocr_pages}} |
| Wall time (download + extract, 6 processes) | {{wall}} |

## Design decisions

These are the choices most likely to come up in review, with the reasoning behind each.

**Gold answers are labeled as character spans, not chunk IDs.** Phase 4 compares chunking strategies. If the gold set pointed at chunk IDs, every chunking change would invalidate the labels, and the comparison would be meaningless. Instead each answer is stored as `(doc_id, start, end)` plus the quoted text. A retrieved chunk counts as relevant when it overlaps a gold span, so one gold set scores any chunker.

**The cleaned text is versioned.** Span labels only mean something against the exact text they were written on. Each document records `cleaner_version` and the SHA-256 of its source PDF. A cleaner change that alters the output bumps the version, and the gold set's quoted text lets its spans be re-anchored automatically.

**Page numbers are physical PDF pages.** The printed numbers disagree with the file: a 2015 transcript prints `1 of 361` on a 299-page PDF. Citations use the physical page, because that is what `source_url#page=N` opens.

**Each document is one string plus `page_starts`, not a list of pages.** Chunkers need continuous text, since sentences cross pages and columns. Citations need page numbers. One `bisect` over `page_starts` maps any character offset back to its page, which serves both.

**The manifest is committed and the PDFs are not.** `data/manifest.jsonl` pins exactly which 293 source URLs make up the corpus, so results stay reproducible even if the Fed reorganizes its site. The PDFs (about 1 GB) are rebuilt from it with one command.

**One database, not two (planned).** Postgres with pgvector for dense search and its built-in full-text search for BM25. At about 50k chunks, a dedicated vector database adds an extra service and nothing measurable, and hybrid retrieval becomes a single SQL query.

**Dev/test split before any tuning (planned).** The gold set is split once, with a fixed seed and stratified by question type. Every experiment is tuned on dev. Test is scored once per final configuration, and each result is reported with a bootstrap 95% confidence interval, so a 2-point gain that falls within the noise gets reported as noise.

## Results

Every row is one change from the row it's compared against, measured on the held-out test split. Blank cells have not been run yet.

| # | Configuration | Recall@5 | Recall@10 | MRR | nDCG@10 | p50 latency |
|---|---|---:|---:|---:|---:|---:|
| 0 | Baseline: 500-token fixed chunks, dense only | | | | | |
| 1 | + chunk overlap | | | | | |
| 2 | Structure-aware chunks (headings / speaker turns) | | | | | |
| 3 | Semantic chunks | | | | | |
| 4 | BM25 only | | | | | |
| 5 | Hybrid BM25 + dense, Reciprocal Rank Fusion | | | | | |
| 6 | + cross-encoder rerank of top 50 | | | | | |
| 7 | + date filters and query rewriting | | | | | |

## Roadmap

- [x] **Phase 1: Ingestion and cleaning.** Crawler, downloader, text-layer extraction with OCR fallback, cleaner, stats, unit tests, CI.
- [ ] **Phase 2: Gold evaluation set.** 150–200 questions, drafted by an LLM from sampled passages and each reviewed by hand. Includes exact-number, paraphrase, multi-passage, and unanswerable questions, stored as versioned JSONL.
- [ ] **Phase 3: Baseline.** Fixed chunks with dense retrieval, plus an eval harness reporting Recall@5/10, MRR, nDCG@10, and latency, with one results row per logged config.
- [ ] **Phase 4: Experiments.** Chunking, BM25, hybrid RRF, reranking, filtering and rewriting, one change at a time.
- [ ] **Phase 5: Answer generation.** Inline citations to exact chunks and pages, abstaining when retrieval confidence is low, and faithfulness evaluation.
- [ ] **Phase 6: Product.** FastAPI, a UI showing retrieved chunks and scores, Docker Compose, deployment.
- [ ] **Phase 7: Write-up.** Key findings and the failure cases that still aren't solved.

## Running it

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ArjunShuklaCSE/fomc-rag && cd fomc-rag
uv sync

uv run python -m fomcrag.ingest --limit 3   # smoke run: 3 docs per type, about 2 minutes
uv run python -m fomcrag.ingest             # full corpus (downloads about 1 GB)
uv run pytest -q
```

Outputs land in `data/processed/`:

| File | Contents |
|---|---|
| `docs.jsonl` | One cleaned document per line: `doc_id`, `doc_type`, `date`, `title`, `source_url`, `n_pages`, `page_starts`, `ocr_pages`, `pdf_sha256`, `cleaner_version`, `text` |
| `ingest_stats.json` | The statistics table above |
| `failures.jsonl` | Every document that failed, with the stage and error. Failures are logged, never silently dropped. |

Downloads are resumable. Existing PDFs are skipped, and files are written to a temporary path and renamed, so an interrupted run never leaves a truncated PDF behind.

## Repository layout

```
src/fomcrag/
  sources.py    crawl federalreserve.gov index pages -> manifest; resumable downloader
  extract.py    PDF -> positioned lines (text layer, font flags, OCR fallback)
  clean.py      lines -> clean text + page offsets (pure functions, unit-tested)
  ingest.py     CLI that runs the pipeline and writes stats
tests/
  test_clean.py one test per cleaning rule
data/
  manifest.jsonl  the pinned list of source documents
```

## Data and license

Code is MIT licensed (see [LICENSE](LICENSE)). The source documents are U.S. government works in the public domain, published by the Board of Governors of the Federal Reserve System at [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm). This project is independent and not affiliated with or endorsed by the Federal Reserve.
