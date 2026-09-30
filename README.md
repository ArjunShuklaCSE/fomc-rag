# fomc-rag

**A retrieval-augmented QA system over 293 Federal Reserve policy documents (1980–2026), built as a controlled experiment: every retrieval choice was measured against a hand-built gold set, and the final configuration lifts Recall@10 from 0.14 to 0.85 on held-out questions.**

[![ci](https://github.com/ArjunShuklaCSE/fomc-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/ArjunShuklaCSE/fomc-rag/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Ask it *"To what level did the FOMC raise the target range in March 2023?"* and it answers from the March 2023 minutes, citing the exact PDF page. Ask it something the corpus cannot answer and it says so.

The system is the output, but the evaluation is the point. The results table below shows what each change did to retrieval quality, with confidence intervals, including the changes that looked good on the development split and then failed to replicate.

---

## Contents

- [Headline results](#headline-results)
- [The corpus](#the-corpus)
- [Architecture](#architecture)
- [Evaluation design](#evaluation-design)
- [Experiments](#experiments)
- [What I learned](#what-i-learned)
- [Answer generation](#answer-generation)
- [Failure cases I have not solved](#failure-cases-i-have-not-solved)
- [Ingestion: making the PDFs usable](#ingestion-making-the-pdfs-usable)
- [Running it](#running-it)
- [Deploying](#deploying)
- [Repository layout](#repository-layout)
- [Data and license](#data-and-license)

---

## Headline results

Held-out **test** split, 65 answerable questions. Every choice in this chain was made on the separate dev split; test was scored once at the end.

| Step | Configuration | Recall@10 (95% CI) | MRR | p50 latency |
|---|---|---:|---:|---:|
| Baseline | 500-token fixed chunks, dense retrieval | 0.138 (0.06–0.23) | 0.062 | 10 ms |
| + chunking | semantic chunks, ≤250 tokens | 0.223 (0.12–0.33) | 0.105 | 12 ms |
| + lexical | BM25 instead of dense | 0.385 (0.27–0.50) | 0.227 | <1 ms |
| + fusion | hybrid BM25 + dense, weighted RRF | 0.369 (0.26–0.48) | 0.226 | 13 ms |
| + reranking | cross-encoder over the fused top 50 | 0.523 (0.42–0.64) | 0.364 | 157 ms |
| + metadata | date and document-type filters parsed from the question | **0.869** (0.79–0.94) | 0.693 | 161 ms |
| + context | "title \| section" header on each chunk (served config) | 0.854 (0.77–0.93) | **0.696** | 146 ms |

Latency is measured per query on a laptop RTX 3060, retrieval only (no LLM).

Two of these steps are real, large, and replicate on both splits: **reranking** (+0.154 Recall@10 on test, paired 95% CI +0.06 to +0.25) and **date filtering** (+0.346, CI +0.24 to +0.45). Two steps that won on dev did **not** replicate on test: hybrid fusion over plain BM25, and contextual chunk headers. Both are within noise on test. [What I learned](#what-i-learned) explains all of this.

## The corpus

The Federal Open Market Committee sets U.S. interest rates. This project indexes three kinds of documents it publishes:

| Type | Years | Docs | Pages | What it looks like |
|---|---|---:|---:|---|
| Minutes | 2008–2026 | 149 | 2,568 | Narrow columns, section headings, footnoted attendance lists. The template changed in 2025. |
| Press conference transcripts | 2011–2026 | 94 | 2,359 | The Chair's statement, then reporter Q&A with speaker labels |
| Meeting transcripts | 1980 (10), 2016–2020 (40) | 50 | 7,663 | Verbatim discussion. The 1980 set is scanned typewriter pages. |
| **Total** | | **293** | **12,590** | 30.4M characters after cleaning, 45,590 chunks in the served index |

Why this corpus:

- **It is public domain.** These are U.S. government works (17 U.S.C. §105), so the UI can show exactly what it retrieved.
- **It is hard in a specific, realistic way.** 149 sets of minutes all discuss "the target range", "inflation" and "labor market conditions" in nearly identical language. The retriever's real job is to find *the right meeting's* version of a passage. A tidier corpus would hide this.
- **Its answers are checkable.** Questions have specific answers ("4-3/4 to 5 percent", "Victoria Guida", "the ECB and the Bank of Japan"), so correctness is not a matter of taste.
- **It has a disclosure lag.** Meeting transcripts are released five years after the meeting, so 2020 is the latest one available.

## Architecture

```mermaid
flowchart LR
    subgraph ingest["Ingestion (offline)"]
        A[federalreserve.gov<br/>index pages] -->|crawl| B[manifest.jsonl<br/>pinned in git]
        B -->|resumable download| C[(293 PDFs)]
        C --> D[PyMuPDF extraction<br/>fonts, positions, OCR fallback]
        D --> E[cleaner<br/>headers, hyphens, speakers,<br/>headings, charts]
        E --> F[(docs.jsonl<br/>text + page offsets)]
    end
    subgraph index["Indexing (cached by config hash)"]
        F --> G[semantic chunker<br/>≤250 tokens]
        G --> H[(BM25<br/>bm25s)]
        G --> I[(dense vectors<br/>bge-base-en-v1.5)]
    end
    subgraph query["Query time"]
        Q[question] --> P[date / doc-type<br/>filter parser]
        P --> H & I
        H & I --> R[weighted RRF<br/>w_dense = 0.3]
        R --> X[cross-encoder rerank<br/>bge-reranker-base, top 50]
        X --> T{top score ≥<br/>calibrated threshold?}
        T -->|no| N["I don't know"]
        T -->|yes| L[LLM with numbered sources<br/>Qwen3-4B local or Claude]
        L --> O[answer with citations<br/>linked to PDF pages]
    end
    subgraph evaluate["Evaluation"]
        GS[(gold set<br/>151 questions,<br/>char-span labels)] -.-> M[Recall@k · MRR · nDCG<br/>bootstrap CIs]
    end
```

**One process, no vector database.** At 45k chunks, exact dense search is a single matrix product that takes a few milliseconds. A vector database would add a service to deploy and approximate-search error to every experiment, without making anything measurably faster. The project brief suggested Postgres with pgvector; I left it out for these reasons, and the index code is small enough to swap one in if the corpus grows by 100x.

## Evaluation design

This is the part of the project I would defend first in a review.

**Gold answers are character spans, not chunk IDs.** Each of the 151 questions stores its evidence as `(doc_id, start, end, quote)` in the cleaned text. A retrieved chunk counts as relevant when it contains at least 50% of a gold span. This is what makes the chunking experiments valid: one gold set scores every chunker, including chunkers that did not exist when the questions were written. It also survived reality. When I fixed four cleaning bugs *after* writing the questions, all 151 spans re-anchored automatically from their stored quotes.

**Question types.** 49 exact-match (numbers, names, dates), 50 paraphrased, 26 multi-passage (one span from the minutes and one from the same meeting's press conference), 6 temporal (a vague date such as "in mid-2021"), and 20 unanswerable. The unanswerable ones are traps:

- wrong-chair premises ("What did Chair Powell say at the March 2014 press conference?"; Yellen held it)
- meetings that never happened
- documents outside the corpus (1985 transcripts; 2022 transcripts, which are not yet released)
- information the Fed never publishes (an individual governor's projection)

**Where the questions came from.** Passages were sampled with a fixed seed, stratified by document type and era. Questions were drafted by an LLM from those passages, not from the retrieval results, so the questions carry no bias toward any retriever. Every evidence quote is machine-verified to appear exactly once in the source. I also checked each unanswerable question's premise against the corpus by grep, and removed three questions whose premises the corpus partly answers. `python -m fomcrag.gold review` walks through every question for human sign-off. Their status is recorded in the file (`drafted` / `human_reviewed` / `rejected`).

**Dev/test split before any tuning.** The split is 76/75, fixed-seed, and stratified by question type. Every choice was made on dev. Test was touched once, for the chain above.

**Metrics.**

- **Recall@5 and Recall@10** are span-level: a two-passage question needs both spans.
- **nDCG@10** credits each span once, so overlapping chunks cannot inflate it.
- **tokens@5** is the context the top five chunks would cost an LLM. Bigger chunks buy recall with context budget, and this metric keeps that cost visible.
- **p50/p95 latency** is per query.
- **Uncertainty:** every Recall@10 carries a bootstrap 95% CI, and every comparison is a *paired* bootstrap over the same queries. That test is much tighter than comparing two independent intervals.

**Limits of this design.** With 65–66 answerable questions per split, differences under about 5 points are noise, and I call them noise below. Relevance is judged only against the passage each question was drafted from. If another passage also answers the question, retrieving it counts as a miss, which biases every configuration's absolute recall downward by the same mechanism.

## Experiments

Every experiment is one config dict in [`configs/experiments.json`](configs/experiments.json). There is no code path per experiment. Per-query results, the config, the index hash and the git commit for every run are in [`results/runs/`](results/runs/). The complete dev table (18 runs) is [`results/dev.md`](results/dev.md), and the test table is [`results/test.md`](results/test.md).

Dev split, all runs:

| Exp | Change | R@10 | Δ vs previous best (paired 95% CI) | MRR | tokens@5 |
|---|---|---:|---|---:|---:|
| E0 | baseline: fixed 500, dense | 0.167 | | 0.130 | 2457 |
| E1 | fixed 500, 20% overlap | 0.159 | −0.008 (−0.09, +0.07) | 0.100 | 2475 |
| E2 | fixed 250 | 0.265 | +0.098 (+0.03, +0.17) | 0.182 | 1241 |
| E3 | fixed 250, 20% overlap | 0.167 | −0.098 (−0.19, −0.01) vs E2 | 0.153 | 1249 |
| E4 | fixed 128 | 0.250 | | 0.157 | 639 |
| E5 | structure-aware 250 (paragraphs, speaker turns, headings) | 0.265 | | 0.176 | 931 |
| E6 | structure-aware 500 | 0.242 | +0.076 vs fixed 500 | 0.161 | 1881 |
| **E7** | **semantic 250** | **0.318** | +0.152 (+0.06, +0.26) vs E0 | 0.149 | 938 |
| E8 | BM25 only | 0.333 | +0.015 (−0.11, +0.14) | 0.227 | |
| E9 | hybrid RRF, w_dense 0.5 | 0.356 | | 0.217 | |
| **E10** | **hybrid RRF, w_dense 0.3** | **0.439** | +0.106 (+0.04, +0.18) vs E8 | 0.249 | |
| E11 | hybrid RRF, w_dense 0.7 | 0.356 | | 0.221 | |
| E12 | hybrid RRF, w_dense 0.1 | 0.333 | | 0.218 | |
| **E13** | **+ rerank top 50** | **0.485** | +0.045 (−0.03, +0.11) | 0.320 | |
| E14 | + date filter | 0.856 | +0.371 (+0.26, +0.49) | 0.634 | |
| **E15** | **+ date and doc-type filter** | **0.856** | tie; MRR +0.013 | 0.647 | |
| E16 | E14 + chunk header | 0.909 | | 0.665 | |
| **E17** | **E15 + chunk header** | **0.909** | +0.053 (+0.00, +0.11) | **0.677** | |

Selection rule, fixed in advance: at each step, keep the configuration with the best dev Recall@10, breaking ties on MRR. Bold rows are the chain.

HyDE query rewriting is the last experiment. Its results are in [Answer generation](#answer-generation) because it needs the LLM.

## What I learned

**1. The corpus is a haystack of near-duplicates, and that one fact explains most of the table.** On the baseline, the median rank of the correct chunk was 204 out of 12,441, even though every gold span was coverable by some chunk. A dense embedding of "What target range did the FOMC set in March 2023?" is close to *every* meeting's target-range paragraph. Nothing in the vector encodes that the question is about March 2023, and the number "4-3/4" barely moves it. For example, q014 ("To what level did the FOMC raise the target range in March 2023?") ranked the correct passage 2,572th under dense retrieval.

**2. Date filtering was worth more than every modeling choice combined (+0.35 Recall@10 on test, CI +0.24 to +0.45).** A regex parses "March 2023", "September 2012" or "1980" from the question and restricts retrieval to documents from that month or year. After reranking had done what it could, the filter took these test questions from not found in the top 10 to rank 1:

- q008: "What did the **December 2008** minutes say amplified the fall in inflation compensation?"
- q043: "At the **September 2012** press conference, how did Bernanke describe the forward guidance adopted that January?"
- q071: "At the **March 2016** FOMC meeting, what market development did Robert Kaplan say he wanted more evidence of?"

This is also the result that most needs a caveat. Most gold questions name a meeting month because they were written about one specific passage. Real users often do the same ("the March minutes"), but a question without a date gets none of this gain. Filtering is also a hard filter: a misparsed date removes the right answer entirely. That is why the filter switches itself off when it would leave fewer than 10 candidate chunks.

**3. BM25 beat dense retrieval on this corpus (+0.162 Recall@10 on test, CI +0.03 to +0.30).** The questions that separate meetings hinge on rare exact tokens that BM25 weights heavily and embeddings smooth away:

- names: q062, "Which Fed governor's continued White House role did Chris Rugaber ask about?"
- numbers: q037, "unemployment drops below 6-1/2 percent"
- distinctive phrases in the 1980 transcripts: q092, q098

All four went from not found to rank 1 or 2.

**4. The cross-encoder was the most reliable modeling win (+0.154 on test).** It reads the question and passage together, so it can check that a passage *answers* the question, not merely that it shares vocabulary with it. Test examples that went from not found to rank 1:

- q007: "Which disease outbreak did the October 2014 minutes say weighed on market sentiment?" (Ebola)
- q090: "At the September 1980 meeting, whose alternative C did Mr. Roos prefer?"

It costs about 145 ms per query on a laptop GPU for 50 pairs. Its dev gain (+0.045) had a CI that crossed zero, and the test gain was larger. Small dev sets can understate a real effect as easily as they overstate a fake one.

**5. Two dev "wins" failed to replicate, and I kept both in the table.**

- **Hybrid fusion:** w_dense = 0.3 beat BM25 alone by +0.106 on dev (CI +0.04 to +0.18) but by −0.015 on test (CI −0.09 to +0.06). The weight sweep (0.1 / 0.3 / 0.5 / 0.7) picked its winner on 66 questions, and part of that gain was the sweep fitting noise.
- **Contextual chunk headers:** +0.053 on dev, −0.015 on test.

I still serve the dev-selected configuration, because switching to the better test number would be tuning on the test set. The honest conclusion is that neither change has shown a real effect at this sample size.

**6. Chunk size mattered more than chunking cleverness.** At 500 tokens, chunks mix several topics and dilute the embedding. Halving to 250 tokens was the biggest single chunking gain (+0.098 on dev). Among the 250-token strategies, semantic chunking (0.318) led fixed windows (0.265) and structure-aware packing (0.265), but those differences are within the confidence intervals. The only statistically clear chunking result is small-over-large.

**7. Overlap hurt, and I do not have an explanation.** Adding 20% overlap to 250-token chunks *lowered* dev Recall@10 by 0.098 (CI −0.19 to −0.01). My first hypothesis was that overlapping windows crowd the top 10 with near-duplicate chunks. I measured that and ruled it out: the number of distinct documents in the top 10 barely changed (9.06 without overlap, 9.00 with). This stays an open question, not a story.

## Answer generation

`answer.py` sends the top five reranked chunks to an LLM as numbered sources, each labelled with document, date and PDF page. The prompt requires a citation on every sentence and the exact reply "I don't know." when the sources do not contain the answer. Citations are parsed and returned as links to `source_url#page=N`, so every claim opens the PDF at the right page.

**Two layers of "I don't know."**

1. **Retrieval gate.** If the cross-encoder's best score is below a threshold, the LLM is never called. The threshold (0.509) was chosen on dev to maximize accuracy on answerable-vs-unanswerable. At that threshold it wrongly abstains on 0 of 66 answerable dev questions but catches only 3 of 10 unanswerable ones. False-premise questions ("Chair Powell at the March 2014 press conference") still retrieve confident passages from the right meeting.
2. **The LLM itself**, which does most of the abstaining.

{{ANSWER_RESULTS}}

## Failure cases I have not solved

From the final configuration's misses on dev and test:

1. **Vague time expressions.** Temporal questions reach only 0.33 Recall@10. "In mid-2021…", "In the summer of 1980…" and "In late 2009…" fall back to a whole-year filter, which still leaves eight or more same-year meetings in play. *Fix I would try next:* map seasons and "early/mid/late" to month ranges before filtering.
2. **Two-part questions.** Multi-passage recall is 0.81 on test, and every miss is the same failure: one of the two spans is found, the other is not. The top 10 fills with chunks from whichever document matches the dominant half of the question. *Next:* decompose the question into sub-queries and interleave their results.
3. **Same meeting, wrong document.** "In December 2019, what did Powell blame for…" retrieves the December 2019 *meeting transcript*, where Powell also speaks, instead of the *press conference*. The doc-type filter only fires on explicit words like "press conference". *Next:* a learned doc-type classifier, or boosting instead of hard filtering.
4. **Tables.** The Summary of Economic Projections tables survive PDF extraction as unordered number soup. The cleaner drops chart-axis paragraphs (1,468 of them) rather than indexing noise, so questions like "What was the median 2024 unemployment projection in June 2023?" are out of reach. *Next:* table-aware extraction.
5. **Absolute recall is a lower bound.** Only the source passage of each question counts as relevant. A second passage that also answers the question is scored as a miss. Pooled relevance judgments over the top results of every configuration would fix that, at the cost of judging a few hundred more passages.

## Ingestion: making the PDFs usable

Every rule in the cleaner exists because of a failure seen in the real documents, and each has a unit test in [`tests/test_clean.py`](tests/test_clean.py).

| Problem | Example from the corpus | Fix |
|---|---|---|
| Running headers and footers inside the text | `Page 6 ___ Federal Open Market Committee`, `150 of 361`, `2/4-5/80`, `-41-` | Drop lines in the top or bottom 10% of the page that repeat, with digits masked, on at least 10% of pages. The threshold is low because the minutes alternate odd/even headers and switch header text mid-document. Speaker labels are exempt. |
| Line-break hyphenation | `Sys-` / `tem`, `longer-` / `run` | Rejoin, unless the *corpus* writes the hyphenated form more often. A corpus-wide vocabulary pass fixed `hightech` → `high-tech`, which a single document's vocabulary could not. |
| Footnote markers glued to words | `Robert J. Tetlow,21`, `9:00 a.m.1` | Drop superscript spans by font flag, plus the 2025 template's small unflagged digits. |
| Speaker turns | `MR. ZEISEL.`, `CHAIR POWELL.`, `VICTORIA GUIDA.`, `SPEAKER(?).`, `QUESTION.` | Every label starts a new paragraph. |
| Headings | Bold in older templates, a larger regular font in 2025+ | Bold or at least 1.5pt above the body size. Font metrics are ignored on scanned pages, where they are OCR guesses, and a lowercase start only counts when it wraps a heading. |
| Unicode fractions | `4¾` became `43/4` under NFKC normalization | Map to `4-3/4`, the 1980s convention, so exact-number queries match across eras. |
| Chart axis labels | `0.7- 0.8 0.9- 1.0 1.1- 1.2 …` | Drop paragraphs that are mostly numeric tokens. |
| Scanned pages | All 1980 transcripts | Keep the Fed's own OCR layer, which measured better than re-OCR (details below). |

**OCR is a fallback, and that decision was measured.** The Fed's text layer on the 1980 scans scores 0.98 word-like tokens at the median. Re-running RapidOCR on the worst page took 8.9 seconds and produced *more* errors (`YeS`, `Ml` for M1). So OCR runs only on image pages whose text layer is empty or garbage, which is 0 pages in this corpus. The first version of the gate also fired on the 2025 cover pages and replaced good text with `0ctober`. That false positive is why the threshold is "empty", not "short".

**The ingestion bugs worth knowing about:**

- **Image bytes in text extraction.** PyMuPDF's `get_text("dict")` embeds every page image's bytes unless told not to. Dropping them made scanned pages about 50x faster, and the corpus now extracts in about a minute.
- **Truncated model downloads.** The model mirror drops long connections without raising an error, so a finished read proved nothing. The downloader now resumes until the byte count matches the hub's listing.

Cleaned documents store `cleaner_version` and the source PDF's SHA-256.

## Running it

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). A CUDA GPU is optional; it is only needed for the local LLM.

```bash
git clone https://github.com/ArjunShuklaCSE/fomc-rag && cd fomc-rag
uv sync

uv run python -m fomcrag.ingest            # download 293 PDFs (175 MB) and clean them, ~5 min
uv run python -m fomcrag.models            # fetch embedder + reranker (add --llm for the local generator)
                                           # huggingface.co blocked on your network? add --source modelscope
uv run python -m fomcrag.evaluate --split dev --only E0,E17   # reproduce two rows of the table
uv run uvicorn fomcrag.api:app --port 8000 # UI at http://localhost:8000
uv run pytest -q
```

With Docker, nothing needs to be installed locally:

```bash
docker compose up --build                  # builds the corpus and index on first start
ANTHROPIC_API_KEY=... docker compose up    # enables /answer with Claude on the CPU image
```

| Command | What it does |
|---|---|
| `python -m fomcrag.gold sample` / `build` / `review` | Sample passages; anchor drafted questions to spans and split dev/test; review questions interactively |
| `python -m fomcrag.evaluate --split dev [--only E7,E8]` | Run experiments; writes `results/runs/`, `results/dev.md`, `results/dev.csv` |
| `python -m fomcrag.evaluate --compare dev E13 E15` | The queries whose rank changed most between two runs |
| `python -m fomcrag.answer "question"` | Answer one question with citations |
| `python -m fomcrag.answer --calibrate` / `--eval --split test` | Abstention threshold on dev; answer-quality evaluation |

The API: `GET /search?q=…&k=10` returns ranked chunks with scores and page links. `POST /answer {"question": "…"}` returns a cited answer. `GET /health` reports the serving config.

## Deploying

The Docker image is CPU-only and runs retrieval (search, filters, reranking) out of the box. `/answer` needs either `ANTHROPIC_API_KEY`, which uses Claude, or a CUDA GPU for the local 4-bit model. The image works unchanged as a Hugging Face Space using the Docker SDK, on Render, or on Fly.io. Mount or bake `data/` so the first start does not re-download the corpus. Retrieval needs about 2 GB of RAM and answers a search in about 0.4 s on CPU (measured locally, reranker included).

A public demo link will be added here once it is deployed.

## Repository layout

```
src/fomcrag/
  sources.py     crawl federalreserve.gov -> manifest; resumable downloader
  extract.py     PDF -> positioned lines with font metadata; OCR fallback
  clean.py       lines -> clean text + page offsets (pure functions, unit-tested)
  ingest.py      corpus pipeline and statistics
  gold.py        gold set: sampling, span anchoring, dev/test split, review CLI
  chunking.py    fixed / structure-aware / semantic chunkers (exact char slices)
  retrieval.py   dense + BM25 indexes, RRF, reranking, filters; config-driven
  evaluate.py    metrics, bootstrap CIs, experiment runner, result tables
  llm.py         local Qwen3-4B (4-bit) or Claude behind one chat() function
  answer.py      cited answers, abstention, answer-quality evaluation
  api.py         FastAPI app; static/index.html is the UI
  models.py      model registry + size-verified downloader
configs/experiments.json   every experiment as a config; "serving" names the deployed one
data/manifest.jsonl        the pinned list of 293 source documents
data/gold/gold_v1.jsonl    the evaluation set
results/                   every run: config, per-query ranks, metrics, commit
```

## Data and license

Code: MIT. Source documents: U.S. government works in the public domain, published by the Board of Governors of the Federal Reserve System at [federalreserve.gov](https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm). Models: [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) and [bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) (MIT), [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (Apache 2.0). This project is independent and not affiliated with the Federal Reserve.
