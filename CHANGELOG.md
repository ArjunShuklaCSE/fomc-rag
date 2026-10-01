# Changelog

## 0.1.0

First release: question answering over 293 FOMC documents (1980–2026) with page-linked citations.

- Ingestion and cleaning for minutes, press conference transcripts and Greenbook/Tealbook PDFs, including OCR repair and heading detection.
- A hand-built gold set of 151 questions, anchored to exact passages, split into dev and test.
- Measured retrieval chain: semantic chunking, BM25, hybrid fusion, cross-encoder reranking and date and document-type filters parsed from the question. Recall@10 on the held-out test split rises from 0.14 to 0.85.
- Cited answer generation with a calibrated retrieval gate and abstention, using a local Qwen3-4B or the Anthropic API.
- Answer-quality evaluation with an audited LLM judge, plus a HyDE experiment that did not help.
- FastAPI service, web UI and CPU Docker image.
