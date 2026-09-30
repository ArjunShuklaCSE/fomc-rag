import numpy as np

from fomcrag.chunking import _pack
from fomcrag.evaluate import covers, score
from fomcrag.retrieval import parse_filters, rrf, top


def span(doc, s, e):
    return {"doc_id": doc, "start": s, "end": e}


def test_chunk_covers_span_only_if_it_holds_half_of_it():
    assert covers(span("d", 100, 200), span("d", 0, 160))       # 60% inside
    assert not covers(span("d", 100, 200), span("d", 0, 140))   # 40% inside
    assert not covers(span("d", 100, 200), span("x", 0, 500))   # other document


def test_score_is_span_level_and_counts_each_span_once():
    gold = [span("a", 0, 10), span("b", 0, 10)]
    ranked = [span("z", 0, 5), span("a", 0, 50), span("a", 0, 60), span("b", 0, 20)]  # two chunks cover span a
    s = score(gold, ranked)
    assert s["first_rank"] == 2 and s["mrr"] == 0.5
    assert s["recall@10"] == 1.0 and s["recall@5"] == 1.0
    ideal = 1 + 1 / np.log2(3)
    assert abs(s["ndcg@10"] - (1 / np.log2(3) + 1 / np.log2(5)) / ideal) < 1e-9  # duplicate hit at rank 3 earns nothing


def test_weighted_rrf_prefers_items_ranked_high_by_both():
    ids, scores = rrf([np.array([1, 2, 3]), np.array([2, 3, 1])], [0.5, 0.5], k=60)
    assert ids[0] == 2 and list(scores) == sorted(scores, reverse=True)
    ids, _ = rrf([np.array([1, 2]), np.array([2, 1])], [0.9, 0.1], k=60)
    assert ids[0] == 1  # dense-heavy weighting follows the dense ranking


def test_top_respects_mask():
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    assert list(top(scores, 2, None)) == [0, 1]
    assert list(top(scores, 2, np.array([False, True, False, True]))) == [1, 3]


def test_filters_prefer_month_over_year_and_need_one_doc_type():
    f = parse_filters("What range did Bernanke give in April 2011 for unemployment in the fourth quarter of 2013?", ["date"])
    assert f == {"months": {"2011-04"}}
    assert parse_filters("In the fall of 1980, what worried a participant?", ["date"]) == {"years": {"1980"}}
    assert parse_filters("minutes and the press conference in March 2020", ["type"]) == {}  # two types: no filter
    assert parse_filters("at the June 2016 press conference", ["type"]) == {"types": {"presconf"}}


def test_pack_breaks_at_headings_and_splits_nothing_under_size():
    units = [(0, 10, 50, True), (11, 20, 50, False), (21, 30, 50, True), (31, 40, 200, False)]
    assert _pack(units, 120, "x" * 40) == [(0, 20), (21, 30), (31, 40)]


def test_fixed_windows_are_exact_slices_with_overlap():
    from fomcrag.chunking import fixed, token_spans
    text = " ".join(f"word{i}" for i in range(300))
    n = len(token_spans(text))
    spans = fixed(text, 100, overlap=20)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(b[0] < a[1] for a, b in zip(spans, spans[1:]))  # consecutive windows overlap
    assert len(spans) == -(-(n - 20) // 80)


def test_sentences_keep_closing_quotes_and_split_on_capitals():
    from fomcrag.chunking import sentences
    text = "He said “stop.” Then it rose. e.g. not a split. Done."
    parts = [text[s:e] for s, e in sentences(text)]
    assert parts == ["He said “stop.”", "Then it rose. e.g. not a split.", "Done."]
