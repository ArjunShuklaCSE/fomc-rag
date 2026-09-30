from fomcrag.clean import Line, clean_document, normalize, page_of


def L(text, page=1, block=0, x0=72.0, y=300.0, bold=False, size=11.0):
    return Line(text, page, block, x0, y, y + 12, 792.0, bold, size)


def clean(lines, n_pages=1):
    return clean_document(lines, n_pages)[0]


def test_running_headers_and_page_numbers_removed():
    lines = []
    for p in range(1, 9):
        lines += [L(f"Page {p} of 8", p, y=20), L("January 27–28, 2015", p, y=40), L(f"Body text {p}.", p, block=1)]
        if p % 2:  # alternating odd-page footer still counts as furniture
            lines.append(L("Minutes of the Meeting of January 27–28, 2015", p, block=2, y=760))
    text = clean(lines, 8)
    assert "Page" not in text and "January" not in text and "Minutes" not in text
    assert text == "\n\n".join(f"Body text {p}." for p in range(1, 9))


def test_hyphenation_uses_document_vocabulary():
    lines = [L("the longer-run goal. The Sys-"), L("tem expects longer-"), L("run rates to rise.")]
    assert clean(lines) == "the longer-run goal. The System expects longer-run rates to rise."


def test_speaker_turns_start_paragraphs():
    lines = [L("CHAIR POWELL. Good afternoon."), L("MICHAEL MCKEE. Thanks. A question"), L("about rates."),
             L("QUESTION. And inflation?"), L("MR. ZEISEL. That is true, yes.")]
    assert clean(lines).split("\n\n") == [
        "CHAIR POWELL. Good afternoon.", "MICHAEL MCKEE. Thanks. A question about rates.",
        "QUESTION. And inflation?", "MR. ZEISEL. That is true, yes."]


def test_sentence_continues_across_pages_and_offsets_map_to_pages():
    lines = [L("The Committee decided to", 1), L("keep the target range unchanged.", 2), L("New paragraph.", 3)]
    text, starts, _ = clean_document(lines, 3)
    assert text == "The Committee decided to keep the target range unchanged.\n\nNew paragraph."
    assert page_of(starts, 0) == 1
    assert page_of(starts, text.index("keep")) == 2
    assert page_of(starts, text.index("New")) == 3


def test_bold_headings_and_footnotes_become_their_own_blocks():
    lines = [L("Staff Review of the Economic Situation", bold=True, size=12), L("Activity rose", block=1),
             L("1 Attended Tuesday’s session only.", block=1, size=8)]
    assert clean(lines).split("\n\n") == [
        "## Staff Review of the Economic Situation", "Activity rose", "1 Attended Tuesday’s session only."]


def test_larger_font_is_a_heading_unless_mid_sentence():
    body = [L("Body text that sets the dominant font size.", block=i, size=10) for i in range(3)]
    lines = [L("Committee Policy Action", size=14), *body, L("continued text in a big font", block=5, size=14)]
    paras = clean(lines).split("\n\n")
    assert paras[0] == "## Committee Policy Action"
    assert "## continued text in a big font" not in paras


def test_normalize():
    assert normalize("0 to ¼  per­cent  ") == "0 to 1/4 percent"
    assert normalize("to 4¾ to 5 percent") == "to 4-3/4 to 5 percent"  # not "43/4"


def test_wrapped_heading_stays_one_heading():
    lines = [L("Participants’ Views on Current Condi-", bold=True), L("tions and the Economic Outlook", bold=True),
             L("In their discussion, participants noted", block=1)]
    assert clean(lines).split("\n\n")[0] == "## Participants’ Views on Current Conditions and the Economic Outlook"


def test_chart_axis_labels_dropped():
    lines = [L("Real prose stays."), L("0.7- 0.8 0.9- 1.0 1.1- 1.2 1.3- 1.4 Percent range 2 4 6 8", block=1)]
    assert clean(lines) == "Real prose stays."
