from scripts.eval_generation_prefix import _ngram_repeat_ratio, _codon_to_aa, _score_stop_behavior


def test_ngram_repeat_ratio_simple():
    seq = ["ATG", "AAA", "CCC", "ATG", "AAA", "CCC"]
    r = _ngram_repeat_ratio(seq, n=3)
    # two 3-grams repeated once each over 4 total grams: uniq=2, total=4 => repeat ratio=1-2/4=0.5
    assert abs(r - 0.5) < 1e-6


def test_codon_to_aa_mapping():
    assert _codon_to_aa("ATG") == "M"
    assert _codon_to_aa("TAA") == "Stop"


def test_stop_behavior_scoring():
    # valid end stop, no early stops
    codons = ["ATG", "AAA", "CCC", "TAG"]
    score, valid, early = _score_stop_behavior(codons, truth_len_codons=4)
    assert valid and not early and score == 1.0

    # no end stop; length error 50% => score decays to >=0 but < 1
    codons2 = ["ATG", "AAA"]
    score2, valid2, _ = _score_stop_behavior(codons2, truth_len_codons=4)
    assert not valid2 and 0.0 <= score2 < 1.0

