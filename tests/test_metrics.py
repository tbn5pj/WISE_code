from wise.metrics import benchmark_scores, twowiki_answer_f1


def test_official_2wiki_special_answers():
    assert twowiki_answer_f1("yes", "yes indeed") == 0.0
    assert twowiki_answer_f1("", "") == 0.0
    assert benchmark_scores("2WikiMultihopQA", "Yes", "yes") == (1.0, 1.0)


def test_hotpot_normalized_f1_em():
    assert benchmark_scores("HotpotQA", "The Eiffel Tower", "eiffel tower") == (1.0, 1.0)
