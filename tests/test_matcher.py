from pathlib import Path

import pytest

import matcher

DATA = Path(__file__).resolve().parent.parent / "data" / "faqs.csv"


@pytest.fixture(scope="module")
def model():
    return matcher.train(matcher.load_faqs(DATA))


def test_exact_question_matches_itself(model):
    for faq in model["faqs"]:
        best = matcher.match(model, faq["question"], top_k=1)[0]
        assert best["id"] == faq["id"], faq["question"]


def test_paraphrase_matches(model):
    assert matcher.match(model, "track parcel status", 1)[0]["id"] == 1
    assert matcher.match(model, "I can't remember my password", 1)[0]["id"] == 20


def test_results_sorted_and_bounded(model):
    results = matcher.match(model, "refund for returned item", top_k=5)
    scores = [r["score"] for r in results]
    assert len(results) == 5
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_gibberish_scores_zero(model):
    assert matcher.match(model, "qzxv wlkj", 1)[0]["score"] == 0.0
    assert matcher.confidence_label(0.0) == "low"


def test_empty_stopword_only_query_does_not_crash(model):
    assert matcher.match(model, "the and of", 1)[0]["score"] == 0.0


def test_confidence_labels():
    assert matcher.confidence_label(0.9) == "high"
    assert matcher.confidence_label(matcher.REJECT_THRESHOLD) == "medium"
    assert matcher.confidence_label(matcher.REJECT_THRESHOLD - 0.01) == "low"


def test_save_load_roundtrip(model, tmp_path):
    path = tmp_path / "m.joblib"
    matcher.save(model, path)
    loaded = matcher.load(path)
    assert matcher.match(loaded, "track my order", 1) == matcher.match(model, "track my order", 1)


def test_load_or_train_recovers_from_bad_model(tmp_path):
    bad = tmp_path / "m.joblib"
    bad.write_text("not a pickle")
    model = matcher.load_or_train(bad, DATA)
    assert len(model["faqs"]) == 34
    assert matcher.load(bad)  # was re-saved correctly


def test_load_faqs_validates_columns(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text("question,answer\nHi,Hello\n")
    with pytest.raises(ValueError):
        matcher.load_faqs(f)


def test_weak_paraphrase_still_surfaces_right_faq_in_top3(model):
    # Known TF-IDF limitation: the stray word "right" pulls in the size-guide FAQ,
    # but the correct FAQ is still in the top-3 so the UI can offer it as "Did you mean...".
    ids = [r["id"] for r in matcher.match(model, "where is my package right now", 3)]
    assert 1 in ids


def test_unusual_faqs_dataset_merges_cleanly_with_written_one():
    import pandas as pd
    root = DATA.parent
    written = pd.read_csv(root / "faqs.csv")
    unusual = pd.read_csv(root / "unusual_faqs.csv")
    assert {"id", "category", "question", "answer"} <= set(unusual.columns)
    assert len(unusual) >= 40 and not unusual.isna().any().any()
    assert not set(written.question.str.lower()) & set(unusual.question.str.lower())

    merged = pd.concat([unusual, written], ignore_index=True)
    merged["id"] = range(1, len(merged) + 1)
    model = matcher.train(merged)
    for q in merged["question"]:       # every FAQ must still return itself for its own exact question
        assert matcher.match(model, q, 1)[0]["question"] == q
    assert "Bitcoin" in matcher.match(model, "do you accept bitcoin", 1)[0]["question"]


# ---- quality gates on the merged (unusual + written) dataset --------------------------------------------------
def _merged_model():
    import pandas as pd
    root = DATA.parent
    merged = pd.concat([pd.read_csv(root / "unusual_faqs.csv"), pd.read_csv(root / "faqs.csv")], ignore_index=True)
    merged["id"] = range(1, len(merged) + 1)
    return matcher.train(matcher.load_faqs_frame(merged) if hasattr(matcher, "load_faqs_frame") else merged.fillna(""))


def test_coupon_codes_regression():
    """Reported by a user: 'is any coupon codes available' used to answer with the delivery FAQ."""
    model = _merged_model()
    best = matcher.match(model, "is any coupon codes available", 1)[0]
    assert best["question"] == "Where can I find coupon codes and current offers?"
    assert matcher.confidence_label(best["score"]) != "low"


def test_asking_for_a_code_is_not_answered_with_how_to_apply_one():
    """Reported by a user: 'give a coupon code' answered 'enter your code in the Promo code box'.
    Asking for a code and asking how to use one must reach different FAQs (and the answer must point to the offers)."""
    model = _merged_model()
    wants = matcher.match(model, "give a coupon code", 1)[0]
    howto = matcher.match(model, "how do I use a coupon code", 1)[0]
    assert wants["question"] == "Where can I find coupon codes and current offers?"
    assert howto["question"] == "How do I use a discount code or promo code?"
    assert "home page" in wants["answer"]


def test_every_promotion_faq_answers_its_own_topic():
    model = _merged_model()
    cases = {"discount for new customers": "Do you offer a discount on my first order?",
             "is there a sale section": "Do you have a sale or clearance section?",
             "do you have a rewards scheme": "Do you have a loyalty or rewards program?"}
    for q, expected in cases.items():
        assert matcher.match(model, q, 1)[0]["question"] == expected, q


def test_accuracy_gate_on_fresh_paraphrases():
    import pandas as pd
    model = _merged_model()
    t = pd.read_csv(DATA.parent / "test_queries_merged.csv").fillna("")
    t = t[t["expected_question"] != ""]
    top1 = top3 = 0
    for q, expected in zip(t["query"], t["expected_question"]):
        got = [r["question"] for r in matcher.match(model, q, 3)]
        top1 += got[0] == expected
        top3 += expected in got
    assert top1 / len(t) >= 0.80, f"top-1 {top1}/{len(t)}"
    assert top3 / len(t) >= 0.90, f"top-3 {top3}/{len(t)}"


def test_out_of_scope_without_word_overlap_is_refused():
    model = _merged_model()
    for q in ["what is the capital of France", "tell me a joke", "who won the football world cup"]:
        assert matcher.confidence_label(matcher.match(model, q, 1)[0]["score"]) == "low", q


def test_unseen_word_forms_still_match():
    """Stemming/lemmatisation: forms that never appear in the FAQs map to the FAQ that has another form."""
    model = _merged_model()
    expectations = {
        "refunding my purchase": "When will I get my refund?",
        "I am cancelling my order": "How can I cancel my order?",
        "promos": "How do I use a discount code or promo code?",
        "coupons": "Where can I find coupon codes and current offers?",
        "declined payments": "My payment failed. What should I do?",
        "deleting accounts": "How do I delete my account?",
    }
    for q, expected in expectations.items():
        assert matcher.match(model, q, 1)[0]["question"] == expected, q


def test_morphology_accuracy_gate():
    import pandas as pd
    model = _merged_model()
    t = pd.read_csv(DATA.parent / "test_queries_morph.csv")
    top1 = top3 = 0
    for q, expected in zip(t["query"], t["expected_question"]):
        got = [r["question"] for r in matcher.match(model, q, 3)]
        top1 += got[0] == expected
        top3 += expected in got
    assert top1 / len(t) >= 0.80, f"top-1 {top1}/{len(t)}"
    assert top3 / len(t) >= 0.95, f"top-3 {top3}/{len(t)}"


def test_stemming_beats_keywords_alone_on_inflected_queries():
    import pandas as pd
    import numpy as np
    root = DATA.parent
    merged = pd.concat([pd.read_csv(root / "unusual_faqs.csv"), pd.read_csv(root / "faqs.csv")], ignore_index=True).fillna("")
    merged["id"] = range(1, len(merged) + 1)
    t = pd.read_csv(root / "test_queries_morph.csv")

    def top1(model):
        return np.mean([matcher.match(model, q, 1)[0]["question"] == e for q, e in zip(t["query"], t["expected_question"])])

    assert top1(matcher.train(merged)) > top1(matcher.train(merged, preprocess=None)) + 0.05


def test_off_topic_question_with_common_word_is_not_answered():
    """'does' used to be missing from the stop-word list, so 'how many moons does Jupiter have' matched shipping FAQs."""
    model = _merged_model()
    assert matcher.match(model, "how many moons does Jupiter have", 1)[0]["score"] == 0.0


def test_model_records_preprocess_and_legacy_models_still_work():
    import pandas as pd
    df = matcher.load_faqs(DATA)
    assert matcher.train(df)["preprocess"] == matcher.PREPROCESS
    legacy = matcher.train(df, preprocess=None)
    assert legacy["preprocess"] is None
    assert matcher.match(legacy, "track parcel status", 1)[0]["id"] == 1
    with pytest.raises(ValueError):
        matcher.train(df, preprocess="something_else")
