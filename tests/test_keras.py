"""Keras export path (skipped automatically when keras/tensorflow is not installed)."""
import json
from pathlib import Path

import joblib
import pytest

keras = pytest.importorskip("keras")
import sklearn  # noqa: E402
from keras import layers  # noqa: E402

import matcher  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "faqs.csv"
QUERIES = ["track parcel status", "I can't remember my password", "do you accept paypal",
           "how long is the warranty", "what is the capital of France"]


def export_like_notebook(out_dir, df):
    """Mirror of the notebook's training + save cells."""
    sk = matcher.train(df)
    n, vocab = sk["matrix"].shape
    inp = keras.Input(shape=(vocab,))
    out = layers.Dense(n, use_bias=False, trainable=False, name="cosine_similarity")(inp)
    km = keras.Model(inp, out)
    km.get_layer("cosine_similarity").set_weights([sk["matrix"].toarray().T.astype("float32")])
    km.save(out_dir / "faq_model.keras")
    joblib.dump(sk["vectorizer"], out_dir / "tfidf_vectorizer.joblib")
    (out_dir / "faq_meta.json").write_text(json.dumps(
        {"faqs": sk["faqs"], "trained_at": "t", "sklearn_version": sklearn.__version__,
         "preprocess": sk["preprocess"]}, default=int))
    return sk


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    d = tmp_path_factory.mktemp("keras_model")
    sk = export_like_notebook(d, matcher.load_faqs(DATA))
    return d, sk


def test_keras_matches_sklearn(exported):
    d, sk = exported
    km = matcher.load_keras(d)
    assert km["engine"] == "keras"
    for q in QUERIES:
        a, b = matcher.match(km, q, 3), matcher.match(sk, q, 3)
        assert [r["id"] for r in a] == [r["id"] for r in b]
        assert [r["score"] for r in a] == pytest.approx([r["score"] for r in b], abs=1e-3)


def test_load_best_prefers_keras(exported, tmp_path):
    d, _ = exported
    assert matcher.load_best(d, DATA)["engine"] == "keras"


def test_load_best_falls_back_when_keras_broken(exported, tmp_path):
    d, _ = exported
    broken = tmp_path / "model"
    broken.mkdir()
    for f in matcher.KERAS_FILES:
        (broken / f).write_bytes((d / f).read_bytes())
    (broken / "faq_model.keras").write_bytes(b"corrupt")
    model = matcher.load_best(broken, DATA)
    assert model.get("engine") != "keras" and len(model["faqs"]) == 34


def test_load_best_without_keras_files_uses_sklearn(tmp_path):
    model = matcher.load_best(tmp_path, DATA)
    assert model.get("engine") != "keras"


def test_api_with_keras_model(exported):
    from app import create_app
    client = create_app(model=matcher.load_keras(exported[0])).test_client()
    assert client.get("/api/health").get_json()["engine"] == "keras"
    data = client.post("/api/ask", json={"question": "How do I track my order?"}).get_json()
    assert data["match"]["id"] == 1 and data["confidence"] == "high"


def test_fallback_uses_merged_faqs_exported_by_notebook(exported, tmp_path):
    """If the Keras model can't load, retrain from the notebook's merged faqs.csv (not the stale data/ copy)."""
    d, _ = exported
    broken = tmp_path / "model"
    broken.mkdir()
    for f in matcher.KERAS_FILES:
        (broken / f).write_bytes((d / f).read_bytes())
    (broken / "faq_model.keras").write_bytes(b"corrupt")
    (broken / "faq_model.joblib").write_bytes(b"stale model that must not be used")
    (broken / "faqs.csv").write_text(
        "id,category,question,answer\n1,A,Do you offer student discounts?,Yes 10% off.\n2,A,Where is my parcel?,Track it.\n")
    model = matcher.load_best(broken, DATA)
    assert len(model["faqs"]) == 2
    assert matcher.match(model, "student discounts", 1)[0]["id"] == 1


def _copy_export(src, dst):
    dst.mkdir()
    for f in matcher.KERAS_FILES:
        (dst / f).write_bytes((src / f).read_bytes())
    return dst


def _pretend_other_sklearn_version(d):
    meta = json.loads((d / "faq_meta.json").read_text())
    meta["sklearn_version"] = "0.0.1"
    (d / "faq_meta.json").write_text(json.dumps(meta))


def test_other_sklearn_version_rebuilds_vectorizer_and_still_uses_keras(exported, tmp_path):
    d = _copy_export(exported[0], tmp_path / "m")
    (d / "faqs.csv").write_bytes(DATA.read_bytes())      # dataset exported next to the model by the notebook
    _pretend_other_sklearn_version(d)
    (d / "tfidf_vectorizer.joblib").write_bytes(b"incompatible pickle - must not be loaded")
    model = matcher.load_keras(d)
    assert model["engine"] == "keras"
    assert matcher.match(model, "track parcel status", 1)[0]["id"] == 1


def test_other_sklearn_version_without_csv_falls_back(exported, tmp_path):
    d = _copy_export(exported[0], tmp_path / "m")
    _pretend_other_sklearn_version(d)
    with pytest.raises(RuntimeError):
        matcher.load_keras(d)
    assert matcher.load_best(d, DATA).get("engine") != "keras"


def test_self_check_rejects_mismatched_model_and_vectorizer(exported, tmp_path):
    d = _copy_export(exported[0], tmp_path / "m")
    other = matcher.train(matcher.load_faqs(DATA).iloc[::-1].reset_index(drop=True))  # different FAQ order
    joblib.dump(other["vectorizer"], d / "tfidf_vectorizer.joblib")
    meta = json.loads((d / "faq_meta.json").read_text())
    meta["faqs"] = meta["faqs"][::-1]
    (d / "faq_meta.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError):
        matcher.load_keras(d)


def test_shipped_model_in_model_dir_loads_and_passes_self_check():
    shipped = Path(__file__).resolve().parent.parent / "model"
    if not (shipped / "faq_model.keras").exists():
        pytest.skip("no Keras model shipped in model/")
    model = matcher.load_keras(shipped)          # raises if the model and vectorizer don't belong together
    assert len(model["faqs"]) >= 78
    assert model["preprocess"] == matcher.PREPROCESS
    for q, expected in {"is any coupon codes available": "Where can I find coupon codes and current offers?",
                        "refunding my purchase": "When will I get my refund?"}.items():
        assert matcher.match(model, q, 1)[0]["question"] == expected, q



def test_legacy_export_without_preprocess_still_loads(tmp_path):
    """Models exported before stemming existed have no 'preprocess' in faq_meta.json and must keep working."""
    df = matcher.load_faqs(DATA)
    sk = matcher.train(df, preprocess=None)
    n, vocab = sk["matrix"].shape
    inp = keras.Input(shape=(vocab,))
    out = layers.Dense(n, use_bias=False, trainable=False, name="cosine_similarity")(inp)
    km = keras.Model(inp, out)
    km.get_layer("cosine_similarity").set_weights([sk["matrix"].toarray().T.astype("float32")])
    km.save(tmp_path / "faq_model.keras")
    joblib.dump(sk["vectorizer"], tmp_path / "tfidf_vectorizer.joblib")
    (tmp_path / "faq_meta.json").write_text(json.dumps(
        {"faqs": sk["faqs"], "trained_at": "t", "sklearn_version": sklearn.__version__}, default=int))
    model = matcher.load_keras(tmp_path)
    assert model["engine"] == "keras" and model["preprocess"] is None
    assert matcher.match(model, "track parcel status", 1)[0]["id"] == 1


def test_export_from_unknown_normaliser_version_is_rejected(exported, tmp_path):
    d = _copy_export(exported[0], tmp_path / "m")
    meta = json.loads((d / "faq_meta.json").read_text())
    meta["preprocess"] = "stem_lemma_v999"
    (d / "faq_meta.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="normaliser"):
        matcher.load_keras(d)


def test_legacy_bigram_export_still_loads_on_another_sklearn_version(tmp_path):
    """First-generation exports used unigrams+bigrams and no normaliser. Reported with a real user upload: after the
    default moved to unigrams, the rebuild produced a different vocabulary and the Keras model was silently dropped."""
    df = matcher.load_faqs(DATA)
    sk = matcher.train(df, preprocess=None, config={"ngram_range": (1, 2)})
    n, vocab = sk["matrix"].shape
    inp = keras.Input(shape=(vocab,))
    out = layers.Dense(n, use_bias=False, trainable=False, name="cosine_similarity")(inp)
    km = keras.Model(inp, out)
    km.get_layer("cosine_similarity").set_weights([sk["matrix"].toarray().T.astype("float32")])
    km.save(tmp_path / "faq_model.keras")
    joblib.dump(sk["vectorizer"], tmp_path / "tfidf_vectorizer.joblib")
    (tmp_path / "faqs.csv").write_bytes(DATA.read_bytes())
    (tmp_path / "faq_meta.json").write_text(json.dumps(
        {"faqs": sk["faqs"], "trained_at": "t", "sklearn_version": "0.0.1"}, default=int))   # "trained elsewhere"
    model = matcher.load_keras(tmp_path)
    assert model["engine"] == "keras" and model["preprocess"] is None
    assert tuple(model["vectorizer"].ngram_range) == (1, 2)
    assert matcher.match(model, "track parcel status", 1)[0]["id"] == 1
