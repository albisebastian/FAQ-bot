"""FAQ matching with TF-IDF + cosine similarity.

The saved model is a plain dict of scikit-learn / Python objects (no custom
classes or functions are pickled), so the same file can be produced by the
Colab notebook, `train.py`, or loaded by the Flask backend:

    {
        "vectorizer":      fitted TfidfVectorizer,
        "matrix":          sparse TF-IDF matrix, one row per FAQ question,
        "faqs":            list of dicts (id, category, question, answer),
        "preprocess":      "stem_lemma_v1" (see normalizer.py) or None for models trained without it,
        "config":          the TfidfVectorizer settings used,
        "sklearn_version": version used for training,
        "trained_at":      ISO timestamp,
    }
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import normalizer
from normalizer import PREPROCESS

# Texts are normalised first (stop-words, lemmatisation, stemming - see normalizer.py), so the vectorizer only has to
# split on whitespace. Unigrams only: word pairs across the keywords column add noise.
DEFAULT_CONFIG = {
    "lowercase": False,
    "strip_accents": None,
    "stop_words": None,
    "token_pattern": r"\S+",
    "ngram_range": (1, 1),
    "sublinear_tf": True,
}
# Config for models trained without normalisation (older exports); the vectorizer does the cleaning itself.
LEGACY_CONFIG = {
    "lowercase": True,
    "strip_accents": "unicode",
    "stop_words": "english",
    "ngram_range": (1, 1),
    "sublinear_tf": True,
}
REQUIRED_COLUMNS = {"id", "category", "question", "answer"}
QUESTION_WEIGHT = 2  # the question text is repeated this many times in the index

# Score tiers used by the API/UI (cosine similarity, 0..1).
# Tune these with the threshold analysis in the Colab notebook.
REJECT_THRESHOLD = 0.15   # below this the bot says "I don't know"
CONFIDENT_THRESHOLD = 0.35  # at/above this the match is shown as high confidence


def load_faqs(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).fillna("")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"FAQ file is missing columns: {sorted(missing)}")
    df = df[df["question"].str.strip() != ""].reset_index(drop=True)
    if df.empty:
        raise ValueError("FAQ file has no questions")
    return df


def index_text(df: pd.DataFrame) -> pd.Series:
    """Text that represents each FAQ in the index: question (x2) + answer (+ optional keywords).

    Answers add useful vocabulary ("ship", "refund", "deliver") that users type
    but that the short question alone does not contain. The question is repeated
    so that asking a FAQ's exact question always returns that FAQ rather than
    another FAQ whose answer happens to mention the same word.
    """
    text = (df["question"] + " ") * QUESTION_WEIGHT + df["answer"]
    if "keywords" in df.columns:
        text = text + " " + df["keywords"]
    return text


def encode(model: dict, texts):
    """Texts -> TF-IDF vectors, normalising them first when the model was trained that way."""
    if model.get("preprocess"):
        texts = [normalizer.normalize(t) for t in texts]
    return model["vectorizer"].transform(texts)


def train(df: pd.DataFrame, config: dict | None = None, preprocess: str | None = PREPROCESS) -> dict:
    """Fit a TF-IDF vectorizer on the (normalised) FAQ index text."""
    if preprocess not in (None, PREPROCESS):
        raise ValueError(f"unknown preprocess {preprocess!r}; this code supports {PREPROCESS!r} or None")
    config = {**(DEFAULT_CONFIG if preprocess else LEGACY_CONFIG), **(config or {})}
    vectorizer = TfidfVectorizer(**config)
    texts = index_text(df)
    if preprocess:
        texts = texts.map(normalizer.normalize)
    matrix = vectorizer.fit_transform(texts)
    return {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "faqs": df[["id", "category", "question", "answer"]].to_dict("records"),
        "preprocess": preprocess,
        "config": config,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save(model: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: str | Path) -> dict:
    model = joblib.load(path)
    if model.get("sklearn_version") != sklearn.__version__:
        raise RuntimeError(
            f"Model was trained with scikit-learn {model.get('sklearn_version')} "
            f"but {sklearn.__version__} is installed"
        )
    return model


def load_or_train(model_path: str | Path, data_path: str | Path) -> dict:
    """Load the saved model; retrain from the CSV if it is missing or incompatible."""
    try:
        return load(model_path)
    except Exception as exc:  # missing file, version mismatch, corrupt pickle
        print(f"[matcher] Could not load {model_path} ({exc}); training from {data_path}")
        model = train(load_faqs(data_path))
        try:
            save(model, model_path)
        except OSError as save_exc:
            print(f"[matcher] Could not save model: {save_exc}")
        return model


KERAS_FILES = ("faq_model.keras", "tfidf_vectorizer.joblib", "faq_meta.json")


def _rebuild_vectorizer(csv_path: Path, preprocess: str | None, vocab_size: int):
    """Refit the vectorizer from the exported dataset, matching the Keras model's input size.

    Exports made before normalisation existed (preprocess=None) were trained either with unigrams or with
    unigrams+bigrams, so try both and keep the one whose vocabulary has the size the Keras model expects.
    """
    df = load_faqs(csv_path)
    candidates = [None] if preprocess else [{"ngram_range": (1, 1)}, {"ngram_range": (1, 2)}]
    for config in candidates:
        vectorizer = train(df, config=config, preprocess=preprocess)["vectorizer"]
        if len(vectorizer.vocabulary_) == vocab_size:
            return vectorizer
    raise RuntimeError(f"could not rebuild a vectorizer with {vocab_size} words from {csv_path.name}")


def load_keras(model_dir: str | Path) -> dict:
    """Load the files exported by the Colab notebook (see notebooks/).

    faq_model.keras         Keras model: TF-IDF query vector -> cosine similarity to every FAQ
    tfidf_vectorizer.joblib fitted TfidfVectorizer (text -> TF-IDF vector)
    faq_meta.json           FAQ records (id, category, question, answer) + metadata
    """
    model_dir = Path(model_dir)
    if not all((model_dir / name).exists() for name in KERAS_FILES):
        raise FileNotFoundError(f"Keras export not found in {model_dir} (need {', '.join(KERAS_FILES)})")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import keras  # imported lazily: only needed when Keras files are present

    meta = json.loads((model_dir / "faq_meta.json").read_text(encoding="utf-8"))
    faqs = meta["faqs"]
    preprocess = meta.get("preprocess")          # absent in exports made before stemming/lemmatisation existed
    if preprocess not in (None, PREPROCESS):
        raise RuntimeError(f"Model was trained with text normaliser {preprocess!r}, this backend has {PREPROCESS!r}")

    keras_model = keras.saving.load_model(model_dir / "faq_model.keras", compile=False)

    if meta.get("sklearn_version") == sklearn.__version__:
        vectorizer = joblib.load(model_dir / "tfidf_vectorizer.joblib")
    else:
        # A pickled vectorizer from another scikit-learn version may not load (or may behave differently).
        # TF-IDF fitting is deterministic, so rebuild it from the dataset exported next to the model;
        # the self-check below proves it matches the Keras weights.
        csv_path = model_dir / "faqs.csv"
        if not csv_path.exists():
            raise RuntimeError(
                f"Vectorizer was trained with scikit-learn {meta.get('sklearn_version')} but "
                f"{sklearn.__version__} is installed, and {csv_path.name} is missing to rebuild it"
            )
        print(f"[matcher] scikit-learn {meta.get('sklearn_version')} -> {sklearn.__version__}: "
              f"rebuilding the TF-IDF vectorizer from {csv_path.name}")
        vectorizer = _rebuild_vectorizer(csv_path, preprocess, keras_model.input_shape[-1])

    model = {
        "engine": "keras",
        "keras_model": keras_model,
        "vectorizer": vectorizer,
        "preprocess": preprocess,
        "faqs": faqs,
        "trained_at": meta.get("trained_at", "unknown"),
    }
    _self_check(model)
    return model


def _self_check(model: dict) -> None:
    """Each FAQ's own question must retrieve that FAQ, otherwise the Keras model and vectorizer don't belong together."""
    faqs = model["faqs"]
    if model["keras_model"].output_shape[-1] != len(faqs):
        raise RuntimeError("Keras model and faq_meta.json do not have the same number of FAQs")
    if model["keras_model"].input_shape[-1] != len(model["vectorizer"].vocabulary_):
        raise RuntimeError("Keras model and vectorizer have different vocabulary sizes")
    vecs = encode(model, [f["question"] for f in faqs]).toarray().astype("float32")
    hits = (model["keras_model"].predict(vecs, verbose=0).argmax(axis=1) == range(len(faqs))).mean()
    if hits < 0.9:
        raise RuntimeError(f"Keras model failed its self-check ({hits:.0%} of FAQs match their own question)")


def load_best(model_dir: str | Path, data_path: str | Path) -> dict:
    """Use the Keras export if present and loadable; otherwise the sklearn model (retraining if needed)."""
    model_dir = Path(model_dir)
    if (model_dir / "faq_model.keras").exists():
        try:
            return load_keras(model_dir)
        except Exception as exc:
            print(f"[matcher] Could not use the Keras model ({exc}); falling back to scikit-learn")
            exported_csv = model_dir / "faqs.csv"   # merged dataset exported by the Colab notebook
            if exported_csv.exists():
                return train(load_faqs(exported_csv))
    return load_or_train(model_dir / "faq_model.joblib", data_path)


def confidence_label(score: float) -> str:
    if score >= CONFIDENT_THRESHOLD:
        return "high"
    if score >= REJECT_THRESHOLD:
        return "medium"
    return "low"


def match(model: dict, query: str, top_k: int = 3) -> list[dict]:
    """Return the top_k FAQs ranked by cosine similarity to the query."""
    query_vec = encode(model, [query])
    if model.get("engine") == "keras":
        scores = model["keras_model"].predict(query_vec.toarray().astype("float32"), verbose=0)[0]
        scores = scores.clip(0.0, 1.0)
    else:
        scores = cosine_similarity(query_vec, model["matrix"]).ravel()
    top_idx = scores.argsort()[::-1][:top_k]
    results = []
    for i in top_idx:
        faq = model["faqs"][i]
        results.append({**faq, "score": round(float(scores[i]), 4)})
    return results
