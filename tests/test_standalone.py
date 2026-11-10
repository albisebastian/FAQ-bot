"""The in-browser (JavaScript) matcher must give the same answers as the Python backend with the Keras model."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("keras")
if shutil.which("node") is None:
    pytest.skip("node is not installed", allow_module_level=True)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
if not (MODEL_DIR / "faq_model.keras").exists():
    pytest.skip("no Keras model in model/", allow_module_level=True)

sys.path.insert(0, str(ROOT / "standalone"))
import build_standalone  # noqa: E402
import matcher  # noqa: E402
import normalizer  # noqa: E402
from app import create_app  # noqa: E402

EXTRA = ["", "the and of", "qzxv wlkj", "CAFÉ déjà vu naïve", "refund refund refund refund", "track!!! my??? order...",
         "can you ship to a yacht", "do you accept bitcoin", "my dog chewed up my package", "Am I talking to a robot?",
         "x " * 200, "What's your return policy?", "pay in 4 installments", "gift card", "I forgot my login password"]


@pytest.fixture(scope="module")
def results():
    questions = (list(pd.read_csv(ROOT / "data" / "test_queries_merged.csv")["query"])
                 + list(pd.read_csv(ROOT / "data" / "test_queries_morph.csv")["query"])
                 + [q for q in EXTRA if q.strip()])
    model = build_standalone.export_model(MODEL_DIR)
    proc = subprocess.run(["node", str(ROOT / "standalone" / "run_queries.js")], capture_output=True, text=True,
                          input=json.dumps({"model": model, "questions": questions}), check=True)
    js = json.loads(proc.stdout)
    client = create_app(model=matcher.load_keras(MODEL_DIR)).test_client()
    py = [client.post("/api/ask", json={"question": q, "top_k": 4}).get_json() for q in questions]
    return questions, js, py


def test_same_decisions_and_answers(results):
    questions, js, py = results
    for q, a, b in zip(questions, js, py):
        assert a["confidence"] == b["confidence"], q
        assert a["matched"] == b["matched"], q
        assert a["answer"] == b["answer"], q
        assert (a["match"] or {}).get("id") == (b["match"] or {}).get("id"), q
        assert [s["id"] for s in a["suggestions"]] == [s["id"] for s in b["suggestions"]], q


def test_scores_agree_to_rounding(results):
    questions, js, py = results
    for q, a, b in zip(questions, js, py):
        sa = [(a["match"] or {"score": 0})["score"]] + [s["score"] for s in a["suggestions"]]
        sb = [(b["match"] or {"score": 0})["score"]] + [s["score"] for s in b["suggestions"]]
        assert sa == pytest.approx(sb, abs=2e-3), q


TEXTS = ["I was refunded for the returned parcels", "can't track my parcel's shipping", "CAFÉ déjà vu naïve",
         "Cancelling orders & coupons!", "my children bought it; they've gone", "", "the and of", "x " * 50,
         "pay4 it_now 2024 abc_def", "Straße ñandú ÀÉÎÕÜ", "personalised gifts for neighbours", "what is left in stock",
         "I got the wrong product", "does it work? please, thanks!", "e-mail—address…", "ﬁnal ﬂight"]


def test_js_stemmer_and_tokenizer_match_python():
    import pydoc_data.topics as topics
    words = sorted(set(re.findall(r"[a-z]+", " ".join(topics.topics.values()).lower()))
                   | set(normalizer.LEMMAS) | set(normalizer.LEMMAS.values()) | {"y", "yes", "abc123", "x_y"})
    model = build_standalone.export_model(MODEL_DIR)
    proc = subprocess.run(["node", str(ROOT / "standalone" / "check_normalizer.js")], capture_output=True, text=True,
                          input=json.dumps({"model": model, "words": words, "texts": TEXTS}), check=True)
    js = json.loads(proc.stdout)
    assert [w for w, s in zip(words, js["stems"]) if normalizer.stem(w) != s] == []
    assert [t for t, toks in zip(TEXTS, js["tokens"]) if normalizer.tokens(t) != toks] == []
