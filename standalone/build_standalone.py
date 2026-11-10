"""Build a server-free version of the bot from the trained Keras model in model/.

Exports the trained model (vocabulary, IDF weights, and the Keras layer's FAQ vectors) to JSON and inlines it with
the frontend into:
  standalone/faq_bot_standalone.html   complete page: double-click it, works offline, no Python needed
  standalone/faq_bot_page.html         same page as a fragment (for publishing as a web artifact)
  standalone/model.json                the exported model (used by the tests)

Usage:  python standalone/build_standalone.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import app as backend_app  # noqa: E402  (reuse the API's messages/limits so both stay in sync)
import matcher  # noqa: E402
import normalizer  # noqa: E402


def export_model(model_dir: Path) -> dict:
    m = matcher.load_keras(model_dir)
    if m.get("preprocess") != normalizer.PREPROCESS:
        raise SystemExit("model/ was trained without stemming/lemmatization: retrain it with the notebook "
                         "(notebooks/FAQ_Matching_Training.ipynb) and unzip the result into model/")
    vec = m["vectorizer"]
    if tuple(vec.ngram_range) != (1, 1):
        raise SystemExit("the browser version only supports unigram models")
    terms = [None] * len(vec.vocabulary_)
    for term, i in vec.vocabulary_.items():
        terms[i] = term
    kernel = m["keras_model"].get_layer("cosine_similarity").get_weights()[0]   # (vocab, n_faqs): the trained weights
    rows = []
    for f in range(kernel.shape[1]):
        nz = np.nonzero(kernel[:, f])[0]
        rows.append([[int(i), round(float(kernel[i, f]), 6)] for i in nz])
    return {
        "vocabulary": terms,
        "idf": [round(float(x), 6) for x in vec.idf_],
        "preprocess": normalizer.PREPROCESS,
        "stop_words": sorted(normalizer.STOP_WORDS),
        "lemmas": normalizer.LEMMAS,
        "rows": rows,
        "faqs": [{**f, "id": int(f["id"])} for f in m["faqs"]],
        "trained_at": m["trained_at"],
        "settings": {
            "reject_threshold": matcher.REJECT_THRESHOLD,
            "confident_threshold": matcher.CONFIDENT_THRESHOLD,
            "min_suggestion_score": backend_app.MIN_SUGGESTION_SCORE,
            "min_suggestion_score_matched": backend_app.MIN_SUGGESTION_SCORE_MATCHED,
            "suggestion_relative": backend_app.SUGGESTION_RELATIVE,
            "max_question_len": backend_app.MAX_QUESTION_LEN,
            "fallback_with_suggestions": backend_app.FALLBACK_WITH_SUGGESTIONS,
            "fallback_no_suggestions": backend_app.FALLBACK_NO_SUGGESTIONS,
        },
    }


def build(model_dir: Path = ROOT / "model") -> None:
    model = export_model(model_dir)
    model_json = json.dumps(model, separators=(",", ":"))
    (ROOT / "standalone" / "model.json").write_text(model_json, encoding="utf-8")

    def read(rel):
        return (ROOT / rel).read_text(encoding="utf-8")

    main_html = re.search(r"<main.*?</main>", read("frontend/index.html"), re.S).group(0)
    safe_json = model_json.replace("</", "<\\/")
    fragment = f"""<title>ShopEase FAQ Bot</title>
<style>
{read("frontend/style.css")}
</style>
{main_html}
<script>const MODEL = {safe_json};</script>
<script>
{read("standalone/tfidf.js")}
</script>
<script>
{read("standalone/local_api.js")}
</script>
<script>window.LOCAL_API = createLocalApi(MODEL);</script>
<script>
{read("frontend/app.js")}
</script>
"""
    (ROOT / "standalone" / "faq_bot_page.html").write_text(fragment, encoding="utf-8")

    head, rest = fragment.split("</title>\n", 1)
    full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n' + head + "</title>\n</head>\n<body>\n"
            + rest + "</body>\n</html>\n")
    (ROOT / "standalone" / "faq_bot_standalone.html").write_text(full, encoding="utf-8")
    print(f"{len(model['faqs'])} FAQs, vocabulary {len(model['vocabulary'])}, "
          f"page {len(fragment) / 1024:.0f} KB")


if __name__ == "__main__":
    build()
