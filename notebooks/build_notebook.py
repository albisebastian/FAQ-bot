"""Generates the short Colab notebook FAQ_Matching_Training.ipynb (embeds data/faqs.csv).

Usage:  python notebooks/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
faqs_csv = (ROOT / "data" / "faqs.csv").read_text().strip()
normalizer_src = (ROOT / "backend" / "normalizer.py").read_text()
assert "'''" not in normalizer_src

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# FAQ Matching Bot — Train & export Keras model

**Steps:** 1) load libraries → 2) dataset → 3) text normalizer (stemming + lemmatization) → 4) train → 5) download Keras model

*How it works:* text is normalized (stop-words removed, words reduced to their lemma and stem), then TF-IDF turns it into vectors. The Keras model stores every FAQ vector in one layer; multiplying a
question vector by that layer gives its **cosine similarity** to every FAQ.
""")

md("## 1. Load libraries")
code("""
import json, zipfile
from datetime import datetime, timezone

import joblib, keras, pandas as pd, sklearn
from keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
""")

md("""
## 2. Dataset = written dataset + your own CSV (merged)

* The **written dataset** below is always used.
* Your **own CSV** (optional) is uploaded in the next cell and **added** to it — it does not replace it.
  Required columns: `question`, `answer` (optional: `category`, `id`, and `keywords`). If an answer contains commas, wrap it in
  double quotes (`"like, this"`) — Excel/Sheets "Save as CSV" does this automatically.
* **`keywords` (optional, recommended):** synonyms and word forms for each FAQ, separated by spaces
  (e.g. `coupon coupons promo promotional discount offer`). TF-IDF only matches words, so this is how the bot learns that
  "coupon" means the same as "promo code". It raised accuracy on our test questions from about 46% to 87%.
""")
code(f'''%%writefile written_faqs.csv
{faqs_csv}
''')
code("""
USE_OWN_DATA = True     # True: a file picker opens - choose your CSV(s). Click Cancel / set False to use only the written dataset.

own_files = []
if USE_OWN_DATA:
    from google.colab import files
    own_files = list(files.upload().keys())
print("own files:", own_files or "none")
""")
code("""
from pathlib import Path


def read_faqs(path):
    bad = []
    df = pd.read_csv(path, encoding="utf-8-sig", engine="python",
                     on_bad_lines=lambda row: bad.append(row) or None).fillna("")   # skip (and report) malformed lines
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"question", "answer"} <= set(df.columns):
        raise ValueError(f"'{path}' doesn't look like an FAQ file: it needs 'question' and 'answer' columns but has "
                         f"{list(df.columns)}. Upload a CSV of questions with their answers (or set USE_OWN_DATA = False).")
    if "category" not in df:
        df["category"] = "General"
    if "keywords" not in df:
        df["keywords"] = ""
    if bad:
        print(f"WARNING {path}: skipped {len(bad)} malformed line(s) - wrap answers that contain commas in double quotes. "
              f"First one: {bad[0][:3]}")
    df["source"] = Path(path).name
    return df[["category", "question", "answer", "keywords", "source"]]


# own data first, so on a duplicate question your version wins over the written one
frames = [read_faqs(f) for f in own_files if f.lower().endswith(".csv")] + [read_faqs("written_faqs.csv")]
faqs = pd.concat(frames, ignore_index=True)
faqs["question"] = faqs["question"].astype(str).str.strip()
faqs = faqs[(faqs["question"] != "") & (faqs["answer"].astype(str).str.strip() != "")]
faqs = faqs.drop_duplicates(subset="question", keep="first").reset_index(drop=True)
faqs.insert(0, "id", range(1, len(faqs) + 1))               # fresh ids, so the two files can't collide
faqs.to_csv("faqs.csv", index=False)                          # merged dataset (also goes in the download)

print(f"{len(faqs)} FAQs in total:")
print(faqs["source"].value_counts().to_string())
faqs.head()
""")

md("""
## 3. Text normalizer: stemming + lemmatization

TF-IDF only matches identical words, so "refunding", "refunded" and "refunds" would count as three different words.
The normalizer reduces them to one: it removes stop-words, **lemmatizes** irregular forms (`bought → buy`,
`children → child`) and **stems** the rest with the Porter algorithm (`shipping, shipped, ships → ship`).
The same code runs in the backend and in the browser version, so training and answering always agree.
(The code is hidden: click *Show code* to read it.)
""")
NORMALIZER_CELL = """
#@title Normalizer code (click "Show code" to read)
NORMALIZER_SRC = r'''@@SRC@@'''
with open("normalizer.py", "w") as f:
    f.write(NORMALIZER_SRC)

from normalizer import normalize, PREPROCESS

for text in ["I was refunded for the returned parcels", "Cancelling orders & coupons", "my children bought it"]:
    print(f"{text!r:45} -> {normalize(text)!r}")
"""
code(NORMALIZER_CELL.replace("@@SRC@@", normalizer_src))

md("## 4. Train")
code("""
# Each FAQ = question (x2) + answer + keywords, normalized (stop-words out, lemmatized, stemmed).
# TF-IDF vectors are L2-normalised, so dot product = cosine similarity.
texts = ((faqs["question"] + " ") * 2 + faqs["answer"] + " " + faqs["keywords"]).map(normalize)
CONFIG = dict(lowercase=False, strip_accents=None, stop_words=None, token_pattern=r"\\S+", ngram_range=(1, 1), sublinear_tf=True)

vectorizer = TfidfVectorizer(**CONFIG)
faq_matrix = vectorizer.fit_transform(texts)              # (n_faqs, vocab)
n_faqs, vocab = faq_matrix.shape

# Keras model: TF-IDF query vector -> cosine similarity with every FAQ
inputs = keras.Input(shape=(vocab,), name="tfidf_query")
outputs = layers.Dense(n_faqs, use_bias=False, trainable=False, name="cosine_similarity")(inputs)
model = keras.Model(inputs, outputs, name="faq_matcher")
model.get_layer("cosine_similarity").set_weights([faq_matrix.toarray().T.astype("float32")])
model.summary()


def ask(question, top_k=3):
    scores = model.predict(vectorizer.transform([normalize(question)]).toarray(), verbose=0)[0]
    for i in scores.argsort()[::-1][:top_k]:
        print(f"  {scores[i]:.2f}  {faqs.loc[i, 'question']}")


# the second and third questions use word forms that appear nowhere in the FAQs ("refunding", "coupons")
for q in ["I forgot my login password", "refunding returned items", "any coupons?"]:
    print("Q:", q); ask(q)
""")

md("""
## 5. Save & download

Downloads **`faq_model_keras.zip`** containing `faq_model.keras`, `tfidf_vectorizer.joblib`, `faq_meta.json` and the merged `faqs.csv`.
Unzip it into the project's `model/` folder and restart the backend.
""")
code("""
model.save("faq_model.keras")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
meta = {
    "faqs": faqs[["id", "category", "question", "answer"]].to_dict("records"),
    "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "sklearn_version": sklearn.__version__,
    "keras_version": keras.__version__,
    "preprocess": PREPROCESS,          # tells the backend which text normalizer this model was trained with
}
with open("faq_meta.json", "w") as f:
    json.dump(meta, f, default=int)

with zipfile.ZipFile("faq_model_keras.zip", "w") as z:
    for name in ["faq_model.keras", "tfidf_vectorizer.joblib", "faq_meta.json", "faqs.csv"]:
        z.write(name)

try:
    from google.colab import files
    files.download("faq_model_keras.zip")
except ImportError:
    print("Saved faq_model_keras.zip in the current folder (not running in Colab).")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "colab": {"name": "FAQ_Matching_Training.ipynb", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = ROOT / "notebooks" / "FAQ_Matching_Training.ipynb"
nbf.write(nb, out)
print("wrote", out)
