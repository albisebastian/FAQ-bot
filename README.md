# FAQ Matching Bot (TF-IDF + Cosine Similarity)

A small NLP project: type a question, and the bot returns the closest predefined FAQ.

* **Model** – `TfidfVectorizer` over the FAQ text, ranked by cosine similarity (scikit-learn)
* **Backend** – Flask JSON API (`backend/`)
* **Frontend** – plain HTML/CSS/JS chat UI (`frontend/`), served by Flask
* **Training notebook** – short Google Colab notebook that exports a Keras model (`notebooks/FAQ_Matching_Training.ipynb`)

```
faq-matching-bot/
├── backend/
│   ├── normalizer.py   # stop-words + lemmatization + Porter stemming (shared by backend, notebook, browser)
│   ├── matcher.py      # train / save / load / match  (the NLP core)
│   ├── train.py        # train from data/faqs.csv + evaluate
│   └── app.py          # Flask API + serves the frontend
├── frontend/           # index.html, style.css, app.js
├── data/
│   ├── faqs.csv        # id, category, question, answer, keywords   (34 sample shop FAQs)
│   ├── unusual_faqs.csv# 44 quirky extra FAQs (boats, Antarctica, bitcoin, dog ate my parcel...) - upload in Colab step 2
│   └── test_queries.csv# paraphrased + out-of-scope queries for evaluation
├── model/              # trained model exported from Colab: faq_model.keras, tfidf_vectorizer.joblib, faq_meta.json, faqs.csv
├── standalone/         # server-free version of the bot (faq_bot_standalone.html) built from the trained model
├── notebooks/FAQ_Matching_Training.ipynb   # <- upload this to Colab
└── tests/              # pytest suite for the matcher and the API
```

## Quick start

```bash
pip install -r requirements.txt
python backend/train.py     # optional: trains + prints accuracy (the backend also trains automatically if no model exists)
python backend/app.py       # open http://127.0.0.1:5000
```

Run the tests:

```bash
python -m pytest -q
```

## How it works

1. Each FAQ is represented by its text: **question (×2) + answer + keywords**. Answers add words users actually type
   ("refund", "ship"); repeating the question makes sure asking a FAQ's exact question returns that FAQ.
2. **The text is normalized** (`backend/normalizer.py`) so that different forms of a word become one token:
   stop-words are removed, irregular forms are **lemmatized** (`bought → buy`, `children → child`, `paid → pay`) and
   everything else is **stemmed** with the Porter algorithm (`refunding, refunded, refunds → refund`; `coupons → coupon`).
   Questions go through the same steps, so a word form that never appears in any FAQ still matches.
3. `TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1,1), sublinear_tf=True)` is fitted on the normalized texts.
   The user's (normalized) question is transformed with the *same* vectorizer and compared with every FAQ using
   cosine similarity.
4. The best score decides the response:

| Best score | Response |
|---|---|
| ≥ 0.35 | answer shown as a **strong match**, plus related questions that are clearly relevant |
| 0.15 – 0.35 | answer shown as a **possible match**, plus related questions |
| < 0.15 | "not sure" + **Did you mean…** suggestions (or nothing if no word overlaps) |

Thresholds live in `backend/matcher.py` (`REJECT_THRESHOLD`, `CONFIDENT_THRESHOLD`); tune them against your own test questions (`data/test_queries.csv`, `python backend/train.py`).

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | status, engine (`keras`/`sklearn`), FAQ count, vocabulary size |
| `GET` | `/api/faqs` | all FAQ questions (no answers) |
| `GET` | `/api/faqs/<id>` | one FAQ with its answer |
| `POST` | `/api/ask` | `{"question": "...", "top_k": 3}` → best match |

```bash
curl -X POST localhost:5000/api/ask -H 'Content-Type: application/json' \
     -d '{"question": "I forgot my login password"}'
```
```json
{
  "matched": true,
  "confidence": "high",
  "answer": "Click 'Log in' > 'Forgot password' and enter your email. ...",
  "match": {"id": 20, "category": "Account", "question": "I forgot my password. How can I reset it?", "score": 0.4474},
  "suggestions": [{"id": 19, "question": "How do I create an account?", "score": 0.11, "category": "Account"}]
}
```
Invalid input (missing/empty/too-long `question`, bad `top_k`, non-JSON body) returns `400` with `{"error": "..."}`.

Environment variables: `PORT` (5000), `HOST` (127.0.0.1), `FAQ_DATA_PATH`, `FAQ_MODEL_DIR`.

## Training in Google Colab (Keras export)

The notebook is short — 5 steps: **load libraries → dataset → text normalizer → train → download**. The normalizer
cell is hidden by default (the same `backend/normalizer.py`), so training and the backend always agree.

1. Open <https://colab.research.google.com> → **File → Upload notebook** → `notebooks/FAQ_Matching_Training.ipynb`.
2. **Runtime → Run all.** Training data = the **written dataset** inside the notebook **+ your own CSV**: a file picker opens
   in step 2 — choose your CSV (columns `question`, `answer`; optional `category`, `id`). It is **added** to the written
   dataset, not replacing it. Duplicate questions keep your version. Set `USE_OWN_DATA = False` (or press Cancel) to skip.
   Answers containing commas must be in double quotes — otherwise that line is skipped and a warning tells you which.
3. It downloads `faq_model_keras.zip`. **Unzip it into the project's `model/` folder** (overwrite the files).
4. Restart `python backend/app.py`. `GET /api/health` now shows `"engine": "keras"`.

What is in the zip:

| File | Purpose |
|---|---|
| `faq_model.keras` | Keras model: TF-IDF query vector → cosine similarity to every FAQ (one frozen `Dense` layer holding the FAQ vectors) |
| `tfidf_vectorizer.joblib` | scikit-learn `TfidfVectorizer` (text → TF-IDF vector) |
| `faq_meta.json` | the FAQ questions/answers + metadata, incl. `"preprocess": "stem_lemma_v1"` (which normalizer was used) |
| `faqs.csv` | the merged dataset (used if the backend has to fall back to scikit-learn) |

**Backend behaviour** (`matcher.load_best`): if the Keras files are in `model/` and `keras` is installed, it uses them.
If the scikit-learn version differs from Colab's, the vectorizer is rebuilt from the bundled `faqs.csv` and checked against
the Keras weights. Exports made before the normalizer existed (no `preprocess` in `faq_meta.json`) still load. Otherwise
(Keras not installed, files missing, a model trained with a newer normalizer than this backend, or a failed self-check)
it falls back to the scikit-learn model, retraining from the bundled `faqs.csv` if needed. So the app always starts.

To use the Keras model locally install TensorFlow: `pip install tensorflow` (skip it if pip can't find a build for your
Python version — the fallback needs nothing extra).

(Regenerate the notebook after changing `data/faqs.csv` with `python notebooks/build_notebook.py`.)

## Interface

A clean white chat box centred on a soft page (`frontend/`). Text is black; blue marks actions and focus, green marks a
strong match and a connected backend, red marks "no close match" and errors. Every state also has a text label, so colour is
never the only signal. The input is a boxed field with the send button inside. Hovering the chat box lifts it slightly with a
deeper blue-tinted shadow, and the example questions, answer chips and input box have their own small hover states.

Small animations show what is happening: a slow ripple mark and status dot, messages that fade in, three pulsing dots while an
answer is looked up, and a match meter that fills while its number counts up (a dashed fill means a weaker "possible" match).
All motion is switched off for visitors who prefer reduced motion. The colours are CSS variables at the top of
`frontend/style.css`; the backend, API and matching are unchanged.

## Standalone page (no Python needed)

`standalone/faq_bot_standalone.html` is the whole bot in one file: **double-click it** and it runs offline in the browser.
It contains a JavaScript port of the normalizer and the TF-IDF + cosine step, plus the trained model's weights (vocabulary, IDF values and the
FAQ vectors from `faq_model.keras`). `tests/test_standalone.py` checks it gives the same answers as the Python backend.

Rebuild it after training a new model (needs the Keras model in `model/` and `pip install tensorflow`):

```bash
python standalone/build_standalone.py
```

## Using your own FAQs

Edit `data/faqs.csv` (columns `id, category, question, answer`), then run `python backend/train.py`.
Add a `keywords` column (space-separated synonyms and word forms per FAQ). When a question is answered wrongly, the fix is usually to add the missing word to the right FAQ's keywords, then retrain.
The frontend title text is in `frontend/index.html`.

## Results & limitations (honest numbers)

TF-IDF matches **words, not meaning**. Two things help it with words it has not seen:

* **Stemming + lemmatization** handle different *forms* of a word ("refunding" ↔ "refund", "coupons" ↔ "coupon").
* The **`keywords` column** handles different *words* with the same meaning ("coupon" ↔ "promo code").

| Test set (top-1 / top-3) | plain TF-IDF | + keywords | + stemming & lemmatization |
|---|---|---|---|
| 83 fresh paraphrased questions over all 78 FAQs (`data/test_queries_merged.csv`) | 53% / 66% | 87% / 94% | **89% / 98%** |
| 50 questions using other forms of FAQ words (`data/test_queries_morph.csv`) | 46% / 58% | 68% / 84% | **88% / 100%** |
| 60 paraphrases of the written FAQs (`data/test_queries.csv`, `python backend/train.py`) | 55% / 77% | 85% / 95% | **92% / 97%** |

Read these with care: the keywords and the test questions were written by the same person, after seeing earlier
misses, so real-world accuracy will be lower. The plain column is the honest baseline. Other limits:

* **Stemming does not catch synonyms or typos.** "voucher" ≠ "coupon" and "refnd" ≠ "refund": add synonyms to `keywords`.
* Porter stemming is aggressive and sometimes merges unrelated words (`personal` and `personalised` both become `person`).
* The lemmatizer is a dictionary of ~140 irregular forms, not a full WordNet lemmatizer; add entries to `LEMMAS` in
  `backend/normalizer.py` if you find missed ones (then bump `PREPROCESS` and retrain).
* A question made of one distinctive word can score high on a short FAQ; the UI labels scores as a "Strong" or "Possible" match, never as certainty.
* Off-topic questions that share a word with an FAQ (e.g. "weather" with "snowstorm") are still answered as a best guess
  (2 of 8 off-topic test questions were). Raising `REJECT_THRESHOLD` trades those for more refused valid questions.

Other ways to improve: character n-grams or spelling correction for typos, or sentence embeddings (same cosine pipeline).

## What the bot understands, and what it does not

The bot **does not understand questions**. It compares the *words* of a question with the words of each FAQ and returns the
written answer of the closest one. The "similarity" shown in the UI is that word overlap, not a measure of whether the
FAQ really answers what the user meant.

Example: "give a coupon code" and "how do I use a coupon code" contain the same content words (`coupon code`), because
"give", "how", "do" are stop-words. The matcher cannot tell *asking for a code* from *asking how to apply one*. They are
now answered correctly only because the data has **both** FAQs ("Where can I find coupon codes and current offers?" →
check the Offers banner on the home page; "How do I use a discount code or promo code?" → enter it at checkout).

So the practical rule is: **a question the bot answers badly usually means the FAQ list is missing that question.**
Add an FAQ (plus `keywords`) for it, retrain, done.

Genuinely understanding intent needs a language model. A common design keeps this retrieval step and adds one more:
a hosted LLM (for example Claude) receives the user's question plus the top matching FAQs and writes an answer *only from
those FAQs*, or says it does not know. That needs an API key, costs money per question, and sends the questions to a third
party, so it is not part of this project; the retrieval code here is exactly the first half of that design.
