"""Flask API for the FAQ matching bot. Also serves the static frontend.

Run:  python backend/app.py        ->  http://127.0.0.1:5000
"""
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import matcher

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
DATA_PATH = Path(os.environ.get("FAQ_DATA_PATH", ROOT / "data" / "faqs.csv"))
MODEL_DIR = Path(os.environ.get("FAQ_MODEL_DIR", ROOT / "model"))

MAX_QUESTION_LEN = 500
MIN_SUGGESTION_SCORE = 0.05          # when nothing matched, show anything remotely related ("Did you mean...")
MIN_SUGGESTION_SCORE_MATCHED = 0.10  # when an answer is shown, alternatives must be clearly relevant ...
SUGGESTION_RELATIVE = 0.4            # ... and at least this fraction of the best score, so weak noise is hidden
FALLBACK_NO_SUGGESTIONS = (
    "Sorry, I couldn't find an answer to that. Try rephrasing your question "
    "or pick one of the example questions below."
)
FALLBACK_WITH_SUGGESTIONS = (
    "I'm not sure I understood that. Did you mean one of these? "
    "You can also try rephrasing your question."
)


def create_app(model: dict | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    app.config["MODEL"] = model or matcher.load_best(MODEL_DIR, DATA_PATH)

    @app.get("/api/health")
    def health():
        m = app.config["MODEL"]
        return jsonify(
            status="ok",
            engine=m.get("engine", "sklearn"),
            faq_count=len(m["faqs"]),
            vocabulary_size=len(m["vectorizer"].vocabulary_),
            trained_at=m["trained_at"],
        )

    @app.get("/api/faqs")
    def faqs():
        return jsonify(
            faqs=[
                {"id": int(f["id"]), "category": f["category"], "question": f["question"]}
                for f in app.config["MODEL"]["faqs"]
            ]
        )

    @app.post("/api/ask")
    def ask():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error="Request body must be JSON"), 400

        question = body.get("question")
        if not isinstance(question, str) or not question.strip():
            return jsonify(error="'question' must be a non-empty string"), 400
        question = question.strip()
        if len(question) > MAX_QUESTION_LEN:
            return jsonify(error=f"'question' must be at most {MAX_QUESTION_LEN} characters"), 400

        top_k = body.get("top_k", 3)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
            return jsonify(error="'top_k' must be an integer between 1 and 10"), 400

        results = matcher.match(app.config["MODEL"], question, top_k=top_k)
        best = results[0]
        confidence = matcher.confidence_label(best["score"])
        matched = confidence != "low"

        def brief(r):
            return {"id": int(r["id"]), "question": r["question"], "category": r["category"],
                    "score": r["score"]}

        # Alternatives: everything except the answer we are showing.
        candidates = results[1:] if matched else results
        floor = (max(MIN_SUGGESTION_SCORE_MATCHED, SUGGESTION_RELATIVE * best["score"]) if matched
                 else MIN_SUGGESTION_SCORE)
        suggestions = [brief(r) for r in candidates if r["score"] >= floor]
        if matched:
            answer = best["answer"]
        else:
            answer = FALLBACK_WITH_SUGGESTIONS if suggestions else FALLBACK_NO_SUGGESTIONS

        return jsonify(
            matched=matched,
            confidence=confidence,
            answer=answer,
            match=brief(best) if matched else None,
            suggestions=suggestions,
        )

    @app.get("/api/faqs/<int:faq_id>")
    def faq_answer(faq_id: int):
        for f in app.config["MODEL"]["faqs"]:
            if int(f["id"]) == faq_id:
                return jsonify(id=faq_id, category=f["category"], question=f["question"],
                               answer=f["answer"])
        return jsonify(error="FAQ not found"), 404

    # ---- frontend ----
    @app.get("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.get("/<path:filename>")
    def static_files(filename):
        if filename.startswith("api/"):
            return jsonify(error="Not found"), 404
        return send_from_directory(FRONTEND_DIR, filename)

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    create_app().run(host=os.environ.get("HOST", "127.0.0.1"), port=port, debug=False)
