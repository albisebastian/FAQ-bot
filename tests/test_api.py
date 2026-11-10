from pathlib import Path

import pytest

import matcher
from app import create_app

DATA = Path(__file__).resolve().parent.parent / "data" / "faqs.csv"


@pytest.fixture(scope="module")
def client():
    app = create_app(model=matcher.train(matcher.load_faqs(DATA)))
    return app.test_client()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["faq_count"] == 34


def test_faqs_list_hides_answers(client):
    faqs = client.get("/api/faqs").get_json()["faqs"]
    assert len(faqs) == 34 and "answer" not in faqs[0]


def test_ask_confident_match(client):
    r = client.post("/api/ask", json={"question": "How do I track my order?"})
    data = r.get_json()
    assert r.status_code == 200
    assert data["matched"] and data["confidence"] == "high"
    assert data["match"]["id"] == 1
    assert "Track package" in data["answer"]
    assert all(s["id"] != 1 for s in data["suggestions"])


def test_ask_out_of_scope_falls_back(client):
    data = client.post("/api/ask", json={"question": "what is the capital of France"}).get_json()
    assert data["matched"] is False
    assert data["match"] is None
    assert "couldn't find" in data["answer"]
    assert data["suggestions"] == []


def test_ask_weak_match_offers_suggestions(client, monkeypatch):
    # With keywords most queries are decisive, so raise the reject threshold to force the "weak match" path.
    monkeypatch.setattr(matcher, "REJECT_THRESHOLD", 0.5)
    data = client.post("/api/ask", json={"question": "where is my package right now"}).get_json()
    assert data["matched"] is False
    assert "Did you mean" in data["answer"]
    assert 1 in [s["id"] for s in data["suggestions"]]


@pytest.mark.parametrize("payload", [
    {}, {"question": ""}, {"question": "   "}, {"question": 5}, {"question": "x" * 501},
    {"question": "hi", "top_k": 0}, {"question": "hi", "top_k": 11}, {"question": "hi", "top_k": "3"},
    {"question": "hi", "top_k": True}, [1, 2],
])
def test_ask_validation(client, payload):
    assert client.post("/api/ask", json=payload).status_code == 400


def test_ask_non_json(client):
    assert client.post("/api/ask", data="hello").status_code == 400


def test_faq_by_id(client):
    assert client.get("/api/faqs/20").get_json()["question"].startswith("I forgot my password")
    assert client.get("/api/faqs/999").status_code == 404


def test_frontend_served(client):
    assert b"FAQ Matching Bot" in client.get("/").data
    assert client.get("/app.js").status_code == 200
    assert client.get("/api/nope").status_code == 404
    assert client.get("/../backend/app.py").status_code in (404, 400)


def test_strong_match_does_not_list_irrelevant_related_questions(client):
    """Reported by a user: a 0.49 match was followed by 'related' questions scoring 0.11-0.12."""
    data = client.post("/api/ask", json={"question": "how do I track my order", "top_k": 5}).get_json()
    assert data["matched"]
    floor = max(0.10, 0.4 * data["match"]["score"])
    assert all(s["score"] >= floor for s in data["suggestions"])

