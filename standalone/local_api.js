// In-browser stand-in for the Flask API (same routes, same JSON), so frontend/app.js runs unchanged
// on a page with no server. Mirrors backend/app.py.
(function (root) {
  const Matcher = root.FAQMatcher || require("./tfidf.js");

  function createLocalApi(model) {
    const matcher = Matcher.create(model);
    const S = model.settings;

    const confidenceLabel = (score) =>
      score >= S.confident_threshold ? "high" : score >= S.reject_threshold ? "medium" : "low";
    const brief = (r) => ({ id: r.id, question: r.question, category: r.category, score: r.score });

    return async function api(path, options) {
      options = options || {};
      const url = path.split("?")[0];

      if (url === "/api/health") {
        return { status: "ok", engine: "browser", faq_count: model.faqs.length,
                 vocabulary_size: model.vocabulary.length, trained_at: model.trained_at };
      }
      if (url === "/api/faqs") {
        return { faqs: model.faqs.map((f) => ({ id: f.id, category: f.category, question: f.question })) };
      }
      const byId = url.match(/^\/api\/faqs\/(\d+)$/);
      if (byId) {
        const f = model.faqs.find((x) => x.id === Number(byId[1]));
        if (!f) throw new Error("FAQ not found");
        return f;
      }
      if (url === "/api/ask" && options.method === "POST") {
        let body;
        try { body = JSON.parse(options.body); } catch (e) { throw new Error("Request body must be JSON"); }
        if (body === null || typeof body !== "object" || Array.isArray(body)) throw new Error("Request body must be JSON");
        if (typeof body.question !== "string" || !body.question.trim()) throw new Error("'question' must be a non-empty string");
        const question = body.question.trim();
        if (question.length > S.max_question_len) throw new Error(`'question' must be at most ${S.max_question_len} characters`);
        const topK = body.top_k === undefined ? 3 : body.top_k;
        if (!Number.isInteger(topK) || topK < 1 || topK > 10) throw new Error("'top_k' must be an integer between 1 and 10");

        const results = matcher.match(question, topK);
        const best = results[0];
        const confidence = confidenceLabel(best.score);
        const matched = confidence !== "low";
        const candidates = matched ? results.slice(1) : results;
        const floor = matched ? Math.max(S.min_suggestion_score_matched, S.suggestion_relative * best.score)
                              : S.min_suggestion_score;
        const suggestions = candidates.filter((r) => r.score >= floor).map(brief);
        const answer = matched ? best.answer
          : suggestions.length ? S.fallback_with_suggestions : S.fallback_no_suggestions;
        return { matched, confidence, answer, match: matched ? brief(best) : null, suggestions };
      }
      throw new Error("Not found");
    };
  }

  root.createLocalApi = createLocalApi;
  if (typeof module !== "undefined") module.exports = createLocalApi;
})(typeof window !== "undefined" ? window : globalThis);
