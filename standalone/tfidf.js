// Browser/Node port of the Python matcher: backend/normalizer.py (stop-words -> lemmatise -> Porter stem) followed by
// scikit-learn's TfidfVectorizer.transform (unigrams, sublinear TF, L2 norm) and cosine similarity.
// tests/test_standalone.py checks that this gives the same tokens, scores and answers as the Python code.
(function (root) {
  // ---- Porter stemmer (regex formulation); mirrors normalizer.stem() ----
  const STEP2 = {
    ational: "ate", tional: "tion", enci: "ence", anci: "ance", izer: "ize", bli: "ble", alli: "al",
    entli: "ent", eli: "e", ousli: "ous", ization: "ize", ation: "ate", ator: "ate", alism: "al",
    iveness: "ive", fulness: "ful", ousness: "ous", aliti: "al", iviti: "ive", biliti: "ble", logi: "log",
  };
  const STEP3 = { icate: "ic", ative: "", alize: "al", iciti: "ic", ical: "ic", ful: "", ness: "" };
  const c = "[^aeiou]", v = "[aeiouy]", C = c + "[^aeiouy]*", V = v + "[aeiou]*";
  const MGR0 = new RegExp("^(" + C + ")?" + V + C);
  const MEQ1 = new RegExp("^(" + C + ")?" + V + C + "(" + V + ")?$");
  const MGR1 = new RegExp("^(" + C + ")?" + V + C + V + C);
  const S_V = new RegExp("^(" + C + ")?" + v);
  const CVC = new RegExp("^" + C + v + "[^aeiouwxy]$");
  const DOUBLE = /([^aeiouylsz])\1$/;
  const STEP2_RE = new RegExp("^(.+?)(" + Object.keys(STEP2).join("|") + ")$");
  const STEP3_RE = new RegExp("^(.+?)(" + Object.keys(STEP3).join("|") + ")$");
  const STEP4_RE = /^(.+?)(al|ance|ence|er|ic|able|ible|ant|ement|ment|ent|ou|ism|ate|iti|ous|ive|ize)$/;
  const STEP4_ION_RE = /^(.+?)(s|t)(ion)$/;

  function stem(w) {
    if (w.length < 3 || !/^[a-z]+$/.test(w)) return w;
    const first = w[0];
    if (first === "y") w = "Y" + w.slice(1);
    let m;

    // step 1a
    if ((m = /^(.+?)(ss|i)es$/.exec(w))) w = m[1] + m[2];
    else if ((m = /^(.+?)([^s])s$/.exec(w))) w = m[1] + m[2];

    // step 1b
    if ((m = /^(.+?)eed$/.exec(w))) {
      if (MGR0.test(m[1])) w = w.slice(0, -1);
    } else if ((m = /^(.+?)(ed|ing)$/.exec(w)) && S_V.test(m[1])) {
      w = m[1];
      if (/(at|bl|iz)$/.test(w)) w += "e";
      else if (DOUBLE.test(w)) w = w.slice(0, -1);
      else if (CVC.test(w)) w += "e";
    }

    // step 1c
    if ((m = /^(.+?)y$/.exec(w)) && S_V.test(m[1])) w = m[1] + "i";

    // step 2
    if ((m = STEP2_RE.exec(w)) && MGR0.test(m[1])) w = m[1] + STEP2[m[2]];

    // step 3
    if ((m = STEP3_RE.exec(w)) && MGR0.test(m[1])) w = m[1] + STEP3[m[2]];

    // step 4
    if ((m = STEP4_RE.exec(w))) {
      if (MGR1.test(m[1])) w = m[1];
    } else if ((m = STEP4_ION_RE.exec(w)) && MGR1.test(m[1] + m[2])) {
      w = m[1] + m[2];
    }

    // step 5
    if ((m = /^(.+?)e$/.exec(w))) {
      const s = m[1];
      if (MGR1.test(s) || (MEQ1.test(s) && !CVC.test(s))) w = s;
    }
    if (/ll$/.test(w) && MGR1.test(w)) w = w.slice(0, -1);

    return first === "y" ? "y" + w.slice(1) : w;
  }

  function stripAccents(s) {
    return s.normalize("NFKD").replace(/[̀-ͯ]/g, "");
  }

  function create(model) {
    const vocab = new Map(model.vocabulary.map((term, i) => [term, i]));
    const stop = new Set(model.stop_words);
    const lemmas = new Map(Object.entries(model.lemmas));
    const rows = model.rows; // rows[f] = [[termIndex, weight], ...], already L2-normalised

    // Same as normalizer.tokens(): lower-case, strip accents, drop apostrophes, stop-words, lemmatise, stem.
    function tokens(text) {
      const cleaned = stripAccents(String(text).toLowerCase()).replace(/['’]/g, "");
      const out = [];
      for (const tok of cleaned.match(/[\p{L}\p{N}_]{2,}/gu) || []) {
        const lemma = lemmas.has(tok) ? lemmas.get(tok) : tok;
        if (stop.has(tok) || stop.has(lemma)) continue;
        out.push(stem(lemma));
      }
      return out;
    }

    // text -> Map(termIndex -> tf-idf weight), L2-normalised (same as TfidfVectorizer.transform)
    function transform(text) {
      const counts = new Map();
      for (const t of tokens(text)) {
        const idx = vocab.get(t);
        if (idx !== undefined) counts.set(idx, (counts.get(idx) || 0) + 1);
      }
      const vec = new Map();
      let norm = 0;
      for (const [idx, n] of counts) {
        const w = (1 + Math.log(n)) * model.idf[idx];
        vec.set(idx, w);
        norm += w * w;
      }
      norm = Math.sqrt(norm);
      if (norm > 0) for (const [idx, w] of vec) vec.set(idx, w / norm);
      return vec;
    }

    // Cosine similarity of the query with every FAQ, best first.
    function match(query, topK) {
      const q = transform(query);
      const scores = rows.map((row, f) => {
        let dot = 0;
        for (const [idx, w] of row) {
          const qw = q.get(idx);
          if (qw !== undefined) dot += qw * w;
        }
        return { index: f, score: Math.min(1, Math.max(0, dot)) };
      });
      scores.sort((a, b) => b.score - a.score || a.index - b.index);
      return scores.slice(0, topK).map((s) => ({ ...model.faqs[s.index], score: Math.round(s.score * 1e4) / 1e4 }));
    }

    return { tokens, transform, match };
  }

  root.FAQMatcher = { create, stem };
  if (typeof module !== "undefined") module.exports = root.FAQMatcher;
})(typeof window !== "undefined" ? window : globalThis);
