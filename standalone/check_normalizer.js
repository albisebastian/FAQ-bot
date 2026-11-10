// Node helper used by tests/test_standalone.py: reads {model, words, texts} JSON on stdin and prints the JS
// stems of `words` and the JS tokens of `texts`, to compare with backend/normalizer.py.
const matcher = require("./tfidf.js");
let input = "";
process.stdin.on("data", (d) => (input += d));
process.stdin.on("end", () => {
  const { model, words, texts } = JSON.parse(input);
  const m = matcher.create(model);
  console.log(JSON.stringify({ stems: words.map(matcher.stem), tokens: texts.map((t) => m.tokens(t)) }));
});
