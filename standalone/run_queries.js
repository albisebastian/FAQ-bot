// Node helper used by tests/test_standalone.py: reads {model, questions} JSON on stdin, prints /api/ask results.
const createLocalApi = require("./local_api.js");
let input = "";
process.stdin.on("data", (d) => (input += d));
process.stdin.on("end", async () => {
  const { model, questions } = JSON.parse(input);
  const api = createLocalApi(model);
  const out = [];
  for (const q of questions) out.push(await api("/api/ask", { method: "POST", body: JSON.stringify({ question: q, top_k: 4 }) }));
  console.log(JSON.stringify(out));
});
