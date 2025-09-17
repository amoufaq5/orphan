import json, pathlib
from src.rag.query_faiss import retrieve

GOLD = pathlib.Path("data/gold/seed_100.jsonl")
ok, total = 0, 0
for line in GOLD.read_text(encoding="utf-8").splitlines():
    if not line.strip(): continue
    ex = json.loads(line)
    q = ex["input"]
    hits = retrieve(q, k=5)
    # naive: count as "found" if any hit contains 3+ words from the input
    need = set(w for w in q.lower().split() if len(w) > 3)
    found = any(len(need & set(h["text"].lower().split())) >= 3 for h in hits)
    ok += int(found); total += 1
print(f"retrieval@5 proxy recall: {ok}/{total} = {ok/total:.1%}")
