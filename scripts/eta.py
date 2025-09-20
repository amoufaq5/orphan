import sys, json, time, statistics
from pathlib import Path

p = Path('out/text_orphgpt/progress.jsonl') if len(sys.argv)<2 else Path(sys.argv[1])
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
if len(rows) < 5:
print("Not enough logs yet."); raise SystemExit(0)
# Use step deltas
pairs = list(zip(rows[:-1], rows[1:]))
spu = [(b["step"]-a["step"]) / (b["time"]-a["time"]) for a,b in pairs if b["time"]>a["time"]]
steps_per_sec = statistics.median(spu)
print(f"steps/sec ~ {steps_per_sec:.3f}")
# Optional: if you know total steps, pass as argv2
if len(sys.argv)>=3:
total = int(sys.argv[2])
cur = rows[-1]["step"]
rem = max(total - cur, 0)
eta_sec = rem / max(steps_per_sec, 1e-6)
print(f"ETA ~ {eta_sec/3600:.2f} h (rough)")
