import orjson, sys
from pathlib import Path

inp = Path(sys.argv[1]); out = Path(sys.argv[2]); n=int(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

idx=0; rows=0; fh=None
with open(inp,'r',encoding='utf-8') as f:
for line in f:
if fh is None or rows>=n:
if fh: fh.close()
idx += 1; rows=0
fh = open(out/f"part-{idx:05d}.jsonl",'w',encoding='utf-8')
fh.write(line); rows += 1
if fh: fh.close()
