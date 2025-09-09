from __future__ import annotations
import sentencepiece as spm, sys
m = spm.SentencePieceProcessor()
m.Load(sys.argv[1])
s = sys.argv[2] if len(sys.argv) > 2 else "hello مرحبا"
print(m.EncodeAsPieces(s))
print(m.EncodeAsIds(s))
