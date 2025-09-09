from __future__ import annotations
import sentencepiece as spm, sys
m = spm.SentencePieceProcessor()
m.Load(sys.argv[1])  # path to .model
for i in range(min(200, m.GetPieceSize())):
    print(i, m.IdToPiece(i))
