import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()
MODEL_DIR = "out/text_orphgpt"

tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto")

class Req(BaseModel):
prompt: str
max_new_tokens: int = 128

@app.post("/generate")
def generate(r: Req):
ids = tok(r.prompt, return_tensors='pt').to(model.device)
with torch.inference_mode():
out = model.generate(**ids, max_new_tokens=r.max_new_tokens, do_sample=True, top_p=0.9, temperature=0.8)
text = tok.decode(out[0], skip_special_tokens=True)
return {"text": text}
