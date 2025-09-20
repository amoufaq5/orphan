import gradio as gr
import requests

API = "http://127.0.0.1:8000/generate"

def infer(prompt, max_new_tokens):
r = requests.post(API, json={"prompt":prompt, "max_new_tokens":int(max_new_tokens)})
return r.json().get("text","<error>")

demo = gr.Interface(
fn=infer,
inputs=[gr.Textbox(label="Prompt", lines=6), gr.Slider(16,512,step=16,value=128,label="Max new tokens")],
outputs=[gr.Textbox(label="Output", lines=12)],
title="OrphGPT Preview"
)

demo.launch()
