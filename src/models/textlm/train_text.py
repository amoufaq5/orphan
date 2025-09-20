from __future__ import annotations
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: total={total:,} trainable={trainable:,}")


def main():
cfg = Cfg(**load_yaml(os.environ.get('TRAIN_YAML','conf/train_text.yaml')))
mcfg = load_yaml('conf/model_config.yaml')

assert torch.cuda.is_available(), "CUDA NOT available — ensure H100 pod image with CUDA and drivers."

tok = AutoTokenizer.from_pretrained(cfg.tokenizer_dir) if os.path.isdir(cfg.tokenizer_dir) else AutoTokenizer.from_pretrained("gpt2")
if tok.pad_token is None:
tok.add_special_tokens({"pad_token":"<|pad|>"})

# Build model config to ~500M (actual printed below)
gcfg = GPT2Config(
vocab_size=int(mcfg.get('vocab_size', 32000)),
n_positions=int(mcfg.get('max_position_embeddings', 2048)),
n_ctx=int(mcfg.get('max_position_embeddings', 2048)),
n_layer=int(mcfg.get('n_layer', 28)),
n_head=int(mcfg.get('n_head', 20)),
n_embd=int(mcfg.get('n_embd', 1280)),
)
model = AutoModelForCausalLM.from_config(gcfg)
model.resize_token_embeddings(len(tok))

if cfg.gradient_checkpointing:
model.gradient_checkpointing_enable()

compute_params(model)

# Stream JSONL → token blocks
files = sorted(glob.glob(cfg.train_jsonl_glob, recursive=True))
assert files, f"No files matched: {cfg.train_jsonl_glob}"
ds = load_dataset("json", data_files=files, split="train", streaming=True)
def _map(batch):
return pack_texts(batch, tok, cfg.block_size, cfg.text_key)
ds = ds.map(lambda x: _map(x), batched=False)

collator = DataCollatorForLanguageModeling(tok, mlm=False)

args = TrainingArguments(
output_dir=cfg.output_dir,
logging_steps=cfg.logging_steps,
per_device_train_batch_size=cfg.batch_size_per_device,
gradient_accumulation_steps=cfg.gradient_accumulation_steps,
num_train_epochs=cfg.num_train_epochs,
learning_rate=cfg.learning_rate,
warmup_ratio=cfg.warmup_ratio,
weight_decay=cfg.weight_decay,
save_steps=cfg.save_steps,
eval_steps=cfg.eval_steps,
save_total_limit=cfg.save_total_limit,
logging_strategy=IntervalStrategy.STEPS,
save_strategy=IntervalStrategy.STEPS,
evaluation_strategy=IntervalStrategy.NO,
bf16=cfg.bf16,
report_to=["none"],
dataloader_pin_memory=True,
)

hb = Heartbeat(os.path.join(cfg.output_dir, "progress.jsonl"))

trainer = Trainer(
model=model, args=args, train_dataset=ds,
data_collator=collator,
tokenizer=tok,
callbacks=[hb],
)

trainer.train()
trainer.save_model(cfg.output_dir)

if __name__ == "__main__":
main()
