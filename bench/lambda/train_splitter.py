"""LoRA SFT of the splitter on one GPU (plan/2026-11-02-finetuning-scope.md, §10). Runs on the rented instance.

Each example is the project's prompt as a user turn, with thinking off, and the knapsack's split as the answer; the
loss covers the answer only. After training, the adapter is merged into the base and saved in the base's own format,
so vLLM serves it like the base model. Usage: python train_splitter.py [key=value ...] (see splitter.yaml)."""
import json
import logging
import math
import os
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, Trainer, TrainingArguments
from transformers import __version__ as transformers_version

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_splitter")
HERE = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config() -> DictConfig:
    cfg = OmegaConf.merge(OmegaConf.load(HERE / "splitter.yaml"), OmegaConf.from_cli())
    for key in ("data_dir", "out_dir"):
        cfg[key] = os.path.expanduser(cfg[key])
    return cfg


def encode(rows: list[dict[str, Any]], tok: Any, max_len: int) -> list[dict[str, list[int]]]:
    """Token ids with the prompt masked out of the labels; an example longer than max_len is an error, not cut."""
    out = []
    for r in rows:
        prompt = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(r["completion"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(c_ids) > max_len:
            raise ValueError(f"an example has {len(p_ids) + len(c_ids)} tokens, more than max_len={max_len}")
        out.append({"input_ids": p_ids + c_ids, "labels": [-100] * len(p_ids) + c_ids})
    return out


class Collator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [self.pad_id] * (n - len(b["input_ids"])) for b in batch]
        labels = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
        mask = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(mask)}


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def main() -> None:
    cfg = load_config()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    OmegaConf.save(cfg, out / "config.yaml")
    env = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers_version,
           "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}
    (out / "env.json").write_text(json.dumps(env, indent=2))
    log.info("config:\n%s\nenv: %s", OmegaConf.to_yaml(cfg), env)

    tok = AutoTokenizer.from_pretrained(cfg.model)
    data = Path(cfg.data_dir)
    train = encode(read(data / "train.jsonl"), tok, cfg.max_len)
    val = encode(read(data / "val.jsonl"), tok, cfg.max_len)
    log.info("%d train and %d val examples; longest %d tokens", len(train), len(val),
             max(len(x["input_ids"]) for x in train + val))

    model = AutoModelForImageTextToText.from_pretrained(cfg.model, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
                                             target_modules=cfg.lora.targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    per_step = cfg.train.batch * cfg.train.grad_accum
    total = cfg.train.max_steps if cfg.train.max_steps > 0 else math.ceil(len(train) / per_step) * cfg.train.epochs
    warmup = max(1, round(cfg.train.warmup_ratio * total))  # Transformers 5 takes warmup in steps only
    log.info("%d optimizer steps, %d of warmup", total, warmup)
    args = TrainingArguments(
        output_dir=str(out / "trainer"), num_train_epochs=cfg.train.epochs, max_steps=cfg.train.max_steps,
        learning_rate=cfg.train.lr, per_device_train_batch_size=cfg.train.batch,
        per_device_eval_batch_size=cfg.train.batch, gradient_accumulation_steps=cfg.train.grad_accum,
        warmup_steps=warmup, lr_scheduler_type="cosine", logging_steps=cfg.train.log_steps,
        eval_strategy="steps", eval_steps=cfg.train.eval_steps, save_strategy="no", bf16=True, report_to="none",
        seed=cfg.seed, data_seed=cfg.seed, remove_unused_columns=False, gradient_checkpointing=True)
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()
    (out / "log.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    model.save_pretrained(out / "adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tok.save_pretrained(out / "merged")
    AutoProcessor.from_pretrained(cfg.model).save_pretrained(out / "merged")  # vLLM loads the checkpoint as the base
    log.info("saved the adapter and the merged model in %s", out)


if __name__ == "__main__":
    main()
