"""Multi-GPU variant of extract_activations.py: shards the model across every visible
GPU with device_map="auto" so a 27B bf16 model fits on two 32GB cards. Identical
math and output format to extract_activations.py (final prompt token, every residual
point, float16 on disk); only the placement differs.

    python src/extract_activations_mp.py --model Qwen/Qwen3.5-27B \
        --convos data/convos_fam_v4.jsonl --out-dir data
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def residual_stack(model, tokenizer, messages: list[dict], device) -> torch.Tensor:
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    ids = enc["input_ids"].to(device)
    out = model(ids, output_hidden_states=True)
    return torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--convos", default="data/convos_fam_v4.jsonl")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = model.get_input_embeddings().weight.device
    print(f"loaded {args.model}; embeddings on {device}; "
          f"devices: {sorted({str(p.device) for p in model.parameters()})}", flush=True)

    rows = [json.loads(l) for l in open(args.convos)]
    acts, ids, labels = [], [], []
    for row in tqdm(rows, desc="extracting"):
        acts.append(residual_stack(model, tokenizer, row["messages"], device))
        ids.append(row["id"])
        labels.append(row["label"])

    short = args.model.split("/")[-1]
    out = Path(args.out_dir) / f"acts_{short}.pt"
    torch.save(
        {"acts": torch.stack(acts).to(torch.float16), "ids": ids, "labels": labels,
         "model": args.model},
        out,
    )
    print(f"saved {len(ids)} x {acts[0].shape} to {out}", flush=True)


if __name__ == "__main__":
    main()
