"""Cache residual-stream activations for every conversation at two token positions:
last token of the final user turn, and the position generating the first assistant
token (same index — the model's state as it begins to respond). Saved per layer.

Output: {out}/acts_{model_short}.pt with
  {"acts": float16 tensor [n_convos, n_layers+1, d_model], "ids": [...], "labels": [...]}
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def residual_stack(model, tokenizer, messages: list[dict], device: str) -> torch.Tensor:
    """Hidden state at the final position of the chat-templated prompt, all layers."""
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    ids = enc["input_ids"].to(device)
    out = model(ids, output_hidden_states=True)
    # hidden_states: tuple of [1, seq, d_model], embeddings + each layer
    return torch.stack([h[0, -1] for h in out.hidden_states])  # [n_layers+1, d_model]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--convos", default="data/convos.jsonl")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()

    with open(args.convos, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    acts, ids, labels = [], [], []
    for row in tqdm(rows, desc="extracting"):
        acts.append(
            residual_stack(model, tokenizer, row["messages"], args.device).cpu()
        )
        ids.append(row["id"])
        labels.append(row["label"])

    short = args.model.split("/")[-1]
    out = Path(args.out_dir) / f"acts_{short}.pt"
    torch.save(
        {
            "acts": torch.stack(acts).to(torch.float16),
            "ids": ids,
            "labels": labels,
            "model": args.model,
        },
        out,
    )
    print(f"saved {len(ids)} x {acts[0].shape} to {out}")


if __name__ == "__main__":
    main()
