"""Steer generation with the probe direction and dump paired transcripts for judging.

For each conversation and each alpha, adds alpha * unit_direction to the residual
stream at the probe's layer (every position) during generation. alpha=0 is the
unsteered control. A matched-norm random direction runs as the mandatory baseline.

Output: out/steered.jsonl — one row per (convo, direction_kind, alpha) with the
generated reply. Judging is a separate step; transcripts must be blinded before any
LLM judge sees them, and 30+ get read by a human first.
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class Steerer:
    def __init__(self, layer_module, direction: torch.Tensor):
        self.direction = direction
        self.alpha = 0.0
        self._handle = layer_module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + self.alpha * self.direction.to(hidden.dtype).to(hidden.device)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    def remove(self):
        self._handle.remove()


@torch.no_grad()
def reply(model, tokenizer, messages, device, max_new_tokens=400) -> str:
    # enable_thinking=False: the HF chat template's switch for suppressing the
    # reasoning preamble (the /no_think soft switch is ollama-side only). Without
    # it the whole token budget goes to analysis and the judge would score that.
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    ids = enc["input_ids"].to(device)
    out = model.generate(
        ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9
    )
    return tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--convos", default="data/convos.jsonl")
    ap.add_argument("--direction", default="out/probe_dir.pt")
    ap.add_argument("--alphas", default="-8,-4,0,4,8")
    ap.add_argument(
        "--label",
        default="neutral",
        help="steer TOWARD lonely on neutral convos (or ablate on lonely)",
    )
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    ap.add_argument("--out", default="out/steered.jsonl")
    args = ap.parse_args()

    blob = torch.load(args.direction)
    direction = blob["direction"].float()
    direction = direction / direction.norm()
    layer = blob["layer"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()
    # hidden_states index L comes out of decoder layer L-1; layer 0 is embeddings
    assert layer >= 1, "probe picked the embedding layer; steering there is meaningless"
    layer_module = model.model.layers[layer - 1]

    torch.manual_seed(args.seed)
    rand_dir = torch.randn_like(direction)
    rand_dir = rand_dir / rand_dir.norm()

    with open(args.convos, encoding="utf-8") as handle:
        all_rows = [json.loads(line) for line in handle]
    rows = [row for row in all_rows if row["label"] == args.label]
    rows = rows[: args.limit]
    alphas = [float(a) for a in args.alphas.split(",")]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    steerer = Steerer(layer_module, direction)
    try:
        with open(args.out, "w", encoding="utf-8") as handle:
            for row in tqdm(rows, desc="steering"):
                for kind, d in (("probe", direction), ("random", rand_dir)):
                    steerer.direction = d
                    for alpha in alphas:
                        if kind == "random" and alpha == 0.0:
                            continue  # alpha=0 is identical for both kinds
                        steerer.alpha = alpha
                        text = reply(model, tokenizer, row["messages"], args.device)
                        handle.write(
                            json.dumps(
                                {
                                    "id": row["id"],
                                    "direction": kind,
                                    "alpha": alpha,
                                    "layer": layer,
                                    "reply": text,
                                }
                            )
                            + "\n"
                        )
    finally:
        steerer.remove()
    print(f"wrote steered generations to {args.out}")
    print("NEXT: blind the transcripts, read 30 yourself, then run the judge.")


if __name__ == "__main__":
    main()
