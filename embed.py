"""Speech Commands -> narrowcast `--embeddings` files.

Deliberately outside narrowcast. The tool takes a dataset and does not know how
it was produced, so a change of modality costs an embedding script and nothing
else — no change to the cascade, the utilities, the splits or the card.

Groups are supplied explicitly. The default rule (first whitespace token) is a
Latin-binomial convention and would make every keyword its own group.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModel

MODEL = "facebook/wav2vec2-base"          # general speech, NOT tuned on this data
SR = 16_000

# Two groups with several members each, matching the arm design used for images
# and text. A single group of N, or N groups of one, does not exercise the group
# fallback at all.
CROWDED = {"three": "digit", "four": "digit", "five": "digit", "nine": "digit",
           "up": "direction", "down": "direction", "left": "direction"}
VARIED = {"three": "digit", "up": "direction", "yes": "response",
          "stop": "command", "bird": "animal", "marvin": "name", "house": "place"}
BACKGROUND = ["bed", "cat", "dog", "happy", "wow", "tree", "visual", "sheila",
              "follow", "learn", "backward", "forward"]


def clips(root: Path, word: str, limit: int):
    """Up to `limit` one-second clips for `word`, zero-padded to a fixed length."""
    out = []
    for f in sorted((root / word).glob("*.wav"))[:limit]:
        x, sr = sf.read(f, dtype="float32")
        if sr != SR:
            continue
        x = x[:SR] if len(x) >= SR else np.pad(x, (0, SR - len(x)))
        out.append(x)
    return out


def embed(waves, fe, model, device, batch=64):
    """Mean-pooled wav2vec2 hidden states — one vector per clip."""
    out = []
    for s in range(0, len(waves), batch):
        inp = fe(waves[s:s + batch], sampling_rate=SR, return_tensors="pt",
                 padding=True).to(device)
        with torch.no_grad():
            h = model(**inp).last_hidden_state
        out.append(h.mean(1).float().cpu().numpy())
    return np.vstack(out).astype("float32")


def write(path, root, spec, fe, model, device, limit):
    waves, labels, groups = [], [], []
    for word, group in spec.items():
        w = clips(root, word, limit)
        waves += w
        labels += [word] * len(w)
        groups += [group] * len(w)
    X = embed(waves, fe, model, device)
    np.savez_compressed(path, descriptor=X, label=np.asarray(labels, dtype=str),
                        group=np.asarray(groups, dtype=str))
    print(f"{path}: {X.shape}, {len(set(labels))} labels, "
          f"{len(set(groups))} groups", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/tmp/sc")
    ap.add_argument("--out-prefix", default="/tmp/kws")
    ap.add_argument("--per-label", type=int, default=400)
    a = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval().to(device)
    print(f"{MODEL} on {device}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params\n", flush=True)

    root = Path(a.root)
    write(f"{a.out_prefix}_crowded.npz", root, CROWDED, fe, model, device, a.per_label)
    write(f"{a.out_prefix}_varied.npz", root, VARIED, fe, model, device, a.per_label)
    write(f"{a.out_prefix}_bg.npz", root, {w: "__OTHER__" for w in BACKGROUND},
          fe, model, device, a.per_label // 4)


if __name__ == "__main__":
    main()
