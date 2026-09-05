"""ESC-50 -> narrowcast `--embeddings`. The hard-within-group audio test.

Speech Commands was the wrong instrument: isolated words, clean audio, thousands
of clips per class. ESC-50 is 2,000 clips over 50 classes -- **40 per class** --
with a documented two-level hierarchy, and within-category confusions that are
real rather than semantic:

    nature/water     rain, sea_waves, water_drops, pouring_water  (all water)
    exterior/urban   airplane, helicopter, engine, chainsaw       (all motors)

Encoder is AST trained on AudioSet: a general audio model, not a speech one, and
not fine-tuned on ESC-50.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModel

MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 16_000

# 2 groups x several members, both acoustically coherent AND internally confusable.
CROWDED = {"rain": "water", "sea_waves": "water", "water_drops": "water",
           "airplane": "motor", "helicopter": "motor"}
# one per major ESC-50 category
VARIED = {"dog": "animals", "rain": "nature", "laughing": "human",
          "clock_tick": "interior", "airplane": "exterior"}
BACKGROUND = ["cow", "crow", "frog", "hen", "crickets", "thunderstorm", "clapping",
              "coughing", "snoring", "can_opening", "keyboard_typing", "siren",
              "car_horn", "church_bells", "fireworks"]


def load_clip(path):
    """5 s at 44.1 kHz -> mono 16 kHz, which is what AST expects."""
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    return resample_poly(x, SR, sr).astype("float32") if sr != SR else x


def embed(waves, fe, model, device, batch=16):
    out = []
    for s in range(0, len(waves), batch):
        inp = fe(waves[s:s + batch], sampling_rate=SR, return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**inp).last_hidden_state
        out.append(h.mean(1).float().cpu().numpy())
    return np.vstack(out).astype("float32")


def write(path, meta, root, spec, fe, model, device):
    waves, labels, groups, folds = [], [], [], []
    for cls, group in spec.items():
        rows = meta[meta.category == cls]
        for _, r in rows.iterrows():
            waves.append(load_clip(root / r.filename))
            labels.append(cls)
            groups.append(group)
            folds.append(f"{cls}-fold{r.fold}")     # cluster: ESC-50's own folds
    X = embed(waves, fe, model, device)
    np.savez_compressed(path, descriptor=X, label=np.asarray(labels, dtype=str),
                        group=np.asarray(groups, dtype=str),
                        cluster=np.asarray(folds, dtype=str))
    print(f"{path}: {X.shape}, {len(set(labels))} labels, {len(set(groups))} groups",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/tmp/esc/ESC-50-master")
    ap.add_argument("--out-prefix", default="/tmp/esc50")
    a = ap.parse_args()

    root = Path(a.root)
    meta = pd.read_csv(root / "meta" / "esc50.csv")
    audio = root / "audio"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval().to(device)
    print(f"{MODEL} on {device}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params\n", flush=True)

    write(f"{a.out_prefix}_crowded.npz", meta, audio, CROWDED, fe, model, device)
    write(f"{a.out_prefix}_varied.npz", meta, audio, VARIED, fe, model, device)
    write(f"{a.out_prefix}_bg.npz", meta, audio,
          {c: "__OTHER__" for c in BACKGROUND}, fe, model, device)


if __name__ == "__main__":
    main()
