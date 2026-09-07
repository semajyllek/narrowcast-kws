"""Speech Commands -> one embedding file carrying word, speaker and clip id.

`embed.py` writes the arms this repo already publishes. The source-mix sweep
needs something it does not provide: the **speaker** for every clip, because the
source variable is a speaker population rather than a corpus.

Subsampled per word for tractability; the cap is recorded in the output.

Usage:
    <plantid>/.venv-mps/bin/python embed_pool.py --per-word 400
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModel

MODEL = "facebook/wav2vec2-base"
SR = 16_000
ROOT = Path("data/speech_commands")


def clips(per_word: int, seed: int = 0):
    """One row per clip: word, speaker id, path. Speaker is the filename hash."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in sorted(p for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")):
        files = sorted(d.glob("*.wav"))
        if len(files) > per_word:
            files = [files[i] for i in rng.choice(len(files), per_word, replace=False)]
        rows += [{"label": d.name, "speaker": f.name.split("_")[0], "path": str(f)}
                 for f in files]
    return rows


def main(per_word: int):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()

    rows = clips(per_word)
    out, keep = [], []
    for start in range(0, len(rows), 32):
        chunk, ok = [], []
        for r in rows[start:start + 32]:
            try:
                w, sr = sf.read(r["path"], dtype="float32")
                assert sr == SR
                chunk.append(np.pad(w, (0, max(0, SR - len(w))))[:SR])
                ok.append(r)
            except Exception:
                continue
        if not chunk:
            continue
        inp = fe(chunk, sampling_rate=SR, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            h = model(**inp).last_hidden_state.mean(1)      # mean-pooled, as embed.py
        out.append(h.float().cpu().numpy())
        keep += ok
        print(f"  {min(start + 32, len(rows))}/{len(rows)}", end="\r", flush=True)

    X = np.vstack(out).astype("float32")
    np.savez_compressed(
        "data/pool.npz", descriptor=X,
        label=np.array([r["label"] for r in keep]),
        speaker=np.array([r["speaker"] for r in keep]))
    print(f"\ndata/pool.npz: {X.shape} over {len({r['label'] for r in keep})} words, "
          f"{len({r['speaker'] for r in keep})} speakers", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-word", type=int, default=400)
    main(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
