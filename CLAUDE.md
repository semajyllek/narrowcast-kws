# narrowcast-kws — orientation

Audio demo for [narrowcast](https://github.com/semajyllek/narrowcast), built as a
**consumer of the installed package** (`pip install narrowcast`), not a fork.
Nothing in the tool changed to support audio: the modality-specific part is
`embed.py` / `embed_esc50.py` in this repo.

## What it establishes

The coarse-answer trap — crowded label sets inflating coverage while quality
collapses — **does not always fire**, and this repo is where that was found.

| dataset | grouping | Δcoverage | Δlabel-level | top-1 crowded |
|---|---|---|---|---|
| Speech Commands | semantic | −0.027 | −0.077 | 0.904 |
| Speech Commands | acoustic (k-means on embeddings) | **+0.196** | +0.030 | 0.883 |
| ESC-50 | documented categories | −0.095 | −0.107 | 0.857 |

Two things were learned the hard way:

1. **Semantic groups are not groups to the encoder.** "direction" (up/down/left)
   had within-group cosine +0.919 against an all-pairs mean of +0.917. Group mass
   never summed, so the cascade never used the group rank.
2. **Speech Commands is the MNIST of audio** and was the wrong instrument — clean
   isolated words, thousands per class. ESC-50 (40 clips/class, water sounds vs
   motor sounds) is the harder test, and there the crowded arm *is* harder
   (top-1 1.000 → 0.857) yet every metric still moves the honest direction.

## The open hypothesis

**Headroom** — coarse-rank accuracy minus fine-rank accuracy — looks like the
quantity that governs whether the trap fires. Large headroom makes retreating to
the group attractive; small headroom means retreating buys nothing.

n=4 arms, and `kws acoustic` already breaks it. **Hypothesis, not a finding.**
Testing it on more arms is the most useful next thing in any of the three repos.

## Practical

Speech Commands: `curl http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz`
ESC-50: `curl -L https://github.com/karolpiczak/ESC-50/archive/master.zip`

Both embed scripts supply `group` **explicitly** — the default
first-whitespace-token rule would make every keyword its own group.
