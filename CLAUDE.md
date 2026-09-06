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

## The hypothesis this repo raised — now tested, and it held

**Headroom** — coarse-rank accuracy minus fine-rank accuracy — governs retreat to
the group rank. Tested in plantid over 1,409 arms (`HEADROOM_FINDINGS.md`):
cross-validated **R² 0.883**, against 0.362 for fine accuracy alone. Fitted on
plants alone it predicts these audio arms with MAE 0.033. Roughly,
**group-answer share ≈ 1.8 × headroom**.

**`kws acoustic` is not the counterexample it looked like.** Decomposing its
coverage gain shows the group answers came out of *declines* (0.430 → 0.199), not
out of label answers — coverage inflated while label share held. Text's came out
of label answers (0.719 → 0.302) and quality collapsed. Same mechanism, two
different shadows.

So the sharper version of what this repo established: **headroom predicts
retreat, not harm.** Whether retreat costs anything depends on which pool the
group answers are drawn from, and both happen. The warning stays a warning about
a risk.

*(Caveats carried over from the test: headroom is very nearly but not exactly the
governing quantity — coarse accuracy weighs ~25% more, which is the same
break-even threshold effect seen twice — and the `1.8 ×` rule **under-predicts at
high headroom**, so read it as a floor. On these audio arms specifically it is
accurate to ~0.02; on text-crowded it is low by 0.136.)*

## Practical

Speech Commands: `curl http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz`
ESC-50: `curl -L https://github.com/karolpiczak/ESC-50/archive/master.zip`

Both embed scripts supply `group` **explicitly** — the default
first-whitespace-token rule would make every keyword its own group.
