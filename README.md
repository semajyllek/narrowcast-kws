# narrowcast-kws — keyword spotting

A third modality for [narrowcast](https://github.com/semajyllek/narrowcast),
built as a **consumer of the installed package**, not a fork of it.

```bash
pip install narrowcast
python embed.py                              # Speech Commands -> embeddings
narrowcast fit --config acoustic_crowded.yaml --out models/kws
```

Google Speech Commands v0.02, 35 spoken words, embedded with mean-pooled
`facebook/wav2vec2-base` (94.4M) — a general speech model, deliberately *not*
one fine-tuned on this dataset.

Nothing in narrowcast changed to support audio. The tool takes a dataset; the
modality-specific part is `embed.py`, 90 lines, in this repo.

## The result: a partial non-replication, on a dataset that was the wrong choice

narrowcast's central finding is that a label set crowded with siblings of one
group buys **coverage** with coarse answers that narrow nothing, so the two
headline metrics move the reassuring way while the model gets worse. It holds on
plants, birds and text. On keyword spotting it does **not** hold in full — and
the way it fails is informative.

| grouping | arm | coverage | precision | label-level | top-1 |
|---|---|---|---|---|---|
| semantic | varied (7 groups) | 0.662 | 0.942 | 0.754 | 0.893 |
| semantic | crowded (2 groups) | 0.635 | 0.954 | 0.677 | 0.904 |
| acoustic | varied (6 groups) | 0.570 | 0.967 | 0.653 | 0.878 |
| **acoustic** | **crowded (2 groups)** | **0.766** | 0.960 | 0.683 | 0.883 |

### Semantic groups are not groups, to this encoder

The first arms grouped by meaning — *digit*, *direction*, *response*. Measured
against wav2vec2's own geometry, those are barely groups at all:

| grouping | mean within-group cosine |
|---|---|
| digit (three, four, five, nine) | +0.942 |
| direction (up, down, left) | +0.919 |
| **all word pairs** | **+0.917** |

"Direction" is +0.002 above chance. The encoder's actual nearest neighbours are
`no/one`, `six/yes`, `tree/two`, `forward/four` — phonetic, not semantic. Group
mass never summed, the cascade never used the group rank, and no coverage
inflation was possible.

### Acoustic groups restore the coverage effect, but not the collapse

Re-grouping by agglomerative clustering on the encoder's own embeddings produced
`{five, follow, forward, four, off}` — a genuinely /f/-onset cluster — and
coverage inverted as predicted: **0.766 crowded against 0.570 varied**, +19.6pp,
with precision unchanged.

But label-level share did **not** collapse: 0.683 crowded against 0.653 varied.
Closed-set top-1 is 0.883 against 0.878 — the crowded arm is not harder.

## What this decomposes

The trap needs **two independent conditions**:

1. **Group cohesion in the encoder's space** — so probability mass sums at group
   rank, γ rises, and declining stops firing.
2. **Within-group difficulty** — so the top label posterior is split and the
   cascade retreats to the group.

Speech Commands supplies the first and not the second. `five` and `four` cluster
acoustically and remain easy to separate; wav2vec2 scores ~0.88 either way.
Plants, birds and text have both: *Sedum* species are visually confusable,
`comp.*` newsgroups share vocabulary.

### This is a fact about the dataset, not about audio

**Speech Commands is the MNIST of audio** — isolated single words, clean
recordings, thousands of examples per class, chosen for exactly the tractability
that makes it a poor test of condition 2. It was the wrong dataset for the
question, and choosing it is the mistake this section records.

Audio supplies genuinely hard within-group discrimination in plenty of places:

- **congeneric bird song** — the acoustic mirror of the bird *image* arm already
  run, with the same genus/species hierarchy and species that genuinely sound alike
- **accented or noisy speech**, where minimal pairs collapse
- **respiratory and heart sounds** — crackle vs wheeze subtypes, murmur classes
- **machine fault types**, where bearing and gear faults share a spectral family
- **dialect and closely-related-language ID**

Any of those would test condition 2 properly. **None has been run here**, so this
repo shows only that condition 1 can be satisfied and condition 2 can be absent —
not that audio lacks condition 2.

**The warning narrowcast prints on a crowded label set is a warning about a risk,
not a prediction of harm.** That much this repo does establish, and it is the
useful part.

## Limits

One encoder, one corpus, six labels per arm, no cluster column (clips are
independent, and the card records that its intervals are anticonservative). The
corpus was chosen for convenience and turned out to be unsuited to half the
question; a hard-within-group audio task remains the right next test.
