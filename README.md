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

## ESC-50: a genuinely hard audio task, and the trap still does not fire

Speech Commands was the wrong instrument, so here is a right one. ESC-50 is 2,000
clips over 50 classes — **40 per class** — with a documented two-level hierarchy
and within-category confusions that are acoustic rather than semantic:
`rain / sea_waves / water_drops` are all water; `airplane / helicopter` are both
motors. Encoder is AST trained on AudioSet: general audio, not speech, not tuned
on ESC-50.

| arm | coverage | precision | label-level | top-1 |
|---|---|---|---|---|
| varied (5 groups) | 0.832 | 0.991 | 0.964 | **1.000** |
| crowded (2 groups) | 0.736 | 0.870 | 0.857 | **0.857** |

**The crowded arm is genuinely harder** — top-1 falls from 1.000 to 0.857, so
condition 2 is satisfied at last. And every metric moves the *honest* direction:
coverage down, precision down, label-level down. There is no trap here, because
there is nothing hidden. A user reading coverage and precision would correctly
conclude this model is worse.

Why: the cascade still never falls back to the group rank (`t_label` = 0.0092).
Coarse accuracy exceeds fine accuracy by only **+0.077** — not enough to make
retreating to the group worth its risk under the declared utility.

### The quantity that looks like it governs this

Across every crowded arm run so far, the gap between coarse and fine accuracy —
call it *headroom* — tracks whether the trap appears:

| crowded arm | fine | coarse | headroom | Δcoverage | Δlabel-level |
|---|---|---|---|---|---|
| text (newsgroups) | 0.771 | 0.965 | **+0.194** | **+0.216** | **−0.326** |
| birds (image) | 0.905 | 0.998 | +0.093 | — | — |
| kws acoustic | 0.887 | 0.974 | +0.087 | +0.196 | +0.030 |
| ESC-50 | 0.913 | 0.990 | +0.077 | −0.095 | −0.107 |
| kws semantic | 0.897 | 0.947 | +0.050 | −0.027 | −0.077 |

Large headroom means a coarse answer is much safer than a fine one, so the
cascade retreats constantly: coverage inflates while label-level collapses. Small
headroom means retreating buys nothing and the model just degrades visibly.

This subsumes the two conditions above — group cohesion without within-group
difficulty leaves fine accuracy high and headroom small; difficulty without
cohesion leaves coarse accuracy no better than fine.

**This is a hypothesis the data suggests, not a finding.** Four arms with both
deltas is far too few for the correlations to mean anything, and `kws acoustic`
already breaks the pattern — small headroom, large coverage inversion. It is
recorded because it is testable, not because it is established.

### Speech Commands was a fact about the dataset, not about audio

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
