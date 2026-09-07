# Audio behaves like dermatology, not like plants

Pre-registered in `SOURCE_MIX_PREREG.md`, written before the dataset finished
downloading. The prediction was that damage at K = 10 would exceed 0.03 in
magnitude. **It is 0.186** — six times the threshold, and roughly what
dermatology shows.

## The result

Speech Commands v0.02, 35 words, 14,000 clips (400 per word), 2,382 speakers,
mean-pooled `wav2vec2-base`. Speakers clustered into two populations; cluster A
trains, cluster B deploys; a reserved 10% of words receive no cluster-B training
rows; both heads scored on held-out cluster-B speakers.

| K | **damage to reserved** | sd | **gain to mixed** | sd | baseline top-1 |
|---|---|---|---|---|---|
| 5 | **−0.2141** | 0.0759 | +0.0758 | 0.0336 | 0.7684 |
| 10 | **−0.1862** | 0.0997 | +0.0923 | 0.0316 | 0.6792 |
| 20 | −0.1807 | 0.0499 | +0.1090 | 0.0228 | 0.6047 |
| 35 | −0.1854 | 0.0357 | +0.1257 | 0.0136 | 0.5450 |

**The damage does not shrink as the label set narrows.** It is flat across the
sweep and if anything largest at K = 5 — the exact opposite of plants, where it
is *zero* at K = 10.

**The premise check passed.** The gain at K = 35 is +0.1257 with sd 0.0136, so
the constructed speaker split does produce a real in-source advantage and the
sweep is interpretable.

**It is not an artifact of the clustering choice.** The pre-registration
specified a mean embedding per speaker; with ~6 clips per speaker that risks
clustering on *which words a speaker happened to say* rather than on their voice,
so a variant residualising the word out first was run as a declared secondary.
The two agree throughout — damage −0.210 / −0.187 / −0.175 / −0.151 against the
preregistered −0.214 / −0.186 / −0.181 / −0.185.

## Three domains on one plane

Against the plant damage-versus-accuracy curve fitted at K = 20
(`damage = −0.194 + 0.181 × baseline`, 28 arms):

| domain | K | baseline | damage | plant curve predicts | ratio |
|---|---|---|---|---|---|
| **audio** | 10 | 0.679 | −0.1862 | −0.0707 | 2.6× |
| **audio** | 20 | 0.605 | −0.1807 | −0.0842 | 2.1× |
| dermatology | 10 | 0.718 | −0.1354 | −0.0637 | 2.1× |
| dermatology | 20 | 0.632 | −0.1475 | −0.0792 | 1.9× |
| plants | 10 | ~0.99 | **0.0000** | — | — |
| plants | 20 | ~0.97 | −0.0093 | — | — |

Two domains that share nothing — different modality, different encoder, different
task, different source variable — land at **roughly twice the plant curve**, while
plants vanish.

## What this supports, and how far

The reading left standing after `narrowcast-derm` was about **what kind of thing
the source variable is**: a property of the *subject* that pervades the signal,
against a property of the *photographer* that frames it. Skin tone is in every
pixel of a lesion; a speaker's voice is in every frame of a clip; Pl@ntNet versus
iNaturalist framing is in neither.

Audio was chosen to test that, and it behaves as the reading predicts.

> **The magnitude is not evidence and the pre-registration says so.** The source
> variable here is *constructed* — k-means finds the most separable speaker split
> available, so how far apart the two populations sit is a choice, not a
> measurement. That audio's ratio (2.0–3.9×) resembles dermatology's (1.3–2.2×)
> is a coincidence this design cannot license. **What is testable is the shape**:
> whether the damage vanishes as K falls. It does not.

Three further limits, two of them declared in advance:

- **There is no clean two-source audio corpus with enough shared classes.**
  ESC-50 and UrbanSound8K overlap on about four labels. That is why the source
  variable had to be constructed at all.
- **The encoder cannot be matched across modalities**, so the cross-encoder
  control that ruled encoder family out in dermatology cannot be run here. Audio
  rests on one embedding set, as dermatology originally did.
- **Speaker identity is what a speech encoder is partly trained to discard.** The
  effect surviving that is mildly surprising and makes the result harder to
  dismiss, not easier — but it also means a different speech encoder could give a
  different level.

## Reproduce

```
<plantid>/.venv-mps/bin/python embed_pool.py --per-word 400
<plantid>/.venv/bin/python k_sweep.py
```
