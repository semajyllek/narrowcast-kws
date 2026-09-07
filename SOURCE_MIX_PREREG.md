# Pre-registration — does the source-mix result replicate in audio?

Written before any head in this design was fitted, and before the dataset had
finished downloading.

## What is being replicated

`narrowcast-plantid` found that giving some classes training rows from the
*deployment* source helps them and **damages** classes denied it, and that both
effects vanish by K = 20 as accuracy approaches its ceiling.
`narrowcast-derm` found the effects persist at every K it can measure, and that
the gap between the two domains is **not** label-set size, class distribution,
baseline accuracy, or encoder family — three encoders with three supervision
signals all produced the same damage on the same images.

The explanation left standing is about **what kind of thing the source variable
is**:

> In dermatology it is a property of the **subject** — skin tone is in every
> pixel of the lesion. In plants it is a property of the **photographer** —
> Pl@ntNet framing versus iNaturalist framing. A class denied deployment-source
> training data is denied something that pervades the signal in one case and
> something that frames it in the other.

Audio is the third modality and the test of that reading.

## The honest constraint, stated first

**There is no clean two-source audio corpus with enough shared classes.** ESC-50
and UrbanSound8K overlap on roughly four labels — dog bark, car horn, siren,
engine — which supports no K sweep at all. Speech Commands v1 against v2 is not
cleanly disjoint.

So the source variable here is **constructed**: speakers are clustered by their
mean embedding into two populations, and those stand in for the two sources. That
is a real limitation and it changes what the experiment can claim.

**It also makes this a better test of the hypothesis than a mechanical
replication would be.** Speaker identity pervades every frame of a clip, exactly
as skin tone pervades every pixel of a lesion, and unlike the framing convention
that separates two photograph corpora. If the reading above is right, audio split
by speaker population should behave like dermatology, not like plants.

## Design

Google Speech Commands v0.02, 35 words, embedded with mean-pooled
`facebook/wav2vec2-base` — the encoder this repo already uses, deliberately not
one fine-tuned on this data. Clips subsampled per word for tractability; the cap
is declared with the result.

1. **Build the source variable.** Mean embedding per speaker, k-means into two
   clusters. Cluster A is the training population, cluster B the deployment one.
2. **Reserve** `r = 0.10` of words, minimum one, from receiving *any* cluster-B
   training rows.
3. **Fit two heads** on identical label sets: `A-only`, and `mixed` = A for every
   word plus B rows for the non-reserved words.
4. **Score both on held-out cluster-B clips**, split by speaker so no speaker
   straddles train and test.
5. **Sweep** `K ∈ {5, 10, 20, 35}`, 15 random word subsets per K — the same
   estimator as the plant and dermatology sweeps.

**Endpoints.** Damage to reserved words, gain to mixed words, and the arm's own
baseline top-1, so the result can be placed on the damage-versus-accuracy plane
next to the other two domains.

## Prediction, declared

**Audio will look like dermatology, not like plants: the damage will remain
substantial at K = 10 and K = 5 rather than vanishing.**

Concretely, damage at K = 10 should exceed 0.03 in magnitude — plants show 0.000
there and dermatology 0.135.

If instead audio vanishes by K = 20 the way plants do, the subject-versus-
photographer reading is wrong, and the plant/dermatology difference needs a
different explanation.

## Ways this comes out uninformative, declared in advance

1. **The source variable is constructed, so its magnitude is chosen rather than
   given.** k-means on speaker embeddings will find *some* split; how far apart
   the two populations are is an artifact of that choice. **What is testable is
   the shape across K, not the level**, and no comparison of the audio damage
   *magnitude* against plants or dermatology is licensed by this design.
2. **The premise check is the gain at K = 35.** If clustering produces two
   populations that are not meaningfully different, there is no in-source
   advantage to capture, the gain will be ~0 at every K, and the sweep says
   nothing about the prediction. That will be reported as uninterpretable rather
   than as a refutation.
3. **Speaker invariance is partly what wav2vec2 is trained for.** If the encoder
   has already removed speaker identity, the constructed split will be weak for
   that reason — which is the same failure as (2) and is not evidence about the
   hypothesis.
4. **The encoder cannot be matched to the other domains.** Images and audio do
   not share an encoder, so the cross-encoder control that ruled encoder family
   out in dermatology cannot be run here.
