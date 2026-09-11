# The few-shot coarse-answer trap replicates in audio, and harder than on plants

Replication of `narrowcast-plantid/TINY_FINDINGS.md` §2 in a second modality.
Speech Commands, `wav2vec2-base`, K=7, 12 draws, **speaker-disjoint** train and
evaluation splits. `fewshot_curve.py`.

Grouping is **acoustic** — k-means over per-word centroids — not semantic, for the
reason this repo established the hard way: "direction" (up/down/left) had
within-group cosine +0.919 against an all-pairs mean of +0.917, so group mass
never summed and the cascade never used the group rank. The crowded arm here is
seven words inside a **single** acoustic cluster; the varied arm is seven words
from seven different ones.

## The result

| shots | CROWDED coverage | CROWDED precision | CROWDED `label_share` | VARIED coverage | VARIED precision | VARIED `label_share` |
|---|---|---|---|---|---|---|
| 1 | **0.363** | **0.909** | 0.0013 | 0.0004 | — | 0.0004 |
| 2 | **0.422** | **0.920** | 0.0005 | 0.0007 | — | 0.0008 |
| 4 | **0.477** | **0.926** | **0.0011** | 0.026 | 0.737 | **0.0304** |
| 8 | 0.502 | 0.926 | 0.070 | 0.103 | 0.850 | 0.125 |
| 16 | 0.522 | 0.925 | 0.210 | 0.233 | 0.891 | 0.281 |
| 64 | 0.619 | 0.933 | 0.455 | 0.441 | 0.933 | 0.535 |
| all | 0.655 | 0.949 | 0.586 | 0.541 | 0.943 | 0.655 |

**At four examples per label the crowded set reports 18× the coverage and names
an actual label 28× less often.** 0.477 against 0.026 on coverage, 0.0011 against
0.0304 on the only number that describes the product.

> **Precision needs its denominator, and at the low end there isn't one.** An arm
> that declines nearly everything computes precision over a handful of rows. Raw
> answered test rows, summed over 12 draws: VARIED answers **8** at one shot and
> **10** at two, so those cells are `—` rather than a number — an earlier draft
> printed 0.248 there, which is noise wearing a decimal point. At four shots
> VARIED answers **455** (~38 per draw), thin but reportable; CROWDED answers
> **20,988** at the same point. So the precision column is quotable from four
> shots up, and the coverage and `label_share` contrast — which rests on all
> 41,570 and 43,966 test rows respectively — carries the finding on its own.
>
> Note `coverage × n_test` does **not** give this count: coverage is reweighted
> to the assumed 20% out-of-catalogue prevalence by `deployment_weights`, so
> multiplying it by a raw row count mixes a weighted rate with an unweighted
> denominator and overstates by ~2×. The counts above come from `per_bucket`,
> which holds raw counts and unweighted rates.

Both headline metrics point the wrong way at once, and the distortion is a
function of how little data there is. The coverage ratio between the arms runs
**907× → 18.6× → 1.21×** at 1, 4 and unlimited shots. With enough data the two
arms converge and the varied arm ends up ahead where it belongs.

This is a cleaner instance than plants, where the crowded arm was at least *worse*
at naming labels. Here at 4 shots it is 28× worse while looking 18× better.

## The mechanism: few shots manufacture headroom

`HEADROOM_FINDINGS.md` establishes that headroom — coarse-rank accuracy minus
fine-rank accuracy — governs retreat to the group rank, at CV R² 0.883. **What
was never measured is that headroom is itself a function of the training-set
size.**

| headroom | 1 shot | 4 shots | all |
|---|---|---|---|
| audio CROWDED | **0.762** | 0.606 | 0.144 |
| audio VARIED | 0.000 | 0.000 | 0.000 |
| plants, `mobileclip2_s0`, K=20 HARD | 0.520 | 0.426 | 0.304 |
| plants, `plantclef24`, K=10 HARD | 0.431 | 0.342 | 0.263 |
| plants, `mobileclip2_s0`, K=20 EASY | 0.047 | 0.031 | 0.021 |

Starving the head collapses **fine**-rank accuracy while **coarse**-rank accuracy
holds up, because telling *nine* from *marvin* needs data and telling the g1
cluster from the g6 cluster does not. Headroom balloons, the cascade correctly
judges a group answer to be worth more than a decline, and coverage inflates
while precision holds — on answers that narrow nothing.

So the chain is: **few shots → fine accuracy collapses → headroom balloons →
retreat → coverage and precision both look good → the product is useless.** Every
step of that is already-established machinery; the new part is that data volume
drives the first step, and nothing warned about it.

The varied arm's headroom is 0.000 by construction — one word per group means a
group answer *is* a label answer — which is why it declines instead of retreating,
and why its precision is undefined-to-poor at one shot. **Declining is the honest
failure.** The crowded arm's 0.909 precision at one shot is the dishonest one.

## Bounds

Speech Commands is the MNIST of audio — clean isolated words, 400 clips per class
— and this repo has already called it the wrong instrument for the *original*
trap question. It is a reasonable instrument here because the manipulation is the
shot count rather than the difficulty, and both arms are drawn from the same
corpus under the same protocol. ESC-50 would be the harder test and is not run.

Groups are k-means clusters, so their number is chosen rather than discovered;
`--groups 8` puts seven words in the largest cluster, which is what makes a K=7
single-group crowded arm possible. A different cluster count gives different arms.
The *shape* replicates; the magnitudes are specific to this partition.

## Reproduce

```
<plantid>/.venv/bin/python fewshot_curve.py --k 7 --draws 12
```
