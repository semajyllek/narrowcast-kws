"""Does the few-shot collapse on crowded label sets replicate in audio?

`narrowcast-plantid/TINY_FINDINGS.md` measures, on plants: at one training example
per label, a **crowded** label set reports coverage 0.63 at precision 0.99 while
naming an actual label on 0.8% of in-list observations, and buying a 7.6x larger
encoder does not move it. On a **separated** set the same encoder reaches 0.54
from the same single example. Both reassuring metrics point the wrong way, and
they point hardest in the few-shot regime.

That is one domain and one modality. The coarse-answer trap earned its
generality through replication -- birds, then 20 Newsgroups -- and this is the
same test for the few-shot version of it.

**Grouping is acoustic, not semantic**, which is this repo's hardest-won lesson:
"direction" (up/down/left) had within-group cosine +0.919 against an all-pairs
mean of +0.917, so group mass never summed and the cascade never used the group
rank. Semantic groups are not groups to the encoder. Groups here are k-means over
per-word centroids, so a crowded arm is genuinely confusable rather than
conceptually related.

**Shots are drawn speaker-disjointly.** Train and evaluation speakers never
overlap, so "8 examples" cannot mean eight clips of one voice scored against a
ninth of the same voice. The speaker is the cluster narrowcast splits and
bootstraps on -- the audio analogue of photographing one plant six times.

Usage:
    <plantid>/.venv/bin/python fewshot_curve.py --k 10 --draws 8
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from narrowcast import build as B

POOL = "data/pool.npz"
SHOTS = [1, 2, 4, 8, 16, 32, 64, 0]      # 0 = every available training clip
N_GROUPS = 8
BG_TRAIN = 800
MIN_ANSWERED = 100      # below this, a precision is not a measurement
P_OOD = 0.20
OTHER = B.OTHER


def _l2(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def load():
    d = np.load(POOL, allow_pickle=True)
    return _l2(d["descriptor"].astype("float32")), d["label"].astype(str), \
        d["speaker"].astype(str)


def acoustic_groups(X, y, n_groups=N_GROUPS, seed=0):
    """k-means over per-word centroids -> word: group id.

    Acoustic rather than semantic, for the reason in the module docstring. The
    centroid is the right unit: clustering raw clips would split one word across
    groups and make the group rank incoherent.
    """
    words = np.array(sorted(set(y)))
    cent = np.stack([X[y == w].mean(0) for w in words])
    lab = KMeans(n_clusters=n_groups, n_init=10, random_state=seed).fit_predict(_l2(cent))
    return dict(zip(words, [f"g{i}" for i in lab]))


def arms(groups, K, seed=0):
    """(crowded, varied) label lists of length K, plus the words left for background.

    Crowded takes whole acoustic groups so siblings stay together; varied takes
    one word per group so no two share one. Mirrors EASY/HARD in the plant sweep
    and CROWDED/VARIED in this repo's published arms.
    """
    rng = np.random.default_rng(seed)
    by_group = {}
    for w, g in groups.items():
        by_group.setdefault(g, []).append(w)
    order = sorted(by_group, key=lambda g: -len(by_group[g]))

    crowded = []
    for g in order:
        crowded += sorted(by_group[g])
        if len(crowded) >= K:
            break
    crowded = crowded[:K]

    varied = []
    for g in rng.permutation(sorted(by_group)):
        pick = sorted(by_group[g])
        varied.append(pick[rng.integers(len(pick))])
        if len(varied) >= K:
            break
    varied = varied[:K]
    return crowded, varied


def one_draw(X, y, spk, groups, chosen, shots, rng):
    """Fit at `shots` clips per label and measure through narrowcast's own path."""
    chosen = list(chosen)
    speakers = np.array(sorted(set(spk)))
    rng.shuffle(speakers)
    tr_spk = set(speakers[: len(speakers) // 2])
    is_tr = np.array([s in tr_spk for s in spk])

    Xtr, ytr = [], []
    for w in chosen:
        idx = np.flatnonzero((y == w) & is_tr)
        if len(idx) == 0:
            return None
        take = idx if shots == 0 else rng.choice(idx, min(shots, len(idx)), replace=False)
        Xtr.append(X[take]); ytr.append(np.full(len(take), w))

    # Negatives: words outside the arm, from training speakers only.
    chosen_groups = {groups[w] for w in chosen}
    out_words = np.array([w for w in sorted(set(y)) if w not in set(chosen)])
    bg_idx = np.flatnonzero(np.isin(y, out_words) & is_tr)
    bg_idx = rng.choice(bg_idx, min(BG_TRAIN, len(bg_idx)), replace=False)
    Xtr.append(X[bg_idx]); ytr.append(np.full(len(bg_idx), OTHER))

    ev = ~is_tr
    in_set = ev & np.isin(y, chosen)
    # near-OOD is a word the arm did not take that *shares an acoustic group*
    # with one it did -- the audio analogue of an unchosen congener, and the case
    # a narrow keyword set actually faces.
    near_w = [w for w in out_words if groups[w] in chosen_groups]
    near = ev & np.isin(y, near_w)
    far = ev & np.isin(y, [w for w in out_words if groups[w] not in chosen_groups])

    idx = np.r_[np.flatnonzero(in_set), np.flatnonzero(near), np.flatnonzero(far)]
    bucket = np.array(["in_catalog"] * int(in_set.sum()) + ["near_ood"] * int(near.sum())
                      + ["distant_ood"] * int(far.sum()))
    truth = np.where(bucket == "in_catalog", y[idx], OTHER)
    ds = B.Dataset(
        X_train=np.vstack(Xtr), y_train=np.concatenate(ytr), frame=pd.DataFrame(),
        X_eval=X[idx], truth=truth, bucket=bucket,
        counts={"in_catalog": int(in_set.sum())},
        # The speaker is the cluster: several clips of one voice are not
        # independent observations, exactly as several photographs of one plant
        # are not. Bootstrapping over clips would understate every interval.
        cluster=spk[idx], group=np.array([groups[w] for w in y[idx]]))
    try:
        m = B.fit_and_measure(B.score_frame(B.fit_head(ds), ds), p_ood=P_OOD, seed=0)
    except ValueError:
        return None
    # How many test rows were actually answered. Precision is a mean over exactly
    # these, and an arm that declines nearly everything can post a precision built
    # on a handful of rows -- VARIED at one shot answers ~8 across 12 draws, where
    # a printed 0.248 would mean nothing. Carried so the caller can refuse to quote
    # a precision that rests on too little.
    #
    # Summed from `per_bucket`, which holds raw counts and unweighted rates.
    # `coverage * n_test` is **not** this number: coverage is reweighted to the
    # assumed deployment prevalence by `deployment_weights`, so multiplying it by
    # a raw row count mixes a weighted rate with an unweighted denominator and
    # overstated the count by ~2x when this was first written.
    answered = int(round(sum(b["n"] * b["answered"] for b in m["per_bucket"].values())))
    return [m["label_share"], m["coverage"], m["precision"], m["closed_set_top1"],
            m["headroom"], answered]


def main(a):
    X, y, spk = load()
    groups = acoustic_groups(X, y, a.groups)
    sizes = {}
    for w, g in groups.items():
        sizes.setdefault(g, []).append(w)
    print(f"\n{len(set(y))} words, {len(set(spk))} speakers, "
          f"{a.groups} acoustic groups", flush=True)
    for g in sorted(sizes):
        print(f"  {g}: {' '.join(sorted(sizes[g]))}", flush=True)

    crowded, varied = arms(groups, a.k)
    print(f"\nCROWDED ({len(crowded)}): {' '.join(crowded)}")
    print(f"  groups: {sorted({groups[w] for w in crowded})}")
    print(f"VARIED  ({len(varied)}): {' '.join(varied)}")
    print(f"  groups: {sorted({groups[w] for w in varied})}\n", flush=True)

    for name, chosen in (("VARIED (separated)", varied), ("CROWDED", crowded)):
        print(f"=== K={a.k}  {name}  [wav2vec2-base] ===", flush=True)
        print(f"  {'shots':>6}  {'label_share':>11}  {'coverage':>9}  "
              f"{'precision':>9}  {'top1':>7}  {'headroom':>8}", flush=True)
        for shots in a.shots:
            rows = [r for i in range(a.draws)
                    if (r := one_draw(X, y, spk, groups, chosen, shots,
                                      np.random.default_rng(i))) is not None]
            if not rows:
                print(f"  {shots or 'all':>6}  (no usable draw)", flush=True)
                continue
            arr = np.array([[np.nan if v is None else v for v in r] for r in rows],
                           dtype=float)
            # An arm that answers nothing has no precision to report -- the mean
            # over zero answered observations is undefined, not zero. Printing 0
            # there would read as "answers everything wrongly", the opposite of
            # what the cascade did, which was decline rather than guess.
            with np.errstate(invalid="ignore"):
                mu = [np.nan if np.isnan(c).all() else np.nanmean(c) for c in arr.T]
            answered = int(np.nansum(arr[:, 5]))
            cell = lambda v, w: ("—".rjust(w) if np.isnan(v) else f"{v:>{w}.4f}")
            # Below this many answered rows over all draws, a precision is not a
            # measurement. Printed as "—" with the count, never as a number that
            # would be read as comparable to the other arm's.
            prec = cell(mu[2], 9) if answered >= MIN_ANSWERED else "—".rjust(9)
            print(f"  {str(shots or 'all'):>6}  {cell(mu[0], 11)}  {cell(mu[1], 9)}  "
                  f"{prec}  {cell(mu[3], 7)}  {cell(mu[4], 8)}"
                  f"   (n={len(rows)}, answered {answered})", flush=True)
        print(flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--groups", type=int, default=N_GROUPS)
    ap.add_argument("--shots", type=int, nargs="+", default=SHOTS)
    sys.exit(main(ap.parse_args()))
