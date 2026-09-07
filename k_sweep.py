"""Does the source-mix result replicate in audio, with speaker as the source?

Pre-registered in `SOURCE_MIX_PREREG.md`. The source variable is constructed --
there is no clean two-source audio corpus with enough shared classes -- and that
is stated there as the design's central limitation.

Two clusterings are run. The pre-registered one is the mean embedding per
speaker. The declared variant residualises the word out first (subtract each
clip's word mean before averaging per speaker), because with ~6 clips per speaker
a raw speaker mean is dominated by *which words that speaker happened to say*
rather than by their voice. The pre-registered version is primary.

Usage:
    .venv/bin/python k_sweep.py
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

KS = (5, 10, 20, 35)
DRAWS = 15
RESERVE = 0.10
TRAIN_FRAC = 0.6      # of deployment-cluster speakers
MIN_CLIPS = 3         # per speaker, to be clustered
SEED = 0


def load(path="data/pool.npz"):
    z = np.load(path, allow_pickle=True)
    X = z["descriptor"].astype(np.float32)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    df = pd.DataFrame({"label": z["label"], "speaker": z["speaker"]})
    df["row"] = np.arange(len(df))
    return df, X


def speaker_clusters(df, X, residualise: bool):
    V = X.copy()
    if residualise:
        for lab, g in df.groupby("label"):
            V[g["row"].to_numpy()] -= V[g["row"].to_numpy()].mean(0)
    keep = df.groupby("speaker").size()
    keep = set(keep.index[keep >= MIN_CLIPS])
    sub = df[df.speaker.isin(keep)]
    means = np.stack([V[g["row"].to_numpy()].mean(0)
                      for _, g in sub.groupby("speaker", sort=True)])
    names = sorted(sub["speaker"].unique())
    lab = KMeans(2, n_init=10, random_state=SEED).fit_predict(means)
    return dict(zip(names, lab))


def fit(X, rows, y):
    return LogisticRegression(max_iter=4000, C=10.0).fit(X[rows], y)


def macro(clf, X, sub, group):
    pred = clf.predict(X[sub["row"].to_numpy()])
    hit = pd.Series((pred == sub["label"].to_numpy()).astype(float))
    return float(np.nanmean(hit.groupby(pd.Series(sub["label"].to_numpy()))
                            .mean().reindex(group).to_numpy()))


def one_draw(df, X, words, K, rng):
    sub = list(rng.choice(np.array(words), K, replace=False))
    n_res = max(1, int(round(RESERVE * K)))
    reserved, mixed = sub[:n_res], sub[n_res:]

    d = df[df.label.isin(sub)]
    A = d[d.cluster == 0]
    B = d[d.cluster == 1]
    spk = np.asarray(B["speaker"].unique(), dtype=object)
    rng.shuffle(spk)
    n_tr = max(1, min(len(spk) - 1, round(len(spk) * TRAIN_FRAC)))
    B_tr = B[B.speaker.isin(spk[:n_tr]) & B.label.isin(mixed)]
    B_te = B[B.speaker.isin(spk[n_tr:])]
    if A.empty or B_tr.empty or B_te.empty:
        return None

    h_A = fit(X, A["row"].to_numpy(), A["label"].to_numpy())
    idx = np.concatenate([A["row"].to_numpy(), B_tr["row"].to_numpy()])
    y = np.concatenate([A["label"].to_numpy(), B_tr["label"].to_numpy()])
    h_mix = fit(X, idx, y)

    out = {}
    for name, group in (("reserved", reserved), ("mixed", mixed)):
        s = B_te[B_te.label.isin(group)]
        if s.empty:
            return None
        out[name] = {"A": macro(h_A, X, s, group), "mix": macro(h_mix, X, s, group)}
    return out


def sweep(residualise: bool):
    df, X = load()
    cl = speaker_clusters(df, X, residualise)
    df = df[df.speaker.isin(cl)].copy()
    df["cluster"] = df["speaker"].map(cl)
    words = sorted(df["label"].unique())
    rows = []
    for K in KS:
        acc = [r for r in (one_draw(df, X, words, min(K, len(words)),
                                    np.random.default_rng(SEED + 100 * d + K))
                           for d in range(DRAWS)) if r]
        if not acc:
            continue
        row = {"clustering": "residualised" if residualise else "preregistered", "K": K,
               "draws": len(acc)}
        for grp in ("reserved", "mixed"):
            v = np.array([a[grp]["mix"] - a[grp]["A"] for a in acc])
            row[f"d_{grp}"] = round(float(v.mean()), 4)
            row[f"sd_{grp}"] = round(float(v.std()), 4)
        row["baseline_top1"] = round(float(np.mean([a["mixed"]["A"] for a in acc])), 4)
        rows.append(row)
    n = pd.Series(list(cl.values())).value_counts().to_dict()
    return pd.DataFrame(rows), n


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    out = []
    for res in (False, True):
        t, n = sweep(res)
        print(f"\n== {'residualised' if res else 'PREREGISTERED'} clustering; "
              f"speakers per cluster {n}")
        print(t.to_string(index=False))
        out.append(t)
    pd.concat(out).to_csv("results_k_sweep.csv", index=False)
