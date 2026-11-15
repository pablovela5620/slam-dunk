"""Bounded recording-local image retrieval using residual-aggregated RootSIFT."""

import os, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from threadpoolctl import threadpool_limits


def candidates(features, ts, rot, ext):
    start = time.monotonic()
    rng = np.random.default_rng(17)
    available = [d for _, d, _ in features if len(d)]
    if sum(len(d) for d in available) < 5000:
        return set()
    pool = np.concatenate([d[:: max(1, len(d) // 12)] for d in available])
    if len(pool) > 16000:
        pool = pool[rng.choice(len(pool), 16000, replace=False)]
    count = int(os.environ.get("VLAD_WORDS", "48"))
    centres = pool[rng.choice(len(pool), count, replace=False)].copy()
    with threadpool_limits(limits=4, user_api="blas"):
        pn = (pool * pool).sum(1)
        for _ in range(8):
            label = (
                pn[:, None] + (centres * centres).sum(1)[None, :] - 2 * pool @ centres.T
            ).argmin(1)
            sums = np.zeros_like(centres)
            np.add.at(sums, label, pool)
            sizes = np.bincount(label, minlength=count)
            ok = sizes > 0
            centres[ok] = sums[ok] / sizes[ok, None]
    cn = (centres * centres).sum(1)

    def aggregate(f):
        d = f[1]
        if not len(d):
            return np.zeros((count, 128), np.float32), np.zeros(count, np.float32)
        label = ((d * d).sum(1)[:, None] + cn[None, :] - 2 * d @ centres.T).argmin(1)
        sums = np.zeros_like(centres)
        np.add.at(sums, label, d - centres[label])
        # Intra-normalization limits repeated textures within one visual word.
        sums /= np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-12)
        return sums, np.bincount(label, minlength=count).astype(np.float32)

    with ThreadPoolExecutor(8) as pool_exec:
        result = list(pool_exec.map(aggregate, features))
    vlad = np.stack([x[0].ravel() for x in result])
    hist = np.stack([x[1] for x in result])
    del result
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
    vlad /= np.maximum(np.linalg.norm(vlad, axis=1, keepdims=True), 1e-12)
    idf = np.log((len(features) + 1) / (np.count_nonzero(hist, axis=0) + 1)) + 0.3
    hist = np.sqrt(hist) * idf
    hist /= np.maximum(np.linalg.norm(hist, axis=1, keepdims=True), 1e-12)
    every = int(os.environ.get("RETRIEVE_EVERY", "2"))
    queries = np.arange(0, len(features), 2 * every)
    # Search both camera views. Gravity is reliable, while yaw may have drifted.
    queries = np.sort(np.r_[queries, queries + 1])
    queries = queries[queries < len(features)]
    with threadpool_limits(limits=6, user_api="blas"):
        sim = vlad[queries] @ vlad.T
        sim += 0.1 * (hist[queries] @ hist.T)
    rotations = np.array([rot[k // 2] @ ext[k % 2][0] for k in range(len(features))])
    pairs = set()
    scores = []
    tops = int(os.environ.get("RETRIEVE_TOP", "3"))
    for row, k in enumerate(queries):
        score = sim[row]
        good = ts[k // 2] - ts[np.arange(len(features)) // 2] > 30e9
        if os.environ.get("RETRIEVE_ORIENT", "1") == "1":
            angle = np.arccos(
                np.clip((np.einsum("nij,ij->n", rotations, rotations[k]) - 1) / 2, -1, 1)
            )
            good &= angle < 1.5
        score[~good] = -1
        chosen = []
        for l in np.argsort(score)[::-1]:
            if score[l] < float(os.environ.get("RETRIEVE_MIN", ".12")):
                break
            if any(abs(ts[l // 2] - ts[p // 2]) < 10e9 for p in chosen):
                continue
            chosen.append(l)
            pairs.add((int(l), int(k), True))
            scores.append(float(score[l]))
            if len(chosen) >= tops:
                break
    print(
        "VLAD retrieval",
        len(pairs),
        "pairs; median",
        np.median(scores) if scores else 0,
        "seconds",
        time.monotonic() - start,
        flush=True,
    )
    return pairs
