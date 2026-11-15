"""Recording-local, atomic checkpoints of verified temporal correspondences."""

import os, json
from image_features import root_descriptors_inplace
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2


def signature():
    return json.dumps(
        {
            k: os.environ.get(k, d)
            for k, d in [
                ("ROOT_SIFT", "1"),
                ("ROOT_POWER", ".5"),
                ("MATCH_RATIO", ".78"),
                ("MATCH_MAX_DISTANCE", ".7"),
                ("ESS_THRESHOLD", ".004"),
            ]
        },
        sort_keys=True,
    )


def build_temporal_cache(features, ts, work, threads):
    if os.environ.get("VERIFY_ALL", "1") != "1":
        return
    if os.environ.get("ROOT_SIFT", "1") == "1":
        power = float(os.environ.get("ROOT_POWER", ".5"))
        if power == 0.5:
            features = root_descriptors_inplace(features)
        else:
            features = [(xy, d**power, r) for xy, d, r in features]
            features = [
                (xy, d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-10), r)
                for xy, d, r in features
            ]
    norms = [(f[1] * f[1]).sum(axis=1) for f in features]
    pairs = set()
    N = len(ts)
    for i in range(N):
        for c in range(2):
            for lag in [int(x) for x in os.environ.get("MATCH_LAGS", "1,2,4,8,16,32").split(",")]:
                if i + lag < N:
                    pairs.add((2 * i + c, 2 * (i + lag) + c))
        pairs.add((2 * i, 2 * i + 1))
        if i + 1 < N:
            pairs.add((2 * i, 2 * (i + 1) + 1))
            pairs.add((2 * i + 1, 2 * (i + 1)))

    def priority(p):
        k, l = p
        lag = abs(k // 2 - l // 2)
        return (0 if lag <= 2 else 2, lag, k, l)

    ratio = float(os.environ.get("MATCH_RATIO", ".78")) ** 2
    cutoff = float(os.environ.get("MATCH_MAX_DISTANCE", ".7")) ** 2
    threshold = float(os.environ.get("ESS_THRESHOLD", ".004"))

    def match(p):
        k, l = p
        _, d1, r1 = features[k]
        _, d2, r2 = features[l]
        empty = np.empty((0, 2), np.int32)
        if min(len(d1), len(d2)) < 15:
            return (k, l, empty)
        distance = -2 * (d1 @ d2.T)
        distance += norms[k][:, None]
        distance += norms[l][None, :]
        back = np.argmin(distance, axis=0)
        win = np.argmin(distance, axis=1)
        ii = np.arange(len(d1))
        best = distance[ii, win].copy()
        distance[ii, win] = np.inf
        second = distance.min(axis=1)
        valid = (best < ratio * second) & (back[win] == ii) & (best < cutoff)
        ix = np.c_[ii[valid], win[valid]].astype(np.int32)
        if len(ix) < 12:
            return (k, l, empty)
        p1 = r1[ix[:, 0], :2] / r1[ix[:, 0], 2:]
        p2 = r2[ix[:, 1], :2] / r2[ix[:, 1], 2:]
        try:
            cv2.setRNGSeed(int((k * 73856093 ^ l * 19349663) & 0x7FFFFFFF))
            E, mask = cv2.findEssentialMat(
                p1, p2, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=threshold, maxIters=200
            )
            if E is None or mask is None:
                return (k, l, empty)
            ix = ix[mask[:, 0] > 0]
        except cv2.error:
            return (k, l, empty)
        return (k, l, ix if len(ix) >= 15 else empty)

    folder = Path(work) / "temporal"
    folder.mkdir(exist_ok=True)
    pending = []
    chunks = 0
    completed = 0

    def checkpoint():
        nonlocal chunks, completed
        lengths = np.array([len(x[2]) for x in pending], np.int64)
        tmp = folder / f"{chunks:04d}.tmp"
        target = folder / f"{chunks:04d}.npz"
        with tmp.open("wb") as f:
            np.savez(
                f,
                pairs=np.array([[x[0], x[1]] for x in pending], np.int32),
                offsets=np.r_[0, np.cumsum(lengths)],
                indices=np.concatenate([x[2] for x in pending]),
                ts=ts,
                config=np.array(signature()),
            )
        os.replace(tmp, target)
        completed += len(pending)
        chunks += 1
        pending.clear()

    with ThreadPoolExecutor(threads) as pool:
        for result in pool.map(match, sorted(pairs, key=priority)):
            pending.append(result)
            if len(pending) >= 512:
                checkpoint()
        if pending:
            checkpoint()
    print("Prefetched", completed, "verified temporal pairs", flush=True)


def load_temporal_cache(work, ts):
    if os.environ.get("VERIFY_ALL", "1") != "1":
        return {}
    cache = {}
    for path in sorted((Path(work) / "temporal").glob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as z:
                if str(z["config"]) != signature():
                    continue
                ct = z["ts"]
                pairs = z["pairs"]
                off = z["offsets"]
                ix = z["indices"]
                for j, (k, l) in enumerate(pairs):
                    if (
                        max(k, l) // 2 >= len(ts)
                        or ct[k // 2] != ts[k // 2]
                        or ct[l // 2] != ts[l // 2]
                    ):
                        continue
                    cache[(int(k), int(l))] = ix[off[j] : off[j + 1]].copy()
        except (OSError, ValueError, KeyError):
            continue
    return cache
