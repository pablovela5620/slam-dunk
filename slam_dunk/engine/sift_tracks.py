from trajectory_io import load_poses
from feature_cache import load_features
from image_features import extract_image, root_descriptors_inplace
import os, sys, json, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2, pycolmap as pc
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

cv2.setNumThreads(1)
src, work = map(Path, sys.argv[1:3])
start = time.monotonic()
cal = json.loads((src / "calibration.json").read_text())
a = load_poses(work / "vio.txt")
step = float(os.environ.get("BA_STEP", ".6"))
nfeat = int(os.environ.get("SIFT_FEATURES", "700"))
nth = int(os.environ.get("SIFT_THREADS", "8"))
sel = [0]
for i in range(1, len(a)):
    if a[i, 0] - a[sel[-1], 0] >= step * 1e9 - 0.01:
        sel.append(i)
if sel[-1] != len(a) - 1:
    sel.append(len(a) - 1)
sel = np.array(sel)
aa = a[sel]
ts = aa[:, 0].astype(np.int64)
N = len(ts)
rot = R.from_quat(aa[:, 4:]).as_matrix()
pp = aa[:, 1:4]
rows = [x.split() for x in (work / "frames.txt").read_text().splitlines()]
fn = {int(x[0]): x[1:] for x in rows}
cams = []
ext = []
for c in range(2):
    j = cal[f"cam{c}"]
    cams.append(
        pc.Camera(
            model=j["model"],
            width=j["resolution"]["width"],
            height=j["resolution"]["height"],
            params=j["params"],
        )
    )
    ext.append((R.from_quat(j["T_b_s"]["qvec"]).as_matrix(), np.array(j["T_b_s"]["tvec"])))
cache_hits = np.zeros(N * 2, dtype=bool)
feature_scales = [None] * (N * 2)
local = threading.local()


def extract(k):
    i, c = divmod(k, 2)
    cached = load_features(work / "features" / f"{int(ts[i])}_{c}.bin", with_scales=True)
    if cached is not None:
        cache_hits[k] = True
        feature_scales[k] = cached[3]
        return cached[:3]
    result = extract_image(
        fn[int(ts[i])][c],
        cams[c],
        local,
        nfeat,
        float(os.environ.get("SIFT_CONTRAST", ".025")),
        os.environ.get("PRECISE_UPSCALE", "0") == "1",
        os.environ.get("SIFT_DARK_ADAPT", "1") == "1",
    )
    feature_scales[k] = local.feature_scales.copy()
    return result


with ThreadPoolExecutor(nth) as pool:
    features = list(pool.map(extract, range(N * 2)))
if os.environ.get("ROOT_SIFT", "1") == "1":
    power = float(os.environ.get("ROOT_POWER", ".5"))
    if power == 0.5:
        features = root_descriptors_inplace(features)
    else:
        features = [(xy, d**power, ray) for xy, d, ray in features]
        features = [
            (xy, d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-10), ray)
            for xy, d, ray in features
        ]
desc_norm = [(f[1] * f[1]).sum(axis=1) for f in features]
print("Cached", int(cache_hits.sum()), "of", N * 2, "images", flush=True)
print(
    "Extracted",
    N * 2,
    "SIFT images",
    sum(len(x[0]) for x in features),
    "features in",
    time.monotonic() - start,
    flush=True,
)
np.savez(work / "sift_features.npz", sel=sel, counts=np.array([len(x[0]) for x in features]))
# Pose-guided temporal and stereo candidates; revisit candidates are verified geometrically.
pairs = set()
for i in range(N):
    for c in range(2):
        for d in [int(x) for x in os.environ.get("MATCH_LAGS", "1,2,4,8,16,32").split(",")]:
            if i + d < N:
                pairs.add((2 * i + c, 2 * (i + d) + c, False))
    pairs.add((2 * i, 2 * i + 1, False))
    # The second camera is also observed in neighboring stereo instants.
    if i + 1 < N:
        pairs.add((2 * i, 2 * (i + 1) + 1, False))
        pairs.add((2 * i + 1, 2 * (i + 1), False))
if os.environ.get("LOOPS", "1") != "0":
    tree = cKDTree(pp)
    for i in range(0, N, 2):
        near = tree.query_ball_point(pp[i], float(os.environ.get("LOOP_RADIUS", "25")))
        near = [j for j in near if j < i and ts[i] - ts[j] > 30e9]
        near.sort(key=lambda j: np.linalg.norm(pp[i] - pp[j]))
        chosen = []
        for j in near:
            if any(abs(ts[j] - ts[k]) < 8e9 for k in chosen):
                continue
            chosen.append(j)
            for c in range(2):
                for c2 in range(2):
                    r1 = rot[i] @ ext[c][0]
                    r2 = rot[j] @ ext[c2][0]
                    ang = np.arccos(np.clip((np.trace(r1.T @ r2) - 1) / 2, -1, 1))
                    if ang < 0.8:
                        pairs.add((2 * j + c2, 2 * i + c, True))
            if len(chosen) >= 2:
                break
pairs = sorted(pairs)
print("Pairs", len(pairs), "loops", sum(x[2] for x in pairs), flush=True)
camrot = []
campos = []
for i in range(N):
    for c in range(2):
        camrot.append(rot[i] @ ext[c][0])
        campos.append(pp[i] + rot[i] @ ext[c][1])
camrot = np.array(camrot)
campos = np.array(campos)
from temporal_cache import load_temporal_cache

temporal_cache = load_temporal_cache(work, ts)
print("Cached temporal pairs", len(temporal_cache), flush=True)
deadline_skips = []


def match(pair):
    k, l, loop = pair
    if not loop and (k, l) in temporal_cache:
        return (k, l, loop, temporal_cache[(k, l)])
    if time.monotonic() - start > float(os.environ.get("SIFT_BUDGET", "10000")) - 8:
        deadline_skips.append((k, l, loop))
        return (k, l, loop, np.empty((0, 2), int))
    xy1, d1, r1 = features[k]
    xy2, d2, r2 = features[l]
    if min(len(d1), len(d2)) < 15:
        return (k, l, loop, np.empty((0, 2), int))
    distance = -2 * (d1 @ d2.T)
    distance += desc_norm[k][:, None]
    distance += desc_norm[l][None, :]
    back = np.argmin(distance, axis=0)
    win = np.argmin(distance, axis=1)
    ii = np.arange(len(d1))
    best = distance[ii, win].copy()
    distance[ii, win] = np.inf
    second = distance.min(axis=1)
    valid = (
        (best < float(os.environ.get("MATCH_RATIO", ".78")) ** 2 * second)
        & (back[win] == ii)
        & (best < float(os.environ.get("MATCH_MAX_DISTANCE", ".7")) ** 2)
    )
    ix = np.c_[ii[valid], win[valid]].astype(np.int32)
    if len(ix) < 12:
        return (k, l, loop, np.empty((0, 2), int))
    if loop or os.environ.get("VERIFY_ALL", "1") == "1":
        p1 = r1[ix[:, 0], :2] / r1[ix[:, 0], 2:]
        p2 = r2[ix[:, 1], :2] / r2[ix[:, 1], 2:]
        try:
            cv2.setRNGSeed(int((k * 73856093 ^ l * 19349663) & 0x7FFFFFFF))
            E, mask = cv2.findEssentialMat(
                p1,
                p2,
                np.eye(3),
                method=cv2.RANSAC,
                prob=0.999,
                threshold=float(os.environ.get("ESS_THRESHOLD", ".004")),
                maxIters=200,
            )
            if E is None or mask is None:
                return (k, l, loop, np.empty((0, 2), int))
            ix = ix[mask[:, 0] > 0]
        except cv2.error:
            return (k, l, loop, np.empty((0, 2), int))
        if len(ix) < (22 if loop else 15):
            return (k, l, loop, np.empty((0, 2), int))
    else:
        # The VIO only supplies a loose epipolar gate, allowing calibration refinement later.
        rel = camrot[l].T @ camrot[k]
        tt = camrot[l].T @ (campos[k] - campos[l])
        tn = np.linalg.norm(tt)
        if tn > 0.015:
            u = r1[ix[:, 0]] @ rel.T
            v = r2[ix[:, 1]]
            normal = np.cross(tt, u)
            err = np.abs(np.einsum("ni,ni->n", normal, v)) / np.maximum(
                np.linalg.norm(normal, axis=1), 1e-9
            )
            ix = ix[err < float(os.environ.get("EPI_GATE", ".025"))]
        if len(ix) < 10:
            ix = np.empty((0, 2), int)
    return (k, l, loop, ix)


# Process short tracks and verified revisits first, then restore the original
# ordering so a completed pass produces the same landmark components.
def priority(pair):
    k, l, loop = pair
    lag = abs(k // 2 - l // 2)
    return (0 if lag <= 2 else (1 if loop else 2), lag, k, l)


with ThreadPoolExecutor(nth) as pool:
    matches = list(pool.map(match, sorted(pairs, key=priority)))
matches.sort(key=lambda x: x[:3])
print(
    "Matched",
    sum(len(x[3]) for x in matches),
    "links;",
    sum(len(x[3]) for x in matches if x[2]),
    "loop links in",
    time.monotonic() - start,
    flush=True,
)
matching_health = {
    "pairs": len(pairs),
    "deadline_skipped_pairs": len(deadline_skips),
    "links": sum(len(x[3]) for x in matches),
    "seconds": time.monotonic() - start,
    "cache_images": int(cache_hits.sum()),
    "images": N * 2,
}
(work / "sift-health.json").write_text(json.dumps(matching_health, indent=2) + "\n")
print("MATCHING_HEALTH", json.dumps(matching_health, sort_keys=True), flush=True)
# Build tracks with a native union-find while preserving deterministic link order.
import ctypes
from pathlib import Path as _Path

off = np.r_[0, np.cumsum([len(x[0]) for x in features])].astype(np.int64)
parent = np.empty(off[-1], dtype=np.int32)
edges = (
    np.concatenate(
        [
            np.ascontiguousarray(ix + np.array([off[k], off[l]]), dtype=np.int32)
            for k, l, loop, ix in matches
            if len(ix)
        ]
    )
    if any(len(x[3]) for x in matches)
    else np.empty((0, 2), np.int32)
)
track_order = os.environ.get("TRACK_ORDER", "original")
if track_order != "original" or os.environ.get("SAVE_LINKS"):
    weights = (
        np.concatenate(
            [
                np.sum((features[k][1][ix[:, 0]] - features[l][1][ix[:, 1]]) ** 2, axis=1)
                for k, l, loop, ix in matches
                if len(ix)
            ]
        )
        if len(edges)
        else np.empty(0, np.float32)
    )
    lags = np.repeat(
        np.array([abs(k // 2 - l // 2) for k, l, loop, ix in matches]),
        [len(ix) for k, l, loop, ix in matches],
    )
    is_loop = np.repeat(
        np.array([loop for k, l, loop, ix in matches]), [len(ix) for k, l, loop, ix in matches]
    )
    if os.environ.get("SAVE_LINKS"):
        np.savez(
            work / "links.npz",
            off=off,
            edges=edges,
            weights=weights,
            lags=lags,
            is_loop=is_loop,
            xy=np.concatenate([x[0] for x in features]),
            ts=ts,
            sel=sel,
        )
    if track_order == "quality":
        edges = np.ascontiguousarray(edges[np.argsort(weights, kind="stable")])
    elif track_order == "local":
        edges = np.ascontiguousarray(edges[np.lexsort((weights, lags, is_loop))])
lib = ctypes.CDLL(str(_Path(__file__).resolve().parent / "native/build/libtrack_union.so"))
fun = lib.union_tracks
fun.restype = ctypes.c_int64
fun.argtypes = [
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]
rejected = fun(
    len(parent),
    len(features),
    off.ctypes.data,
    len(edges),
    edges.ctypes.data,
    parent.ctypes.data,
    int(os.environ.get("TRACK_CONSISTENCY", "1") == "1"),
)
print("Conflicting links rejected", rejected, flush=True)
cnt = np.bincount(parent, minlength=len(parent))
minimum = int(os.environ.get("SIFT_MIN_TRACK", "3"))
records = []
scales = []
for k, (xy, desc, ray) in enumerate(features):
    i, c = divmod(k, 2)
    ids = parent[off[k] : off[k + 1]]
    valid = cnt[ids] >= minimum
    if valid.any():
        records.append(
            np.c_[
                np.full(valid.sum(), ts[i]),
                np.full(valid.sum(), c),
                ids[valid],
                xy[valid],
                np.full(valid.sum(), sel[i]),
            ]
        )
        scales.append(feature_scales[k][valid])
tracks = np.concatenate(records) if records else np.zeros((0, 6))
tracks.astype(np.float64).tofile(work / "tracks_ba.bin.tmp")
os.replace(work / "tracks_ba.bin.tmp", work / "tracks_ba.bin")
(
    np.concatenate(scales).astype(np.float32).tofile(work / "track_scales.bin.tmp")
    if scales
    else np.empty(0, np.float32).tofile(work / "track_scales.bin.tmp")
)
os.replace(work / "track_scales.bin.tmp", work / "track_scales.bin")
print("Wrote", len(tracks), "observations in", time.monotonic() - start, flush=True)

if os.environ.get("SAVE_MATCHES"):
    import pickle

    with (work / "matches.pkl").open("wb") as f:
        pickle.dump(
            dict(
                features=features,
                matches=matches,
                camrot=camrot,
                campos=campos,
                aa=aa,
                sel=sel,
                base=a,
                ext=ext,
            ),
            f,
            protocol=5,
        )
if os.environ.get("POSE_GRAPH", "1") == "1":
    from loop_graph import close_loops

    a = close_loops(features, matches, camrot, campos, aa, sel, a, work, ext)
