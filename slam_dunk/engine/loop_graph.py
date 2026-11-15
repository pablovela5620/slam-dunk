import os, time
from pathlib import Path
import numpy as np, cv2
from scipy.spatial.transform import Rotation as R, Slerp


def close_loops(features, matches, camrot, campos, aa, sel, base, work, ext):
    import gtsam as g

    started = time.monotonic()
    N = len(aa)
    world = [np.full((len(x[0]), 3), np.nan) for x in features]
    quality = [np.zeros(len(x[0])) for x in features]
    for k, l, loop, ix in matches:
        if loop or not len(ix):
            continue
        d1 = features[k][2][ix[:, 0]] @ camrot[k].T
        d2 = features[l][2][ix[:, 1]] @ camrot[l].T
        o1 = campos[k]
        o2 = campos[l]
        w = o1 - o2
        b = np.einsum("ni,ni->n", d1, d2)
        d = d1 @ w
        e = d2 @ w
        den = 1 - b * b
        s = (b * e - d) / np.maximum(den, 1e-9)
        t = (e - b * d) / np.maximum(den, 1e-9)
        p1 = o1 + s[:, None] * d1
        p2 = o2 + t[:, None] * d2
        xyz = (p1 + p2) / 2
        err = np.linalg.norm(p1 - p2, axis=1) / np.maximum(np.minimum(s, t), 1e-6)
        valid = (
            (den > 0.0001)
            & (s > 0.4)
            & (t > 0.4)
            & (s < 150)
            & (t < 150)
            & (err < 0.01)
            & np.isfinite(xyz).all(axis=1)
        )
        for kk, col in [(k, 0), (l, 1)]:
            fi = ix[:, col]
            ok = valid & (den > quality[kk][fi])
            world[kk][fi[ok]] = xyz[ok]
            quality[kk][fi[ok]] = den[ok]
    loops = []
    for k, l, loop, ix in matches:
        if not loop or len(ix) < 22:
            continue
        # Estimate the later camera from 3-D points triangulated solely near the earlier view.
        ok = np.isfinite(world[k][ix[:, 0]]).all(axis=1)
        if ok.sum() < 20:
            continue
        pts = world[k][ix[ok, 0]]
        rays = features[l][2][ix[ok, 1]]
        uv = rays[:, :2] / rays[:, 2:]
        try:
            cv2.setRNGSeed(int((k * 83492791 ^ l * 1299709) & 0x7FFFFFFF))
            success, rv, tv, inliers = cv2.solvePnPRansac(
                pts,
                uv,
                np.eye(3),
                None,
                iterationsCount=200,
                reprojectionError=0.008,
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not success or inliers is None or len(inliers) < max(18, int(len(pts) * 0.35)):
                continue
            ii = inliers[:, 0]
            rv, tv = cv2.solvePnPRefineLM(pts[ii], uv[ii], np.eye(3), None, rv, tv)
            rcw = cv2.Rodrigues(rv)[0]
            pos = -rcw.T @ tv[:, 0]
            rwc = rcw.T
            c = l % 2
            rb = rwc @ ext[c][0].T
            pb = pos - rb @ ext[c][1]
            # Verify pose with an angular error, including fisheye peripheral rays.
            re = (pts[ii] - pos) @ rcw.T
            re /= np.linalg.norm(re, axis=1)[:, None]
            er = 240 * np.linalg.norm(np.cross(re, rays[ii]), axis=1)
            if (
                np.median(er) > 1.4
                or np.percentile(er, 90) > 3
                or np.linalg.norm(pb - aa[l // 2, 1:4]) > 40
            ):
                continue
            rr = R.from_quat(aa[k // 2, 4:]).as_matrix()
            pp = aa[k // 2, 1:4]
            relr = rr.T @ rb
            relt = rr.T @ (pb - pp)
            loops.append((k // 2, l // 2, relr, relt, len(ii), float(np.median(er))))
        except cv2.error:
            continue
    print("PnP loop edges", len(loops), "in", time.monotonic() - started, flush=True)
    if not loops:
        return base
    poses = [g.Pose3(g.Rot3(R.from_quat(x[4:]).as_matrix()), x[1:4]) for x in aa]
    graph = g.NonlinearFactorGraph()
    vals = g.Values()
    n_odo = g.noiseModel.Diagonal.Sigmas(np.array([0.0015] * 3 + [0.04] * 3))
    n_prior = g.noiseModel.Isotropic.Sigma(6, 1e-7)
    n_g = g.noiseModel.Isotropic.Sigma(2, 0.006)
    n_loop = g.noiseModel.Robust.Create(
        g.noiseModel.mEstimator.Huber.Create(2),
        g.noiseModel.Diagonal.Sigmas(
            np.array(
                [float(os.environ.get("LOOP_ROT_NOISE", ".005"))] * 3
                + [float(os.environ.get("LOOP_TRANS_NOISE", ".15"))] * 3
            )
        ),
    )
    for i, p in enumerate(poses):
        vals.insert(i, p)
        if i:
            graph.add(g.BetweenFactorPose3(i - 1, i, poses[i - 1].between(p), n_odo))
        bgravity = R.from_quat(aa[i, 4:]).inv().apply([0, 0, 1])
        graph.add(
            g.AttitudeFactorPose3(i, g.Unit3(np.array([0.0, 0, 1.0])), n_g, g.Unit3(bgravity))
        )
    graph.add(g.PriorFactorPose3(0, poses[0], n_prior))
    for i, j, rr, tt, n, err in loops:
        graph.add(g.BetweenFactorPose3(i, j, g.Pose3(g.Rot3(rr), tt), n_loop))
    params = g.LevenbergMarquardtParams()
    params.setMaxIterations(30)
    params.setRelativeErrorTol(1e-5)
    result = g.LevenbergMarquardtOptimizer(graph, vals, params).optimize()
    adj = np.array(
        [
            [
                *result.atPose3(i).translation(),
                *R.from_matrix(result.atPose3(i).rotation().matrix()).as_quat(),
            ]
            for i in range(N)
        ]
    )
    print(
        "Pose graph error",
        graph.error(vals),
        "->",
        graph.error(result),
        "max correction",
        np.max(np.linalg.norm(adj[:, :3] - aa[:, 1:4], axis=1)),
        flush=True,
    )
    t0 = base[0, 0]
    tt = (base[:, 0] - t0) * 1e-9
    st = (aa[:, 0] - t0) * 1e-9
    dpos = adj[:, :3] - aa[:, 1:4]
    pos = base[:, 1:4] + np.column_stack([np.interp(tt, st, dpos[:, j]) for j in range(3)])
    drot = R.from_quat(adj[:, 3:]) * R.from_quat(aa[:, 4:]).inv()
    q = (Slerp(st, drot)(np.clip(tt, st[0], st[-1])) * R.from_quat(base[:, 4:])).as_quat()
    out = np.c_[base[:, 0], pos, q]
    np.savetxt(work / "vio_before_loops.txt", base, fmt=["%d"] + ["%.12f"] * 7)
    checkpoint = work / "vio_loops.tmp"
    np.savetxt(checkpoint, out, fmt=["%d"] + ["%.12f"] * 7)
    os.replace(checkpoint, work / "vio.txt")
    return out
