"""Summarize the two physical effects SLAM Dunk estimates, per recording (full config).

Focal drift (px and per mille, min..max over 5 s chunks), focal-vs-temperature slope,
stereo rotation/baseline excursion, and the gyro matrix G (scale ppm, misalignment mrad).

    python repro/physics_summary.py RUNS_DIR PREPARED_DIR OUT.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def summarize(run: Path, prep: Path) -> dict | None:
    work = run / "full" / "work"
    if not (work / "ba.txt.focals").is_file() or not (run / "full" / "trajectory.txt").is_file():
        return None
    foc = np.loadtxt(work / "ba.txt.focals", ndmin=2)
    diag = np.loadtxt(work / "ba.txt.diag", ndmin=2)
    G = np.loadtxt(work / "ba.txt.gyromat").reshape(3, 3)
    kf = np.loadtxt(work / "ba.txt", ndmin=2)
    t0 = kf[0, 0] * 1e-9
    chunk_t = t0 + 5.0 * (np.arange(len(foc)) + 0.5)
    meta = np.loadtxt(prep / "cam0/meta.csv", delimiter=",", ndmin=2)
    temp = np.interp(chunk_t, meta[:, 0] * 1e-9, meta[:, 3])
    f = foc[:, [0, 2]].mean(1)  # mean fx of both cameras
    slope = float(np.polyfit(temp, f, 1)[0]) if np.ptp(temp) > 1 else float("nan")
    q = diag[:, [8, 9, 10, 7]]
    rot = (Rotation.from_quat(q[:1]).inv() * Rotation.from_quat(q)).magnitude() * 1e3
    base = np.linalg.norm(diag[:, 11:14], axis=1) * 1e3
    off = G - np.diag(np.diag(G))
    return dict(
        chunks=len(foc),
        focal_range_px=float(np.ptp(f)),
        focal_range_permille=float(np.ptp(f) / f.mean() * 1e3),
        temp_range_c=float(np.ptp(temp)),
        focal_slope_px_per_c=slope,
        stereo_rot_max_mrad=float(rot.max()),
        baseline_range_mm=float(np.ptp(base)),
        gyro_scale_ppm=(np.diag(G) * 1e6).round(0).tolist(),
        gyro_offdiag_max_mrad=float(np.abs(off).max() * 1e3),
    )


def main() -> None:
    runs, prep, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rows = {}
    for run in sorted(p for p in runs.iterdir() if p.is_dir()):
        s = summarize(run, prep / run.name)
        if s:
            rows[run.name] = s
    md = ["| recording | chunks | focal range px (‰) | temp range °C | focal slope px/°C | stereo rot max mrad | baseline range mm | gyro scale ppm (x,y,z) | gyro misalign max mrad |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, s in rows.items():
        md.append(f"| {k} | {s['chunks']} | {s['focal_range_px']:.2f} ({s['focal_range_permille']:.1f}) | {s['temp_range_c']:.1f} | "
                  f"{s['focal_slope_px_per_c']:.3f} | {s['stereo_rot_max_mrad']:.2f} | {s['baseline_range_mm']:.2f} | "
                  f"{', '.join(f'{v:+.0f}' for v in s['gyro_scale_ppm'])} | {s['gyro_offdiag_max_mrad']:.2f} |")
    out.write_text("\n".join(md) + "\n")
    hdr = ["recording", "5 s chunks", "focal range px (‰)", "temp range °C", "focal slope px/°C",
           "stereo rotation max mrad", "baseline range mm", "gyro scale ppm x, y, z", "gyro misalignment max mrad"]
    h = ['<h3>What the full model measured, per recording</h3>',
         '<p class="sub">From the full config\'s saved states (<code>ba.txt.focals</code>, <code>ba.txt.diag</code>, '
         '<code>ba.txt.gyromat</code>). Focal = mean f<sub>x</sub> of both SLAM cameras; slope = least-squares fit of focal '
         'against camera-record temperature. Paper: ≈0.03 px/°C, 1–6 mrad stereo rotation.</p>',
         '<div class="tscroll"><table class="repro"><thead><tr>' + "".join(f"<th>{x}</th>" for x in hdr) + "</tr></thead><tbody>"]
    for k, s in rows.items():
        h.append(f"<tr><th>{k}</th><td>{s['chunks']}</td><td>{s['focal_range_px']:.2f} ({s['focal_range_permille']:.1f})</td>"
                 f"<td>{s['temp_range_c']:.1f}</td><td>{s['focal_slope_px_per_c']:.3f}</td><td>{s['stereo_rot_max_mrad']:.2f}</td>"
                 f"<td>{s['baseline_range_mm']:.2f}</td><td>{', '.join(f'{v:+.0f}' for v in s['gyro_scale_ppm'])}</td>"
                 f"<td>{s['gyro_offdiag_max_mrad']:.2f}</td></tr>")
    h.append("</tbody></table></div>")
    out.with_suffix(".html").write_text("\n".join(h) + "\n")
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
