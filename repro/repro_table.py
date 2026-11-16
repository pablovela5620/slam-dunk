"""Build the ours-vs-paper CP table (markdown + HTML fragment + JSON) from run scores.

Paper per-recording values are digitized from Fig. 4c (baseline/full, ±0.1 CP);
camera_only/gyro_only exist only as 13-record means (Table 1).

    python repro/repro_table.py RUNS_DIR OUT_STEM
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COHORT = ["R_11_5cp", "R_12_10cp", "R_13_15cp", "sequence_1_19", "sequence_1_20", "sequence_2_11",
          "sequence_2_12", "sequence_3_17", "sequence_3_18", "sequence_4_10", "sequence_4_11",
          "sequence_5_11", "sequence_5_12"]
PAPER = {  # (baseline, full) from Fig. 4c
    "sequence_3_17": (52.6, 87.6), "sequence_4_11": (60.9, 93.3), "sequence_2_12": (43.0, 70.3),
    "sequence_3_18": (63.1, 88.8), "sequence_1_19": (81.1, 94.3), "sequence_2_11": (83.7, 95.5),
    "R_11_5cp": (86.2, 97.4), "sequence_4_10": (81.4, 91.1), "sequence_5_11": (81.5, 87.4),
    "R_13_15cp": (83.0, 88.3), "R_12_10cp": (81.1, 82.1), "sequence_5_12": (7.0, 6.0),
    "sequence_1_20": (94.1, 89.4),
}
PAPER_MEAN = {"baseline": 69.14, "gyro_only": 71.47, "camera_only": 77.13, "full": 82.42}
CONFIGS = ["vio", "baseline", "gyro_only", "camera_only", "full"]


def score(run: Path, cfg: str) -> float | None:
    path = run / ("cp-vio.json" if cfg == "vio" else f"score-{cfg}.json")
    try:
        return float(json.loads(path.read_text())["cp_score"])
    except (OSError, KeyError, ValueError, TypeError):
        return None


def ate(run: Path, cfg: str) -> float | None:
    try:
        v = json.loads((run / f"score-{cfg}.json").read_text()).get("ate_rmse_m")
        return None if v is None else float(v)
    except (OSError, ValueError):
        return None


def fmt3(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.1f}"


def main() -> None:
    runs, stem = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    for seq in COHORT:
        ours = {c: score(runs / seq, c) for c in CONFIGS}
        pb, pf = PAPER[seq]
        rows.append(dict(seq=seq, ours=ours, paper_baseline=pb, paper_full=pf,
                         ate={c: ate(runs / seq, c) for c in ("baseline", "full")}))
    done = [r for r in rows if r["ours"]["full"] is not None and r["ours"]["baseline"] is not None]
    n = len(done)
    mean = {c: (sum(r["ours"][c] for r in done) / n if n and all(r["ours"][c] is not None for r in done) else None)
            for c in CONFIGS}
    paper_sub = (sum(r["paper_baseline"] for r in done) / n, sum(r["paper_full"] for r in done) / n) if n else (None, None)

    ate_rows = [r for r in done if r["ate"]["full"] is not None and r["ate"]["baseline"] is not None]
    ate_mean = {c: (sum(r["ate"][c] for r in ate_rows) / len(ate_rows) if ate_rows else None) for c in ("baseline", "full")}
    md = ["| recording | VIO seed | baseline (ours) | baseline (paper) | Δ | gyro_only | camera_only | full (ours) | full (paper) | Δ |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        o = r["ours"]
        db = None if o["baseline"] is None else o["baseline"] - r["paper_baseline"]
        df = None if o["full"] is None else o["full"] - r["paper_full"]
        md.append(f"| {r['seq']} | {fmt(o['vio'])} | {fmt(o['baseline'])} | {r['paper_baseline']:.1f} | "
                  f"{'—' if db is None else f'{db:+.1f}'} | {fmt(o['gyro_only'])} | {fmt(o['camera_only'])} | "
                  f"{fmt(o['full'])} | {r['paper_full']:.1f} | {'—' if df is None else f'{df:+.1f}'} |")
    md.append(f"| **mean over {n} done** | {fmt(mean['vio'])} | **{fmt(mean['baseline'])}** | {fmt(paper_sub[0])} | | "
              f"{fmt(mean['gyro_only'])} | {fmt(mean['camera_only'])} | **{fmt(mean['full'])}** | {fmt(paper_sub[1])} | |")
    md.append(f"| paper Table 1 (13) | | | {PAPER_MEAN['baseline']} | | {PAPER_MEAN['gyro_only']} | "
              f"{PAPER_MEAN['camera_only']} | | {PAPER_MEAN['full']} | |")
    stem.with_suffix(".md").write_text("\n".join(md) + "\n")

    def cell(x: float | None, paper: float | None = None) -> str:
        if x is None:
            return "<td class=na>—</td>"
        cls = ""
        if paper is not None:
            cls = " class=ok" if abs(x - paper) <= 3 else (" class=warn" if abs(x - paper) <= 8 else " class=bad")
        return f"<td{cls}>{x:.1f}</td>"

    html = ['<div class="tscroll"><table class="repro"><thead><tr><th>recording</th><th>VIO seed</th><th>baseline<br>ours</th>'
            '<th>baseline<br>paper</th><th>gyro_only</th><th>camera_only</th><th>full<br>ours</th><th>full<br>paper</th>'
            '<th>ATE m<br>baseline</th><th>ATE m<br>full</th></tr></thead><tbody>']
    for r in rows:
        o = r["ours"]
        html.append(f"<tr><th>{r['seq']}</th>{cell(o['vio'])}{cell(o['baseline'], r['paper_baseline'])}"
                    f"<td>{r['paper_baseline']:.1f}</td>{cell(o['gyro_only'])}{cell(o['camera_only'])}"
                    f"{cell(o['full'], r['paper_full'])}<td>{r['paper_full']:.1f}</td>"
                    f"<td>{fmt3(r['ate']['baseline'])}</td><td>{fmt3(r['ate']['full'])}</td></tr>")
    html.append(f"<tr class=mean><th>mean ({n} done)</th>{cell(mean['vio'])}{cell(mean['baseline'], paper_sub[0])}"
                f"{cell(paper_sub[0])}{cell(mean['gyro_only'])}{cell(mean['camera_only'])}"
                f"{cell(mean['full'], paper_sub[1])}{cell(paper_sub[1])}"
                f"<td>{fmt3(ate_mean['baseline'])}</td><td>{fmt3(ate_mean['full'])}</td></tr>")
    html.append(f"<tr class=mean><th>paper Table 1 (13)</th><td></td><td></td><td>{PAPER_MEAN['baseline']}</td>"
                f"<td>{PAPER_MEAN['gyro_only']}</td><td>{PAPER_MEAN['camera_only']}</td><td></td><td>{PAPER_MEAN['full']}</td>"
                f"<td>1.018 (21)</td><td>0.430 (21)</td></tr>")
    html.append("</tbody></table></div>")
    stem.with_suffix(".html").write_text("\n".join(html) + "\n")
    stem.with_suffix(".json").write_text(json.dumps(dict(rows=rows, n_done=n, mean=mean, paper_subset_mean=paper_sub,
                                                         paper_table1=PAPER_MEAN), indent=2) + "\n")
    four = [r for r in rows if all(r["ours"][c] is not None for c in ("baseline", "gyro_only", "camera_only", "full"))]
    abl = ['<div class="tscroll"><table class="repro"><thead><tr><th>config</th><th>camera path</th><th>gyro G</th>'
           f'<th>ours, mean CP ({len(four)} rec.)</th><th>paper, mean CP (13)</th></tr></thead><tbody>']
    for c, cam, gyr in (("baseline", "static", "off"), ("gyro_only", "static", "on"),
                        ("camera_only", "5 s chunks", "off"), ("full", "5 s chunks", "on")):
        m = sum(r["ours"][c] for r in four) / len(four) if four else None
        abl.append(f"<tr><th>{c}</th><td>{cam}</td><td>{gyr}</td>{cell(m)}<td>{PAPER_MEAN[c]}</td></tr>")
    abl.append("</tbody></table></div>")
    if len(four) == len(COHORT):
        abl.append('<p class="sub">All 13 CP recordings, the same cohort as the paper\'s Table 1; one frontend packet per recording '
                   "shared by the four configs.</p>")
    else:
        abl.append(f'<p class="sub">Recordings with all four configs: {", ".join(r["seq"] for r in four) or "none yet"}. '
                   "The paper's means are over all 13; ours are over this subset, so compare the ordering and gaps, not the absolute level.</p>")
    # Fig. 6 removal ablations: CP lost vs full on recordings with that config done.
    removal = {"no_accelerometer": ("accelerometer intrinsics", 3.3), "no_clock": ("clock offsets", 2.3),
               "no_thermal": ("thermal terms", 0.5), "no_auxiliary": ("all three auxiliary groups", 5.6),
               "no_final_refinement": ("final refinement pass", None)}
    rem_rows = []
    for c, (label, paper_loss) in removal.items():
        both = [(score(runs / r["seq"], c), r["ours"]["full"]) for r in rows]
        both = [(a, f) for a, f in both if a is not None and f is not None]
        if both:
            loss = sum(f - a for a, f in both) / len(both)
            rem_rows.append(f"<tr><th>{c}</th><td>{label}</td><td>{len(both)}</td>{cell(loss)}"
                            f"<td>{'—' if paper_loss is None else paper_loss}</td></tr>")
    if rem_rows:
        abl.append('<h3>Removing one term from full (paper Fig. 6)</h3><div class="tscroll"><table class="repro"><thead><tr>'
                   '<th>config</th><th>removed</th><th>recordings</th><th>ours: mean CP lost vs full</th>'
                   '<th>paper: mean CP lost (13)</th></tr></thead><tbody>' + "".join(rem_rows) + "</tbody></table></div>")
    stem.with_name("ablation.html").write_text("\n".join(abl) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
