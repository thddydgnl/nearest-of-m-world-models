#!/usr/bin/env python3
"""Appendix figure fig_samples: decoded next-frame samples of the nine A2-div models of Table 2.

Revision r1, analysis h (main/revision/an_analyses/REPORT.md, section h). For the seed-0 A2-div checkpoints of
Table 2 (tab:a2), 16 samples per context (torch seed 123) were generated on the pod and decoded by the position
readout (main/code/imlewm/r1_samples.py); this script only draws the stored results. Rows: the first three
training contexts and the first held-out context picked by a fixed rule (in the order of check.memo_model's
context permutation with seed 0, the contexts whose K = 4 possible next positions are distinct and at least
6 px apart). Columns: the nine configurations, in the order of the table.

usage (from the repository root):
    python3 paper/build/make_fig_samples.py [--png PREVIEW.png]
writes paper/tmlr/figures/fig_samples.pdf; with --png also a 300-dpi PNG preview of the same figure.

Conventions follow paper/build/make_figures.py: drawn at the printed size (6.5 in = \\textwidth, to be included
with width=\\linewidth, i.e. at scale 1), titles 8 pt, legend and annotations 7 pt, Okabe-Ito colors, TrueType
fonts embedded (pdf.fonttype 42). Needs only numpy and matplotlib.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

SRC = "main/revision/an_analyses/raw/an.1/out/h_samples"
OUT = "paper/tmlr/figures/fig_samples.pdf"

# ------------------------------------------------------------------ style (as in paper/build/make_figures.py)
WIDTH = 6.5          # in; = \textwidth of the TMLR style
FS_LABEL = 8         # column titles, row labels
FS_TEXT = 7          # legend, corner counts
INK = "#222222"      # annotation text
plt.rcParams.update({
    "font.size": FS_TEXT, "axes.labelsize": FS_LABEL, "axes.titlesize": FS_LABEL, "legend.fontsize": FS_TEXT,
    "pdf.fonttype": 42, "figure.dpi": 150, "legend.frameon": False,
    "legend.handlelength": 1.2, "legend.handletextpad": 0.4, "legend.columnspacing": 1.6,
    "legend.borderaxespad": 0.2, "legend.borderpad": 0.2,
})
SAMPLE_C = "#D55E00"   # Okabe-Ito vermillion: decoded samples; the dark outline keeps them apart from the frame
SAMPLE_KW = dict(ls="none", mfc=SAMPLE_C, mec=INK, mew=0.35)   # in gray-level print
DOT_MS, TRI_MS = 3.0, 3.6   # marker sizes (pt) of a sample and of a sample outside the crop
REAL_C = "#0072B2"     # Okabe-Ito blue: realized mode
OTHER_C = INK          # the other three modes
FRAME_ALPHA = 0.32     # the last context frame is shown in gray levels, at this opacity over white
WIN = 38               # crop size in px (frames are 64 px); the same in every panel
UP = 8                 # integer upscaling of the crop (nearest neighbour), so that viewers do not smooth it

# ------------------------------------------------------------------ what is drawn
ROWS = [  # (row label, sample file, frame file, split)
    ("train 1", "samples.json", "context_frames.npz", "train"),
    ("train 2", "samples_ctx2.json", "context_frames_ctx2.npz", "train"),
    ("train 3", "samples_ctx3.json", "context_frames_ctx3.npz", "train"),
    ("held-out", "samples.json", "context_frames.npz", "test"),
]
COLS = [  # (key in the JSON, group header, column title)
    ("Regression", None, "Regression"),
    ("FM-50", None, "FM-50"),
    ("FM-2", None, "FM-2"),
    ("iMF", None, "iMF"),
    ("Shortcut", None, "Shortcut"),
    ("Energy score m=8", "Energy score", "$m{=}8$"),
    ("Energy score m=16", "Energy score", "$m{=}16$"),
    ("IMLE-WM m=4", "IMLE-WM", "$m{=}4$"),
    ("IMLE-WM m=16", "IMLE-WM", "$m{=}16$"),
]
# (t, realized mode) per row and the effective samples per mode, copied from REPORT.md section h; the script
# stops if the JSON or the re-computed monitor assignment disagrees with them
EXPECTED_T = [(7412, 2), (7395, 3), (39101, 3), (1855, 1)]
EXPECTED = {
    "Regression": [(0, 0, 16, 0), (0, 0, 0, 16), (0, 0, 0, 16), (0, 0, 0, 0)],
    "FM-50": [(2, 1, 3, 2), (0, 2, 3, 6), (3, 0, 5, 2), (0, 0, 5, 5)],
    "FM-2": [(1, 1, 4, 2), (1, 0, 3, 4), (0, 1, 3, 0), (0, 0, 7, 1)],
    "iMF": [(1, 0, 1, 2), (1, 1, 3, 3), (1, 0, 0, 0), (0, 1, 0, 1)],
    "Shortcut": [(0, 1, 6, 1), (1, 2, 3, 4), (0, 1, 0, 0), (0, 1, 4, 3)],
    "Energy score m=8": [(0, 0, 7, 2), (0, 0, 0, 16), (0, 0, 0, 16), (0, 0, 0, 16)],
    "Energy score m=16": [(0, 0, 16, 0), (0, 0, 0, 16), (0, 0, 0, 16), (0, 0, 0, 11)],
    "IMLE-WM m=4": [(2, 7, 6, 0), (0, 0, 0, 13), (3, 0, 0, 12), (2, 0, 0, 3)],
    "IMLE-WM m=16": [(5, 4, 3, 2), (2, 3, 5, 5), (4, 3, 1, 3), (5, 2, 2, 2)],
}


def monitor_assign(ps, bp, tau_cap=4.0):
    """As check.balance: every sample goes to its nearest mode j and counts as effective if it lies closer than
    tau_j = min(0.45 x (distance from mode j to the nearest other mode), 4 px)."""
    K = len(bp)
    dp = np.linalg.norm(ps[:, None] - bp[None], axis=-1)
    sep = (np.linalg.norm(bp[:, None] - bp[None], axis=-1) + np.eye(K) * 1e9).min(1)
    tau = np.minimum(0.45 * sep, tau_cap)
    j = dp.argmin(1)
    ok = dp[np.arange(len(ps)), j] < tau[j]
    return j, ok, tau


def crop_window(bp):
    """WIN x WIN px window centered on the mean of the K possible next positions, moved (by whole pixels) only as
    far as needed to stay inside the 64 px frame."""
    c = bp.mean(0)
    return tuple(int(np.clip(round(v - WIN / 2), 0, 64 - WIN)) for v in c)


def frame_image(frame, x0, y0):
    """Gray-level crop of a 64x64 RGB frame, lightened (opacity FRAME_ALPHA over white), upscaled by UP."""
    lum = frame[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114]) / 255.0
    g = 1.0 - FRAME_ALPHA * (1.0 - lum)
    g = np.kron(g[y0:y0 + WIN, x0:x0 + WIN], np.ones((UP, UP)))
    return np.repeat(g[..., None], 3, axis=-1)


def load_rows():
    rows = []
    for r, (label, fjson, fnpz, split) in enumerate(ROWS):
        c = json.load(open(os.path.join(SRC, fjson)))["contexts"][split]
        frames = np.load(os.path.join(SRC, fnpz))[f"{split}_frames"]   # frames t-2 .. t+1
        assert (c["t"], c["realized"]) == EXPECTED_T[r], (label, c["t"], c["realized"])
        bp = np.array(c["branch_pos"], float)
        panels = []
        for key, _, _ in COLS:
            m = c["models"][key]
            ps = np.array(m["pos"], float)
            j, ok, tau = monitor_assign(ps, bp)
            counts = np.bincount(j[ok], minlength=len(bp))
            assert list(counts) == list(m["valid_counts"]) == list(EXPECTED[key][r]), (label, key, counts)
            assert list(j) == list(m["nearest_branch"]), (label, key)
            panels.append(dict(key=key, ps=ps, ok=ok, a=int(counts[c["realized"]]), b=int(counts.sum())))
        rows.append(dict(label=label, t=c["t"], realized=c["realized"], bp=bp, tau=tau, frame=frames[2],
                         ball=np.array(c["ctx_pos"][-1]), radius=float(c["app"][6]), win=crop_window(bp),
                         panels=panels))
    return rows


def pick_corner(row, box_w=19.0, box_h=8.0):
    """Corner (of the crop) for the a/b label that covers the fewest marks over the nine panels of a row."""
    x0, y0 = row["win"]
    corners = {"upper left": (x0, y0), "upper right": (x0 + WIN - box_w, y0),
               "lower left": (x0, y0 + WIN - box_h), "lower right": (x0 + WIN - box_w, y0 + WIN - box_h)}
    pts = np.concatenate([p["ps"] for p in row["panels"]])
    best, best_score = None, None
    for name, (bx, by) in corners.items():
        inside = ((pts[:, 0] > bx - 1) & (pts[:, 0] < bx + box_w + 1) &
                  (pts[:, 1] > by - 1) & (pts[:, 1] < by + box_h + 1)).sum()
        hits = 0
        for (cx, cy), rad in [*zip(row["bp"], row["tau"]), (row["ball"], row["radius"])]:
            nx, ny = np.clip(cx, bx, bx + box_w), np.clip(cy, by, by + box_h)   # nearest point of the box
            hits += np.hypot(cx - nx, cy - ny) < rad + 0.5
        score = inside + 6 * hits
        if best_score is None or score < best_score:
            best, best_score = name, score
    return best


def draw_panel(ax, row, panel):
    x0, y0 = row["win"]
    x1, y1 = x0 + WIN, y0 + WIN
    ax.imshow(frame_image(row["frame"], x0, y0), extent=(x0, x1, y1, y0), interpolation="none", zorder=0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.4)
        s.set_color("#808080")
    # the four possible next positions, radius = the monitor's tolerance; the realized one on top
    for k, (p, rad) in enumerate(zip(row["bp"], row["tau"])):
        if k != row["realized"]:
            ax.add_patch(Circle(p, rad, fill=False, ec=OTHER_C, lw=0.6, zorder=2))
    ax.add_patch(Circle(row["bp"][row["realized"]], row["tau"][row["realized"]], fill=False, ec=REAL_C, lw=1.4,
                        zorder=3))
    # decoded samples; those outside the crop as triangles on the panel edge, pointing outward
    ps = panel["ps"]
    inside = (ps[:, 0] >= x0) & (ps[:, 0] <= x1) & (ps[:, 1] >= y0) & (ps[:, 1] <= y1)
    pin = np.unique(ps[inside], axis=0)   # identical samples (all 16 of regression) drawn once, not overdrawn
    ax.plot(pin[:, 0], pin[:, 1], marker="o", ms=DOT_MS, zorder=5, clip_on=False,  # a dot on the edge is drawn
            **SAMPLE_KW)                                                          # whole rather than cut
    inset = 1.8   # px, so that the whole triangle stays inside the panel
    for x, y in ps[~inside]:
        dx = (x0 - x) if x < x0 else (x - x1) if x > x1 else 0.0
        dy = (y0 - y) if y < y0 else (y - y1) if y > y1 else 0.0
        if dx >= dy:
            mk = "<" if x < x0 else ">"
        else:
            mk = "^" if y < y0 else "v"   # the y axis points down, markers are drawn in screen space
        ax.plot(np.clip(x, x0 + inset, x1 - inset), np.clip(y, y0 + inset, y1 - inset), marker=mk, ms=TRI_MS,
                zorder=5, clip_on=False, **SAMPLE_KW)
    # a/b: effective samples on the realized mode / effective samples on any mode
    up = row["corner"].startswith("upper")
    left = row["corner"].endswith("left")
    label = ax.text(0.035 if left else 0.965, 0.955 if up else 0.045, f"{panel['a']}/{panel['b']}",
                    transform=ax.transAxes, ha="left" if left else "right", va="top" if up else "bottom",
                    fontsize=FS_TEXT, color=INK, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))
    return int((~inside).sum()), label, ps[inside]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=None, help="also write a 300-dpi PNG preview to this path")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    rows = load_rows()
    for row in rows:
        row["corner"] = pick_corner(row)

    # layout in inches: row labels on the left, two title lines on top, legend at the bottom
    left, right, gap_x = 0.20, 0.01, 0.035
    pw = (WIDTH - left - right - (len(COLS) - 1) * gap_x) / len(COLS)          # square panels
    gap_y, gap_split, top, bottom = 0.035, 0.11, 0.34, 0.25
    height = top + 4 * pw + 2 * gap_y + gap_split + bottom
    fig = plt.figure(figsize=(WIDTH, height))
    W, H = WIDTH, height
    ys = []   # bottom edge of each row
    y = height - top
    for r in range(4):
        y -= pw
        ys.append(y)
        y -= gap_split if r == 2 else gap_y
    xs = [left + c * (pw + gap_x) for c in range(len(COLS))]

    outside, labels = {}, []
    for r, row in enumerate(rows):
        for c, panel in enumerate(row["panels"]):
            ax = fig.add_axes([xs[c] / W, ys[r] / H, pw / W, pw / H])
            n_out, label, ps_in = draw_panel(ax, row, panel)
            outside[(row["label"], panel["key"])] = n_out
            labels.append((row["label"], panel["key"], ax, label, ps_in))
        fig.text(left * 0.42 / W, (ys[r] + pw / 2) / H, row["label"], rotation=90, ha="center", va="center",
                 fontsize=FS_LABEL, color=INK)

    # column titles (baseline just above the panels) and group headers over the two-column groups
    y_title = height - top + 0.055
    for c, (_, _, title) in enumerate(COLS):
        fig.text((xs[c] + pw / 2) / W, y_title / H, title, ha="center", va="baseline", fontsize=FS_LABEL,
                 color=INK)
    y_rule = y_title + 0.135
    groups = {}
    for c, (_, grp, _) in enumerate(COLS):
        if grp:
            groups.setdefault(grp, []).append(c)
    for grp, cs in groups.items():
        xa, xb = xs[cs[0]] + 0.04, xs[cs[-1]] + pw - 0.04
        fig.add_artist(Line2D([xa / W, xb / W], [y_rule / H] * 2, lw=0.6, color=INK))
        fig.text((xa + xb) / 2 / W, (y_rule + 0.035) / H, grp, ha="center", va="baseline", fontsize=FS_LABEL,
                 color=INK)

    handles = [
        Line2D([], [], ls="none", marker="o", ms=8, mfc="none", mec=REAL_C, mew=1.4, label="realized mode"),
        Line2D([], [], ls="none", marker="o", ms=8, mfc="none", mec=OTHER_C, mew=0.6, label="other modes"),
        Line2D([], [], marker="o", ms=DOT_MS, label="decoded sample", **SAMPLE_KW),
        Line2D([], [], marker=">", ms=TRI_MS, label="sample outside the crop", **SAMPLE_KW),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=((left + (W - left - right) / 2) / W, 0.0))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out)   # no bbox cropping: the file keeps the exact 6.5 in width
    print(f"wrote {a.out} ({WIDTH:.2f} x {height:.2f} in, panel {pw:.3f} in, crop {WIN} px)")
    if a.png:
        fig.savefig(a.png, dpi=300)
        print("wrote", a.png)

    # report: a/b per panel and the samples drawn on the panel edge
    print("a/b per panel (a: effective samples on the realized mode, b: effective samples on any mode)")
    print(f"{'':18s}" + "".join(f"{row['label']:>10s}" for row in rows))
    for c, (key, _, _) in enumerate(COLS):
        print(f"{key:18s}" + "".join(f"{str(row['panels'][c]['a']) + '/' + str(row['panels'][c]['b']):>10s}"
                                     for row in rows))
    print("windows (x0, y0):", {row["label"]: row["win"] for row in rows})
    print("corners:", {row["label"]: row["corner"] for row in rows})
    print("samples outside the crop:", {k: v for k, v in outside.items() if v})
    # no sample may hide under an a/b label (sample centers within 1.5 pt of the label box); redraw at the
    # figure dpi first, so that the label boxes and the data transforms are in the same display units
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pt = fig.dpi / 72
    hidden, clearance = {}, np.inf
    for rlabel, key, ax, label, ps_in in labels:
        bb = label.get_bbox_patch().get_window_extent(renderer)
        q = ax.transData.transform(ps_in)
        gap = np.hypot(np.maximum(np.maximum(bb.x0 - q[:, 0], q[:, 0] - bb.x1), 0),
                       np.maximum(np.maximum(bb.y0 - q[:, 1], q[:, 1] - bb.y1), 0)) / pt   # pt from the box
        if len(gap):
            clearance = min(clearance, gap.min())
            if (gap < 1.5).any():
                hidden[(rlabel, key)] = int((gap < 1.5).sum())
    print(f"samples under an a/b label: {hidden or 'none'} (smallest distance of a sample center from a label "
          f"box: {clearance:.1f} pt)")


if __name__ == "__main__":
    main()
