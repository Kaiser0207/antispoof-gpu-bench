"""Report figures (matplotlib, Agg backend, PNG at 150 dpi). Never imports torch.

Every function takes plain Python data prepared by ``agbench.report`` and writes one PNG.
Colours come from a colour-blind-checked categorical palette; identity is never carried by
colour alone (legends, hatching, line styles and value labels are always present).
"""

from __future__ import annotations

from pathlib import Path

DPI = 150

# Chart chrome (light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots. Measured GPUs take MEASURED_COLORS in order; our reference GPUs keep a
# fixed colour so the same machine has the same colour in every report.
MEASURED_COLORS = ("#2a78d6", "#008300", "#e34948", "#e87ba4", "#eda100")
REFERENCE_COLORS = {
    "rtx-3090-24gb": "#eb6834",
    "rtx-a4500-20gb": "#1baf7a",
    "a100-40gb-colab": "#4a3aa7",
}
OTHER_REFERENCE_COLOR = MUTED


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "hatch.linewidth": 0.8,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })
    return plt


def _style(ax, grid_axis: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, length=3)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _lighten(hex_color: str, amount: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def fmt_value(v: float) -> str:
    """Three significant figures without scientific notation (168, 56.3, 5.21)."""
    if v is None:
        return "-"
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _bar_kwargs(style: str, color: str) -> dict:
    if style == "hollow":
        return {"facecolor": SURFACE, "edgecolor": color, "linewidth": 1.6}
    if style == "hatched":
        return {"facecolor": _lighten(color, 0.72), "edgecolor": color, "hatch": "////", "linewidth": 0.8}
    if style == "oom":
        return {"facecolor": _lighten(color, 0.85), "edgecolor": color, "hatch": "xx", "linewidth": 0.8}
    return {"facecolor": color, "edgecolor": color, "linewidth": 0.0}


def _legend_patch(style: str, color: str, label: str):
    from matplotlib.patches import Patch

    return Patch(label=label, **_bar_kwargs(style, color))


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def grouped_hbars(path: Path, *, groups: list[str], series: list[dict], title: str, xlabel: str,
                  empty_text: str = "not measured") -> Path:
    """Horizontal grouped bars.

    groups: category labels, drawn top to bottom.
    series: [{"label": str, "color": hex, "style": "solid"|"hollow"|"hatched",
              "values": {group: (value, note_or_empty)}}], drawn in this order inside each group.
    """
    plt = _plt()
    rows = []  # (y, series or None, value, note)
    ticks, ticklabels = [], []
    y = 0.0
    for g in groups:
        present = [(s, s["values"][g]) for s in series
                   if g in s["values"] and (s["values"][g][0] is not None or s["values"][g][1])]
        start = y
        if not present:
            rows.append((y, None, None, empty_text))
            y += 1
        for s, (v, note) in present:
            rows.append((y, s if v is not None else None, None if v is None else float(v), note))
            y += 1
        ticks.append((start + y - 1) / 2)
        ticklabels.append(g)
        y += 0.7
    height = max(2.6, 0.26 * y + 1.4 + 0.18 * len(series))
    fig, ax = plt.subplots(figsize=(8.2, height), layout="constrained")
    _style(ax, "x")
    vmax = max([r[2] for r in rows if r[2] is not None] + [1.0])
    for yy, s, v, note in rows:
        if s is None:
            ax.text(vmax * 0.01, yy, note, va="center", ha="left", color=MUTED, fontsize=8, style="italic")
            continue
        ax.barh(yy, v, height=0.8, **_bar_kwargs(s["style"], s["color"]))
        label = s.get("fmt", fmt_value)(v) + (f"  {note}" if note else "")
        ax.text(v + vmax * 0.012, yy, label, va="center", ha="left", color=INK2, fontsize=8)
    ax.set_yticks(ticks, ticklabels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y - 1.1, -0.8)
    ax.set_xlim(0, vmax * 1.28)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    handles = [_legend_patch(s["style"], s["color"], s["label"]) for s in series]
    if handles:
        fig.legend(handles=handles, loc="outside lower center", ncol=1, frameon=False)
    return _save(fig, path)


def memory_fit(path: Path, *, rows: list[str], series: list[dict], caps: list[dict], guides: list[dict],
               title: str) -> Path:
    """Peak reserved memory per scenario against VRAM caps.

    rows: scenario labels top to bottom.
    series: [{"label", "color", "values": {row: (gib, note, is_oom)}}] (one per measured GPU).
    caps: [{"label", "value", "color"}] solid vertical lines (usable cap per GPU).
    guides: [{"label", "value"}] dashed grey vertical lines (typical 8 GB / 16 GB caps).
    """
    plt = _plt()
    from matplotlib.lines import Line2D

    placed = []
    ticks, ticklabels = [], []
    y = 0.0
    for r in rows:
        present = [(s, s["values"][r]) for s in series if r in s["values"] and s["values"][r][0] is not None]
        if not present:
            continue
        start = y
        for s, val in present:
            placed.append((y, s, val))
            y += 1
        ticks.append((start + y - 1) / 2)
        ticklabels.append(r)
        y += 0.5
    height = max(2.6, 0.24 * y + 1.6 + 0.16 * (len(series) + len(caps) + len(guides)))
    fig, ax = plt.subplots(figsize=(8.2, height), layout="constrained")
    _style(ax, "x")
    xmax = max([v[2][0] for v in placed] + [c["value"] for c in caps] + [g["value"] for g in guides] + [1.0])
    for yy, s, (gib, note, is_oom) in placed:
        ax.barh(yy, gib, height=0.8, **_bar_kwargs("oom" if is_oom else "solid", s["color"]))
        text = fmt_value(gib) + (f"  {note}" if note else "")
        ax.text(gib + xmax * 0.01, yy, text, va="center", ha="left", color=INK2, fontsize=7.5, zorder=5,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.8, "alpha": 0.9})
    handles = [_legend_patch("solid", s["color"], s["label"]) for s in series]
    if any(v[2][2] for v in placed):
        handles.append(_legend_patch("oom", MUTED, "out of memory at every batch split (last attempt shown)"))
    for c in caps:
        ax.axvline(c["value"], color=c["color"], linewidth=1.4, linestyle="-")
        handles.append(Line2D([0], [0], color=c["color"], linewidth=1.4, label=c["label"]))
    for g in guides:
        ax.axvline(g["value"], color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
        handles.append(Line2D([0], [0], color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), label=g["label"]))
    ax.set_yticks(ticks, ticklabels)
    ax.tick_params(axis="y", length=0, labelsize=8)
    ax.set_ylim(max(y - 0.9, 0.5), -0.8)
    ax.set_xlim(0, xmax * 1.3)
    ax.set_xlabel("peak reserved GPU memory (GiB)")
    ax.set_title(title, loc="left")
    fig.legend(handles=handles, loc="outside lower center", ncol=1, frameon=False)
    return _save(fig, path)


def length_scaling(path: Path, *, series: list[dict], title: str) -> Path:
    """Two panels against clip length: clips/s and ms per optimizer step.

    series: [{"label", "color", "points": [(seconds, clips_per_s, step_ms)]}]
    """
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.4), layout="constrained")
    for ax in (a1, a2):
        _style(ax, "both")
    for s in series:
        pts = sorted(s["points"])
        xs = [p[0] for p in pts]
        a1.plot(xs, [p[1] for p in pts], color=s["color"], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=s["label"])
        a2.plot(xs, [p[2] for p in pts], color=s["color"], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=s["label"])
        if pts:
            a1.annotate(fmt_value(pts[-1][1]), (xs[-1], pts[-1][1]), textcoords="offset points", xytext=(4, 4),
                        color=INK2, fontsize=8)
            a2.annotate(fmt_value(pts[-1][2]), (xs[-1], pts[-1][2]), textcoords="offset points", xytext=(4, -10),
                        color=INK2, fontsize=8)
    a1.set_xlabel("clip length (s)")
    a1.set_ylabel("clips per second (higher is better)")
    a1.set_title("Throughput", loc="left")
    a2.set_xlabel("clip length (s)")
    a2.set_ylabel("ms per optimizer step (lower is better)")
    a2.set_title("Time per step (effective batch 16)", loc="left")
    a1.set_ylim(bottom=0)
    a2.set_ylim(bottom=0)
    a1.set_xlim(0, 11)
    a2.set_xlim(0, 11)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=10)
    if len(series) > 1:
        fig.legend(handles=a1.get_lines(), loc="outside lower center", ncol=min(len(series), 3), frameon=False)
    return _save(fig, path)


def cpu_pipeline(path: Path, *, series: list[dict], demand: list[dict], title: str) -> Path:
    """Loader throughput against worker count, with the GPU's demand as horizontal lines.

    series: [{"label", "color", "dashed": bool, "points": [(workers, clips_per_s)]}]
    demand: [{"label", "color", "value"}]
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.2, 4.2), layout="constrained")
    _style(ax, "both")
    xmax = 1
    for s in series:
        pts = sorted(s["points"])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        xmax = max(xmax, max(xs))
        ax.plot(xs, [p[1] for p in pts], color=s["color"], linewidth=2,
                linestyle=(0, (5, 3)) if s.get("dashed") else "-",
                marker="s" if s.get("dashed") else "o", markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=s["label"])
        ax.annotate(fmt_value(pts[-1][1]), (xs[-1], pts[-1][1]), textcoords="offset points", xytext=(5, 0),
                    va="center", color=INK2, fontsize=8)
    for d in demand:
        ax.axhline(d["value"], color=d["color"], linewidth=1.2, linestyle=(0, (1, 2)), label=d["label"])
    ax.set_xlim(0, xmax + 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("DataLoader worker processes")
    ax.set_ylabel("clips per second delivered (6 s crops)")
    ax.set_title(title, loc="left")
    fig.legend(loc="outside lower center", ncol=1, frameon=False)
    return _save(fig, path)


def sustained(path: Path, *, series: list[dict], title: str) -> Path:
    """Small multiples over time for the sustained run.

    series: [{"label", "color", "windows": [(t_s, clips_per_s)],
              "telemetry": [(t_s, temp_c, sm_clock_mhz, power_w)]}]
    """
    plt = _plt()
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 7.2), sharex=True, layout="constrained")
    ylabels = ("clips/s per 30 s window", "GPU temperature (deg C)", "SM clock (MHz)", "board power (W)")
    for ax, ylabel in zip(axes, ylabels):
        _style(ax, "both")
        ax.set_ylabel(ylabel)
    for s in series:
        w = sorted(s.get("windows") or [])
        if w:
            axes[0].plot([p[0] for p in w], [p[1] for p in w], color=s["color"], linewidth=2, marker="o",
                         markersize=5, markeredgecolor=SURFACE, label=s["label"])
        tel = sorted(s.get("telemetry") or [])
        for k, ax in ((1, axes[1]), (2, axes[2]), (3, axes[3])):
            pts = [(p[0], p[k]) for p in tel if p[k] is not None]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=s["color"], linewidth=1.5)
    # Zero-based axes (temperature from 20 deg C) so small drifts are not exaggerated.
    for ax in (axes[0], axes[2], axes[3]):
        ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1] * 1.08, 1.0))
    lo, hi = axes[1].get_ylim()
    axes[1].set_ylim(min(20.0, lo), max(90.0, hi + 3))
    axes[-1].set_xlabel("time since start of the sustained run (s)")
    axes[0].set_title(title, loc="left")
    if len(series) > 1:
        fig.legend(handles=axes[0].get_lines(), loc="outside lower center", ncol=min(len(series), 3),
                   frameon=False)
    return _save(fig, path)
