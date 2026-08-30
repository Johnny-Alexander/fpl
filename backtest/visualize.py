"""
Worm graph for the season backtest.

Light surface with strong gridlines and large type, sized to stay legible
projected rather than on a laptop.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1 and 2 of the reference palette, in fixed order.
SERIES_COLORS = ["#2a78d6", "#eb6834"]
REFERENCE_COLOR = "#6f6e69"  # benchmark line: neutral, not a competing series
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#c9c8c2"


def _cumulative(scores, gameweeks):
    running, out = 0.0, []
    for gw in gameweeks:
        running += scores.get(gw, 0.0)
        out.append(running)
    return out


def plot_worm_graph(runs, output_path, first_gw=1, last_gw=38, actual_total=None):
    """
    runs:         {label: {gameweek: points}}
    actual_total: the manager's real season total, drawn as an even-pace reference.
                  Per-gameweek history is not retrievable once a season ends, so
                  only the endpoint is real -- the line between is interpolation
                  and is labelled as such.
    """
    gameweeks = list(range(first_gw, last_gw + 1))

    fig, ax = plt.subplots(figsize=(15, 8.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.grid(True, which="major", color=GRID, linewidth=1.1, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

    if actual_total:
        pace = [
            actual_total * (i + 1) / len(gameweeks) for i in range(len(gameweeks))
        ]
        # The endpoint is not labelled directly: the total is already named in the
        # legend, and a strategy finishing near it would collide with that label.
        ax.plot(gameweeks, pace, linestyle=(0, (6, 4)), color=REFERENCE_COLOR,
                linewidth=2.0, zorder=2,
                label=f"Actual season total ({actual_total:,}) — even pace")

    for index, (label, scores) in enumerate(runs.items()):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        cumulative = _cumulative(scores, gameweeks)
        ax.plot(gameweeks, cumulative, color=color, linewidth=2.6, zorder=4 + index,
                label=label, solid_capstyle="round")
        ax.annotate(f"{cumulative[-1]:,.0f}", xy=(gameweeks[-1], cumulative[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    color=color, fontsize=14, fontweight="bold", va="center")

    ax.set_xlabel("Gameweek", fontsize=16, color=INK_SECONDARY, labelpad=10)
    ax.set_ylabel("Cumulative points", fontsize=16, color=INK_SECONDARY, labelpad=10)
    ax.set_title("FPL backtest: model-managed season vs actual",
                 fontsize=21, fontweight="bold", color=INK, pad=18, loc="left")

    ticks = [gw for gw in gameweeks if gw % 5 == 0 or gw in (first_gw, last_gw)]
    ax.set_xticks(ticks)
    ax.tick_params(axis="both", labelsize=13, colors=INK_SECONDARY, length=0)
    ax.set_xlim(first_gw, last_gw + 1.8)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)

    legend = ax.legend(loc="upper left", fontsize=14, frameon=True,
                       facecolor=SURFACE, edgecolor=GRID, borderpad=0.8)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  worm graph -> {output_path}")
