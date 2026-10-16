"""
The weekly email report: charts and HTML.

Email is a static medium -- no JavaScript, and remote images and data: URIs are
blocked by default in most clients -- so charts are rendered to PNG and attached
by content-id, and the layout uses tables and inline styles rather than modern
CSS. That constraint is why there is no hover layer here; everything a reader
needs has to be on the face of the chart.

Charts render on an explicit light surface rather than a transparent one. A
transparent PNG dropped onto a dark-mode client's background turns dark ink
invisible, and a PNG cannot restyle itself the way an HTML chart can.
"""

import base64
import datetime as dt
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory

from attribution import FACTOR_NOTES, FACTORS

# Reference palette, used unmodified so its published validation applies.
# Categorical slots 1-2 for the two-series tracker and 1-5 for the factors; the
# documented blue<->red diverging pair, gray midpoint, for fixture difficulty
# and for factor contributions (blue adds points, red takes them away).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SERIES = CATEGORICAL[:2]
DIVERGING_EASY, DIVERGING_MID, DIVERGING_HARD = "#2a78d6", "#f0efec", "#e34948"

# Slots in attribution.FACTORS order, never reassigned, so a factor keeps its
# colour from one week's email to the next.
FACTOR_COLOURS = dict(zip(FACTORS, CATEGORICAL))
FACTOR_SHORT = {"Price & ownership": "Price", "Form": "Form", "Minutes": "Mins",
                "Underlying stats": "Stats", "Fixture": "Fixt"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

POSITION_NAME = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

DIFFICULTY_CMAP = LinearSegmentedColormap.from_list(
    "fdr", [DIVERGING_EASY, DIVERGING_MID, DIVERGING_HARD]
)


def _style(ax):
    """Recessive chrome: hairline grid, no top/right spines, muted tick ink."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=1, alpha=1)
    ax.set_axisbelow(True)


def _relative_luminance(rgba):
    """WCAG relative luminance, used to pick readable ink on a coloured cell."""
    channels = []
    for value in rgba[:3]:
        channels.append(value / 12.92 if value <= 0.03928
                        else ((value + 0.055) / 1.055) ** 2.4)
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _readable_ink(rgba):
    """
    Whichever of dark or white ink contrasts better with this cell.

    Picking by a luminance threshold needs a magic number and gets the middle of
    a diverging ramp wrong -- the pale pinks are light enough that white text on
    them fails. Comparing the two contrast ratios directly has no such tuning.
    """
    background = _relative_luminance(rgba)
    against_white = 1.05 / (background + 0.05)
    against_ink = (background + 0.05) / (_relative_luminance((0.043, 0.043, 0.043)) + 0.05)
    return "#ffffff" if against_white > against_ink else INK


def _to_png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, facecolor=SURFACE,
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return buffer.getvalue()


def fixture_outlook(fixtures, bootstrap, team_ids, start_gw, weeks=5):
    """
    {team_id: {gameweek: [(opponent_short, difficulty, is_home), ...]}}

    A gameweek may hold two fixtures for a team, or none; both are represented
    faithfully rather than flattened to a single number.
    """
    shorts = {int(t["id"]): t["short_name"] for t in bootstrap["teams"]}
    wanted = set(range(start_gw, start_gw + weeks))
    outlook = {t: {gw: [] for gw in wanted} for t in team_ids}

    for fixture in fixtures:
        gameweek = fixture.get("event")
        if gameweek not in wanted:
            continue
        for side, other, difficulty_key, home in (
            ("team_h", "team_a", "team_h_difficulty", True),
            ("team_a", "team_h", "team_a_difficulty", False),
        ):
            team = int(fixture[side])
            if team in outlook:
                outlook[team][gameweek].append(
                    (shorts.get(int(fixture[other]), "?"),
                     int(fixture.get(difficulty_key) or 3), home)
                )
    return outlook


def chart_fixture_grid(squad, outlook, start_gw, weeks=5):
    """
    Difficulty of each squad player's next fixtures.

    The one view the model cannot act on for itself: the multi-gameweek planner
    was measured and did not work, so fixture runs are left to the reader. Cells
    carry the opponent and are shaded by difficulty, with the number of fixtures
    shown where a gameweek is blank or doubled.
    """
    players = sorted(squad, key=lambda p: (p["position"], -p["predicted"]))
    gameweeks = list(range(start_gw, start_gw + weeks))

    height = max(3.2, 0.34 * len(players) + 1.1)
    fig, ax = plt.subplots(figsize=(7.4, height))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for row, player in enumerate(players):
        entries = outlook.get(player["team"], {})
        for column, gameweek in enumerate(gameweeks):
            matches = entries.get(gameweek, [])
            if not matches:
                ax.add_patch(plt.Rectangle((column, row), 0.94, 0.9,
                                           facecolor="#f4f3f0", edgecolor=SURFACE,
                                           linewidth=2))
                ax.text(column + 0.47, row + 0.45, "blank", ha="center", va="center",
                        fontsize=7, color=INK_MUTED, style="italic")
                continue

            mean_difficulty = float(np.mean([m[1] for m in matches]))
            # 1 (easiest) to 5 (hardest), neutral at 3.
            shade = DIFFICULTY_CMAP((mean_difficulty - 1) / 4)
            ax.add_patch(plt.Rectangle((column, row), 0.94, 0.9,
                                       facecolor=shade, edgecolor=SURFACE,
                                       linewidth=2))
            label = ", ".join(
                f"{opponent}{'(H)' if home else ''}" for opponent, _, home in matches
            )
            if len(matches) > 1:
                label = f"{label}  x{len(matches)}"
            # Choose ink from the cell's actual luminance, not from the
            # difficulty value: the ramp is light through most of its middle and
            # only dark at the two extremes, so a difficulty threshold puts white
            # text on pale cells.
            tone = _readable_ink(shade)
            ax.text(column + 0.47, row + 0.45, label, ha="center", va="center",
                    fontsize=7.5, color=tone, fontweight="medium")

    ax.set_xlim(0, len(gameweeks))
    ax.set_ylim(0, len(players))
    ax.set_xticks([i + 0.47 for i in range(len(gameweeks))])
    ax.set_xticklabels([f"GW{g}" for g in gameweeks], fontsize=9, color=INK_SECONDARY)
    ax.set_yticks([i + 0.45 for i in range(len(players))])
    ax.set_yticklabels(
        [f"{POSITION_NAME.get(p['position'], '')}  {p['name']}" for p in players],
        fontsize=8.5, color=INK_SECONDARY,
    )
    ax.xaxis.tick_top()
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.invert_yaxis()
    ax.set_title("Fixture difficulty, next five gameweeks",
                 fontsize=12, color=INK, fontweight="bold", pad=26, loc="left")
    # As an x-label rather than figure text: figure coordinates below zero make
    # the tight bounding box grow, leaving a large gap under the grid.
    ax.set_xlabel("Blue easier, red harder.  (H) = home.  FPL difficulty rating.",
                  fontsize=8.5, color=INK_MUTED, labelpad=12, loc="left")
    return _to_png(fig)


def chart_season_tracker(rows):
    """
    Model-recommended points against the manager's actual, cumulative.

    Returns None until two gameweeks have been scored -- a line through one point
    is not a chart, and the honest thing early in a season is to show nothing.
    """
    scored = [r for r in rows if r.get("scored") and "model_cumulative" in r]
    if len(scored) < 2:
        return None

    gameweeks = [r["gameweek"] for r in scored]
    model = [r["model_cumulative"] for r in scored]
    human = [r["human_cumulative"] for r in scored]

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)

    for values, colour, label in ((model, SERIES[0], "Model"), (human, SERIES[1], "You")):
        ax.plot(gameweeks, values, color=colour, linewidth=2,
                marker="o", markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2, label=label, solid_capstyle="round")
        # Direct label at the series end; two series, so both get one.
        ax.annotate(f"{label} {values[-1]:.0f}",
                    xy=(gameweeks[-1], values[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=INK, fontweight="bold")

    ax.set_xlabel("Gameweek", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("Cumulative points", fontsize=9.5, color=INK_SECONDARY)
    ax.set_xticks(gameweeks)
    ax.set_title("Model recommendation vs your team, season to date",
                 fontsize=11.5, color=INK, fontweight="bold", pad=12, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left",
              labelcolor=INK_SECONDARY)
    ax.margins(x=0.12)
    return _to_png(fig)


MARKERS = {"owned": "●", "buy": "▶"}
# In HTML the triangle carries the text-presentation selector, or iOS Mail draws
# it as a blue emoji button. Not in charts, where the font has no such glyph.
HTML_MARKERS = {key: glyph + "\ufe0e" for key, glyph in MARKERS.items()}


def chart_top_players(players, base, gameweek):
    """
    The game's highest predictions, each split into the factors behind it.

    Bars diverge from a zero line rather than starting at zero points: the base
    every player shares is stated once in the caption, and what is left is the
    part that distinguishes them. Factors that add points stack to the right,
    those that take points away to the left, and the prediction itself is
    printed at the end of the row -- with anything on the left of the line, the
    bar's right end overstates it, so the number is the reading to trust.

    Returns None when nothing can be explained, for a model that is not a tree
    ensemble.
    """
    rows = [p for p in players if p.get("factors")]
    if not rows or base is None:
        return None

    fig, ax = plt.subplots(figsize=(7.4, 0.38 * len(rows) + 2.0))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)

    lowest = highest = 0.0
    for index, player in enumerate(rows):
        right = left = 0.0
        for factor in FACTORS:
            value = player["factors"][factor]
            if abs(value) < 0.005:
                continue
            start = right if value > 0 else left
            # A surface-coloured edge is the 2px gap between touching segments.
            ax.barh(index, value, left=start, height=0.62,
                    color=FACTOR_COLOURS[factor], edgecolor=SURFACE,
                    linewidth=1.2, zorder=2)
            if value > 0:
                right += value
            else:
                left += value
        lowest, highest = min(lowest, left), max(highest, right)

    ax.axvline(0, color=AXIS, linewidth=1, zorder=3)
    ax.set_xlim(min(lowest, 0) - 0.15, highest + 0.15)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x:+g}"))
    ax.tick_params(axis="x", labelsize=10.5)

    labels = []
    for player in rows:
        marker = MARKERS.get(player.get("status"), "")
        labels.append(f"{marker + ' ' if marker else ''}{player['name']}   "
                      f"{player['team_short']} £{player['price']:.1f}m")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=11.5)
    for tick, player in zip(ax.get_yticklabels(), rows):
        mine = player.get("status") is not None
        tick.set_color(INK if mine else INK_SECONDARY)
        tick.set_fontweight("bold" if mine else "normal")

    # The prediction, in a column of its own past the plot.
    column = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.015, -0.95, "xPts", transform=column, ha="left", va="center",
            fontsize=10, color=INK_MUTED)
    for index, player in enumerate(rows):
        ax.text(1.015, index, f"{player['predicted']:.1f}", transform=column,
                ha="left", va="center", fontsize=12, color=INK, fontweight="bold")

    handles = [Patch(facecolor=FACTOR_COLOURS[f], edgecolor="none", label=f) for f in FACTORS]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.035),
              ncol=len(FACTORS), frameon=False, fontsize=10.5, handlelength=1.0,
              handleheight=1.0, handletextpad=0.4, columnspacing=1.0,
              labelcolor=INK_SECONDARY, borderaxespad=0)
    ax.set_title(f"Top {len(rows)} predicted players, GW{gameweek}", loc="left",
                 fontsize=14, color=INK, fontweight="bold", pad=40)
    ax.set_xlabel(
        f"Points each factor adds to, or takes from, an average player-gameweek "
        f"({base:.1f}).\n{MARKERS['owned']} in your squad   "
        f"{MARKERS['buy']} recommended transfer in",
        fontsize=10.5, color=INK_MUTED, labelpad=10, loc="left", linespacing=1.6,
    )
    return _to_png(fig)


# ──────────────────────────── HTML ────────────────────────────

STYLE_TABLE = ("width:100%;border-collapse:collapse;font-size:14px;"
               "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif")
STYLE_TH = (f"text-align:left;padding:6px 8px;border-bottom:1px solid {AXIS};"
            f"color:{INK_SECONDARY};font-weight:600;font-size:12px;"
            "text-transform:uppercase;letter-spacing:0.04em")
STYLE_TD = f"padding:6px 8px;border-bottom:1px solid {GRID};color:{INK}"


def _esc(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _transfer_block(option, index):
    if option["n_transfers"] == 0:
        return (f'<p style="margin:8px 0;color:{INK_SECONDARY}">'
                "No transfer improves on the current squad — save it.</p>")

    rows = []
    for move in option["moves"]:
        out, into = move["out"], move["in"]
        rows.append(
            f'<tr>'
            f'<td style="{STYLE_TD}">{_esc(out["name"])} '
            f'<span style="color:{INK_MUTED}">£{out["price"]:.1f}m · {out["predicted"]:.1f} pred</span></td>'
            f'<td style="{STYLE_TD};color:{INK_MUTED}">&rarr;</td>'
            f'<td style="{STYLE_TD}"><strong>{_esc(into["name"])}</strong> '
            f'<span style="color:{INK_MUTED}">£{into["price"]:.1f}m · {into["predicted"]:.1f} pred</span></td>'
            f'</tr>'
        )

    hit = (f'<span style="color:#d03b3b">−{option["hit_points"]} hit</span>'
           if option["hit_points"] else
           f'<span style="color:#006300">free</span>')
    heading = "Recommended" if index == 1 else f"Alternative {index - 1}"
    return (
        f'<h3 style="margin:18px 0 6px;font-size:15px;color:{INK}">{heading} '
        f'<span style="font-weight:400;color:{INK_SECONDARY};font-size:13px">'
        f'· {option["n_transfers"]} transfer{"s" if option["n_transfers"] != 1 else ""} · {hit} '
        f'· net {option["net_gain"]:+.2f} pts</span></h3>'
        f'<table style="{STYLE_TABLE}">{"".join(rows)}</table>'
    )


def _mix(colour, weight):
    """`colour` laid over the surface at `weight`, as hex: a tint, not a block."""
    surface, ink = np.array(to_rgb(SURFACE)), np.array(to_rgb(colour))
    return to_hex(surface + (ink - surface) * weight)


def _contribution_cell(value, scale=2.0):
    """A factor's points as a table cell, tinted blue when it adds and red when it subtracts."""
    if value is None:
        return f'<td style="{STYLE_TD};padding:6px 4px;text-align:center;color:{INK_MUTED}">—</td>'
    if abs(value) < 0.05:
        return (f'<td style="{STYLE_TD};padding:6px 4px;text-align:right;color:{INK_MUTED};'
                f'font-size:13px;font-variant-numeric:tabular-nums">0.0</td>')
    pole = DIVERGING_EASY if value > 0 else DIVERGING_HARD
    tint = _mix(pole, 0.65 * min(abs(value) / scale, 1.0))
    ink = _readable_ink(to_rgb(tint))
    return (f'<td style="{STYLE_TD};padding:6px 4px;text-align:right;background:{tint};'
            f'color:{ink};border-bottom:1px solid {SURFACE};font-size:13px;'
            f'font-variant-numeric:tabular-nums">{value:+.1f}</td>')


# Header cells that carry a label as typed, not in the uppercase section style.
STYLE_TH_PLAIN = f"{STYLE_TH};text-transform:none;letter-spacing:0"


def _factor_heading(factor):
    # The swatch sits above the label rather than beside it: side by side, the
    # headings are wider than the numbers under them and set the column width.
    swatch = (f'<span style="display:inline-block;width:8px;height:8px;border-radius:2px;'
              f'background:{FACTOR_COLOURS[factor]}"></span>')
    return (f'<td style="{STYLE_TH_PLAIN};font-size:11px;text-align:right;padding:6px 4px;'
            f'line-height:1.2;vertical-align:bottom" title="{_esc(factor)}">'
            f'{swatch}<br>{FACTOR_SHORT[factor]}</td>')


def _squad_table(squad):
    starters = [p for p in squad if p.get("is_starter")]
    bench = [p for p in squad if not p.get("is_starter")]
    rows = []
    for group, label in ((starters, "Starting XI"), (bench, "Bench")):
        rows.append(f'<tr><td colspan="4" style="{STYLE_TH};padding-top:14px">{label}</td></tr>')
        for player in sorted(group, key=lambda p: (p["position"], -p["predicted"])):
            mark = ' <span style="color:#eb6834;font-weight:600">(C)</span>' if player.get("is_captain") else ""
            rows.append(
                f'<tr>'
                f'<td style="{STYLE_TD};color:{INK_MUTED};width:44px">{POSITION_NAME.get(player["position"], "")}</td>'
                f'<td style="{STYLE_TD}">{_esc(player["name"])}{mark}</td>'
                f'<td style="{STYLE_TD};text-align:right;color:{INK_SECONDARY}">£{player["price"]:.1f}m</td>'
                f'<td style="{STYLE_TD};text-align:right">{player["predicted"]:.1f}</td>'
                f'</tr>'
            )
    return f'<table style="{STYLE_TABLE}">{"".join(rows)}</table>'


def _squad_factor_table(squad, factors):
    """
    The squad table with each prediction split by factor.

    Position and price move under the name so that five factor columns still
    fit a phone: the plain table's separate columns would push it past the
    width of the screen.
    """
    columns = 2 + len(FACTORS)
    rows = [
        f'<tr><td style="{STYLE_TH}"></td>'
        f'<td style="{STYLE_TH_PLAIN};text-align:right;padding-right:6px">xPts</td>'
        + "".join(_factor_heading(f) for f in FACTORS) + '</tr>'
    ]
    starters = [p for p in squad if p.get("is_starter")]
    bench = [p for p in squad if not p.get("is_starter")]
    for group, label in ((starters, "Starting XI"), (bench, "Bench")):
        rows.append(f'<tr><td colspan="{columns}" style="{STYLE_TH};padding-top:14px">{label}</td></tr>')
        for player in sorted(group, key=lambda p: (p["position"], -p["predicted"])):
            mark = ' <span style="color:#eb6834;font-weight:600">(C)</span>' if player.get("is_captain") else ""
            parts = factors.get(player["element_id"])
            rows.append(
                f'<tr><td style="{STYLE_TD};line-height:1.3">{_esc(player["name"])}{mark}<br>'
                f'<span style="color:{INK_MUTED};font-size:12px">'
                f'{POSITION_NAME.get(player["position"], "")} · £{player["price"]:.1f}m</span></td>'
                f'<td style="{STYLE_TD};text-align:right;padding-right:6px;font-weight:600;'
                f'font-variant-numeric:tabular-nums">{player["predicted"]:.1f}</td>'
                + "".join(_contribution_cell(parts[f] if parts else None) for f in FACTORS)
                + '</tr>'
            )
    return (
        f'<table style="{STYLE_TABLE}">{"".join(rows)}</table>'
        f'<p style="margin:8px 0 0;color:{INK_MUTED};font-size:12px;line-height:1.5">'
        'Each prediction split by factor, as in the chart above: blue adds points, red '
        'takes them away. A dash means the model\'s number was overridden, usually '
        'because the player is ruled out.</p>'
    )


POSITION_PLURAL = {1: "Goalkeepers", 2: "Defenders", 3: "Midfielders", 4: "Forwards"}


def _fixture_cell(matches):
    """One gameweek of a team's fixture ticker, shaded by FPL difficulty."""
    base = (f"padding:5px 2px;text-align:center;font-size:11px;line-height:1.25;"
            f"border:2px solid {SURFACE};white-space:nowrap")
    if not matches:
        return f'<td style="{base};background:#f4f3f0;color:{INK_MUTED}">—</td>'
    shade = DIFFICULTY_CMAP((float(np.mean([m[1] for m in matches])) - 1) / 4)
    # The ticker convention: capitals for a home game, lower case away. "(H)"
    # as in the squad grid would make each cell twice as wide, and five of them
    # would no longer fit a phone.
    label = "<br>".join(opponent.upper() if home else opponent.lower()
                        for opponent, _, home in matches)
    return (f'<td style="{base};background:{to_hex(shade)};'
            f'color:{_readable_ink(shade)}">{label}</td>')


def _position_table(by_position, fixtures, weeks):
    """The strongest predictions in each position, with each team's next fixtures."""
    head = (f'<tr><td style="{STYLE_TH}"></td>'
            f'<td style="{STYLE_TH_PLAIN};text-align:right;padding-right:8px">xPts</td>'
            + "".join(f'<td style="{STYLE_TH_PLAIN};text-align:center;padding:6px 2px">GW{g}</td>'
                      for g in weeks)
            + '</tr>')
    rows = [head]
    for position, players in by_position.items():
        rows.append(f'<tr><td colspan="{2 + len(weeks)}" style="{STYLE_TH};padding-top:14px">'
                    f'{POSITION_PLURAL.get(position, "")}</td></tr>')
        for player in players:
            marker = HTML_MARKERS.get(player.get("status"), "")
            marker = (f'<span style="color:{INK};font-size:10px">{marker}</span> '
                      if marker else "")
            owned = player["owned_pct"]
            owned = f"{owned:.0f}%" if owned >= 1 else f"{owned:.1f}%"
            rows.append(
                f'<tr><td style="{STYLE_TD};line-height:1.3">{marker}'
                f'<strong>{_esc(player["name"])}</strong><br>'
                f'<span style="color:{INK_MUTED};font-size:12px">'
                f'{_esc(player["team_short"])} · £{player["price"]:.1f}m · {owned} owned</span></td>'
                f'<td style="{STYLE_TD};text-align:right;padding-right:8px;font-weight:600;'
                f'font-variant-numeric:tabular-nums">{player["predicted"]:.1f}</td>'
                + "".join(_fixture_cell(fixtures.get(player["team"], {}).get(g, []))
                          for g in weeks)
                + '</tr>'
            )
    return (
        f'<table style="{STYLE_TABLE}">{"".join(rows)}</table>'
        f'<p style="margin:8px 0 0;color:{INK_MUTED};font-size:12px;line-height:1.5">'
        f'{HTML_MARKERS["owned"]} in your squad · {HTML_MARKERS["buy"]} recommended transfer in. '
        'Fixtures shaded by FPL difficulty, blue easier and red harder; capitals are '
        'home games, a dash is a blank.</p>'
    )


def _factor_key(base):
    """What each factor in the chart and squad table is made of."""
    rows = "".join(
        f'<tr><td style="padding:3px 8px 3px 0;vertical-align:top;white-space:nowrap">'
        f'<span style="display:inline-block;width:9px;height:9px;border-radius:2px;'
        f'background:{FACTOR_COLOURS[f]};margin-right:6px"></span>'
        f'<span style="color:{INK}">{_esc(f)}</span></td>'
        f'<td style="padding:3px 0;color:{INK_SECONDARY}">{_esc(FACTOR_NOTES[f])}</td></tr>'
        for f in FACTORS
    )
    return (
        f'<table style="border-collapse:collapse;font-size:12px;margin-top:10px;'
        f'font-family:inherit">{rows}</table>'
        f'<p style="margin:8px 0 0;color:{INK_MUTED};font-size:12px;line-height:1.5">'
        f'Every bar starts from an average player-gameweek of {base:.1f} points, most of '
        'which are players who do not play; the factors add up exactly to the prediction. '
        'They describe how the model reached its number, not a scouting report: a negative '
        'Form means recent returns pull the number down once the other factors are taken '
        'into account, not that the player is out of form.</p>'
    )


def _review_block(review):
    """Last gameweek's recommended XI: what each player was predicted, and what they scored."""
    hit = (f' after a −{review["model_hit"]:.0f} hit' if review.get("model_hit") else "")
    you = (f'; you scored <strong>{review["human_points"]:.0f}</strong>'
           if review.get("human_points") is not None else "")
    summary = (
        f'<p style="margin:0 0 6px;color:{INK_SECONDARY};font-size:14px">'
        f'The recommended team was predicted <strong>{review["predicted_total"] or 0:.1f}</strong> '
        f'and scored <strong>{review["model_points"]:.0f}</strong>{hit}{you}.</p>'
    )
    rows = [
        f'<tr><td colspan="2" style="{STYLE_TH}">Starting XI</td>'
        f'<td style="{STYLE_TH};text-align:right">Pred</td>'
        f'<td style="{STYLE_TH};text-align:right">Actual</td>'
        f'<td style="{STYLE_TH};text-align:right">Diff</td></tr>'
    ]
    starters = sorted((p for p in review["players"] if p["is_starter"]),
                      key=lambda p: (p["position"], -p["predicted"]))
    for player in starters:
        captain = (' <span style="color:#eb6834;font-weight:600">(C)</span>'
                   if player["is_captain"] else "")
        actual = f'{player["actual"]:.0f}'
        if not player["minutes"]:
            actual = f'<span style="color:{INK_MUTED};font-size:12px">DNP</span> {actual}'
        diff = player["actual"] - player["predicted"]
        tone = "#006300" if diff >= 0 else "#d03b3b"
        rows.append(
            f'<tr><td style="{STYLE_TD};color:{INK_MUTED};width:44px">{POSITION_NAME.get(player["position"], "")}</td>'
            f'<td style="{STYLE_TD}">{_esc(player["name"])}{captain}</td>'
            f'<td style="{STYLE_TD};text-align:right;color:{INK_SECONDARY};font-variant-numeric:tabular-nums">{player["predicted"]:.1f}</td>'
            f'<td style="{STYLE_TD};text-align:right;font-weight:600;font-variant-numeric:tabular-nums">{actual}</td>'
            f'<td style="{STYLE_TD};text-align:right;color:{tone};font-variant-numeric:tabular-nums">{diff:+.1f}</td></tr>'
        )
    note = (f'<p style="margin:8px 0 0;color:{INK_MUTED};font-size:12px;line-height:1.5">'
            'Points before the captain is doubled. The team total includes the captaincy '
            'and any automatic substitutions.</p>')
    return summary + f'<table style="{STYLE_TABLE}">{"".join(rows)}</table>' + note


def _chip_block(chips_data):
    if not chips_data:
        return ""
    rows = []
    for chip in chips_data:
        if chip["verdict"] == "play":
            verdict = f'<strong style="color:#006300">PLAY</strong>'
        elif chip["verdict"] == "hold":
            verdict = f'<span style="color:{INK_MUTED}">hold</span>'
        else:
            verdict = f'<span style="color:{INK_MUTED}">run --wildcard to value</span>'
        detail = (f'{chip["value"]:.1f} vs bar {chip["bar"]:.1f}'
                  if chip.get("value") is not None else "—")
        rows.append(
            f'<tr><td style="{STYLE_TD};width:60px"><strong>{chip["chip"]}</strong></td>'
            f'<td style="{STYLE_TD};color:{INK_SECONDARY}">{detail}</td>'
            f'<td style="{STYLE_TD};text-align:right">{verdict}</td>'
            f'<td style="{STYLE_TD};text-align:right;color:{INK_MUTED}">'
            f'{chip["weeks_left"]}w left</td></tr>'
        )
    return f'<table style="{STYLE_TABLE}">{"".join(rows)}</table>'


def _heading(text, top=26, bottom=8):
    return (f'<h2 style="margin:{top}px 0 {bottom}px;font-size:17px;color:{INK}">'
            f'{_esc(text)}</h2>')


def render_html(recommendation, history_rows, images, insights=None):
    """
    The email body.

    Inline styles and tables throughout: email clients strip <style> blocks and
    do not support grid or flex reliably. Images are referenced by content-id,
    which is the only form most clients display without the reader clicking
    "load images".

    Forward-looking sections come first -- the decision, then the wider market,
    then the squad -- and the look back at last gameweek and the season after.
    `insights` (from insights.build) adds the market and review sections; without
    it the email is the recommendation alone.
    """
    insights = insights or {}
    deadline = recommendation.get("deadline")
    when = ""
    if deadline:
        moment = dt.datetime.fromisoformat(deadline.replace("Z", "+00:00"))
        hours = (moment - dt.datetime.now(dt.timezone.utc)).total_seconds() / 3600
        when = (f'{moment.strftime("%a %d %b, %H:%M")} UTC '
                f'<span style="color:{INK_MUTED}">· {hours:.0f}h away</span>')

    captain = recommendation.get("captain") or {}
    options = recommendation.get("options") or []

    scored = [r for r in history_rows if r.get("scored") and "model_points" in r]
    if scored:
        model_total = scored[-1]["model_cumulative"]
        human_total = scored[-1]["human_cumulative"]
        gap = model_total - human_total
        tracking = (
            f'<p style="margin:6px 0;color:{INK_SECONDARY};font-size:14px">'
            f'Over {len(scored)} tracked gameweek{"s" if len(scored) != 1 else ""}: '
            f'model <strong>{model_total:.0f}</strong>, you <strong>{human_total:.0f}</strong> '
            f'({gap:+.0f}).</p>'
        )
    else:
        tracking = (f'<p style="margin:6px 0;color:{INK_MUTED};font-size:14px">'
                    "First tracked gameweek — comparison starts once it is played.</p>")

    blocks = [
        f'<div style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Helvetica,Arial,sans-serif;'
        f'max-width:680px;margin:0 auto;padding:24px;background:{SURFACE};color:{INK}">',
        f'<p style="margin:0 0 4px;color:{INK_MUTED};font-size:12px;text-transform:uppercase;'
        f'letter-spacing:0.06em">FPL model · {_esc(recommendation.get("season", ""))}</p>',
        f'<h1 style="margin:0 0 2px;font-size:24px;color:{INK}">Gameweek {recommendation["gameweek"]}</h1>',
        f'<p style="margin:0 0 14px;color:{INK_SECONDARY};font-size:14px">Deadline {when}</p>',
        f'<div style="padding:14px 16px;background:#f4f6fb;border-left:3px solid {SERIES[0]};'
        f'border-radius:3px;margin-bottom:6px">'
        f'<div style="font-size:13px;color:{INK_SECONDARY}">Captain</div>'
        f'<div style="font-size:19px;font-weight:600;color:{INK}">{_esc(captain.get("name", "—"))}'
        f'<span style="font-weight:400;color:{INK_SECONDARY};font-size:14px">'
        f' · {captain.get("predicted", 0):.1f} predicted, doubled</span></div>'
        f'<div style="font-size:13px;color:{INK_SECONDARY};margin-top:6px">'
        f'Projected XI total <strong>{recommendation.get("predicted_xi_points") or 0:.1f}</strong> pts</div>'
        f'</div>',
        tracking,
        f'<h2 style="margin:22px 0 0;font-size:17px;color:{INK}">Transfers</h2>',
    ]
    blocks += [_transfer_block(option, i) for i, option in enumerate(options, 1)]

    if images.get("top") and insights.get("base") is not None:
        top = insights.get("top") or []
        alt = ", ".join(f'{p["name"]} {p["predicted"]:.1f}' for p in top[:10])
        blocks += [
            _heading("Who the model rates this week"),
            f'<img src="cid:top" alt="Highest predicted players, split by factor: {_esc(alt)}" '
            f'style="width:100%;max-width:640px;display:block;border:0">',
            _factor_key(insights["base"]),
        ]
    if insights.get("by_position"):
        blocks += [
            _heading("Best options by position", bottom=4),
            _position_table(insights["by_position"], insights.get("fixtures") or {},
                            insights.get("fixture_weeks") or []),
        ]

    if images.get("fixtures"):
        blocks.append(
            _heading("Fixtures ahead")
            + f'<img src="cid:fixtures" alt="Fixture difficulty grid for the squad, next five gameweeks" '
            f'style="width:100%;max-width:640px;display:block;border:0">'
        )

    blocks += [
        _heading("Squad", bottom=4),
        (_squad_factor_table(recommendation.get("squad") or [], insights["squad_factors"])
         if any((insights.get("squad_factors") or {}).values())
         else _squad_table(recommendation.get("squad") or [])),
    ]
    if recommendation.get("chips"):
        blocks += [_heading("Chips", bottom=4), _chip_block(recommendation["chips"])]

    review = insights.get("review")
    if review:
        blocks += [_heading(f"Last gameweek: GW{review['gameweek']}", bottom=4),
                   _review_block(review)]
    if images.get("tracker"):
        blocks.append(
            _heading("Season tracker")
            + f'<img src="cid:tracker" alt="Cumulative model points against your team" '
            f'style="width:100%;max-width:640px;display:block;border:0">'
        )

    warnings = recommendation.get("warnings") or []
    caveat = (
        f'<div style="margin-top:28px;padding-top:14px;border-top:1px solid {GRID};'
        f'color:{INK_MUTED};font-size:12px;line-height:1.55">'
        f'<p style="margin:0 0 6px"><strong>Read this as a second opinion, not an instruction.</strong> '
        f'Across four backtested seasons the model averages the 70th percentile of managers — '
        f'above average, not expert.</p>'
        f'<p style="margin:0 0 6px">Sent 24h out, which is often <em>before</em> Friday press '
        f'conferences, so a named player may be ruled out after this was written. '
        f'Check team news before confirming.</p>'
        f'<p style="margin:0">Model {_esc(recommendation.get("model"))} · '
        f'{recommendation.get("n_features")} features · '
        f'{recommendation.get("training_rows", 0):,} training rows · '
        f'form through GW{recommendation.get("form_through_gw")}</p>'
        + ("".join(f'<p style="margin:6px 0 0;color:#d03b3b">{_esc(w)}</p>' for w in warnings))
        + '</div>'
    )
    blocks += [caveat, "</div>"]
    return "".join(blocks)
