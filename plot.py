import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Impect pitch defaults
X_MIN, X_MAX = -52.5, 52.5
Y_MIN, Y_MAX = -34.0, 34.0


def _sec_to_match_minute(game_time_sec: float) -> float:
    if pd.isna(game_time_sec):
        return np.nan
    if game_time_sec >= 10000:
        return (game_time_sec - 10000) / 60.0
    return game_time_sec / 60.0


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _interp_color(colorscale, value, vmax):
    if vmax <= 0:
        return colorscale[0][1]
    t = max(0.0, min(1.0, value / vmax))
    for i in range(len(colorscale) - 1):
        t0, c0 = colorscale[i]
        t1, c1 = colorscale[i + 1]
        if t0 <= t <= t1:
            if t1 == t0:
                return c1
            local_t = (t - t0) / (t1 - t0)
            r0, g0, b0 = _hex_to_rgb(c0)
            r1, g1, b1 = _hex_to_rgb(c1)
            r = int(r0 + (r1 - r0) * local_t)
            g = int(g0 + (g1 - g0) * local_t)
            b = int(b0 + (b1 - b0) * local_t)
            return f"#{r:02x}{g:02x}{b:02x}"
    return colorscale[-1][1]


def _add_colorbar(fig, colorscale):
    gradient = np.linspace(0, 1, 200)
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                size=0.1,
                color=gradient,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    len=0.9,
                    y=0.5,
                    x=1.02,
                    thickness=12,
                    outlinewidth=0,
                    tickvals=[],
                    ticktext=[]
                )
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )


def _pitch_x_range(pitch: str) -> Tuple[float, float]:
    pitch = (pitch or "full").strip().lower()
    if pitch == "full":
        return X_MIN, X_MAX
    if pitch in ("half", "own half", "own_half", "own-half"):
        return X_MIN, 0.0
    if pitch in ("opp half", "opponent half", "opp_half", "opp-half"):
        return 0.0, X_MAX
    return X_MIN, X_MAX


def _grid_bins(grid: str) -> Tuple[int, int]:
    if not grid:
        return 0, 0
    grid = str(grid).lower().strip()
    if grid in ("none", "null"):
        return 0, 0
    if grid == "set piece":
        return 0, 0
    m = re.match(r"^(\d+)\s*x\s*(\d+)$", grid)
    if not m:
        return 0, 0
    return int(m.group(1)), int(m.group(2))


def _draw_grid_lines(fig, x_min, x_max, y_min, y_max, orientation, x_edges, y_edges):
    line_color = "rgba(120,120,120,0.8)"
    for y_val in x_edges:
        if orientation == "vertical":
            xs, ys = [y_min, y_max], [y_val, y_val]
        else:
            xs, ys = [y_val, y_val], [y_min, y_max]
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=line_color, width=0.8),
            hoverinfo="skip",
            showlegend=False,
        ))
    for x_val in y_edges:
        if orientation == "vertical":
            xs, ys = [x_val, x_val], [x_min, x_max]
        else:
            xs, ys = [x_min, x_max], [x_val, x_val]
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=line_color, width=0.8),
            hoverinfo="skip",
            showlegend=False,
        ))


def _draw_set_piece_grid(fig, orientation):
    # Zones for corners / throw-ins (Impect coordinates)
    zones_right = [
        (27.0, 52.5, -34.00, -20.16),
        (36.0, 52.5, -20.16, -9.16),
        (36.0, 41.5, -9.16, 9.16),
        (36.0, 52.5, 9.16, 20.16),
        (27.0, 52.5, 20.16, 34.00),
        (41.5, 47.0, -9.16, -3.05),
        (47.0, 52.5, -9.16, -3.05),
        (41.5, 47.0, -3.05, 3.05),
        (47.0, 52.5, -3.05, 3.05),
        (41.5, 47.0, 3.05, 9.16),
        (47.0, 52.5, 3.05, 9.16),
        (27.0, 36.0, -20.16, 20.16),
    ]
    zones_left = [
        (27.0, 52.5, -34.00, -20.16),
        (36.0, 52.5, -20.16, -9.16),
        (36.0, 41.5, -9.16, 9.16),
        (36.0, 52.5, 9.16, 20.16),
        (27.0, 52.5, 20.16, 34.00),
        (41.5, 47.0, -9.16, -3.05),
        (47.0, 52.5, -9.16, -3.05),
        (41.5, 47.0, -3.05, 3.05),
        (47.0, 52.5, -3.05, 3.05),
        (41.5, 47.0, 3.05, 9.16),
        (47.0, 52.5, 3.05, 9.16),
        (27.0, 36.0, -20.16, 20.16),
    ]
    zones = list({z for z in (zones_right + zones_left)})
    line_color = "rgba(120,120,120,0.8)"
    for x0, x1, y0, y1 in zones:
        if orientation == "vertical":
            xs, ys = zip(*[(y0, x0), (y1, x0), (y1, x1), (y0, x1), (y0, x0)])
        else:
            xs, ys = zip(*[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=line_color, width=0.8),
            hoverinfo="skip",
            showlegend=False,
        ))


def _build_full_pitch_line_traces(orientation="vertical"):
    traces = []
    line_color = "black"

    def _swap(x, y):
        return (y, x) if orientation == "vertical" else (x, y)

    outline = [(Y_MIN, X_MIN), (Y_MAX, X_MIN), (Y_MAX, X_MAX), (Y_MIN, X_MAX), (Y_MIN, X_MIN)]
    xs, ys = zip(*[_swap(x, y) for x, y in outline])
    traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))

    xh, yh = _swap(Y_MIN, 0)
    xh2, yh2 = _swap(Y_MAX, 0)
    traces.append(go.Scatter(x=[xh, xh2], y=[yh, yh2], mode="lines", line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))

    circle_t = np.linspace(0, 2 * np.pi, 200)
    cx = np.cos(circle_t) * 9.15
    cy = np.sin(circle_t) * 9.15
    cx2, cy2 = _swap(cx, cy)
    traces.append(go.Scatter(x=cx2, y=cy2, mode="lines", line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))
    sx, sy = _swap(0, 0)
    traces.append(go.Scatter(x=[sx], y=[sy], mode="markers", marker=dict(size=4, color="black"), hoverinfo="skip", showlegend=False))

    return traces


def _build_half_pitch_line_traces(x_min: float, x_max: float, orientation="vertical"):
    traces = []
    line_color = "black"

    def _swap(x, y):
        return (y, x) if orientation == "vertical" else (x, y)

    outline = [(Y_MIN, x_min), (Y_MAX, x_min), (Y_MAX, x_max), (Y_MIN, x_max), (Y_MIN, x_min)]
    xs, ys = zip(*[_swap(x, y) for x, y in outline])
    traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))

    if abs(x_max) < 1e-6 or abs(x_min) < 1e-6:
        xh, yh = _swap(Y_MIN, 0)
        xh2, yh2 = _swap(Y_MAX, 0)
        traces.append(go.Scatter(x=[xh, xh2], y=[yh, yh2], mode="lines", line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))

    circle_t = np.linspace(0, 2 * np.pi, 200)
    cx = np.cos(circle_t) * 9.15
    cy = np.sin(circle_t) * 9.15
    cx2, cy2 = _swap(cx, cy)
    traces.append(go.Scatter(x=cx2, y=cy2, mode="lines", line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))
    sx, sy = _swap(0, 0)
    traces.append(go.Scatter(x=[sx], y=[sy], mode="markers", marker=dict(size=4, color="black"), hoverinfo="skip", showlegend=False))

    return traces


def build_canvas(
    pitch: str,
    grid: str,
    orientation: str,
    filtercontent: List[str],
    filtertype: List[str],
) -> go.Figure:
    fig = go.Figure()

    orientation = (orientation or "vertical").strip().lower()
    x_min, x_max = _pitch_x_range(pitch)
    x_bins, y_bins = _grid_bins(grid)

    base_traces = []
    if pitch == "full":
        base_traces.extend(_build_full_pitch_line_traces(orientation=orientation))
    else:
        base_traces.extend(_build_half_pitch_line_traces(x_min, x_max, orientation=orientation))

    for trace in base_traces:
        fig.add_trace(trace)

    grid_key = str(grid).lower().strip() if grid is not None else ""
    if x_bins > 0 and y_bins > 0:
        x_edges = np.linspace(x_min, x_max, x_bins + 1)
        y_edges = np.linspace(Y_MIN, Y_MAX, y_bins + 1)
        _draw_grid_lines(fig, x_min, x_max, Y_MIN, Y_MAX, orientation, x_edges, y_edges)
    elif grid_key in ("own third", "own_third", "own-third", "middle third", "middle_third", "middle-third", "final third", "final_third", "final-third"):
        third = (X_MAX - X_MIN) / 3.0
        x_edges = [X_MIN + third, X_MIN + 2 * third]
        _draw_grid_lines(fig, x_min, x_max, Y_MIN, Y_MAX, orientation, x_edges, [])
    elif grid_key == "set piece":
        _draw_set_piece_grid(fig, orientation)

    filtercontent = filtercontent or []
    filtertype = filtertype or []
    if not isinstance(filtercontent, list):
        filtercontent = [filtercontent]
    if not isinstance(filtertype, list):
        filtertype = [filtertype]
    filtertype = filtertype or ["dropdown"]

    values = ["All"]
    buttons = [dict(label="All", method="update", args=[{"visible": [True] * len(fig.data)}])]

    fig.update_layout(
        width=520,
        height=520,
        margin=dict(l=10, r=10, t=40, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        hoverlabel=dict(namelength=-1),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.5,
                y=1.05,
                xanchor="center",
                yanchor="bottom",
                buttons=buttons,
            )
        ] if (filtertype and filtertype[0] == "dropdown") else [],
    )

    if filtertype and filtertype[0] == "slider":
        fig.update_layout(
            sliders=[
                dict(
                    active=0,
                    x=0.08,
                    y=1.05,
                    len=0.84,
                    pad=dict(t=10, b=0),
                    steps=[
                        dict(label=str(val), method="update", args=[{"visible": [True] * len(fig.data)}])
                        for val in values
                    ],
                )
            ]
        )

    if orientation == "vertical":
        fig.update_xaxes(range=[Y_MAX, Y_MIN], autorange=False, visible=False, fixedrange=True)
        fig.update_yaxes(range=[x_min, x_max], autorange=False, visible=False, fixedrange=True, scaleanchor="x", scaleratio=1)
    else:
        fig.update_xaxes(range=[x_min, x_max], autorange=False, visible=False, fixedrange=True)
        fig.update_yaxes(range=[Y_MIN, Y_MAX], autorange=False, visible=False, fixedrange=True, scaleanchor="x", scaleratio=1)

    return fig
