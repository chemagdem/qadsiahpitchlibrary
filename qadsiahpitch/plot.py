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


def _normalize_pitch(pitch: str) -> str:
    key = (pitch or "full").strip().lower()
    key = key.replace("_", " ").replace("-", " ")
    key = " ".join(key.split())
    if key in ("full",):
        return "full"
    if key in ("half", "own half"):
        return "own half"
    if key in ("opp half", "opponent half"):
        return "opp half"
    return key


def _pitch_x_range(pitch: str) -> Tuple[float, float]:
    pitch = _normalize_pitch(pitch)
    if pitch == "full":
        return X_MIN, X_MAX
    if pitch == "own half":
        return X_MIN, 0.0
    if pitch == "opp half":
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
    def _swap(x, y):
        return (x, y) if orientation == "vertical" else (y, x)
    for y_val in x_edges:
        (x0, y0) = _swap(y_min, y_val)
        (x1, y1) = _swap(y_max, y_val)
        xs, ys = [x0, x1], [y0, y1]
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=line_color, width=0.8),
            hoverinfo="skip",
            showlegend=False,
        ))
    for x_val in y_edges:
        (x0, y0) = _swap(x_val, x_min)
        (x1, y1) = _swap(x_val, x_max)
        xs, ys = [x0, x1], [y0, y1]
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=line_color, width=0.8),
            hoverinfo="skip",
            showlegend=False,
        ))


def _set_piece_zones():
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
    zones = list({z for z in zones_right})
    return zones


def _draw_set_piece_grid(fig, orientation):
    zones = _set_piece_zones()
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


def _grid_edges_for_option(grid: str, pitch_key: str):
    grid_key = str(grid).lower().strip()
    if grid_key in ("none", "null", ""):
        return None, None
    if grid_key == "set piece":
        return None, None
    x_min, x_max = _pitch_x_range(pitch_key)
    x_bins, y_bins = _grid_bins(grid_key)
    if x_bins > 0 and y_bins > 0:
        x_edges = np.linspace(x_min, x_max, x_bins + 1)
        y_edges = np.linspace(Y_MIN, Y_MAX, y_bins + 1)
        return x_edges, y_edges
    if grid_key in ("own third", "own_third", "own-third",
                    "middle third", "middle_third", "middle-third",
                    "final third", "final_third", "final-third"):
        third = (X_MAX - X_MIN) / 3.0
        if grid_key.startswith("own"):
            start, end = X_MIN, X_MIN + third
        elif grid_key.startswith("middle"):
            start, end = X_MIN + third, X_MIN + 2 * third
        else:
            start, end = X_MIN + 2 * third, X_MAX
        start = max(start, x_min)
        end = min(end, x_max)
        if end <= start:
            return None, None
        lane = (Y_MAX - Y_MIN) / 3.0
        x_edges = np.array([start, end])
        y_edges = np.array([Y_MIN, Y_MIN + lane, Y_MIN + 2 * lane, Y_MAX])
        return x_edges, y_edges
    return None, None


def add_grid_heatmap(
    fig: go.Figure,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    pitch: str,
    grid: str,
    orientation: str,
    against: int = 0,
    opacity: float = 0.7,
):
    pitch_key = _normalize_pitch(pitch)
    grid_key = str(grid).lower().strip() if grid is not None else ""
    if grid_key == "set piece":
        zones = _set_piece_zones()
        if not zones:
            return {"vmax": 0, "nonzero": 0}
        counts = np.zeros(len(zones))
        for x, y in zip(x_vals, y_vals):
            if x is None or y is None:
                continue
            for idx, (x0, x1, y0, y1) in enumerate(zones):
                if x0 <= x <= x1 and y0 <= y <= y1:
                    counts[idx] += 1
                    break
        vmax = counts.max() if counts.max() > 0 else 1
        if against == 0:
            colorscale = [[0.0, "#ffffff"], [1.0, "#AA2D3A"]]
        else:
            colorscale = [[0.0, "#ffffff"], [0.5, "#b2d8d8"], [1.0, "#004c4c"]]
        min_value = vmax * 0.25
        for idx, (x0, x1, y0, y1) in enumerate(zones):
            if counts[idx] <= 0:
                continue
            if orientation == "vertical":
                xs = [y0, y1, y1, y0, y0]
                ys = [x0, x0, x1, x1, x0]
            else:
                xs = [x0, x1, x1, x0, x0]
                ys = [y0, y0, y1, y1, y0]
            color = _interp_color(colorscale, max(counts[idx], min_value), vmax)
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                fill="toself",
                mode="lines",
                line=dict(width=0),
                fillcolor=color,
                opacity=opacity,
                hoverinfo="skip",
                showlegend=False,
            ))
        _draw_set_piece_grid(fig, orientation)
        return {"vmax": float(vmax), "nonzero": int((counts > 0).sum())}

    x_edges, y_edges = _grid_edges_for_option(grid, pitch_key)
    if x_edges is None or y_edges is None:
        return {"vmax": 0, "nonzero": 0}

    counts = np.zeros((len(x_edges) - 1, len(y_edges) - 1))
    for x, y in zip(x_vals, y_vals):
        if x is None or y is None:
            continue
        if x < x_edges[0] or x > x_edges[-1]:
            continue
        if y < y_edges[0] or y > y_edges[-1]:
            continue
        xi = np.digitize(x, x_edges) - 1
        yi = np.digitize(y, y_edges) - 1
        if 0 <= xi < counts.shape[0] and 0 <= yi < counts.shape[1]:
            counts[xi, yi] += 1

    vmax = counts.max() if counts.max() > 0 else 1
    if against == 0:
        colorscale = [[0.0, "#ffffff"], [1.0, "#AA2D3A"]]
    else:
        colorscale = [[0.0, "#ffffff"], [0.5, "#b2d8d8"], [1.0, "#004c4c"]]

    min_value = vmax * 0.25
    for i in range(len(x_edges) - 1):
        for j in range(len(y_edges) - 1):
            count = counts[i, j]
            if count <= 0:
                continue
            x0, x1 = x_edges[i], x_edges[i + 1]
            y0, y1 = y_edges[j], y_edges[j + 1]
            if orientation == "vertical":
                xs = [y0, y1, y1, y0, y0]
                ys = [x0, x0, x1, x1, x0]
            else:
                xs = [x0, x1, x1, x0, x0]
                ys = [y0, y0, y1, y1, y0]
            color = _interp_color(colorscale, max(count, min_value), vmax)
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                fill="toself",
                mode="lines",
                line=dict(width=0),
                fillcolor=color,
                opacity=opacity,
                hoverinfo="skip",
                showlegend=False,
            ))

    _draw_grid_lines(
        fig,
        x_edges[0],
        x_edges[-1],
        Y_MIN,
        Y_MAX,
        orientation,
        x_edges,
        y_edges,
    )
    return {"vmax": float(vmax), "nonzero": int((counts > 0).sum())}


def add_event_markers(
    fig: go.Figure,
    df: pd.DataFrame,
    orientation: str,
    markertype: str = "point",
    line_color: str = "#4a4a4a",
    line_width: float = 1.5,
    marker_color: str = "#4a4a4a",
    marker_size: int = 7,
    opacity: float = 0.7,
):
    if df.empty:
        return
    markertype = (markertype or "point").strip().lower()

    def _map_xy(x, y):
        if orientation == "vertical":
            return y, x
        return x, y

    if markertype == "arrow":
        if "x_end" not in df.columns or "y_end" not in df.columns:
            return
        xs = []
        ys = []
        for _, row in df.iterrows():
            x0, y0 = _map_xy(row["x"], row["y"])
            x1, y1 = _map_xy(row["x_end"], row["y_end"])
            xs.extend([x0, x1, None])
            ys.extend([y0, y1, None])
        fig.add_scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=line_color, width=line_width),
            opacity=opacity,
            hoverinfo="skip",
            showlegend=False,
            name="event-arrow",
        )
        fig.add_scatter(
            x=[_map_xy(x, y)[0] for x, y in zip(df["x_end"], df["y_end"])],
            y=[_map_xy(x, y)[1] for x, y in zip(df["x_end"], df["y_end"])],
            mode="markers",
            marker=dict(size=max(3, int(marker_size * 0.7)), color=marker_color),
            opacity=opacity,
            hoverinfo="skip",
            showlegend=False,
            name="event-dot",
        )
        return

    fig.add_scatter(
        x=[_map_xy(x, y)[0] for x, y in zip(df["x"], df["y"])],
        y=[_map_xy(x, y)[1] for x, y in zip(df["x"], df["y"])],
        mode="markers",
        marker=dict(size=marker_size, color=marker_color, line=dict(color="black", width=0.5), opacity=opacity),
        hoverinfo="skip",
        showlegend=False,
        name="event-point",
    )

def _build_full_pitch_line_traces(orientation="vertical"):
    traces = []
    line_color = "black"

    def _swap(x, y):
        return (x, y) if orientation == "vertical" else (y, x)

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

    area_w = 40.32
    area_d = 16.5
    small_w = 18.32
    small_d = 5.5

    # Goal lines (small line behind each end line)
    gx, gy = _swap(-3.66, X_MIN)
    gx2, gy2 = _swap(3.66, X_MIN)
    traces.append(go.Scatter(x=[gx, gx2], y=[gy, gy2], mode="lines", line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))
    gx, gy = _swap(-3.66, X_MAX)
    gx2, gy2 = _swap(3.66, X_MAX)
    traces.append(go.Scatter(x=[gx, gx2], y=[gy, gy2], mode="lines", line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))

    # Bottom
    rect = [(-area_w / 2, X_MIN), (area_w / 2, X_MIN), (area_w / 2, X_MIN + area_d),
            (-area_w / 2, X_MIN + area_d), (-area_w / 2, X_MIN)]
    xs, ys = zip(*[_swap(x, y) for x, y in rect])
    traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))
    rect = [(-small_w / 2, X_MIN), (small_w / 2, X_MIN), (small_w / 2, X_MIN + small_d),
            (-small_w / 2, X_MIN + small_d), (-small_w / 2, X_MIN)]
    xs, ys = zip(*[_swap(x, y) for x, y in rect])
    traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))
    px, py = _swap(0, X_MIN + 11.0)
    traces.append(go.Scatter(x=[px], y=[py], mode="markers", marker=dict(size=4, color="black"), hoverinfo="skip", showlegend=False))

    # Top
    rect = [(-area_w / 2, X_MAX - area_d), (area_w / 2, X_MAX - area_d), (area_w / 2, X_MAX),
            (-area_w / 2, X_MAX), (-area_w / 2, X_MAX - area_d)]
    xs, ys = zip(*[_swap(x, y) for x, y in rect])
    traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))
    rect = [(-small_w / 2, X_MAX - small_d), (small_w / 2, X_MAX - small_d), (small_w / 2, X_MAX),
            (-small_w / 2, X_MAX), (-small_w / 2, X_MAX - small_d)]
    xs, ys = zip(*[_swap(x, y) for x, y in rect])
    traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))
    px, py = _swap(0, X_MAX - 11.0)
    traces.append(go.Scatter(x=[px], y=[py], mode="markers", marker=dict(size=4, color="black"), hoverinfo="skip", showlegend=False))

    return traces


def _build_half_pitch_line_traces_defensive(extend_center=6.0, orientation="vertical"):
    traces = []
    line_color = "black"
    def _swap(x, y):
        return (x, y) if orientation == "vertical" else (y, x)
    outline = [(Y_MIN, X_MIN), (Y_MAX, X_MIN), (Y_MAX, 0), (Y_MIN, 0), (Y_MIN, X_MIN)]
    xs, ys = zip(*[_swap(x, y) for x, y in outline])
    traces.append(go.Scatter(x=xs, y=ys, mode="lines",
                             line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))
    xs, ys = zip(*[_swap(x, y) for x, y in [(Y_MIN, 0), (Y_MAX, 0)]])
    traces.append(go.Scatter(x=list(xs), y=list(ys), mode="lines",
                             line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))
    if extend_center > 0:
        xs, ys = zip(*[_swap(x, y) for x, y in [(Y_MIN, 0), (Y_MIN, extend_center)]])
        traces.append(go.Scatter(x=list(xs), y=list(ys), mode="lines",
                                 line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))
        xs, ys = zip(*[_swap(x, y) for x, y in [(Y_MAX, 0), (Y_MAX, extend_center)]])
        traces.append(go.Scatter(x=list(xs), y=list(ys), mode="lines",
                                 line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))
    rect = [(-20.16, X_MIN), (20.16, X_MIN), (20.16, X_MIN + 16.5), (-20.16, X_MIN + 16.5), (-20.16, X_MIN)]
    xs, ys = zip(*[_swap(x, y) for x, y in rect])
    traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=line_color, width=1),
                             hoverinfo="skip", showlegend=False))
    rect = [(-9.16, X_MIN), (9.16, X_MIN), (9.16, X_MIN + 5.5), (-9.16, X_MIN + 5.5), (-9.16, X_MIN)]
    xs, ys = zip(*[_swap(x, y) for x, y in rect])
    traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=line_color, width=1),
                             hoverinfo="skip", showlegend=False))
    px, py = _swap(0, X_MIN + 11.0)
    traces.append(go.Scatter(x=[px], y=[py], mode="markers",
                             marker=dict(size=4, color="black"), hoverinfo="skip", showlegend=False))
    gx, gy = _swap(-3.66, X_MIN - 0.2)
    gx2, gy2 = _swap(3.66, X_MIN - 0.2)
    traces.append(go.Scatter(x=[gx, gx2], y=[gy, gy2], mode="lines",
                             line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))
    circle_t = np.linspace(0, 2 * np.pi, 200)
    cx = np.cos(circle_t) * 9.15
    cy = np.sin(circle_t) * 9.15
    cx2, cy2 = _swap(cx, cy)
    traces.append(go.Scatter(x=cx2, y=cy2, mode="lines",
                             line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))
    sx, sy = _swap(0, 0)
    traces.append(go.Scatter(x=[sx], y=[sy], mode="markers",
                             marker=dict(size=4, color="black"), hoverinfo="skip", showlegend=False))
    return traces


def _build_half_pitch_line_traces_attacking(extend_center=6.0, orientation="vertical"):
    traces = []
    line_color = "black"
    def _swap(x, y):
        return (x, y) if orientation == "vertical" else (y, x)
    outline = [(Y_MIN, 0), (Y_MAX, 0), (Y_MAX, X_MAX), (Y_MIN, X_MAX), (Y_MIN, 0)]
    xs, ys = zip(*[_swap(x, y) for x, y in outline])
    traces.append(go.Scatter(x=xs, y=ys, mode="lines",
                             line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))
    xs, ys = zip(*[_swap(x, y) for x, y in [(Y_MIN, 0), (Y_MAX, 0)]])
    traces.append(go.Scatter(x=list(xs), y=list(ys), mode="lines",
                             line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))
    if extend_center > 0:
        xs, ys = zip(*[_swap(x, y) for x, y in [(Y_MIN, 0), (Y_MIN, -extend_center)]])
        traces.append(go.Scatter(x=list(xs), y=list(ys), mode="lines",
                                 line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))
        xs, ys = zip(*[_swap(x, y) for x, y in [(Y_MAX, 0), (Y_MAX, -extend_center)]])
        traces.append(go.Scatter(x=list(xs), y=list(ys), mode="lines",
                                 line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))
    rect = [(-20.16, X_MAX - 16.5), (20.16, X_MAX - 16.5), (20.16, X_MAX), (-20.16, X_MAX), (-20.16, X_MAX - 16.5)]
    xs, ys = zip(*[_swap(x, y) for x, y in rect])
    traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=line_color, width=1),
                             hoverinfo="skip", showlegend=False))
    rect = [(-9.16, X_MAX - 5.5), (9.16, X_MAX - 5.5), (9.16, X_MAX), (-9.16, X_MAX), (-9.16, X_MAX - 5.5)]
    xs, ys = zip(*[_swap(x, y) for x, y in rect])
    traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=line_color, width=1),
                             hoverinfo="skip", showlegend=False))
    px, py = _swap(0, X_MAX - 11.0)
    traces.append(go.Scatter(x=[px], y=[py], mode="markers",
                             marker=dict(size=4, color="black"), hoverinfo="skip", showlegend=False))
    gx, gy = _swap(-3.66, X_MAX + 0.2)
    gx2, gy2 = _swap(3.66, X_MAX + 0.2)
    traces.append(go.Scatter(x=[gx, gx2], y=[gy, gy2], mode="lines",
                             line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))
    circle_t = np.linspace(0, 2 * np.pi, 200)
    cx = np.cos(circle_t) * 9.15
    cy = np.sin(circle_t) * 9.15
    cx2, cy2 = _swap(cx, cy)
    traces.append(go.Scatter(x=cx2, y=cy2, mode="lines",
                             line=dict(color=line_color, width=1), hoverinfo="skip", showlegend=False))
    sx, sy = _swap(0, 0)
    traces.append(go.Scatter(x=[sx], y=[sy], mode="markers",
                             marker=dict(size=4, color="black"), hoverinfo="skip", showlegend=False))
    return traces


def build_canvas(
    provider: str,
    pitch: str,
    grid: str,
    orientation: str,
    filtercontent: List[str],
    filtertype: List[str],
) -> go.Figure:
    fig = go.Figure()

    orientation = (orientation or "vertical").strip().lower()
    # Impect and SkillCorner share the same pitch dimensions; provider kept for compatibility.
    _ = (provider or "impect").strip().lower()
    pitch_key = _normalize_pitch(pitch)
    x_min, x_max = _pitch_x_range(pitch_key)
    x_bins, y_bins = _grid_bins(grid)

    base_traces = []
    if pitch_key == "full":
        base_traces.extend(_build_full_pitch_line_traces(orientation=orientation))
    elif pitch_key == "opp half" and orientation == "vertical":
        # Build from defensive half and flip to keep goal at the top
        base_traces.extend(_build_half_pitch_line_traces_defensive(extend_center=6.0, orientation=orientation))
        for tr in base_traces:
            if getattr(tr, "y", None) is not None:
                tr.y = [(-v if v is not None else None) for v in tr.y]
    elif pitch_key == "opp half":
        base_traces.extend(_build_half_pitch_line_traces_attacking(extend_center=6.0, orientation=orientation))
    else:
        base_traces.extend(_build_half_pitch_line_traces_defensive(extend_center=6.0, orientation=orientation))

    for trace in base_traces:
        fig.add_trace(trace)

    # Grid (if requested)
    grid_key = str(grid).lower().strip() if grid is not None else ""
    if grid_key and grid_key not in ("none", "null"):
        if grid_key == "set piece":
            _draw_set_piece_grid(fig, orientation)
        elif grid_key in (
            "own third", "own_third", "own-third",
            "middle third", "middle_third", "middle-third",
            "final third", "final_third", "final-third",
        ):
            third = (X_MAX - X_MIN) / 3.0
            if grid_key.startswith("own"):
                start, end = X_MIN, X_MIN + third
            elif grid_key.startswith("middle"):
                start, end = X_MIN + third, X_MIN + 2 * third
            else:
                start, end = X_MIN + 2 * third, X_MAX
            # Clip to current pitch window (full/half)
            start = max(start, x_min)
            end = min(end, x_max)
            if end > start:
                lane = (Y_MAX - Y_MIN) / 3.0
                y_edges = [Y_MIN, Y_MIN + lane, Y_MIN + 2 * lane, Y_MAX]
                _draw_grid_lines(
                    fig,
                    start,
                    end,
                    Y_MIN,
                    Y_MAX,
                    orientation,
                    [start, end],
                    y_edges,
                )
        else:
            if x_bins > 0 and y_bins > 0:
                x_edges = np.linspace(x_min, x_max, x_bins + 1)
                y_edges = np.linspace(Y_MIN, Y_MAX, y_bins + 1)
                _draw_grid_lines(fig, x_min, x_max, Y_MIN, Y_MAX, orientation, x_edges, y_edges)

    filtercontent = filtercontent or []
    filtertype = filtertype or []
    if not isinstance(filtercontent, list):
        filtercontent = [filtercontent]
    if not isinstance(filtertype, list):
        filtertype = [filtertype]
    filtertype = filtertype or ["dropdown"]

    values = ["All"]
    buttons = [dict(label="All", method="update", args=[{"visible": [True] * len(fig.data)}])]

    is_full = pitch_key == "full"
    is_horizontal = orientation == "horizontal"
    fig.update_layout(
        width=700 if is_horizontal else 520,
        height=520 if is_horizontal else (700 if is_full else 520),
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
        if pitch_key == "full":
            fig.update_yaxes(range=[-60.0, 60.0], autorange=False, visible=False, fixedrange=True, scaleanchor="x", scaleratio=1)
        elif pitch_key == "own half":
            fig.update_yaxes(range=[X_MIN, 5.0], autorange=False, visible=False, fixedrange=True, scaleanchor="x", scaleratio=1)
        else:  # opp half
            fig.update_yaxes(range=[0.0, X_MAX], autorange=False, visible=False, fixedrange=True, scaleanchor="x", scaleratio=1)
    else:
        if pitch_key == "full":
            fig.update_xaxes(range=[60.0, -60.0], autorange=False, visible=False, fixedrange=True)
        elif pitch_key == "own half":
            fig.update_xaxes(range=[5.0, X_MIN], autorange=False, visible=False, fixedrange=True)
        else:  # opp half
            fig.update_xaxes(range=[X_MAX, 0.0], autorange=False, visible=False, fixedrange=True)
        fig.update_yaxes(range=[Y_MAX, Y_MIN], autorange=False, visible=False, fixedrange=True, scaleanchor="x", scaleratio=1)

    return fig
