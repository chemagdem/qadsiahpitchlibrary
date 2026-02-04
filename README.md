# qadsiahpitch - v0.1.1

A lightweight Python library for building football pitch maps in Plotly.
It focuses on the **canvas** (pitch + lines + grid + UI), while you add your own data traces.

What it includes:
- Pitch canvas (full / own half / opp half, horizontal / vertical).
- Configurable grids and frequency heatmaps.
- UI filters (dropdown / slider).
- Event markers as points or arrows (start â†’ end).
- BigQuery helpers for Impect/SkillCorner/StatsBomb pipelines.

---

## Installation

```bash
pip install git+https://github.com/chemagdem/qadsiahpitchlibrary.git
```

---

## Core concepts

- **Canvas only**: `build_canvas(...)` draws the field, lines, grids and UI.
- **You add data**: use `add_grid_heatmap(...)` and `add_event_markers(...)` with your own data.
- **Provider aware**: `provider` controls coordinate defaults.
- **Orientation aware**: vertical/horizontal switch with consistent geometry.
- **Half views**: `own half` and `opp half` crop the field correctly.

---

## Main payload parameters

### Provider
Defines coordinate system and default columns.

- `provider`: `"impect"`, `"skillcorner"`, `"statsbomb"`
  - Impect / SkillCorner: X [-52.5, 52.5], Y [-34, 34]
  - StatsBomb: X [0, 120], Y [0, 80]

### Pitch / layout

- `pitch`: `"full"`, `"own half"`, `"opp half"`
- `orientation`: `"horizontal"` or `"vertical"`

### Grid

- `grid` options:
  - `"none"` / `null`
  - `"5x3"`, `"3x3"`, `"5x5"`, `"20x20"`
  - `"set piece"`
  - `"wings"` (lane + third grid with box lines)
  - `"own third"`, `"middle third"`, `"final third"` (each third is split into 3 lanes)

Grid lines are always gray with 60% opacity (`rgba(120,120,120,0.6)`).

### Heatmap color

- `gridcolor` controls the heatmap colors. It does **not** depend on `against`.

Options:
- `"whitetoteal"` (default)
- `"whitetored"`
- Custom list of 2â€“5 colors (hex or rgb strings)

Examples:
```json
"gridcolor": "whitetoteal"
"gridcolor": ["#ffffff", "#fdae61", "#2c7bb6"]
"gridcolor": ["rgb(255,255,255)", "rgb(255,140,0)", "rgb(70,130,180)", "rgb(138,43,226)"]
```

### UI filters

- `filtertype`: `"dropdown"` or `"slider"`
  - Can also be a list: `["dropdown", "dropdown"]`
- `filtercontent`: BigQuery column(s) used by the filter UI
  - Example: `"playerName"` or `["playerName", "playerId"]`

### Marker type

- `markertype`: `"point"` or `"arrow"`
  - `"point"`: dot at event location
  - `"arrow"`: line `x/y or x_end/y_end` + end dot
- `markeralpha`: float (0/1) controlling marker opacity
- `markercolor`: hex/rgb string to control marker and arrow color

### Team selection

- `against`: used **only** to determine which team to analyze
  - `0` = same as `squadId`
  - `1` = opponent team
  - When `against=1`, the library flips X and Y so opponent events attack in the same direction and keep left/right consistency.

---

## Attack direction arrow

The canvas always draws an attack direction arrow:
- **Vertical** pitch: arrow points **up** on the right side.
- **Horizontal** pitch: arrow points **left**, below the field.

---

## Core API

### `build_canvas(provider, pitch, grid, orientation, filtercontent, filtertype)`
Builds the pitch, axes, lines, and grid lines only.
Returns a `plotly.graph_objects.Figure`.

### `add_grid_heatmap(fig, x_vals, y_vals, pitch, grid, orientation, gridcolor=None, opacity=0.7)`
Paints the grid by event frequency (heatmap cells drawn **under** pitch lines).
Returns: `{"vmax": ..., "nonzero": ...}`.

### `add_event_markers(fig, df, orientation, markertype="point", markeralpha=None, ...)`
Draws events:
- `point`: dot at `x/y`.
- `arrow`: line `x/y â†’ x_end/y_end` + end dot.

Requires `x`, `y` and optionally `x_end`, `y_end`.

---

## BigQuery helpers

### `parse_list(raw)`
Converts anything into a list:
- `"a,b" â†’ ["a","b"]`, `["a"] â†’ ["a"]`, `None â†’ []`

### `parse_metric_filters(raw_metric)`
Supports the syntax:
```
FROM <column> import <value1, value2>
```
Example:
```
["SHOT_XG", "FROM actionType import SHOT", "FROM result import SUCCESS"]
```
Returns `(metrics, filters)`.

### `resolve_match_ids(client, request_json, team_id, match_data_table, default_iteration_id=1469)`
Resolves match IDs:
- Uses `matchIds`/`matchId` if provided.
- Uses `matchweeks` with `iterationId` when present.
- Otherwise returns all matches for the team.

### `resolve_analysis_team_ids(client, match_ids, squad_id, opponent_id, against, match_info_table)`
Determines which team to analyze.

### `resolve_coord_columns(provider)`
Maps start/end coordinate columns by provider:
- Impect: `startAdjCoordinatesX/Y`, `endAdjCoordinatesX/Y`
- SkillCorner: `x_start/y_start`, `x_end/y_end`
- StatsBomb: `x/y`, `end_x/end_y`

---

## Providers

### `fetch_events_impect(client, events_table, match_ids, squad_ids, metrics, filters=None, coord_columns=None)`
Queries BigQuery from `Impect.events`:
- Returns `x`, `y`, `x_end`, `y_end`, `playerName`, `playerId`, `gameTimeInSec`.
- Accepts column filters (`eventType`, `event`, etc.).
- Accepts numeric `metrics` to bring extra columns.

---

## Minimal example (canvas only)

```python
from qadsiahpitch.plot import build_canvas

fig = build_canvas(
    provider="impect",
    pitch="full",
    grid="5x3",
    orientation="horizontal",
    filtercontent=["playerName"],
    filtertype=["dropdown"],
)
```

---

## Full example (with heatmap + markers)

```python
from qadsiahpitch.plot import build_canvas, add_grid_heatmap, add_event_markers

fig = build_canvas(
    provider="impect",
    pitch="full",
    grid="5x3",
    orientation="horizontal",
    filtercontent=["playerName"],
    filtertype=["dropdown"],
)

add_grid_heatmap(
    fig,
    df["x"].values,
    df["y"].values,
    pitch="full",
    grid="5x3",
    orientation="horizontal",
    gridcolor=["#ffffff", "#fdae61", "#2c7bb6"],
)

add_event_markers(fig, df, orientation="horizontal", markertype="arrow", markeralpha=0.6)
```

---

## Minimal Use Example (all payload options)

This is a compact, provider-agnostic Impect example that includes **every payload parameter** from this README (provider, pitch, orientation, grid, gridcolor, filtertype/content, squadId/opponentId, matchIds/matchweeks/iterationId, against, metric, eventType/event, markertype/markeralpha/markercolor), plus dropdown filtering for heatmaps and markers.

See the example script in this repo:

```
examples/minimal_use_example.py
```

Run it locally to generate the HTML output.

---

## Custom Plotly layer example

Add your own traces on top of the pitch:

```python
import plotly.graph_objects as go
from qadsiahpitch.plot import build_canvas

fig = build_canvas(
    provider="impect",
    pitch="full",
    grid="5x5",
    orientation="vertical",
    filtercontent=["playerName"],
    filtertype=["dropdown"],
)

# Example: custom markers
fig.add_trace(go.Scatter(
    x=[0, 10, -15],
    y=[20, -5, 30],
    mode="markers",
    marker=dict(size=8, color="#ffcc00"),
    name="Custom points",
))

fig.write_html("custom_layer_example.html", include_plotlyjs="cdn")
```

---

## Example payload (test)

```json
{
  "provider": "impect",
  "pitch": "full",
  "orientation": "vertical",
  "grid": "5x3",
  "gridcolor": "whitetoteal",
  "filtertype": "dropdown",
  "filtercontent": "playerName",
  "squadId": 5067,
  "matchIds": 228025,
  "against": 0,
  "metric": ["FROM actionType import PASS", "FROM result import SUCCESS"],
  "markertype": "arrow",
  "markeralpha": 0.6
}
```
