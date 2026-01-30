# qadsiahpitch

Lightweight library for building Plotly football pitch maps in an agnostic way.  
It includes:

- Pitch canvas (full/own half/opp half, horizontal/vertical).
- Configurable grid and frequency heatmap.
- UI filters (dropdown/slider).
- BigQuery helpers (match/team resolution and event fetch).
- Event markers as points or arrows (start → end).

The library **does not impose business logic**. It only gives you a fast, consistent base to draw from.

---

## Installation

```bash
pip install git+https://github.com/chemagdem/qadsiahpitchlibrary.git
```

---

## Main payload parameters

### Provider
Defines the coordinate system and default columns.

- `provider`: `"impect"`, `"skillcorner"`, `"statsbomb"`
  - Impect / SkillCorner: X [-52.5, 52.5], Y [-34, 34]
  - StatsBomb: X [0, 120], Y [0, 80]

### Pitch / layout

- `pitch`: `"full"`, `"own half"`, `"opp half"`
- `orientation`: `"horizontal"` or `"vertical"`

### Grid

- `grid`:  
  - `"none"` / `null`
  - `"5x3"`, `"3x3"`, `"5x5"`, `"20x20"`
  - `"set piece"`
  - `"own third"`, `"middle third"`, `"final third"`  
    (each third is split into **3 lanes**)

> Grid lines are always gray with 80% opacity (`rgba(120,120,120,0.8)`).

### UI Filters

- `filtertype`: `"dropdown"` or `"slider"`  
  Also accepts a list, e.g. `["dropdown", "dropdown"]`.
- `filtercontent`: BigQuery column(s) used by the filter UI  
  Example: `"playerName"` or `["playerName", "playerId"]`.

### Marker type

- `markertype`: `"point"` or `"arrow"`
  - `"point"`: draw a dot at the event location.
  - `"arrow"`: draw a line start → end + end dot.

---

## Core API

### `build_canvas(provider, pitch, grid, orientation, filtercontent, filtertype)`
Builds the pitch, axes, lines, and grid lines only.  
Returns a `plotly.graph_objects.Figure`.

### `add_grid_heatmap(fig, x_vals, y_vals, pitch, grid, orientation, against=0, opacity=0.7)`
Paints the grid by event frequency:

- `against=0` → white → red scale (`#AA2D3A`)
- `against=1` → white → teal scale

Returns a small debug dict: `{"vmax": ..., "nonzero": ...}`.

### `add_event_markers(fig, df, orientation, markertype="point", ...)`
Draws events:

- `point`: dot at `x/y`.
- `arrow`: line `x/y → x_end/y_end` + end dot.

Requires `x`, `y` and optionally `x_end`, `y_end`.

---

## BigQuery helpers

### `parse_match_ids(raw)`
Parses `matchIds` from list, string or number.

### `parse_matchweeks(raw)`
Parses `matchweeks` from list, string or number.

### `parse_list(raw)`
Converts anything into a list:
`"a,b" → ["a","b"]`, `["a"] → ["a"]`, `None → []`.

### `parse_metric_filters(raw_metric)`
Supports the syntax:

```
FROM <column> import <value1, value2>
```

Example:
`["SHOT_XG", "FROM actionType import SHOT"]`  
→ Metric `SHOT_XG` + filter `actionType=SHOT`.

Returns `(metrics, filters)`.

### `resolve_match_ids(client, request_json, team_id, match_data_table, default_iteration_id=1469)`
Resolves match IDs:

- If `matchIds`/`matchId` is present, uses them.
- If `matchweeks` is present, queries `matchData` with `iterationId`.
- If no `matchweeks`, returns all matches for the team.

### `resolve_analysis_team_ids(client, match_ids, squad_id, opponent_id, against, match_info_table)`
Resolves which team to analyze:

- If `opponent_id` is provided, it is used.
- If `against=1`, it returns opponents from `matchInfo`.

### `resolve_coord_columns(provider)`
Defines start/end columns by provider:

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

## Full example

```python
from qadsiahpitch.plot import build_canvas, add_grid_heatmap, add_event_markers

provider = "impect"
pitch = "full"
orientation = "horizontal"
grid = "5x3"
against = 1
markertype = "arrow"

fig = build_canvas(
    provider=provider,
    pitch=pitch,
    grid=grid,
    orientation=orientation,
    filtercontent=["playerName"],
    filtertype=["dropdown"],
)

# df must include x/y and optionally x_end/y_end
add_grid_heatmap(fig, df["x"].values, df["y"].values, pitch, grid, orientation, against=against)
add_event_markers(fig, df, orientation=orientation, markertype=markertype)
```

---

## Example payload (test)

```json
{
  "provider": "impect",
  "pitch": "full",
  "orientation": "vertical",
  "grid": "5x3",
  "filtertype": "dropdown",
  "filtercontent": "playerName",
  "squadId": 5067,
  "matchIds": 228025,
  "against": 1,
  "metric": ["FROM actionType import PASS"],
  "markertype": "arrow"
}
```
