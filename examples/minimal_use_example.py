import os
import pandas as pd
from google.cloud import bigquery
# Core helpers: parse payload, resolve match/team IDs, and coordinate columns.
from qadsiahpitch.core import parse_list, parse_metric_filters, resolve_match_ids, resolve_analysis_team_ids, resolve_coord_columns
# Plot helpers: canvas, heatmap, markers, and pitch range utilities.
from qadsiahpitch.plot import build_canvas, add_grid_heatmap, add_event_markers, Y_MIN, Y_MAX, _pitch_x_range, _normalize_pitch, _reorder_heatmap_traces
# Provider fetcher: swap to skillcorner/statsbomb equivalent if needed.
from qadsiahpitch.providers.impect import fetch_events_impect  # change to providers.skillcorner, providers.statsbomb and/or fetch_events_skillcorner, fetch_events_statsbomb

# BigQuery config (Impect).
PROJECT_ID = "prj-alqadsiahplatforms-0425"
DATASET = "Impect"
T_EVENTS = f"`{PROJECT_ID}.{DATASET}.events`"
T_MATCHDATA = f"{PROJECT_ID}.{DATASET}.matchData"
T_MATCHINFO = f"{PROJECT_ID}.{DATASET}.matchInfo"

# Full payload example: includes all supported parameters.
payload = {
    "provider": "impect",
    "pitch": "full",
    "orientation": "vertical",
    "grid": "5x5",
    "gridcolor": "whitetored",
    "filtertype": "dropdown",
    "filtercontent": "playerName",
    "squadId": 5067,
    "opponentId": None,
    "matchIds": 228025,
    "matchweeks": None,
    "iterationId": 1469,
    "against": 1,
    "metric": ["FROM actionType import SHOT"],
    "eventType": None,
    "event": None,
    "markertype": "arrow",
    "markeralpha": 0.5,
    "markercolor": "black",
}

# Normalize payload options.
provider = (payload.get("provider") or "impect").lower()
pitch = payload.get("pitch", "full")
orientation = payload.get("orientation", "vertical")
grid = payload.get("grid", "5x5")
gridcolor = payload.get("gridcolor", "whitetoteal")
markertype = payload.get("markertype", "point")
markeralpha = payload.get("markeralpha", 0.6)
markercolor = payload.get("markercolor", "#333333")
against = int(payload.get("against", 0))

# Team resolution (squadId/opponentId).
squad_id = payload.get("squadId")
opponent_id = payload.get("opponentId")
team_id = opponent_id if opponent_id is not None else squad_id

# Parse metric filters and optional event filters.
metrics, filters = parse_metric_filters(payload.get("metric"))
event_types = parse_list(payload.get("eventType"))
events = parse_list(payload.get("event"))
if event_types:
    filters.setdefault("eventType", []).extend(event_types)
if events:
    filters.setdefault("event", []).extend(events)

# Resolve match IDs and analysis teams from payload.
client = bigquery.Client(project=PROJECT_ID)
match_ids = resolve_match_ids(client, payload, team_id, T_MATCHDATA)
analysis_team_ids = resolve_analysis_team_ids(client, match_ids, squad_id, opponent_id, against, T_MATCHINFO)

# Fetch provider events with the correct coordinate columns.
coord_columns = resolve_coord_columns(provider)
df = fetch_events_impect(client, T_EVENTS, match_ids, analysis_team_ids, metrics, filters, coord_columns)

# Filter events to the requested pitch window.
pitch_key = _normalize_pitch(pitch)
x_min, x_max = _pitch_x_range(pitch_key)
df = df[
    (df["x"].astype(float) >= float(x_min))
    & (df["x"].astype(float) <= float(x_max))
    & (df["y"].astype(float) >= float(Y_MIN))
    & (df["y"].astype(float) <= float(Y_MAX))
]

# Build the base pitch canvas.
fig = build_canvas(provider, pitch, grid, orientation, parse_list(payload.get("filtercontent")), parse_list(payload.get("filtertype")))
if df.empty:
    fig.write_html(os.path.abspath("qadsiahpitch_impect_minimal_all.html"), include_plotlyjs="cdn")
    raise SystemExit("No data")

# Add "All" heatmap layer.
add_grid_heatmap(fig, df["x"].values, df["y"].values, pitch, grid, orientation, gridcolor, draw_grid_lines=False, show_colorbar=True, group_key="all", reorder_below_lines=False, against=against)
players = sorted(df.get("playerName", pd.Series(dtype=str)).dropna().unique().tolist())
for p in players:
    d = df[df["playerName"] == p]
    # Add per-player heatmaps for dropdown filtering.
    add_grid_heatmap(fig, d["x"].values, d["y"].values, pitch, grid, orientation, gridcolor, draw_grid_lines=False, show_colorbar=False, group_key=p, reorder_below_lines=False, against=against)
_reorder_heatmap_traces(fig)
marker_traces_by_player = {}
for p in players:
    # Add per-player markers for dropdown filtering.
    before = len(fig.data)
    add_event_markers(fig, df[df["playerName"] == p], orientation, markertype, markeralpha=markeralpha, markercolor=markercolor, against=against)
    if len(fig.data) > before:
        marker_traces_by_player[p] = list(fig.data[before:])

# Build dropdown visibility mapping for base, heatmaps, and markers.
marker_name_set = {"event-point", "event-arrow", "event-dot"}
heatmap_by_key = {}
base_idxs = []
for i, tr in enumerate(fig.data):
    name = getattr(tr, "name", "") or ""
    if name.startswith("heatmap-cell:"):
        heatmap_by_key.setdefault(name.split(":", 1)[1], []).append(i)
    elif name not in marker_name_set:
        base_idxs.append(i)
all_heatmap = heatmap_by_key.get("all", [])
id_to_index = {id(tr): i for i, tr in enumerate(fig.data)}
all_marker = []
marker_by_player = {}
for p in players:
    idxs = [id_to_index[id(tr)] for tr in marker_traces_by_player.get(p, []) if id(tr) in id_to_index]
    marker_by_player[p] = idxs
    all_marker.extend(idxs)

# Visibility for "All".
def _vis_all():
    vis = [False] * len(fig.data)
    for i in base_idxs + all_heatmap + all_marker:
        vis[i] = True
    return vis

# Visibility for a single player.
def _vis_player(name):
    vis = [False] * len(fig.data)
    for i in base_idxs + heatmap_by_key.get(name, []) + marker_by_player.get(name, []):
        vis[i] = True
    return vis

# Dropdown UI for switching players.
buttons = [dict(label="All", method="update", args=[{"visible": _vis_all()}])]
buttons += [dict(label=p, method="update", args=[{"visible": _vis_player(p)}]) for p in players]
fig.update_layout(updatemenus=[dict(type="dropdown", direction="down", x=0.5, y=1.05, xanchor="center", yanchor="bottom", buttons=buttons)])

# Export to HTML.
out_path = os.path.abspath("qadsiahpitch_impect_minimal_all.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"OK -> {out_path}")
