import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from google.cloud import bigquery

from qadsiahpitch.core import (
    parse_list,
    parse_metric_filters,
    resolve_match_ids,
    resolve_analysis_team_ids,
    resolve_coord_columns,
)
from qadsiahpitch.plot import (
    build_canvas,
    add_grid_heatmap,
    add_event_markers,
    _pitch_x_range,
    _normalize_pitch,
    _reorder_heatmap_traces,
    Y_MIN,
    Y_MAX,
)
from qadsiahpitch.providers.impect import fetch_events_impect

PROJECT_ID = "prj-alqadsiahplatforms-0425"
DATASET = "Impect"
T_EVENTS = f"`{PROJECT_ID}.{DATASET}.events`"
T_MATCHDATA = f"{PROJECT_ID}.{DATASET}.matchData"
T_MATCHINFO = f"{PROJECT_ID}.{DATASET}.matchInfo"


def _map_points_for_orientation(df: pd.DataFrame, orientation: str) -> Tuple[np.ndarray, np.ndarray]:
    if orientation == "vertical":
        return df["y"].values, df["x"].values
    return df["x"].values, df["y"].values


def _fetch_recent_matches(client: bigquery.Client, team_id: int, limit: int = 5) -> List[int]:
    query = f"""
        SELECT id
        FROM {T_MATCHDATA}
        WHERE (homeSquadId = @teamId OR awaySquadId = @teamId)
        ORDER BY matchDayIndex DESC
        LIMIT @limit
    """
    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("teamId", "INT64", int(team_id)),
                bigquery.ScalarQueryParameter("limit", "INT64", int(limit)),
            ]
        ),
    )
    df = job.to_dataframe()
    return df["id"].astype(int).tolist() if not df.empty else []


def _game_minute(series: pd.Series) -> pd.Series:
    minutes = series.astype(float)
    minutes = minutes.apply(lambda v: v - 10000 if v >= 10000 else v)
    return minutes / 60.0


def plot_from_bq(body: Dict) -> Any:
    provider = (body.get("provider") or "impect").lower()
    pitch = body.get("pitch", "full")
    orientation = body.get("orientation", "vertical")
    grid = body.get("grid", "none")
    against = int(body.get("against", 0))

    squad_id = body.get("squadId")
    opponent_id = body.get("opponentId")
    team_id = opponent_id if opponent_id is not None else squad_id
    metrics, filters = parse_metric_filters(body.get("metric"))
    event_types = parse_list(body.get("eventType"))
    events = parse_list(body.get("event"))
    if event_types:
        filters.setdefault("eventType", []).extend(event_types)
    if events:
        filters.setdefault("event", []).extend(events)

    client = bigquery.Client(project=PROJECT_ID)
    match_ids = resolve_match_ids(client, body, team_id, T_MATCHDATA)
    if not match_ids and team_id is not None:
        match_ids = _fetch_recent_matches(client, int(team_id), limit=5)

    analysis_team_ids = resolve_analysis_team_ids(
        client=client,
        match_ids=match_ids,
        squad_id=squad_id,
        opponent_id=opponent_id,
        against=against,
        match_info_table=T_MATCHINFO,
    )

    if provider != "impect":
        raise ValueError("This only works if provider=impect.")

    coord_columns = resolve_coord_columns(provider)
    df = fetch_events_impect(
        client=client,
        events_table=T_EVENTS,
        match_ids=match_ids,
        squad_ids=analysis_team_ids,
        metrics=metrics,
        filters=filters,
        coord_columns=coord_columns,
    )
    if match_ids:
        df = df[df["match_id"].isin([int(m) for m in match_ids])]
    print(f"[DEBUG] match_ids: {match_ids}")
    print(f"[DEBUG] unique matches in df: {df['match_id'].nunique() if not df.empty else 0}")

    pitch_key = _normalize_pitch(pitch)
    x_min, x_max = _pitch_x_range(pitch_key)
    df = df[
        (df["x"].astype(float) >= float(x_min))
        & (df["x"].astype(float) <= float(x_max))
        & (df["y"].astype(float) >= float(Y_MIN))
        & (df["y"].astype(float) <= float(Y_MAX))
    ]

    if df.empty and match_ids:
        df_probe = fetch_events_impect(
            client=client,
            events_table=T_EVENTS,
            match_ids=match_ids,
            squad_ids=[],
            metrics=[],
        )
        if df_probe.empty:
            print("[DEBUG] No events found in Impect.events for those matchIds.")
        else:
            squads = sorted(df_probe["squad_id"].dropna().unique().tolist())
            print(f"[DEBUG] Events exist for matchIds. Squads in events: {squads}")
            if analysis_team_ids and not set(analysis_team_ids).intersection(set(squads)):
                print("[DEBUG] The provided squadId/opponentId is not in those matches.")

    filtercontent = parse_list(body.get("filtercontent"))
    filtertype_list = parse_list(body.get("filtertype"))
    fig = build_canvas(
        provider=provider,
        pitch=pitch,
        grid=grid,
        orientation=orientation,
        filtercontent=filtercontent,
        filtertype=filtertype_list,
    )

    if df.empty:
        return fig

    players = sorted(df.get("playerName", pd.Series(dtype=str)).dropna().unique().tolist())
    is_dropdown = (filtertype_list or ["dropdown"])[0] == "dropdown"

    gridcolor = body.get("gridcolor")
    grid_debug = add_grid_heatmap(
        fig=fig,
        x_vals=df["x"].values,
        y_vals=df["y"].values,
        pitch=pitch,
        grid=grid,
        orientation=orientation,
        gridcolor=gridcolor,
        draw_grid_lines=False,
        show_colorbar=True,
        group_key="all",
        reorder_below_lines=False,
    )
    print(f"[DEBUG] grid heatmap: {grid_debug}")

    if is_dropdown and players:
        for player in players:
            df_player = df[df["playerName"] == player]
            add_grid_heatmap(
                fig=fig,
                x_vals=df_player["x"].values,
                y_vals=df_player["y"].values,
                pitch=pitch,
                grid=grid,
                orientation=orientation,
                gridcolor=gridcolor,
                draw_grid_lines=False,
                show_colorbar=False,
                group_key=player,
                reorder_below_lines=False,
            )

    _reorder_heatmap_traces(fig)

    base_trace_indices = []
    all_heatmap_indices = []
    colorbar_indices = []
    heatmap_traces_by_player = {}
    for idx, tr in enumerate(fig.data):
        name = getattr(tr, "name", "") or ""
        if name == "heatmap-scale":
            colorbar_indices.append(idx)
            continue
        if name.startswith("heatmap-cell:"):
            key = name.split(":", 1)[1]
            if key == "all":
                all_heatmap_indices.append(idx)
            else:
                heatmap_traces_by_player.setdefault(key, []).append(idx)
            continue
        base_trace_indices.append(idx)

    markertype = body.get("markertype", "point")
    markeralpha = body.get("markeralpha")
    marker_traces_by_player = {}
    for player in players:
        df_player = df[df["playerName"] == player]
        before = len(fig.data)
        add_event_markers(
            fig=fig,
            df=df_player,
            orientation=orientation,
            markertype=markertype,
            markeralpha=markeralpha,
        )
        marker_traces_by_player[player] = list(range(before, len(fig.data)))

    if is_dropdown and players:
        buttons = []
        total_traces = len(fig.data)

        def _vis_all():
            vis = [False] * total_traces
            for idx in base_trace_indices:
                vis[idx] = True
            for idx in all_heatmap_indices:
                vis[idx] = True
            for idxs in marker_traces_by_player.values():
                for idx in idxs:
                    vis[idx] = True
            for idx in colorbar_indices:
                vis[idx] = True
            return vis

        def _vis_for_player(name: str):
            vis = [False] * total_traces
            for idx in base_trace_indices:
                vis[idx] = True
            for idx in heatmap_traces_by_player.get(name, []):
                vis[idx] = True
            for idx in marker_traces_by_player.get(name, []):
                vis[idx] = True
            for idx in colorbar_indices:
                vis[idx] = True
            return vis

        initial_vis = _vis_all()
        for idx, visible in enumerate(initial_vis):
            fig.data[idx].visible = visible

        buttons.append(dict(label="All", method="update", args=[{"visible": initial_vis}]))
        for player in players:
            buttons.append(dict(label=player, method="update", args=[{"visible": _vis_for_player(player)}]))

        fig.update_layout(
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
            ]
        )
    return fig


if __name__ == "__main__":
    test_body = {
        "provider": "impect",
        "pitch": "full",
        "orientation": "horizontal",
        "grid": "5x3", #choose between "5x3", "20x20", "set piece", "own third", "final third", "5x5"
        "filtertype": "dropdown",
        "filtercontent": "playerName",
        "squadId": 5067,
        "matchIds": 228025,
        "against": 0, #used to determine the team plotted. (=0 is the same as squadId, =1 is the opponent)
        "metric": ["FROM actionType import PASS", "FROM result import SUCCESS"],
        "markertype": "arrow", #choose between "point", "arrow"
        "markeralpha": 0.5, #marker opacity
        "gridcolor": "whitetored" #choose between "whitetoteal", "blacktoteal" or put between 2-5 rgb/hex colors (e.g. "gridcolor": ["rgb(255,255,255)", "rgb(255,140,0)", "rgb(70,130,180)", "rgb(138,43,226)"]')
    }

    print(f"[DEBUG] test_body pitch={test_body.get('pitch')} grid={test_body.get('grid')}")
    fig = plot_from_bq(test_body)
    html = fig.to_html(include_plotlyjs="cdn")
    safe_pitch = str(test_body.get("pitch", "plot")).replace(" ", "_")
    safe_grid = str(test_body.get("grid", "grid")).replace(" ", "_")
    out_name = f"test_plot_from_bq_{safe_pitch}_{safe_grid}.html"
    out_path = os.path.abspath(out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"OK -> {out_path}")
