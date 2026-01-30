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
from qadsiahpitch.plot import build_canvas, add_grid_heatmap, add_event_markers
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
        raise ValueError("Este ejemplo solo implementa provider=impect.")

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

    fig = build_canvas(
        provider=provider,
        pitch=pitch,
        grid=grid,
        orientation=orientation,
        filtercontent=parse_list(body.get("filtercontent")),
        filtertype=parse_list(body.get("filtertype")),
    )

    if df.empty:
        return fig

    grid_debug = add_grid_heatmap(
        fig=fig,
        x_vals=df["x"].values,
        y_vals=df["y"].values,
        pitch=pitch,
        grid=grid,
        orientation=orientation,
        against=against,
    )
    print(f"[DEBUG] grid heatmap: {grid_debug}")

    markertype = body.get("markertype", "point")
    base_count = len(fig.data)
    players = sorted(df.get("playerName", pd.Series(dtype=str)).dropna().unique().tolist())
    for player in players:
        df_player = df[df["playerName"] == player]
        add_event_markers(
            fig=fig,
            df=df_player,
            orientation=orientation,
            markertype=markertype,
        )

    if (body.get("filtertype") or "dropdown") == "dropdown" and players:
        buttons = []
        total_traces = len(fig.data)

        def _vis_for_player(name: str):
            vis = [True] * base_count
            for idx in range(base_count, total_traces):
                vis.append(players[idx - base_count] == name)
            return vis

        buttons.append(dict(label="All", method="update", args=[{"visible": [True] * total_traces}]))
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
        "orientation": "vertical",
        "grid": "5x5",
        "filtertype": "dropdown",
        "filtercontent": "playerName",
        "squadId": 5067,
        "matchIds": 228025,
        "against": 0,
        "metric": ["FROM actionType import PASS"],
        "markertype": "arrow",
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
