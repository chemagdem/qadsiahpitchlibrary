import re
from typing import Dict, List, Optional

import pandas as pd
from google.cloud import bigquery


def fetch_events_impect(
    client: bigquery.Client,
    events_table: str,
    match_ids: List[int],
    squad_ids: List[int],
    metrics: List[str],
    filters: Optional[Dict[str, List[str]]] = None,
    coord_columns: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    if not match_ids:
        return pd.DataFrame()
    metric_selects = []
    for m in metrics:
        metric_selects.append(f"CAST({m} AS FLOAT64) AS m_{m}")
    metrics_sql = ",\n        " + ",\n        ".join(metric_selects) if metric_selects else ""

    params = [bigquery.ArrayQueryParameter("matchIds", "INT64", match_ids)]
    squad_clause = ""
    if squad_ids:
        squad_clause = "AND CAST(squadId AS INT64) IN UNNEST(@squadIds)"
        params.append(bigquery.ArrayQueryParameter("squadIds", "INT64", [int(s) for s in squad_ids]))

    filter_clause = ""
    if filters:
        safe_filters = {}
        for col, values in filters.items():
            if not col or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", col):
                continue
            vals = [str(v) for v in values if str(v) != ""]
            if vals:
                safe_filters[col] = vals
        if safe_filters:
            parts = []
            for col, vals in safe_filters.items():
                param_name = f"f_{col}"
                parts.append(f"{col} IN UNNEST(@{param_name})")
                params.append(bigquery.ArrayQueryParameter(param_name, "STRING", vals))
            filter_clause = "AND " + " AND ".join(parts)

    start_x = (coord_columns or {}).get("start_x", "startAdjCoordinatesX")
    start_y = (coord_columns or {}).get("start_y", "startAdjCoordinatesY")
    end_x = (coord_columns or {}).get("end_x", "endAdjCoordinatesX")
    end_y = (coord_columns or {}).get("end_y", "endAdjCoordinatesY")

    query = f"""
        SELECT
            CAST(matchId AS INT64) AS match_id,
            CAST({start_x} AS FLOAT64) AS x,
            CAST({start_y} AS FLOAT64) AS y,
            CAST({end_x} AS FLOAT64) AS x_end,
            CAST({end_y} AS FLOAT64) AS y_end,
            CAST(squadId AS INT64) AS squad_id,
            playerName,
            CAST(playerId AS INT64) AS playerId,
            CAST(gameTimeInSec AS FLOAT64) AS gameTimeInSec
            {metrics_sql}
        FROM {events_table}
        WHERE CAST(matchId AS INT64) IN UNNEST(@matchIds)
          AND {start_x} IS NOT NULL
          AND {start_y} IS NOT NULL
          {squad_clause}
          {filter_clause}
    """
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    return client.query(query, job_config=job_config).to_dataframe()
