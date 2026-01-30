import json
import re
from typing import Dict, List

from google.cloud import bigquery


def parse_match_ids(raw) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [int(x) for x in raw if str(x).strip() != ""]
    if isinstance(raw, (int, float)):
        return [int(raw)]
    if isinstance(raw, str):
        s = raw.strip()
        try:
            if s.startswith("[") and s.endswith("]"):
                arr = json.loads(s)
                return [int(x) for x in arr]
            return [int(x.strip()) for x in s.split(",") if x.strip()]
        except Exception:
            pass
    return []


def parse_matchweeks(raw) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [int(x) for x in raw]
    if isinstance(raw, (int, float)):
        return [int(raw)]
    if isinstance(raw, str):
        s = raw.strip()
        try:
            if s.startswith("[") and s.endswith("]"):
                arr = json.loads(s)
                return [int(x) for x in arr]
            return [int(x.strip()) for x in s.split(",") if x.strip()]
        except Exception:
            pass
    return []


def parse_list(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        try:
            if s.startswith("[") and s.endswith("]"):
                return json.loads(s)
        except Exception:
            pass
        return [x.strip() for x in s.split(",") if x.strip()]
    return [raw]


def parse_metric_filters(raw_metric):
    """
    Parse metric entries. Normal metrics are treated as column names.
    Special syntax: "FROM <column> import <value1, value2>" adds filters.
    Returns (metrics, filters_dict).
    """
    metrics = []
    filters = {}
    items = parse_list(raw_metric)
    for item in items:
        if not isinstance(item, str):
            metrics.append(item)
            continue
        text = item.strip()
        m = re.match(r"^FROM\s+([A-Za-z_][A-Za-z0-9_]*)\s+import\s+(.+)$", text, flags=re.IGNORECASE)
        if not m:
            metrics.append(text)
            continue
        col = m.group(1)
        raw_vals = m.group(2).strip()
        vals = [v.strip() for v in raw_vals.split(",") if v.strip()]
        if vals:
            filters.setdefault(col, []).extend(vals)
    return metrics, filters


def resolve_match_ids(
    client: bigquery.Client,
    request_json: dict,
    team_id: int,
    match_data_table: str,
    default_iteration_id: int = 1469,
) -> List[int]:
    match_ids = parse_match_ids(request_json.get("matchIds", request_json.get("matchId")))
    if match_ids:
        return match_ids

    iteration_id = request_json.get("iterationId")
    matchweeks = parse_matchweeks(request_json.get("matchweeks"))
    if iteration_id is None:
        iteration_id = default_iteration_id

    if not matchweeks:
        if team_id is None:
            return []
        query_all = f"""
            SELECT id
            FROM `{match_data_table}`
            WHERE (homeSquadId = @teamId OR awaySquadId = @teamId)
              AND iterationId = @iterationId
            ORDER BY matchDayIndex
        """
        job = client.query(
            query_all,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("teamId", "INT64", int(team_id)),
                    bigquery.ScalarQueryParameter("iterationId", "INT64", int(iteration_id)),
                ]
            ),
        )
        df_all = job.to_dataframe()
        if df_all.empty:
            return []
        return df_all["id"].astype(int).tolist()

    if team_id is None:
        raise ValueError("squadId u opponentId es requerido cuando se usan iterationId/matchweeks.")

    params = [
        bigquery.ScalarQueryParameter("iterationId", "INT64", int(iteration_id)),
        bigquery.ScalarQueryParameter("teamId", "INT64", int(team_id)),
        bigquery.ArrayQueryParameter("matchWeeks", "INT64", matchweeks),
    ]
    query = f"""
        SELECT id
        FROM `{match_data_table}`
        WHERE iterationId = @iterationId
          AND (homeSquadId = @teamId OR awaySquadId = @teamId)
          AND matchDayIndex IN UNNEST(@matchWeeks)
        ORDER BY matchDayIndex
    """
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    df_plan = client.query(query, job_config=job_config).to_dataframe()
    if df_plan.empty:
        return []
    return df_plan["id"].astype(int).tolist()


def resolve_analysis_team_ids(
    client: bigquery.Client,
    match_ids: List[int],
    squad_id: int,
    opponent_id: int,
    against: int,
    match_info_table: str,
) -> List[int]:
    if opponent_id is not None:
        return [opponent_id]
    if squad_id is None:
        return []
    if against != 1:
        return [squad_id]

    match_ids_str = ", ".join([str(mid) for mid in match_ids])
    query = f"""
        SELECT id, squadHome_id, squadAway_id
        FROM `{match_info_table}`
        WHERE id IN ({match_ids_str})
    """
    df = client.query(query).to_dataframe()
    if df.empty:
        return [squad_id]
    opponent_ids = set()
    for _, row in df.iterrows():
        home = int(row["squadHome_id"])
        away = int(row["squadAway_id"])
        if squad_id == home:
            opponent_ids.add(away)
        elif squad_id == away:
            opponent_ids.add(home)
        else:
            opponent_ids.update([home, away])
    return list(opponent_ids) if opponent_ids else [squad_id]
