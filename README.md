# qadsiahpitchlibrary
Lightweight library for building agnostic football pitch maps in Plotly. It standardizes match filtering, team selection (against), grid heatmaps, and player/event points, with provider adapters (Impect/SkillCorner/StatsBomb). Supports pitch regions, orientation, and filter UI (dropdown/slider).

Lightweight library for building agnostic football pitch maps in Plotly. It standardizes match filtering, team selection (against), grid heatmaps, and player/event points, with provider adapters (Impect/SkillCorner/StatsBomb). Supports pitch regions, orientation, and filter UI (dropdown/slider).

## Parameters (schema)

Required:
- `provider`: `"impect"`, `"skillcorner"`, `"statsbomb"`
- `metric`: string or list of strings (provider-specific metrics)

At least one:
- `squadId` or `opponentId`

Match selection:
- `matchweeks`: list like `[1, 2, 3, 4]`
- `iterationId`: integer (default: `1469`)
- or `matchId` / `matchIds` (overrides matchweeks)

Pitch / layout:
- `pitch`: `"full"`, `"own half"`, `"opp half"`, `"own third"`, `"opp third"`
- `orientation`: `"horizontal"` or `"vertical"`
- `grid`: `"5x3"`, `"3x3"`, `"5x5"`, `"set piece"` or `null`
- `against`: `0` or `1`

Filters:
- `filtertype`: `"dropdown"` or `"slider"`  
  Can be a list, e.g. `["dropdown", "dropdown"]`.
- `filtercontent`: field(s) used in filter UI, e.g. `"playerName"`  
  or `["playerName", "playerId"]`.

Notes:
- Multiple filters are combined into a single selector (e.g. `"Name | Id"`).
- Providers expose different metric names; validate per provider.

## Example payload

```json
{
  "provider": "impect",
  "squadId": 5067,
  "matchweeks": [1, 2, 3, 4],
  "iterationId": 1469,
  "against": 0,
  "pitch": "own third",
  "orientation": "vertical",
  "grid": "3x3",
  "filtertype": "dropdown",
  "filtercontent": "playerName",
  "metric": ["OPP_PXT_BALL_LOSS"]
}
```
