# qadsiahpitch

Lightweight library for building agnostic football pitch maps in Plotly. It standardizes match filtering, team selection (against), grid heatmaps, and player/event points, with provider adapters (Impect/SkillCorner/StatsBomb). Supports pitch regions, orientation, and filter UI (dropdown/slider).

## Parameters (schema)


Provider:
- `provider`: `"impect"`, `"skillcorner"`, `"statsbomb"`
  - Impect: x [-52.5, 52.5], y [-34, 34]
  - StatsBomb: x [0, 120], y [0, 80]
  - SkillCorner: x [-52.5, 52.5], y [-34, 34] (same as Impect)
  - The canvas uses the provider dimensions you pass/implement.


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
- This library only builds the pitch + grid + filter UI (canvas).  
  You add the data traces separately.

## Example payload

```json
{
  "pitch": "own third",
  "orientation": "vertical",
  "grid": "3x3",
  "filtertype": "dropdown",
  "filtercontent": "playerName",
  "against": 0
}
```
