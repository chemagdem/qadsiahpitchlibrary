# qadsiahpitch

Lightweight library for building Plotly football pitch canvases. It focuses on pitch geometry, orientation, and grid overlays, plus a basic filter UI (dropdown/slider). Data traces are added by the caller.

## Parameters (schema)

Provider:
- `provider`: `"impect"`, `"skillcorner"`, `"statsbomb"`
  - Impect/SkillCorner: x [-52.5, 52.5], y [-34, 34]
  - StatsBomb: x [0, 120], y [0, 80]

Pitch / layout:
- `pitch`: `"full"`, `"own half"`, `"opp half"`
- `orientation`: `"horizontal"` or `"vertical"`
- `grid`:
  - `"none"` / `null`
  - `"5x3"`, `"3x3"`, `"5x5"`, `"20x20"`
  - `"set piece"`
  - `"own third"`, `"middle third"`, `"final third"` (each third is split into 3 lanes)

Filters:
- `filtertype`: `"dropdown"` or `"slider"`  
  Can be a list, e.g. `["dropdown", "dropdown"]`.
- `filtercontent`: field(s) used in filter UI, e.g. `"playerName"`  
  or `["playerName", "playerId"]`.

Notes:
- This library only builds the pitch + grid + filter UI (canvas).  
  You add the data traces separately.
- Grid lines are always gray with 80% opacity (`rgba(120,120,120,0.8)`).
- Impect and SkillCorner share the same pitch dimensions and layout rules.

## Example payload

```json
{
  "provider": "skillcorner",
  "pitch": "own half",
  "orientation": "vertical",
  "grid": "own third",
  "filtertype": "dropdown",
  "filtercontent": "playerName"
}
```
