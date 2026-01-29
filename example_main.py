import json
import traceback

import functions_framework
from flask import Request, make_response

from qadsiahpitch import build_canvas


def _cors_resp(data, code=200):
    resp = make_response(data, code)
    resp.headers["Content-Type"] = "application/json"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


def _err(code, msg, detail=None):
    payload = {"error": msg}
    if detail:
        payload["detail"] = detail
    return _cors_resp(json.dumps(payload), code)


@functions_framework.http
def example_canvas(request: Request):
    if request.method == "OPTIONS":
        return _cors_resp("", 204)
    if request.method != "POST":
        return _err(405, "Method Not Allowed. Use POST with JSON body.")

    try:
        body = request.get_json(force=True) or {}
        # provider is read from schema but the canvas uses Impect dimensions for now
        pitch = body.get("pitch", "full")
        orientation = body.get("orientation", "vertical")
        grid = body.get("grid", "3x3")
        filtertype = body.get("filtertype", "dropdown")
        filtercontent = body.get("filtercontent", "playerName")
        output = (body.get("output") or "plotly").lower()

        fig = build_canvas(
            pitch=pitch,
            grid=grid,
            orientation=orientation,
            filtercontent=filtercontent,
            filtertype=filtertype,
        )
        if output == "html":
            html = fig.to_html(include_plotlyjs="cdn")
            return _cors_resp(json.dumps({"data": html}), 200)

        plotly_json = fig.to_plotly_json()
        return _cors_resp(
            json.dumps({"data": {"data": plotly_json.get("data"), "layout": plotly_json.get("layout")}}, default=str),
            200,
        )
    except Exception as e:
        return _err(500, "Internal server error", f"{e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    class MockRequest:
        def __init__(self, json_data):
            self.json_data = json_data
            self.method = "POST"

        def get_json(self, force=False):
            return self.json_data

    test_body = {
        "provider": "impect",
        "pitch": "own half",
        "orientation": "vertical",
        "grid": "3x3",
        "filtertype": "dropdown",
        "filtercontent": "playerName",
        "output": "html",
    }

    resp = example_canvas(MockRequest(test_body))
    print(resp)
