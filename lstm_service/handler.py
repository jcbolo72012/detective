"""AWS Lambda entry point (Function URL). Public URL, so: shared-secret header, input caps, JSON only."""
import base64
import hmac
import json
import os

import forecast


def _resp(code, body):
    return {"statusCode": code, "headers": {"content-type": "application/json"}, "body": json.dumps(body)}


def handler(event, context):
    secret = os.environ.get("FORECAST_TOKEN", "")
    if not secret:
        return _resp(500, {"error": "Server is missing FORECAST_TOKEN."})  # fail closed
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    if not hmac.compare_digest(headers.get("x-forecast-token", ""), secret):
        return _resp(401, {"error": "Unauthorized."})
    try:
        raw = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            raw = base64.b64decode(raw).decode("utf-8")
        payload = json.loads(raw)
        return _resp(200, forecast.run(payload))
    except ValueError as e:  # includes bad JSON and validate() errors
        return _resp(400, {"error": str(e)})
    except Exception as e:  # never leak a traceback
        print("forecast failed:", repr(e))
        return _resp(500, {"error": "Forecast failed on the server."})
