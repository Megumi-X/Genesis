from __future__ import annotations

import json
from typing import Any


class GeneralIRValidationError(RuntimeError):
    pass


def extract_first_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < 0 or end <= start:
            raise GeneralIRValidationError("Model output does not contain a JSON object.")
        snippet = stripped[start : end + 1]
        try:
            payload = json.loads(snippet)
        except json.JSONDecodeError as exc:
            raise GeneralIRValidationError(f"Model output JSON parse failed: {exc}") from exc

    if not isinstance(payload, dict):
        raise GeneralIRValidationError("Model output must be a JSON object.")
    return payload


def sanitize_payload(payload: dict[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    actions = normalized.get("actions")
    if not isinstance(actions, list):
        return normalized

    sanitized: list[object] = []
    for action_any in actions:
        if not isinstance(action_any, dict):
            sanitized.append(action_any)
            continue
        action = dict(action_any)
        op = action.get("op")
        if isinstance(op, str):
            action["op"] = op.strip().lower()
        if action.get("op") == "observe":
            fields = action.get("fields")
            if isinstance(fields, list):
                normalized_fields: list[str] = []
                for item in fields:
                    if not isinstance(item, str):
                        continue
                    field = item.strip().lower()
                    if field not in normalized_fields:
                        normalized_fields.append(field)
                action["fields"] = normalized_fields
        sanitized.append(action)

    normalized["actions"] = sanitized
    return normalized
