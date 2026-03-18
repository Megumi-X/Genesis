from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from ...ir_schema import SingleRigidIR
from ..client import OpenAIResponsesClient, coerce_content_to_text
from ..constraints.general_constraints import extract_first_json_object, parse_sanitize_validate
from ...tool_library import GeneralIRAgentToolLibrary
from .prompt_utils import truncate_prompt_text


class IRGenerationError(RuntimeError):
    pass


@dataclass(slots=True)
class IRGenerationRoundLog:
    round: int
    assistant_content: str | None
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    validation_error: str | None


@dataclass(slots=True)
class IRGenerationResult:
    model: str
    rounds: int
    program: SingleRigidIR
    ir_json: dict[str, Any]
    logs: list[IRGenerationRoundLog]


IR_SYSTEM_PROMPT = (
    "You are an IR planning agent for the Genesis rigid-scene IR. "
    "Use tools to fetch guide/schema and validate candidate IR before final output. "
    "When done, output exactly one IR JSON object and nothing else."
)


def _compact_tool_call(raw_call: Any) -> dict[str, Any]:
    if not isinstance(raw_call, dict):
        return {"raw": raw_call}

    function = raw_call.get("function", {})
    name = function.get("name") if isinstance(function, dict) else None
    arguments = function.get("arguments") if isinstance(function, dict) else None
    return {"id": raw_call.get("id"), "name": name, "arguments": arguments}


def _tool_result_message(tool_call_id: str, name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": json.dumps(result, ensure_ascii=False),
    }


def _build_user_prompt(task: str, *, additional_requirements: str | None = None) -> str:
    lines = [
        "Process requirements:",
        "- Call get_generation_guide and get_single_rigid_ir_schema first.",
        "- Call get_observation_field_guide before finalizing observe actions.",
        "- Use top-level `bodies` list, not legacy single `body` root.",
        "- Use `bodies[].fixed=true` for static primitive or URDF props, targets, and supports. For MJCF, express a fixed base in the XML itself.",
        "- `observe`, `set_pose`, and `apply_external_wrench` may target a single body or a list of body names via the `entity` field.",
        "- Keep the IR as concise as possible without changing behavior; merge repeated `observe`, `set_pose`, or `apply_external_wrench` actions into one action with an `entity` list whenever the payload and timing are identical.",
        "- Render is mandatory for generated IR; ensure scene.render is present.",
        "- The IR may contain multiple bodies in `bodies`, but at most one body may be articulated (`mjcf` or `urdf`).",
        "- If articulated structure is needed and tool is available, call generate_articulated_xml before finalizing IR.",
        "- For articulated bodies, define actuators only in IR (`bodies[].actuators`), not in XML.",
        "- Use `set_target_pos` only with position actuators and `set_torque` only with motor actuators.",
        "- If task specifies target simulation duration, pass it to validate_ir as target_sim_duration_sec.",
        "- Draft candidate IR and call validate_ir.",
        "- If validate_ir fails, revise and validate again.",
        "- Return only final valid IR JSON.",
        "",
        "Task:",
        task.strip(),
    ]
    if additional_requirements:
        lines.extend(["", "Additional hard requirements:", additional_requirements.strip()])
    return "\n".join(lines)


def _build_revision_prompt_sections(
    *,
    previous_ir_json: dict[str, Any] | None,
    previous_xml_text: str | None,
) -> list[str]:
    if previous_ir_json is None and previous_xml_text is None:
        return []

    lines = [
        "",
        "Revision mode:",
        "- This is not a fresh generation pass.",
        "- You are revising the previous candidate based on critic feedback.",
        "- Preserve working parts of the previous candidate unless the feedback requires changing them.",
        "- Prefer targeted edits over broad regeneration.",
    ]
    if previous_ir_json is not None:
        lines.extend(
            [
                "",
                "Previous validated IR JSON to revise:",
                truncate_prompt_text(json.dumps(previous_ir_json, ensure_ascii=False, indent=2)),
            ]
        )
    if previous_xml_text is not None and previous_xml_text.strip():
        lines.extend(
            [
                "",
                "Previous articulated XML to revise if XML changes are needed:",
                truncate_prompt_text(previous_xml_text.strip()),
            ]
        )
    return lines


def _build_prompt_cache_key(tool_specs: list[dict[str, Any]]) -> str:
    signature = {
        "system": IR_SYSTEM_PROMPT,
        "tools": tool_specs,
    }
    digest = hashlib.sha1(json.dumps(signature, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"single_rigid_ir_agent:{digest}"


def _build_hosted_prompt_ref(
    *,
    hosted_prompt_id: str | None,
    hosted_prompt_version: str | None,
    task: str,
    additional_requirements: str | None,
    previous_ir_json: dict[str, Any] | None,
    previous_xml_text: str | None,
) -> dict[str, Any] | None:
    if hosted_prompt_id is None:
        return None
    prompt: dict[str, Any] = {
        "id": hosted_prompt_id,
        "variables": {
            "task": task,
            "additional_requirements": "" if additional_requirements is None else additional_requirements,
            "previous_ir_json": "" if previous_ir_json is None else truncate_prompt_text(
                json.dumps(previous_ir_json, ensure_ascii=False, indent=2)
            ),
            "previous_xml_text": "" if previous_xml_text is None else truncate_prompt_text(previous_xml_text),
        },
    }
    if hosted_prompt_version is not None:
        prompt["version"] = hosted_prompt_version
    return prompt


def generate_ir_with_tool_agent(
    *,
    task: str,
    model: str,
    client: OpenAIResponsesClient,
    tool_library: GeneralIRAgentToolLibrary,
    max_rounds: int = 12,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    normalize: bool = True,
    additional_requirements: str | None = None,
    previous_ir_json: dict[str, Any] | None = None,
    previous_xml_text: str | None = None,
    hosted_prompt_id: str | None = None,
    hosted_prompt_version: str | None = None,
) -> IRGenerationResult:
    if max_rounds < 1:
        raise ValueError("`max_rounds` must be >= 1.")

    system_message = {"role": "system", "content": IR_SYSTEM_PROMPT}
    user_prompt = _build_user_prompt(task, additional_requirements=additional_requirements)
    revision_sections = _build_revision_prompt_sections(
        previous_ir_json=previous_ir_json,
        previous_xml_text=previous_xml_text,
    )
    if revision_sections:
        user_prompt = "\n".join([user_prompt, *revision_sections])

    pending_messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
    previous_response_id: str | None = None
    tool_specs = tool_library.tool_specs()
    prompt_cache_key = _build_prompt_cache_key(tool_specs)
    hosted_prompt = _build_hosted_prompt_ref(
        hosted_prompt_id=hosted_prompt_id,
        hosted_prompt_version=hosted_prompt_version,
        task=task,
        additional_requirements=additional_requirements,
        previous_ir_json=previous_ir_json,
        previous_xml_text=previous_xml_text,
    )

    if hosted_prompt is not None:
        pending_messages = []
    base_messages = [] if hosted_prompt is not None else [system_message]

    logs: list[IRGenerationRoundLog] = []
    validated_with_tool = False

    for round_idx in range(1, max_rounds + 1):
        assistant_message = client.responses_completion(
            model=model,
            messages=[*base_messages, *pending_messages],
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            prompt=hosted_prompt,
            previous_response_id=previous_response_id,
            prompt_cache_key=prompt_cache_key,
            tools=tool_specs,
            tool_choice="auto",
        )
        response_id = assistant_message.get("_response_id")
        if isinstance(response_id, str) and response_id:
            previous_response_id = response_id

        assistant_content = coerce_content_to_text(assistant_message.get("content"))
        raw_tool_calls = assistant_message.get("tool_calls")
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        if isinstance(raw_tool_calls, list) and len(raw_tool_calls) > 0:
            next_messages: list[dict[str, Any]] = []
            for call_index, raw_call in enumerate(raw_tool_calls):
                compact_call = _compact_tool_call(raw_call)
                tool_calls.append(compact_call)

                name = compact_call.get("name")
                arguments_json = compact_call.get("arguments")
                call_id = compact_call.get("id")

                if not isinstance(name, str):
                    name = "unknown_tool"
                if not isinstance(arguments_json, str):
                    arguments_json = "{}"
                if not isinstance(call_id, str):
                    call_id = f"synthetic_tool_call_{round_idx}_{call_index}"

                result = tool_library.execute_tool_call(name=name, arguments_json=arguments_json)
                tool_results.append({"id": call_id, "name": name, "result": result})
                next_messages.append(_tool_result_message(call_id, name, result))
                if name == "validate_ir" and isinstance(result, dict) and result.get("ok") is True:
                    validated_with_tool = True

            pending_messages = next_messages

            logs.append(
                IRGenerationRoundLog(
                    round=round_idx,
                    assistant_content=assistant_content or None,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    validation_error=None,
                )
            )
            continue

        validation_error: str | None = None
        try:
            payload = extract_first_json_object(assistant_content)
            if not validated_with_tool:
                raise ValueError("Must call validate_ir and obtain ok=true before final output.")
            program = parse_sanitize_validate(payload, normalize=normalize)
            program = tool_library.apply_parameter_overrides(program)
            constraint_errors = tool_library.validate_program_constraints(program)
            if constraint_errors:
                raise ValueError("; ".join(constraint_errors))
            logs.append(
                IRGenerationRoundLog(
                    round=round_idx,
                    assistant_content=assistant_content or None,
                    tool_calls=[],
                    tool_results=[],
                    validation_error=None,
                )
            )
            return IRGenerationResult(
                model=model,
                rounds=round_idx,
                program=program,
                ir_json=program.model_dump(mode="json"),
                logs=logs,
            )
        except Exception as exc:  # noqa: BLE001
            validation_error = str(exc)

        logs.append(
            IRGenerationRoundLog(
                round=round_idx,
                assistant_content=assistant_content or None,
                tool_calls=[],
                tool_results=[],
                validation_error=validation_error,
            )
        )

        pending_messages = [
            {
                "role": "user",
                "content": (
                    "Your last response was not valid IR JSON. "
                    f"Validation error: {validation_error}. "
                    "Use tools and return corrected JSON only."
                ),
            }
        ]

    last_error = None
    for log in reversed(logs):
        if log.validation_error:
            last_error = log.validation_error
            break

    raise IRGenerationError(
        f"Failed to generate valid IR within {max_rounds} rounds. Last validation error: {last_error}"
    )
