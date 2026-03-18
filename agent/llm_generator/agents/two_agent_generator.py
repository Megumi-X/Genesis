from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..client import OpenAIResponsesClient
from ...tool_library import GeneralIRAgentToolLibrary, GeneratorParameterOverrides
from .ir_agent import IRGenerationResult, generate_ir_with_tool_agent
from .xml_agent import XMLGenerationResult, generate_articulated_xml_with_openai


@dataclass(slots=True)
class TwoAgentGenerationResult:
    model: str
    mode: str
    articulated_requested: bool
    ir_result: IRGenerationResult
    xml_result: XMLGenerationResult | None

    @property
    def ir_json(self) -> dict[str, Any]:
        return self.ir_result.ir_json


def _merge_xml_task(base_task: str, xml_feedback_requirements: str | None) -> str:
    if xml_feedback_requirements is None or not xml_feedback_requirements.strip():
        return base_task
    return "\n\n".join(
        [
            base_task.strip(),
            "XML-specific repair requirements for this round:",
            xml_feedback_requirements.strip(),
        ]
    )


def generate_ir_two_agent(
    *,
    task: str,
    model: str,
    client: OpenAIResponsesClient,
    xml_model: str | None = None,
    max_rounds: int = 12,
    xml_max_attempts: int = 4,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    normalize: bool = True,
    assets_dir: str | Path = "agent/generated_assets",
    force_primitive_mode: bool = False,
    additional_requirements: str | None = None,
    xml_feedback_requirements: str | None = None,
    previous_ir_json: dict[str, Any] | None = None,
    previous_xml_text: str | None = None,
    hosted_prompt_id: str | None = None,
    hosted_prompt_version: str | None = None,
    parameter_overrides: GeneratorParameterOverrides | None = None,
) -> TwoAgentGenerationResult:
    xml_out_dir = Path(assets_dir)
    xml_out_dir.mkdir(parents=True, exist_ok=True)
    xml_task_default = _merge_xml_task(task, xml_feedback_requirements)

    def _xml_generation_fn(xml_task: str | None, file_stem: str | None) -> XMLGenerationResult:
        effective_xml_task = xml_task if isinstance(xml_task, str) and xml_task.strip() else xml_task_default
        if (
            xml_feedback_requirements is not None
            and xml_feedback_requirements.strip()
            and xml_feedback_requirements.strip() not in effective_xml_task
        ):
            effective_xml_task = _merge_xml_task(effective_xml_task, xml_feedback_requirements)
        result = generate_articulated_xml_with_openai(
            task=effective_xml_task,
            model=xml_model or model,
            client=client,
            output_dir=xml_out_dir,
            file_stem=file_stem or "generated_articulated_body",
            previous_xml_text=previous_xml_text,
            max_attempts=xml_max_attempts,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        return result

    tool_library = GeneralIRAgentToolLibrary(
        allowed_shape_kinds=(
            ("sphere", "box", "cylinder")
            if force_primitive_mode
            else ("sphere", "box", "cylinder", "mjcf")
        ),
        enforce_articulated_actuator_control=True,
        xml_generation_fn=None if force_primitive_mode else _xml_generation_fn,
        xml_task_default=xml_task_default,
        parameter_overrides=parameter_overrides,
    )

    requirement_lines: list[str] = []
    if previous_ir_json is not None or (previous_xml_text is not None and previous_xml_text.strip()):
        requirement_lines.extend(
            [
                "This is a refinement pass, not a fresh generation pass.",
                "Modify the previous candidate based on feedback instead of starting over from scratch.",
                "Preserve working parts of the previous IR/XML unless the feedback requires changing them.",
            ]
        )
    if force_primitive_mode:
        requirement_lines.append("Primitive-only mode: all bodies[].shape.kind must be one of sphere/box/cylinder.")
    else:
        requirement_lines.extend(
            [
                "You may generate multiple bodies in one IR.",
                "At most one body may be articulated (`mjcf` or `urdf`); all other bodies must be primitive shapes.",
                "Use `fixed=true` on primitive or URDF bodies that should stay anchored in the world. For MJCF bodies, express a fixed base in the XML itself.",
                "If articulated motion is needed, call `generate_articulated_xml` and bind returned xml_path to one body in `bodies` with `shape.kind='mjcf'`.",
                "Do not define actuators inside XML; define actuators only on the articulated body in `bodies[].actuators`.",
                "For articulated bodies, do not use `set_pose` / `set_dofs_position` / `set_dofs_velocity`; "
                "use that body's actuators plus actuator commands (`set_target_pos` for position actuators, "
                "`set_torque` for motor actuators).",
            ]
        )
    requirement_lines.append(
        "If the task specifies a target simulation duration, enforce it via "
        "validate_ir(target_sim_duration_sec=..., sim_duration_tolerance_sec=...)."
    )
    merged_requirements = "\n".join(requirement_lines)
    if additional_requirements:
        merged_requirements = "\n\n".join([merged_requirements, additional_requirements.strip()])

    ir_result = generate_ir_with_tool_agent(
        task=task,
        model=model,
        client=client,
        tool_library=tool_library,
        max_rounds=max_rounds,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        normalize=normalize,
        additional_requirements=merged_requirements,
        previous_ir_json=previous_ir_json,
        previous_xml_text=previous_xml_text,
        hosted_prompt_id=hosted_prompt_id,
        hosted_prompt_version=hosted_prompt_version,
    )

    xml_result = tool_library.generated_xml_result
    articulated_requested = any(body.shape.kind in {"mjcf", "urdf"} for body in ir_result.program.bodies)
    if force_primitive_mode:
        mode = "single_agent_primitive"
    elif xml_result is not None:
        mode = "ir_agent_triggered_xml"
    else:
        mode = "ir_agent_no_xml"

    return TwoAgentGenerationResult(
        model=model,
        mode=mode,
        articulated_requested=articulated_requested,
        ir_result=ir_result,
        xml_result=xml_result,
    )
