from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..ir_schema import RigidIR
from ..llm_generator.constraints.general_constraints import parse_sanitize_validate
from .overrides import GeneratorParameterOverrides, apply_generator_parameter_overrides
from .program_constraints import validate_program_constraints
from .tool_specs import (
    build_generation_guide_payload,
    build_observation_field_guide_payload,
    build_schema_payload,
    build_tool_specs,
)

ToolFunc = Callable[[dict[str, Any]], dict[str, Any]]
if TYPE_CHECKING:
    from ..llm_generator.agents.xml_agent import XMLGenerationResult

XMLGenerationFunc = Callable[[str | None, str | None], "XMLGenerationResult"]


class GeneralIRAgentToolLibrary:
    """Tool registry for IR agent (supports primitive + articulated)."""

    def __init__(
        self,
        *,
        required_shape_kind: str | None = None,
        required_shape_file: str | None = None,
        allowed_shape_kinds: tuple[str, ...] | None = None,
        allowed_articulated_joint_names: tuple[str, ...] | None = None,
        enforce_articulated_actuator_control: bool = False,
        xml_generation_fn: XMLGenerationFunc | None = None,
        xml_task_default: str | None = None,
        target_sim_duration_sec: float | None = None,
        sim_duration_tolerance_sec: float = 0.75,
        parameter_overrides: GeneratorParameterOverrides | None = None,
    ) -> None:
        self.required_shape_kind = required_shape_kind
        self.required_shape_file = required_shape_file
        self.allowed_shape_kinds = allowed_shape_kinds
        self.allowed_articulated_joint_names = allowed_articulated_joint_names
        self.enforce_articulated_actuator_control = enforce_articulated_actuator_control
        self.xml_generation_fn = xml_generation_fn
        self.xml_task_default = xml_task_default
        self.target_sim_duration_sec = target_sim_duration_sec
        self.sim_duration_tolerance_sec = sim_duration_tolerance_sec
        self.parameter_overrides = parameter_overrides
        self._generated_xml_result: XMLGenerationResult | None = None
        self._tool_funcs: dict[str, ToolFunc] = {
            "get_generation_guide": self._get_generation_guide,
            "get_observation_field_guide": self._get_observation_field_guide,
            "get_rigid_ir_schema": self._get_rigid_ir_schema,
            "validate_ir": self._validate_ir,
        }
        if self.xml_generation_fn is not None:
            self._tool_funcs["generate_articulated_xml"] = self._generate_articulated_xml

    def tool_specs(self) -> list[dict[str, Any]]:
        return build_tool_specs(xml_generation_enabled=self.xml_generation_fn is not None)

    def execute_tool_call(self, *, name: str, arguments_json: str | None) -> dict[str, Any]:
        if name not in self._tool_funcs:
            return {"ok": False, "error": f"Unknown tool `{name}`."}

        try:
            args = self._parse_arguments(arguments_json)
            return self._tool_funcs[name](args)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def validate_program_constraints(
        self,
        program: RigidIR,
        *,
        target_sim_duration_sec: float | None = None,
        sim_duration_tolerance_sec: float | None = None,
    ) -> list[str]:
        program = self.apply_parameter_overrides(program)
        return validate_program_constraints(
            program,
            required_shape_kind=self.required_shape_kind,
            required_shape_file=self.required_shape_file,
            allowed_shape_kinds=self.allowed_shape_kinds,
            allowed_articulated_joint_names=self.allowed_articulated_joint_names,
            enforce_articulated_actuator_control=self.enforce_articulated_actuator_control,
            xml_generation_enabled=self.xml_generation_fn is not None,
            generated_xml_shape_file=self.required_shape_file,
            target_sim_duration_sec=(
                self.target_sim_duration_sec if target_sim_duration_sec is None else target_sim_duration_sec
            ),
            sim_duration_tolerance_sec=(
                self.sim_duration_tolerance_sec if sim_duration_tolerance_sec is None else sim_duration_tolerance_sec
            ),
        )

    @property
    def generated_xml_result(self) -> XMLGenerationResult | None:
        return self._generated_xml_result

    def apply_parameter_overrides(self, program: RigidIR) -> RigidIR:
        return apply_generator_parameter_overrides(program, self.parameter_overrides)

    @staticmethod
    def _parse_arguments(arguments_json: str | None) -> dict[str, Any]:
        if arguments_json is None or arguments_json.strip() == "":
            return {}
        parsed = json.loads(arguments_json)
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments root must be an object")
        return parsed

    def _get_generation_guide(self, _: dict[str, Any]) -> dict[str, Any]:
        return build_generation_guide_payload(
            required_shape_kind=self.required_shape_kind,
            required_shape_file=self.required_shape_file,
            allowed_shape_kinds=self.allowed_shape_kinds,
            allowed_articulated_joint_names=self.allowed_articulated_joint_names,
            enforce_articulated_actuator_control=self.enforce_articulated_actuator_control,
            target_sim_duration_sec=self.target_sim_duration_sec,
            duration_tolerance_sec=self.sim_duration_tolerance_sec,
            xml_generation_enabled=self.xml_generation_fn is not None,
            generated_xml_path=(
                None if self._generated_xml_result is None else self._generated_xml_result.xml_path
            ),
            parameter_overrides=self.parameter_overrides,
        )

    @staticmethod
    def _get_observation_field_guide(_: dict[str, Any]) -> dict[str, Any]:
        return build_observation_field_guide_payload()

    @staticmethod
    def _get_rigid_ir_schema(_: dict[str, Any]) -> dict[str, Any]:
        return build_schema_payload()

    def _generate_articulated_xml(self, args: dict[str, Any]) -> dict[str, Any]:
        if self.xml_generation_fn is None:
            return {"ok": False, "errors": ["`generate_articulated_xml` is not available in current mode."]}

        xml_task = args.get("xml_task")
        file_stem = args.get("file_stem")
        xml_task_value = xml_task if isinstance(xml_task, str) and xml_task.strip() else self.xml_task_default
        file_stem_value = file_stem if isinstance(file_stem, str) and file_stem.strip() else None

        result = self.xml_generation_fn(xml_task_value, file_stem_value)
        from ..llm_generator.agents.xml_agent import list_named_joint_names

        self._generated_xml_result = result
        self.required_shape_kind = "mjcf"
        self.required_shape_file = result.xml_path
        self.allowed_articulated_joint_names = list_named_joint_names(result.xml_path)
        return {
            "ok": True,
            "xml_path": result.xml_path,
            "joint_names": list(self.allowed_articulated_joint_names),
            "attempts": result.attempts,
        }

    def _validate_ir(self, args: dict[str, Any]) -> dict[str, Any]:
        candidate = args.get("candidate_ir")
        normalize = args.get("normalize", True)
        target_sim_duration_sec = args.get("target_sim_duration_sec")
        sim_duration_tolerance_sec = args.get("sim_duration_tolerance_sec")

        if not isinstance(candidate, dict):
            return {"ok": False, "errors": ["`candidate_ir` must be a JSON object."]}
        if not isinstance(normalize, bool):
            return {"ok": False, "errors": ["`normalize` must be a boolean."]}
        if target_sim_duration_sec is not None:
            if not isinstance(target_sim_duration_sec, (int, float)) or float(target_sim_duration_sec) <= 0:
                return {"ok": False, "errors": ["`target_sim_duration_sec` must be a positive number."]}
            target_sim_duration_sec = float(target_sim_duration_sec)
        if sim_duration_tolerance_sec is not None:
            if not isinstance(sim_duration_tolerance_sec, (int, float)) or float(sim_duration_tolerance_sec) <= 0:
                return {"ok": False, "errors": ["`sim_duration_tolerance_sec` must be a positive number."]}
            sim_duration_tolerance_sec = float(sim_duration_tolerance_sec)

        try:
            program = parse_sanitize_validate(candidate, normalize=normalize)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "errors": [str(exc)]}

        errors = self.validate_program_constraints(
            program,
            target_sim_duration_sec=target_sim_duration_sec,
            sim_duration_tolerance_sec=sim_duration_tolerance_sec,
        )
        if errors:
            return {"ok": False, "errors": errors}

        program = self.apply_parameter_overrides(program)
        return {"ok": True, "errors": [], "normalized_ir": program.model_dump(mode="json")}
