from __future__ import annotations

from typing import Any

from .overrides import GeneratorParameterOverrides
from .tool_specs import (
    build_generation_guide_payload,
    build_observation_field_guide_payload,
    build_schema_payload,
    build_tool_specs,
)


def build_generator_tool_context(
    *,
    xml_generation_enabled: bool = True,
    parameter_overrides: GeneratorParameterOverrides | None = None,
) -> dict[str, Any]:
    guide = build_generation_guide_payload(
        required_shape_kind=None,
        required_shape_file=None,
        allowed_shape_kinds=None,
        allowed_articulated_joint_names=None,
        enforce_articulated_actuator_control=True,
        target_sim_duration_sec=None,
        duration_tolerance_sec=0.75,
        xml_generation_enabled=xml_generation_enabled,
        generated_xml_path=None,
        parameter_overrides=parameter_overrides,
    )
    constraints = dict(guide["constraints"])
    constraints["direct_state_actions_pre_step_only"] = [
        "set_pose",
        "set_dofs_position",
        "set_dofs_velocity",
    ]
    constraints["implementable_fix_rule"] = (
        "Only recommend changes expressible through the current generator tool library, IR fields, "
        "MJCF generation path, and supported action ops."
    )
    return {
        "tool_specs": build_tool_specs(xml_generation_enabled=xml_generation_enabled),
        "generation_guide": {
            "ok": guide["ok"],
            "mode": guide["mode"],
            "constraints": constraints,
            "templates": guide["templates"],
        },
        "observation_field_guide": build_observation_field_guide_payload(),
        "schema": build_schema_payload()["schema"],
    }
