from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..ir_schema import CollisionIR, SingleRigidIR
from ..llm_generator.constraints import ensure_program_has_render, synchronize_render_timing


@dataclass(slots=True, frozen=True)
class GeneratorParameterOverrides:
    sim_dt: float | None = None
    render_every_n_steps: int | None = None
    render_res: tuple[int, int] | None = None
    primitive_density: float | None = None
    ground_friction: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def apply_generator_parameter_overrides(
    program: SingleRigidIR,
    overrides: GeneratorParameterOverrides | None,
) -> SingleRigidIR:
    if overrides is None:
        return program

    patched = ensure_program_has_render(program.model_copy(deep=True))

    if overrides.sim_dt is not None:
        patched.scene.sim.dt = overrides.sim_dt

    if patched.scene.render is not None:
        if overrides.render_every_n_steps is not None:
            patched.scene.render.render_every_n_steps = overrides.render_every_n_steps
        if overrides.render_res is not None:
            patched.scene.render.res = overrides.render_res

    if overrides.ground_friction is not None and patched.scene.add_ground:
        if patched.scene.ground_collision is None:
            patched.scene.ground_collision = CollisionIR()
        patched.scene.ground_collision.friction = overrides.ground_friction

    if overrides.primitive_density is not None:
        for body in patched.bodies:
            if body.shape.kind in {"sphere", "box", "cylinder"}:
                body.rho = overrides.primitive_density

    return synchronize_render_timing(patched)
