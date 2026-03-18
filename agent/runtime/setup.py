from __future__ import annotations

import os
from typing import Any

from ..ir_schema import SingleRigidIR
from .actuators import configure_actuators
from .builders import apply_collision_overrides, build_body_morph, build_rigid_material
from .helpers import get_follow_target_entity
from .models import RuntimeContext, RuntimeState


def _configure_follow_camera(program: SingleRigidIR, runtime: RuntimeContext) -> None:
    render = runtime.render
    camera = runtime.camera
    if render is None or camera is None or render.follow_entity is None:
        return

    follow_cfg = render.follow_entity
    target_entity = runtime.entities.get(follow_cfg.entity)
    if target_entity is None:
        raise ValueError(f"Cannot follow unknown entity `{follow_cfg.entity}`.")

    camera.follow_entity(
        get_follow_target_entity(target_entity),
        fixed_axis=tuple(follow_cfg.fixed_axis),
        smoothing=follow_cfg.smoothing,
        fix_orientation=follow_cfg.fix_orientation,
    )


def configure_headless_if_needed(program: SingleRigidIR) -> None:
    if not program.scene.show_viewer:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYGLET_HEADLESS", "1")


def ensure_genesis_initialized(gs: Any, program: SingleRigidIR) -> None:
    requested_backend = gs.cpu if program.scene.backend == "cpu" else gs.gpu
    if getattr(gs, "_initialized", False):
        active_backend = getattr(gs, "backend", None)
        # if active_backend != requested_backend:
        #     raise ValueError(
        #         "Genesis already initialized with a different backend. "
        #         f"Active backend={active_backend}, requested backend={requested_backend}."
        #     )
        return
    gs.init(backend=requested_backend)


def create_runtime_context(gs: Any, program: SingleRigidIR) -> RuntimeContext:
    viewer_options = None
    if program.scene.viewer is not None:
        viewer = program.scene.viewer
        viewer_options = gs.options.ViewerOptions(
            camera_pos=tuple(viewer.camera_pos),
            camera_lookat=tuple(viewer.camera_lookat),
            camera_fov=viewer.camera_fov,
        )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=program.scene.sim.dt,
            gravity=tuple(program.scene.sim.gravity),
        ),
        viewer_options=viewer_options,
        show_viewer=program.scene.show_viewer,
    )

    render = program.scene.render
    camera = None
    if render is not None:
        camera = scene.add_camera(
            res=tuple(render.res),
            pos=tuple(render.camera_pos),
            lookat=tuple(render.camera_lookat),
            up=tuple(render.camera_up),
            fov=render.camera_fov,
            near=render.near,
            far=render.far,
            GUI=render.gui,
        )

    entities: dict[str, Any] = {}
    body_entities: dict[str, Any] = {}
    if program.scene.add_ground:
        ground_kwargs: dict[str, Any] = {
            "morph": gs.morphs.Plane(),
            "name": "ground",
        }
        ground_material = build_rigid_material(gs, rho=None, collision=program.scene.ground_collision)
        if ground_material is not None:
            ground_kwargs["material"] = ground_material
        entities["ground"] = scene.add_entity(**ground_kwargs)

    for body in program.bodies:
        add_entity_kwargs: dict[str, Any] = {
            "morph": build_body_morph(gs, body),
            "visualize_contact": body.visualize_contact,
            "name": body.name,
        }
        body_material = build_rigid_material(gs, rho=body.rho, collision=body.collision)
        if body_material is not None:
            add_entity_kwargs["material"] = body_material
        body_entity = scene.add_entity(**add_entity_kwargs)
        entities[body.name] = body_entity
        body_entities[body.name] = body_entity

    return RuntimeContext(
        scene=scene,
        camera=camera,
        render=render,
        entities=entities,
        body_entities=body_entities,
    )


def build_runtime_context(program: SingleRigidIR, runtime: RuntimeContext, state: RuntimeState) -> None:
    for body in program.bodies:
        apply_collision_overrides(runtime.body_entities[body.name], body.collision)
    if program.scene.add_ground:
        apply_collision_overrides(runtime.entities["ground"], program.scene.ground_collision)
    state.actuators_by_entity = {
        body.name: configure_actuators(runtime.body_entities[body.name], body.actuators)
        for body in program.bodies
    }
    _configure_follow_camera(program, runtime)

    if runtime.camera is not None and runtime.render is not None:
        runtime.camera.start_recording()
        state.recording_started = True
        if runtime.render.include_initial_frame:
            runtime.camera.render(force_render=runtime.render.force_render)
            state.rendered_frames += 1
