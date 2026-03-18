from __future__ import annotations

import os
from typing import Any

from ..ir_schema import RenderIR


def to_serializable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def count_true(mask: Any) -> int:
    data = to_serializable(mask)
    if not isinstance(data, list):
        return int(bool(data))

    count = 0
    stack: list[Any] = [data]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(current)
        else:
            count += int(bool(current))
    return count


def count_contacts(contact_data: dict[str, Any]) -> int:
    if "valid_mask" in contact_data:
        return count_true(contact_data["valid_mask"])

    geom_a = to_serializable(contact_data.get("geom_a", []))
    if isinstance(geom_a, list):
        if geom_a and isinstance(geom_a[0], list):
            return len(geom_a[0])
        return len(geom_a)
    return 0


def is_floating_base_entity(entity: Any) -> bool:
    try:
        return int(entity.n_qs) >= 7 and int(entity.n_dofs) >= 6
    except Exception:  # noqa: BLE001
        return False


def _as_float_vector(value: Any) -> list[float] | None:
    data = to_serializable(value)
    if isinstance(data, list) and data and isinstance(data[0], list):
        data = data[0]
    if not isinstance(data, list):
        return None
    output: list[float] = []
    for component in data:
        if not isinstance(component, int | float) or isinstance(component, bool):
            return None
        output.append(float(component))
    return output


def get_floating_base_root_state(entity: Any) -> dict[str, list[float]] | None:
    if not is_floating_base_entity(entity):
        return None

    qpos = _as_float_vector(entity.get_qpos())
    dofs_velocity = _as_float_vector(entity.get_dofs_velocity())
    if qpos is None or dofs_velocity is None:
        return None
    if len(qpos) < 7 or len(dofs_velocity) < 6:
        return None

    return {
        "pos": qpos[:3],
        "quat": qpos[3:7],
        "vel": dofs_velocity[:3],
        "ang": dofs_velocity[3:6],
    }


def get_entity_root_pos(entity: Any, envs_idx: Any = None) -> Any:
    if not is_floating_base_entity(entity):
        return entity.get_pos(envs_idx)
    qpos = entity.get_qpos(envs_idx=envs_idx)
    return qpos[..., :3]


class FollowRootPositionProxy:
    def __init__(self, entity: Any) -> None:
        self._entity = entity

    def get_pos(self, envs_idx: Any = None) -> Any:
        return get_entity_root_pos(self._entity, envs_idx=envs_idx)


def get_follow_target_entity(entity: Any) -> Any:
    if is_floating_base_entity(entity):
        return FollowRootPositionProxy(entity)
    return entity


def capture_entity_snapshot(entity: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    floating_base_root_state = get_floating_base_root_state(entity)
    getter_by_field = {
        "pos": entity.get_pos,
        "quat": entity.get_quat,
        "vel": entity.get_vel,
        "ang": entity.get_ang,
        "qpos": entity.get_qpos,
        "dofs_position": entity.get_dofs_position,
        "dofs_velocity": entity.get_dofs_velocity,
    }

    for field, getter in getter_by_field.items():
        try:
            if floating_base_root_state is not None and field in floating_base_root_state:
                snapshot[field] = floating_base_root_state[field]
            else:
                snapshot[field] = to_serializable(getter())
        except Exception:  # noqa: BLE001
            continue

    try:
        snapshot["contacts"] = {"count": count_contacts(entity.get_contacts())}
    except Exception:  # noqa: BLE001
        pass

    return snapshot


def finalize_recording(camera: Any, render: RenderIR) -> None:
    output_dir = os.path.dirname(render.output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    camera.stop_recording(save_to_filename=render.output_video, fps=render.fps)
