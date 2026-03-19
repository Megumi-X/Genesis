from __future__ import annotations

import math
from typing import cast

from pydantic import BaseModel, ConfigDict

IR_VERSION = "genesis.rigid.v1"

Vec3 = tuple[float, float, float]
QuatWXYZ = tuple[float, float, float, float]
ScalarOrSequence = float | tuple[float, ...]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def dedupe_non_empty_names(names: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if len(names) == 0:
        raise ValueError(f"`{field_name}` cannot be empty.")
    normalized: list[str] = []
    seen: set[str] = set()
    for name in names:
        stripped = name.strip()
        if not stripped:
            raise ValueError(f"`{field_name}` cannot contain empty strings.")
        if stripped not in seen:
            seen.add(stripped)
            normalized.append(stripped)
    return tuple(normalized)


def validate_non_negative_indices(indices: tuple[int, ...], *, field_name: str) -> tuple[int, ...]:
    if len(indices) == 0:
        raise ValueError(f"`{field_name}` cannot be empty.")
    if any(index < 0 for index in indices):
        raise ValueError(f"`{field_name}` must contain non-negative indices.")
    return indices


def length_if_sequence(value: ScalarOrSequence | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, tuple):
        return len(value)
    return None


def normalize_quat(quat: QuatWXYZ) -> QuatWXYZ:
    norm = math.sqrt(sum(component * component for component in quat))
    if norm <= 1e-12:
        raise ValueError("Quaternion norm is zero; cannot normalize.")
    normalized = tuple(component / norm for component in quat)
    return cast(QuatWXYZ, normalized)
