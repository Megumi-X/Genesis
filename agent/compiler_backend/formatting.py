from __future__ import annotations


def fmt_tuple(values: tuple[float, ...]) -> str:
    return "(" + ", ".join(f"{float(value):.10g}" for value in values) + ")"


def fmt_vec3(values: tuple[float, float, float]) -> str:
    return "(" + ", ".join(f"{float(value):.10g}" for value in values) + ")"


def fmt_int_tuple(values: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(int(value)) for value in values) + ")"


def fmt_str_tuple(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(repr(value) for value in values) + ")"


def fmt_scalar_or_tuple(values: float | tuple[float, ...]) -> str:
    if isinstance(values, tuple):
        return fmt_tuple(values)
    return f"{float(values):.10g}"


def safe_var_name(name: str) -> str:
    stem = "".join(char if char.isalnum() else "_" for char in name)
    if not stem or stem[0].isdigit():
        stem = f"_{stem}"
    return f"entity_{stem}"

