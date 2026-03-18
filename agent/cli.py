from __future__ import annotations

import argparse
from pathlib import Path

from .compiler_backend import compile_single_rigid_ir_to_file
from .io_utils import dump_json, load_json_object
from .ir_schema import SingleRigidIR, normalize_ir, parse_ir_payload
from .runtime import build_llm_event_pack, run_single_rigid_ir


def _cmd_schema(args: argparse.Namespace) -> None:
    dump_json(SingleRigidIR.model_json_schema(), args.out)


def _cmd_validate(args: argparse.Namespace) -> None:
    payload = load_json_object(args.ir, label="IR")
    program = parse_ir_payload(payload)
    if not args.no_normalize:
        program = normalize_ir(program)
    dump_json(program.model_dump(mode="json"), args.out)


def _cmd_compile(args: argparse.Namespace) -> None:
    payload = load_json_object(args.ir, label="IR")
    artifact = compile_single_rigid_ir_to_file(payload, args.out)
    print(f"Compiled IR -> {args.out} ({len(artifact.source.splitlines())} lines)")


def _cmd_run(args: argparse.Namespace) -> None:
    if args.llm_only and args.llm_friendly:
        raise ValueError("`--llm-only` and `--llm-friendly` are mutually exclusive.")

    payload = load_json_object(args.ir, label="IR")
    program = parse_ir_payload(payload)
    if not args.no_normalize:
        program = normalize_ir(program)

    raw_result = run_single_rigid_ir(program, normalize=False)
    llm_pack = None
    if args.llm_friendly or args.llm_only or args.llm_out is not None:
        llm_pack = build_llm_event_pack(program, raw_result)

    if args.llm_out is not None:
        if llm_pack is None:
            raise RuntimeError("Internal error: expected llm event pack to be generated.")
        dump_json(llm_pack, args.llm_out)

    if args.llm_only:
        if llm_pack is None:
            raise RuntimeError("Internal error: expected llm event pack to be generated.")
        dump_json(llm_pack, args.out)
        return

    if args.llm_friendly:
        if llm_pack is None:
            raise RuntimeError("Internal error: expected llm event pack to be generated.")
        dump_json({"raw_result": raw_result, "llm_event_pack": llm_pack}, args.out)
        return

    dump_json(raw_result, args.out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rigid-scene IR toolchain for Genesis (multiple primitive bodies plus at most one articulated body)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_schema = subparsers.add_parser("schema", help="Print IR JSON schema.")
    parser_schema.add_argument("--out", type=Path, default=None, help="Optional output file path.")
    parser_schema.set_defaults(func=_cmd_schema)

    parser_validate = subparsers.add_parser("validate", help="Validate and normalize IR JSON.")
    parser_validate.add_argument("--ir", type=Path, required=True, help="Input IR JSON path.")
    parser_validate.add_argument("--out", type=Path, default=None, help="Optional output file path.")
    parser_validate.add_argument("--no-normalize", action="store_true", help="Disable quaternion normalization.")
    parser_validate.set_defaults(func=_cmd_validate)

    parser_compile = subparsers.add_parser("compile", help="Compile IR JSON into executable python code.")
    parser_compile.add_argument("--ir", type=Path, required=True, help="Input IR JSON path.")
    parser_compile.add_argument("--out", type=Path, required=True, help="Output python path.")
    parser_compile.set_defaults(func=_cmd_compile)

    parser_run = subparsers.add_parser("run", help="Execute IR directly and emit events.")
    parser_run.add_argument("--ir", type=Path, required=True, help="Input IR JSON path.")
    parser_run.add_argument("--out", type=Path, default=None, help="Optional output JSON path.")
    parser_run.add_argument("--no-normalize", action="store_true", help="Disable quaternion normalization.")
    parser_run.add_argument(
        "--llm-friendly",
        "--event-pack",
        dest="llm_friendly",
        action="store_true",
        help="Embed an additional LLM-friendly structured event pack in output.",
    )
    parser_run.add_argument(
        "--llm-only",
        "--event-pack-only",
        dest="llm_only",
        action="store_true",
        help="Output only the LLM-friendly structured event pack.",
    )
    parser_run.add_argument(
        "--llm-out",
        "--event-pack-out",
        dest="llm_out",
        type=Path,
        default=None,
        help="Optional separate output path for the LLM-friendly structured event pack.",
    )
    parser_run.set_defaults(func=_cmd_run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
