from __future__ import annotations

import argparse
from pathlib import Path

from ..io_utils import dump_json
from ..llm_generator.client import OpenAIResponsesClient, REASONING_EFFORT_VALUES
from .critic import CriticEvaluationInput, evaluate_prompt_event_video


def _cmd_evaluate(args: argparse.Namespace) -> None:
    client = OpenAIResponsesClient.from_env(
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
        timeout_sec=args.timeout_sec,
    )
    eval_input = CriticEvaluationInput(
        task=args.task,
        ir_path=args.ir,
        event_pack_path=args.event_pack,
        video_path=args.video,
        xml_path=args.xml,
        sample_every_sec=args.sample_every_sec,
        max_frames=args.max_frames,
        max_width=args.max_width,
    )
    result = evaluate_prompt_event_video(
        client=client,
        model=args.model,
        eval_input=eval_input,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
    )

    dump_json(result.analysis_json, args.out)

    if args.log_out is not None:
        dump_json(
            {
                "model": result.model,
                "reasoning_effort": args.reasoning_effort,
                "task": args.task,
                "ir": str(args.ir),
                "xml": str(args.xml) if args.xml is not None else None,
                "event_pack": str(args.event_pack),
                "video": str(args.video),
                "sample_every_sec": args.sample_every_sec,
                "frames_used": result.frames_used,
                "input_digest": result.input_digest,
                "raw_response_text": result.raw_response_text,
                "analysis_json": result.analysis_json,
            },
            args.log_out,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate simulation quality from prompt + event pack + video.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_eval = subparsers.add_parser("evaluate", help="Run multimodal critique.")
    parser_eval.add_argument("--task", type=str, required=True, help="Original natural-language task prompt.")
    parser_eval.add_argument("--ir", type=Path, required=True, help="Path to generated IR JSON.")
    parser_eval.add_argument("--xml", type=Path, default=None, help="Optional path to XML asset used by IR.")
    parser_eval.add_argument("--event-pack", type=Path, required=True, help="Path to event_pack.json.")
    parser_eval.add_argument("--video", type=Path, required=True, help="Path to rendered video (mp4).")
    parser_eval.add_argument("--model", type=str, default="gpt-5.2", help="OpenAI model name.")
    parser_eval.add_argument(
        "--sample-every-sec",
        type=float,
        default=0.5,
        help="Sample one frame every N seconds.",
    )
    parser_eval.add_argument("--max-frames", type=int, default=24, help="Hard cap on sampled frames.")
    parser_eval.add_argument("--max-width", type=int, default=640, help="Max frame width when sampling.")
    parser_eval.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature. Omitted by default.",
    )
    parser_eval.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        choices=REASONING_EFFORT_VALUES,
        help="Optional Responses API reasoning effort.",
    )
    parser_eval.add_argument("--timeout-sec", type=float, default=180.0, help="OpenAI request timeout in seconds.")
    parser_eval.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY", help="API key env var name.")
    parser_eval.add_argument("--base-url-env", type=str, default="OPENAI_BASE_URL", help="Base URL env var name.")
    parser_eval.add_argument("--out", type=Path, default=None, help="Output analysis JSON path; default stdout.")
    parser_eval.add_argument("--log-out", type=Path, default=None, help="Optional debug log JSON path.")
    parser_eval.set_defaults(func=_cmd_evaluate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
