#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path


def latest_success_dir(output_dir):
    runs_dir = Path(output_dir) / "runs"
    candidates = [path for path in runs_dir.glob("*/success") if path.is_dir()]
    if not candidates:
        raise RuntimeError(f"No success output found under {runs_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_generate(args):
    from generate.pipeline import run_pipeline

    print(f"Starting generation pipeline with {args.data} -> {args.out}")
    run_pipeline(
        args.data,
        args.out,
        only_ids_path=args.only_ids,
        max_rows=args.max_rows,
        backend=args.backend,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        concurrency=args.concurrency,
        timeout=args.timeout,
        retries=args.retries,
        max_tokens=args.max_tokens,
    )


def run_eval(args):
    from eval.pipeline import run_pipeline

    print(f"Starting eval pipeline with {args.input} -> {args.out}")
    run_pipeline(
        args.input,
        args.out,
        max_rows=args.max_rows,
        backend=args.backend,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        concurrency=args.concurrency,
        timeout=args.timeout,
        retries=args.retries,
    )


def run_all(args):
    from generate.pipeline import run_pipeline as run_generate_pipeline
    from eval.pipeline import run_pipeline as run_eval_pipeline

    print(f"Starting generation pipeline with {args.data} -> {args.out}")
    run_generate_pipeline(
        args.data,
        args.out,
        only_ids_path=args.only_ids,
        max_rows=args.generate_max_rows,
        backend=args.generate_backend,
        model=args.generate_model,
        api_key=args.api_key,
        base_url=args.base_url,
        concurrency=args.concurrency,
        timeout=args.timeout,
        retries=args.retries,
        max_tokens=args.generate_max_tokens,
    )

    eval_input = args.eval_input or str(latest_success_dir(args.out))
    print(f"Starting eval pipeline with {eval_input} -> {args.eval_out}")
    run_eval_pipeline(
        eval_input,
        args.eval_out,
        max_rows=args.max_rows,
        backend=args.backend,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        concurrency=args.concurrency,
        timeout=args.timeout,
        retries=args.retries,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Agent-STITCH-S generation and eval pipeline entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Run STITCH-S generation only.")
    generate.add_argument(
        "--data",
        default="voidful/agent-sft",
        help="Input records (.parquet path or Hugging Face dataset repo, e.g. voidful/agent-sft).",
    )
    generate.add_argument("--out", default="./out_full", help="Output SFT dataset directory.")
    generate.add_argument(
        "--only-ids",
        default=None,
        help="Optional text/CSV/JSONL file containing ids to run. For CSV/text, the first column is treated as the id.",
    )
    generate.add_argument("--max-rows", type=int, default=None, help="Optional API smoke-test row limit.")
    generate.add_argument("--backend", choices=["local", "api"], default="local", help="Generation backend.")
    generate.add_argument("--model", default=None, help="API generation model when --backend api.")
    generate.add_argument("--api-key", default=None, help="API key when --backend api.")
    generate.add_argument("--base-url", default=None, help="OpenAI-compatible base URL when --backend api.")
    generate.add_argument("--concurrency", type=int, default=None, help="Parallel API requests when --backend api.")
    generate.add_argument("--timeout", type=int, default=None, help="API request timeout seconds.")
    generate.add_argument("--retries", type=int, default=None, help="Retries per row for API backend.")
    generate.add_argument("--max-tokens", type=int, default=None, help="Max output tokens per row for API backend.")
    generate.set_defaults(func=run_generate)

    eval_parser = subparsers.add_parser("eval", help="Run eval scoring only.")
    eval_parser.add_argument("--input", required=True, help="Input assembled STITCH-S parquet path/directory.")
    eval_parser.add_argument("--out", required=True, help="Output eval scoring directory.")
    eval_parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row limit.")
    eval_parser.add_argument("--backend", choices=["local", "api"], default="local", help="Eval backend.")
    eval_parser.add_argument("--model", default=None, help="API judge model when --backend api.")
    eval_parser.add_argument("--api-key", default=None, help="API key when --backend api.")
    eval_parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL when --backend api.")
    eval_parser.add_argument("--concurrency", type=int, default=None, help="Parallel API requests when --backend api.")
    eval_parser.add_argument("--timeout", type=int, default=None, help="API request timeout seconds.")
    eval_parser.add_argument("--retries", type=int, default=None, help="Retries per row for API backend.")
    eval_parser.set_defaults(func=run_eval)

    all_parser = subparsers.add_parser("all", help="Run generation, then eval scoring.")
    all_parser.add_argument(
        "--data",
        default="voidful/agent-sft",
        help="Input records (.parquet path or Hugging Face dataset repo, e.g. voidful/agent-sft).",
    )
    all_parser.add_argument("--out", default="./out_full", help="Output SFT dataset directory.")
    all_parser.add_argument("--eval-out", required=True, help="Output eval scoring directory.")
    all_parser.add_argument(
        "--eval-input",
        default=None,
        help="Optional eval input override. Defaults to the latest generated runs/*/success directory under --out.",
    )
    all_parser.add_argument(
        "--only-ids",
        default=None,
        help="Optional text/CSV/JSONL file containing ids to run. For CSV/text, the first column is treated as the id.",
    )
    all_parser.add_argument("--max-rows", type=int, default=None, help="Optional eval row limit.")
    all_parser.add_argument("--generate-max-rows", type=int, default=None, help="Optional generate row limit for API backend.")
    all_parser.add_argument("--generate-backend", choices=["local", "api"], default="local", help="Generation backend.")
    all_parser.add_argument("--generate-model", default=None, help="API generation model when --generate-backend api.")
    all_parser.add_argument("--generate-max-tokens", type=int, default=None, help="Max output tokens per row for API generation.")
    all_parser.add_argument("--backend", choices=["local", "api"], default="local", help="Eval backend.")
    all_parser.add_argument("--model", default=None, help="API judge model when --backend api.")
    all_parser.add_argument("--api-key", default=None, help="API key when --backend api.")
    all_parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL when --backend api.")
    all_parser.add_argument("--concurrency", type=int, default=None, help="Parallel API requests when --backend api.")
    all_parser.add_argument("--timeout", type=int, default=None, help="API request timeout seconds.")
    all_parser.add_argument("--retries", type=int, default=None, help="Retries per row for API backend.")
    all_parser.set_defaults(func=run_all)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
