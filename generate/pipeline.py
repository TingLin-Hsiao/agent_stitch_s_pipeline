#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse


def run_pipeline(
    input_path,
    output_dir,
    only_ids_path=None,
    max_rows=None,
    backend="local",
    model=None,
    api_key=None,
    base_url=None,
    concurrency=None,
    timeout=None,
    retries=None,
    max_tokens=None,
):
    if backend == "api":
        from generate.api import run_pipeline as run_generate_pipeline

        kwargs = {}
        if max_rows is not None:
            kwargs["max_rows"] = max_rows
        if model is not None:
            kwargs["model"] = model
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        if concurrency is not None:
            kwargs["concurrency"] = concurrency
        if timeout is not None:
            kwargs["timeout"] = timeout
        if retries is not None:
            kwargs["retries"] = retries
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return run_generate_pipeline(input_path, output_dir, only_ids_path=only_ids_path, **kwargs)

    if backend != "local":
        raise ValueError(f"Unsupported generate backend: {backend}")

    from generate.local import run_pipeline as run_generate_pipeline

    return run_generate_pipeline(input_path, output_dir, only_ids_path=only_ids_path)


def main():
    parser = argparse.ArgumentParser(description="STITCH-S SFT generation pipeline.")
    parser.add_argument(
        "--data",
        default="voidful/agent-sft",
        help="Input records (.parquet path or Hugging Face dataset repo, e.g. voidful/agent-sft).",
    )
    parser.add_argument("--out", default="./out_full", help="Output SFT dataset directory.")
    parser.add_argument(
        "--only-ids",
        default=None,
        help="Optional text/CSV/JSONL file containing ids to run. For CSV/text, the first column is treated as the id.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional API smoke-test row limit.")
    parser.add_argument("--backend", choices=["local", "api"], default="local", help="Generation backend.")
    parser.add_argument("--model", default=None, help="API generation model when --backend api.")
    parser.add_argument("--api-key", default=None, help="API key when --backend api.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL when --backend api.")
    parser.add_argument("--concurrency", type=int, default=None, help="Parallel API requests when --backend api.")
    parser.add_argument("--timeout", type=int, default=None, help="API request timeout seconds.")
    parser.add_argument("--retries", type=int, default=None, help="Retries per row for API backend.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max output tokens per row for API backend.")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
