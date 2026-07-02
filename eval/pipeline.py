#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse


def run_pipeline(
    input_path,
    output_dir,
    max_rows=None,
    backend="local",
    model=None,
    api_key=None,
    base_url=None,
    concurrency=None,
    timeout=None,
    retries=None,
):
    if backend == "api":
        from eval.api import run_pipeline as run_eval_pipeline

        kwargs = {}
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
        return run_eval_pipeline(input_path, output_dir, max_rows=max_rows, **kwargs)

    if backend != "local":
        raise ValueError(f"Unsupported eval backend: {backend}")

    from eval.local import run_pipeline as run_eval_pipeline

    return run_eval_pipeline(input_path, output_dir, max_rows=max_rows)


def main():
    parser = argparse.ArgumentParser(description="Local Gemma Agent-STITCH-S eval scoring pipeline.")
    parser.add_argument("--input", required=True, help="Input assembled STITCH-S parquet path/directory.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument("--backend", choices=["local", "api"], default="local", help="Eval backend.")
    parser.add_argument("--model", default=None, help="API judge model when --backend api.")
    parser.add_argument("--api-key", default=None, help="API key when --backend api.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL when --backend api.")
    parser.add_argument("--concurrency", type=int, default=None, help="Parallel API requests when --backend api.")
    parser.add_argument("--timeout", type=int, default=None, help="API request timeout seconds.")
    parser.add_argument("--retries", type=int, default=None, help="Retries per row for API backend.")
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
