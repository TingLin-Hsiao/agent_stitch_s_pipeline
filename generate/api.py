#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq

from generate.common import (
    SYSTEM_PROMPT,
    assemble_agent_stitch_s,
    build_user_payload,
    canonicalize_row,
    generate_sft_row,
    has_tool_call_tool_result_and_reference,
    loads_json_field,
    load_only_ids,
    tool_reference_filter_failure,
)


DEFAULT_MODEL = (
    os.environ.get("GENERATE_MODEL")
    or os.environ.get("GENERATE_API_MODEL")
    or os.environ.get("API_MODEL")
    or os.environ.get("MODEL_ID", "google/gemma-4-26B-A4B-it")
)
DEFAULT_CONCURRENCY = int(os.environ.get("API_CONCURRENCY", "4"))
DEFAULT_TIMEOUT = int(os.environ.get("API_TIMEOUT", "300"))
DEFAULT_RETRIES = int(os.environ.get("API_RETRIES", "3"))
DEFAULT_MAX_TOKENS = int(os.environ.get("GENERATE_MAX_OUTPUT_TOKENS", os.environ.get("MAX_OUTPUT_TOKENS", "4096")))
PREVIEW_ROWS = int(os.environ.get("PREVIEW_ROWS", "100"))


class ApiKeyPool:
    def __init__(self, raw_keys):
        if isinstance(raw_keys, str):
            keys = [key.strip() for key in raw_keys.split(",") if key.strip()]
        elif raw_keys is None:
            keys = []
        else:
            keys = [key.strip() for key in raw_keys if key and key.strip()]
        if not keys:
            raise RuntimeError("No API key found. Set API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY.")
        self.keys = keys
        self.lock = threading.Lock()
        self.index = 0

    def next_key(self):
        with self.lock:
            key = self.keys[self.index % len(self.keys)]
            self.index += 1
            return key

    def first_key(self):
        return self.keys[0]

    def __len__(self):
        return len(self.keys)


def make_run_id():
    now = datetime.now().astimezone()
    return now, now.strftime("%Y%m%d_%H%M%S")


def infer_base_url(model, api_key, base_url):
    if base_url:
        return base_url
    key = api_key or ""
    if model.startswith("google/") or key.startswith("AIza") or key.startswith("AQ."):
        return "https://generativelanguage.googleapis.com/v1beta/openai/"
    return "https://api.openai.com/v1/"


def infer_model_name(model, base_url):
    if "generativelanguage.googleapis.com" in base_url and model.startswith("google/"):
        return model[len("google/") :]
    return model


def iter_parquet_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(path.glob("*.parquet"))


def read_input_rows(input_path, max_rows=None):
    is_hf_dataset = (
        input_path.startswith("hf://")
        or ("/" in input_path and not os.path.exists(input_path))
    )
    if is_hf_dataset:
        import datasets

        dataset_name = input_path.replace("hf://datasets/", "").replace("hf://", "")
        if "/" in dataset_name:
            parts = dataset_name.split("/")
            if len(parts) >= 2:
                dataset_name = f"{parts[0]}/{parts[1]}"
        hf_ds = datasets.load_dataset(dataset_name)
        if isinstance(hf_ds, datasets.DatasetDict):
            hf_ds = hf_ds[list(hf_ds.keys())[0]]
        if max_rows is not None:
            hf_ds = hf_ds.select(range(min(max_rows, len(hf_ds))))
        return list(hf_ds)

    rows = []
    for file_path in iter_parquet_files(input_path):
        table = pq.read_table(file_path)
        for row in table.to_pylist():
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                return rows
    return rows


def call_chat_completion(base_url, model, key_pool, messages, max_tokens, timeout, retries):
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")

    last_error = None
    for attempt in range(retries + 1):
        api_key = key_pool.next_key()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
                return parsed["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = ""
            last_error = f"HTTP {exc.code} {exc.reason}: {error_body}"
            if attempt >= retries:
                break
            delay = min(60, 2 ** attempt)
            if exc.code == 429 or "quota" in error_body.lower() or "rate" in error_body.lower():
                delay = max(delay, 30)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            delay = min(60, 2 ** attempt)
            if "429" in str(exc) or "quota" in str(exc).lower() or "rate" in str(exc).lower():
                delay = max(delay, 30)
            time.sleep(delay)
    raise RuntimeError(f"api_generation_request_failed: {last_error}")


def build_messages(canonical_row):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_payload(canonical_row)},
    ]


def process_canonical_row(row, base_url, model, key_pool, max_tokens, timeout, retries):
    canonical_row = {
        "id": row["id"],
        "source": row["source"],
        "user_request": row["user_request"],
        "steps_full": loads_json_field(row.get("steps_full_json"), []),
        "available_tools": loads_json_field(row.get("available_tools_json"), []),
        "tool_steps": loads_json_field(row.get("tool_steps_json"), []),
        "final_answer_hint": row.get("final_answer_hint", ""),
        "language": row.get("language", "zh-TW"),
        "context": loads_json_field(row.get("context_json"), []),
    }
    raw_patch = ""
    try:
        raw_patch = call_chat_completion(
            base_url,
            model,
            key_pool,
            build_messages(row),
            max_tokens,
            timeout,
            retries,
        )
        assembled = assemble_agent_stitch_s(canonical_row, raw_patch)
        if assembled["drop_reason"] is not None:
            return None, {
                "id": row["id"],
                "status": "dropped",
                "drop_reason": assembled["drop_reason"],
                "raw_patch": raw_patch,
                "error": assembled.get("error"),
                "sft_data": None,
            }
        return generate_sft_row(canonical_row, assembled), None
    except Exception as exc:
        return None, {
            "id": row.get("id"),
            "status": "dropped",
            "drop_reason": "api_generation_failed",
            "raw_patch": raw_patch,
            "error": str(exc),
            "sft_data": None,
        }


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows[:PREVIEW_ROWS]:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_parquet_rows(rows, output_path):
    os.makedirs(output_path, exist_ok=True)
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), os.path.join(output_path, "part-00000.parquet"))


def run_pipeline(
    input_path,
    output_sft_dir,
    only_ids_path=None,
    max_rows=None,
    model=DEFAULT_MODEL,
    api_key=None,
    base_url=None,
    concurrency=DEFAULT_CONCURRENCY,
    timeout=DEFAULT_TIMEOUT,
    retries=DEFAULT_RETRIES,
    max_tokens=DEFAULT_MAX_TOKENS,
):
    started_at, run_id = make_run_id()
    raw_api_key = api_key or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    key_pool = ApiKeyPool(raw_api_key)
    base_url = infer_base_url(model, key_pool.first_key(), base_url or os.environ.get("API_BASE_URL") or os.environ.get("OPENAI_BASE_URL"))
    model = infer_model_name(model, base_url)

    print(f"Run timestamp: {started_at.isoformat()} (run_id={run_id})")
    print(f"API generate config: model={model}, base_url={base_url}, key_count={len(key_pool)}, concurrency={concurrency}, timeout={timeout}, retries={retries}, max_tokens={max_tokens}")
    rows = read_input_rows(input_path, max_rows=max_rows)
    only_ids = load_only_ids(only_ids_path)
    if only_ids is not None:
        rows = [row for row in rows if (row.get("id") or "unknown_id") in only_ids]
    print(f"Loaded {len(rows)} input rows.")

    canonical_rows = [canonicalize_row(row) for row in rows]
    failures = [
        tool_reference_filter_failure(row)
        for row in canonical_rows
        if not has_tool_call_tool_result_and_reference(row)
    ]
    canonical_rows = [row for row in canonical_rows if has_tool_call_tool_result_and_reference(row)]
    print(f"Eligible rows after tool/reference filter: {len(canonical_rows)}")

    successes = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(process_canonical_row, row, base_url, model, key_pool, max_tokens, timeout, retries): row["id"]
            for row in canonical_rows
        }
        completed = 0
        for future in as_completed(futures):
            success, failure = future.result()
            if success is not None:
                successes.append(success)
            if failure is not None:
                failures.append(failure)
            completed += 1
            print(f"processed {completed}/{len(canonical_rows)} rows", flush=True)

    run_output_dir = os.path.join(output_sft_dir, "runs", run_id)
    success_output_path = os.path.join(run_output_dir, "success")
    failed_output_path = os.path.join(run_output_dir, "failed")
    preview_output_path = os.path.join(run_output_dir, "preview")
    write_parquet_rows(successes, success_output_path)
    write_parquet_rows(failures, failed_output_path)
    write_jsonl(successes, os.path.join(preview_output_path, "success.jsonl"))
    write_jsonl(failures, os.path.join(preview_output_path, "failed.jsonl"))

    print(f"Pipeline finished. SFT datasets written to {success_output_path}")
    print(f"Run output directory: {run_output_dir}")
    print(f"Success rows: {len(successes)}")
    print(f"Failed rows: {len(failures)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API STITCH-S SFT generation pipeline.")
    parser.add_argument("--data", default="voidful/agent-sft", help="Input records (.parquet path or Hugging Face dataset repo).")
    parser.add_argument("--out", default="./out_full", help="Output SFT dataset directory.")
    parser.add_argument("--only-ids", default=None, help="Optional id filter file.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="API generation model.")
    parser.add_argument("--api-key", default=None, help="API key. Defaults to API_KEY/OPENAI_API_KEY/GEMINI_API_KEY.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Parallel API requests.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="API request timeout seconds.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retries per row.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max output tokens per row.")
    args = parser.parse_args()
    run_pipeline(
        args.data,
        args.out,
        only_ids_path=args.only_ids,
        max_rows=args.max_rows,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        concurrency=args.concurrency,
        timeout=args.timeout,
        retries=args.retries,
        max_tokens=args.max_tokens,
    )
