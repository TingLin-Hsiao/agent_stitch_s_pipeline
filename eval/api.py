#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq

from eval.scoring import (
    SPLIT_BUCKETS,
    SYSTEM_PROMPT,
    build_judge_payload,
    parse_json_response,
    table_for_eval_bucket,
    validate_score,
)


DEFAULT_MODEL = (
    os.environ.get("EVAL_MODEL")
    or os.environ.get("EVAL_API_MODEL")
    or os.environ.get("API_MODEL")
    or os.environ.get("MODEL_ID", "google/gemma-4-26B-A4B-it")
)
DEFAULT_CONCURRENCY = int(os.environ.get("API_CONCURRENCY", "4"))
DEFAULT_TIMEOUT = int(os.environ.get("API_TIMEOUT", "180"))
DEFAULT_RETRIES = int(os.environ.get("API_RETRIES", "3"))
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


def compact(value, limit):
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "...[TRUNCATED]"


def make_run_id():
    now = datetime.now().astimezone()
    return now, now.strftime("%Y%m%d_%H%M%S")


def iter_parquet_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(path.glob("*.parquet"))


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


def build_messages(row, row_index):
    payload = build_judge_payload(row, row_index)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "請評分以下單筆 Agent-STITCH-S 資料，只輸出 JSON。\n\n"
            + json.dumps(payload, ensure_ascii=False, default=str),
        },
    ], payload


def call_chat_completion(base_url, model, key_pool, messages, timeout, retries):
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")

    last_error = None
    for attempt in range(retries + 1):
        api_key = key_pool.next_key()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        raw_text = ""
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_text = resp.read().decode("utf-8")
                parsed = json.loads(raw_text)
                return parsed["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            delay = min(60, 2 ** attempt)
            if "429" in str(exc) or "quota" in str(exc).lower() or "rate" in str(exc).lower():
                delay = max(delay, 30)
            time.sleep(delay)
    raise RuntimeError(f"api_judge_request_failed: {last_error}")


def error_score(error, raw_text=None, fallback_silence=None):
    return validate_score(
        {
            "scores": {},
            "max_silence_gap_sec": fallback_silence,
            "timing_estimated": True,
            "temporal_causality_errors": [],
            "grounding_errors": [],
            "critical_errors": [error],
            "minor_errors": [],
            "deduction_reasons": [error],
            "customer_service_comment": "API judge output could not be parsed or requested.",
            "comment": "API judge output could not be parsed or requested.",
            "eval_error": error,
            "raw_judge_output": compact(raw_text or "", 2000),
        },
        fallback_silence,
    )


def score_row(row, row_index, base_url, model, key_pool, timeout, retries):
    messages, payload = build_messages(row, row_index)
    fallback_silence = payload.get("deterministic_signals", {}).get("estimated_max_silence_gap_sec")
    raw = ""
    try:
        raw = call_chat_completion(base_url, model, key_pool, messages, timeout, retries)
        score = validate_score(parse_json_response(raw), fallback_silence)
        score["eval_backend"] = "api"
        return score
    except Exception as exc:
        score = error_score(str(exc), raw, fallback_silence)
        score["eval_backend"] = "api"
        return score


def append_score_columns(table, score_objs):
    arrays = [table[name] for name in table.column_names]
    fields = [field for field in table.schema]
    arrays.append(pa.array([json.dumps(score, ensure_ascii=False) for score in score_objs], type=pa.string()))
    fields.append(pa.field("eval_scores", pa.string()))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def write_preview(rows, preview_path):
    os.makedirs(os.path.dirname(preview_path), exist_ok=True)
    with open(preview_path, "w", encoding="utf-8") as f:
        for row in rows[:PREVIEW_ROWS]:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return min(len(rows), PREVIEW_ROWS)


def process_rows(rows, base_url, model, key_pool, concurrency, timeout, retries, start_index=0):
    scored = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(score_row, row, start_index + i, base_url, model, key_pool, timeout, retries): i
            for i, row in enumerate(rows)
        }
        completed = 0
        for future in as_completed(futures):
            i = futures[future]
            scored[i] = future.result()
            completed += 1
            if completed % 25 == 0:
                print(f"scored {completed}/{len(rows)} rows", flush=True)
    return scored


def run_pipeline(
    input_path,
    output_dir,
    max_rows=None,
    model=DEFAULT_MODEL,
    api_key=None,
    base_url=None,
    concurrency=DEFAULT_CONCURRENCY,
    timeout=DEFAULT_TIMEOUT,
    retries=DEFAULT_RETRIES,
):
    started_at, run_id = make_run_id()
    raw_api_key = api_key or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    key_pool = ApiKeyPool(raw_api_key)

    base_url = infer_base_url(model, key_pool.first_key(), base_url or os.environ.get("API_BASE_URL") or os.environ.get("OPENAI_BASE_URL"))
    model = infer_model_name(model, base_url)
    print(f"Run timestamp: {started_at.isoformat()} (run_id={run_id})")
    print(f"API eval config: model={model}, base_url={base_url}, key_count={len(key_pool)}, concurrency={concurrency}, timeout={timeout}, retries={retries}")

    files = iter_parquet_files(input_path)
    if not files:
        raise RuntimeError(f"No parquet files found under {input_path}")

    run_output_dir = os.path.join(output_dir, "runs", run_id)
    scored_output_path = os.path.join(run_output_dir, "scored")
    preview_path = os.path.join(run_output_dir, "preview", "scored.jsonl")
    os.makedirs(scored_output_path, exist_ok=True)
    split_output_paths = {
        bucket: os.path.join(run_output_dir, bucket)
        for bucket in SPLIT_BUCKETS
    }
    for split_output_path in split_output_paths.values():
        os.makedirs(split_output_path, exist_ok=True)

    global_row = 0
    written = 0
    preview_rows = []
    split_preview_rows = {bucket: [] for bucket in SPLIT_BUCKETS}
    split_written = {bucket: 0 for bucket in SPLIT_BUCKETS}
    for file_path in files:
        if max_rows is not None and written >= max_rows:
            break
        table = pq.read_table(file_path)
        rows = table.to_pylist()
        if max_rows is not None:
            rows = rows[: max_rows - written]
            table = table.slice(0, len(rows))
        if not rows:
            continue

        score_objs = process_rows(rows, base_url, model, key_pool, concurrency, timeout, retries, global_row)
        out_table = append_score_columns(table, score_objs)
        out_file = os.path.join(scored_output_path, file_path.name)
        pq.write_table(out_table, out_file)

        out_rows = out_table.to_pylist()
        if len(preview_rows) < PREVIEW_ROWS:
            preview_rows.extend(out_rows[: PREVIEW_ROWS - len(preview_rows)])
        for bucket in SPLIT_BUCKETS:
            split_table = table_for_eval_bucket(out_table, bucket)
            split_written[bucket] += split_table.num_rows
            if split_table.num_rows:
                split_file = os.path.join(split_output_paths[bucket], file_path.name)
                pq.write_table(split_table, split_file)
                if len(split_preview_rows[bucket]) < PREVIEW_ROWS:
                    split_rows = split_table.to_pylist()
                    split_preview_rows[bucket].extend(
                        split_rows[: PREVIEW_ROWS - len(split_preview_rows[bucket])]
                    )
        global_row += len(rows)
        written += len(rows)
        print(f"wrote {len(rows)} rows -> {out_file}", flush=True)

    preview_count = write_preview(preview_rows, preview_path)
    print(f"Scored dataset written to {scored_output_path}")
    print(f"Preview rows written to {preview_path}: {preview_count}")
    for bucket in SPLIT_BUCKETS:
        split_preview_path = os.path.join(run_output_dir, "preview", f"{bucket}.jsonl")
        split_preview_count = write_preview(split_preview_rows[bucket], split_preview_path)
        print(
            f"{bucket.title()} dataset written to {split_output_paths[bucket]} "
            f"(rows: {split_written[bucket]}, preview rows: {split_preview_count})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Agent-STITCH-S eval scoring pipeline.")
    parser.add_argument("--input", required=True, help="Input assembled STITCH-S parquet path/directory.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="API judge model.")
    parser.add_argument("--api-key", default=None, help="API key. Defaults to API_KEY/OPENAI_API_KEY/GEMINI_API_KEY.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Parallel API requests.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="API request timeout seconds.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retries per row.")
    args = parser.parse_args()
    run_pipeline(
        args.input,
        args.out,
        max_rows=args.max_rows,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        concurrency=args.concurrency,
        timeout=args.timeout,
        retries=args.retries,
    )
