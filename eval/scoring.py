#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCORE_KEYS = [
    "speech_first",
    "tool_waiting_safety",
    "temporal_causality",
    "incremental_update",
    "stitch_markers",
    "silence_gap",
    "grounding",
    "customer_service_quality",
    "tool_validity",
]
MAX_SCORE = len(SCORE_KEYS) * 2
SPLIT_BUCKETS = ("keep", "review")
HARD_FAIL_SCORE_KEYS = {
    "speech_first",
    "tool_waiting_safety",
    "temporal_causality",
    "grounding",
    "tool_validity",
}

SAY_RE = re.compile(r"<SAY>(.*?)</SAY>", re.DOTALL)
SOPR_RE = re.compile(r"\[SOPR\]", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<TOOL_CALL>(.*?)</TOOL_CALL>", re.DOTALL)
TOOL_RESULT_RE = re.compile(r"<TOOL_RESULT>(.*?)</TOOL_RESULT>", re.DOTALL)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
EVENT_RE = re.compile(
    r"(<SAY>.*?</SAY>|\[SOPR\].*?\[EOPR\]|<TOOL_CALL>.*?</TOOL_CALL>|<TOOL_RESULT>.*?</TOOL_RESULT>|\[EOR\])",
    re.DOTALL,
)


def load_prompt_file(filename):
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / filename
    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read().strip()


SYSTEM_PROMPT = load_prompt_file("eval_stitch_s.txt")


def compact(value, limit):
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "...[TRUNCATED]"


def compact_head_tail(value, limit, head_ratio=0.6):
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    head_len = max(1, int((limit - 30) * head_ratio))
    tail_len = max(1, limit - 30 - head_len)
    return text[:head_len] + "...[TRUNCATED]..." + text[-tail_len:]


def strip_wrapping(text, open_tag, close_tag):
    if text.startswith(open_tag) and text.endswith(close_tag):
        return text[len(open_tag): -len(close_tag)].strip()
    return text.strip()


def summarize_target_events(target_sequence):
    """
    Preserve event order and all user-audible SAY text, while compressing bulky
    tool payloads. This keeps the judge focused on speech quality without losing
    enough causal context to score grounding and tool-result timing.
    """
    msg = target_sequence or ""
    events = []
    tool_call_index = 0
    tool_result_index = 0

    for match in EVENT_RE.finditer(msg):
        raw = match.group(1).strip()
        if raw.startswith("<SAY>"):
            events.append({
                "type": "say",
                "text": strip_wrapping(raw, "<SAY>", "</SAY>"),
            })
        elif raw.startswith("[SOPR]"):
            events.append({
                "type": "private_reasoning",
                "text": compact(strip_wrapping(raw, "[SOPR]", "[EOPR]"), 700),
            })
        elif raw.startswith("<TOOL_CALL>"):
            tool_call_index += 1
            call_text = strip_wrapping(raw, "<TOOL_CALL>", "</TOOL_CALL>")
            events.append({
                "type": "tool_call",
                "index": tool_call_index,
                "text": compact_head_tail(call_text, 1200),
            })
        elif raw.startswith("<TOOL_RESULT>"):
            tool_result_index += 1
            result_text = strip_wrapping(raw, "<TOOL_RESULT>", "</TOOL_RESULT>")
            events.append({
                "type": "tool_result",
                "index": tool_result_index,
                "text": compact_head_tail(result_text, 1600),
            })
        elif raw == "[EOR]":
            events.append({"type": "eor"})

    return events


def summarize_tool_steps(tool_steps):
    if not isinstance(tool_steps, list):
        return tool_steps

    summarized = []
    for i, step in enumerate(tool_steps, 1):
        if not isinstance(step, dict):
            summarized.append(step)
            continue
        summarized.append({
            "order": step.get("order", i),
            "tool_call_line": compact_head_tail(step.get("tool_call_line", ""), 1200),
            "tool_result_line": compact_head_tail(step.get("tool_result_line", ""), 1600),
        })
    return summarized


def as_py(value):
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def row_to_plain(row):
    return {k: as_py(v) for k, v in row.items()}


def normalize_row(row, row_index):
    input_obj = row.get("input") or {}
    if isinstance(input_obj, str):
        try:
            input_obj = json.loads(input_obj) if input_obj else {}
        except Exception:
            input_obj = {}
    if not isinstance(input_obj, dict):
        input_obj = {}
    return {
        "row_index": row_index,
        "id": row.get("id"),
        "source": row.get("source"),
        "user_utterance": input_obj.get("user") or row.get("user"),
        "translated_user": row.get("user"),
        "target_sequence": row.get("msg"),
        "session_context": input_obj.get("context"),
        "available_tools": input_obj.get("available_tools"),
        "tool_steps": input_obj.get("tool_steps"),
        "reference_answer": input_obj.get("reference_answer"),
        "raw_time_fields": extract_time_fields(row),
    }


def extract_time_fields(obj):
    names = {
        "timestamp",
        "start_time",
        "end_time",
        "duration",
        "duration_sec",
        "audio_duration",
        "latency",
        "tool_latency",
    }
    found = {}

    def walk(prefix, value, depth=0):
        if depth > 4:
            return
        if isinstance(value, dict):
            for k, v in value.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                if str(k) in names:
                    found[key] = v
                walk(key, v, depth + 1)
        elif isinstance(value, list):
            for i, item in enumerate(value[:20]):
                walk(f"{prefix}[{i}]", item, depth + 1)

    walk("", obj)
    return found


def estimate_say_duration(text):
    cjk_count = len(CJK_RE.findall(text or ""))
    return max(0.8, cjk_count / 4.5)


def estimate_tool_latency(tool_call_text):
    text = (tool_call_text or "").lower()
    if any(x in text for x in ["search", "web", "map", "geo", "place", "restaurant"]):
        return 5.0
    if any(x in text for x in ["calendar", "profile", "preference", "user"]):
        return 2.5
    if text:
        return 3.0
    return 4.0


def deterministic_signals(target_sequence):
    msg = target_sequence or ""
    first_sopr = SOPR_RE.search(msg)
    say_before_sopr = False
    first_say_text = ""
    if first_sopr:
        prefix = msg[: first_sopr.start()]
        say_match = SAY_RE.search(prefix)
        say_before_sopr = bool(say_match)
        first_say_text = say_match.group(1).strip() if say_match else ""

    tool_calls = TOOL_CALL_RE.findall(msg)
    tool_results = TOOL_RESULT_RE.findall(msg)
    say_blocks = SAY_RE.findall(msg)

    max_silence = 0.0
    for call in tool_calls:
        max_silence = max(max_silence, estimate_tool_latency(call))
    for say in say_blocks:
        max_silence = max(0.0, max_silence - estimate_say_duration(say))

    return {
        "has_say_before_first_sopr": say_before_sopr,
        "first_say_text": compact(first_say_text, 300),
        "tool_call_count": len(tool_calls),
        "tool_result_count": len(tool_results),
        "say_count": len(say_blocks),
        "has_eor": "[EOR]" in msg,
        "estimated_max_silence_gap_sec": round(max_silence, 1),
        "timing_estimated": True,
    }


def build_judge_payload(row, row_index):
    normalized = normalize_row(row, row_index)
    signals = deterministic_signals(normalized.get("target_sequence"))
    target_events = summarize_target_events(normalized.get("target_sequence"))
    return {
        "row": {
            "row_index": row_index,
            "id": normalized.get("id"),
            "source": normalized.get("source"),
            "user_utterance": normalized.get("user_utterance"),
            "translated_user": normalized.get("translated_user"),
            "target_events": target_events,
            "target_sequence_excerpt": compact_head_tail(normalized.get("target_sequence") or "", 4000),
            "session_context": normalized.get("session_context"),
            "available_tools": normalized.get("available_tools"),
            "tool_steps": summarize_tool_steps(normalized.get("tool_steps")),
            "reference_answer": normalized.get("reference_answer"),
            "raw_time_fields": normalized.get("raw_time_fields"),
        },
        "deterministic_signals": signals,
    }


def parse_json_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_score(obj, fallback_silence=None):
    scores = obj.get("scores") or {}
    clean_scores = {}
    for key in SCORE_KEYS:
        value = scores.get(key, 0)
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 0
        clean_scores[key] = max(0, min(2, value))

    total = sum(clean_scores.values())
    obj["scores"] = clean_scores
    obj["total"] = total
    obj["score_total"] = total
    obj["percentage"] = round(total / MAX_SCORE * 100, 2)
    if obj.get("max_silence_gap_sec") is None:
        obj["max_silence_gap_sec"] = fallback_silence
    obj["timing_estimated"] = bool(obj.get("timing_estimated", True))
    for key in ["temporal_causality_errors", "grounding_errors", "deduction_reasons"]:
        if not isinstance(obj.get(key), list):
            obj[key] = []
    for key in ["critical_errors", "minor_errors"]:
        if not isinstance(obj.get(key), list):
            obj[key] = []
    if not isinstance(obj.get("customer_service_comment"), str):
        obj["customer_service_comment"] = str(obj.get("customer_service_comment") or "")
    if not isinstance(obj.get("comment"), str):
        obj["comment"] = str(obj.get("comment") or obj.get("customer_service_comment") or "")

    hard_fail = any(clean_scores.get(key, 0) == 0 for key in HARD_FAIL_SCORE_KEYS)
    hard_fail = hard_fail or bool(obj.get("critical_errors"))
    if "keep" not in obj:
        obj["keep"] = total >= 17 and not hard_fail
    else:
        obj["keep"] = bool(obj.get("keep")) and not hard_fail
    if not isinstance(obj.get("bucket"), str) or obj["bucket"] not in {"keep", "review", "reject"}:
        if obj["keep"]:
            obj["bucket"] = "keep"
        elif total >= 14 and not hard_fail:
            obj["bucket"] = "review"
        else:
            obj["bucket"] = "reject"
    return obj


def bucket_from_eval_scores(eval_scores):
    if isinstance(eval_scores, str):
        try:
            eval_scores = json.loads(eval_scores) if eval_scores else {}
        except Exception:
            eval_scores = {}
    if not isinstance(eval_scores, dict):
        return "reject"

    bucket = eval_scores.get("bucket")
    if bucket in {"keep", "review", "reject"}:
        return bucket
    if bool(eval_scores.get("keep")):
        return "keep"
    return "reject"


def row_eval_bucket(row):
    return bucket_from_eval_scores(row.get("eval_scores"))


def table_for_eval_bucket(table, bucket):
    rows = [row for row in table.to_pylist() if row_eval_bucket(row) == bucket]
    if not rows:
        return table.slice(0, 0)
    return pa.Table.from_pylist(rows, schema=table.schema)


def call_chat_completion(base_url, model, api_key, payload, temperature, timeout, retries):
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "請評分以下單筆 Agent-STITCH-S 資料，只輸出 JSON。\n\n"
                + json.dumps(payload, ensure_ascii=False, default=str),
            },
        ],
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                parsed = json.loads(resp.read().decode("utf-8"))
                return parsed["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(8, 2**attempt))
    raise RuntimeError(f"judge request failed: {last_error}")


def score_row(row, row_index, args, api_key):
    payload = build_judge_payload(row, row_index)
    fallback_silence = payload["deterministic_signals"]["estimated_max_silence_gap_sec"]
    if args.dry_run:
        scores = {key: 1 for key in SCORE_KEYS}
        if payload["deterministic_signals"]["has_say_before_first_sopr"]:
            scores["speech_first"] = 2
        else:
            scores["speech_first"] = 0
        if payload["deterministic_signals"]["has_eor"]:
            scores["stitch_markers"] = 1
        else:
            scores["stitch_markers"] = 0
        return validate_score(
            {
                "scores": scores,
                "max_silence_gap_sec": fallback_silence,
                "timing_estimated": True,
                "temporal_causality_errors": [],
                "grounding_errors": [],
                "deduction_reasons": ["dry_run heuristic score; not a real judge score"],
                "customer_service_comment": "dry_run only",
            },
            fallback_silence,
        )

    raw = call_chat_completion(
        args.judge_base_url,
        args.judge_model,
        api_key,
        payload,
        args.temperature,
        args.timeout,
        args.retries,
    )
    return validate_score(parse_json_response(raw), fallback_silence)


def append_score_columns(table, score_objs):
    arrays = [table[name] for name in table.column_names]
    fields = [field for field in table.schema]

    def add(name, values, typ):
        arrays.append(pa.array(values, type=typ))
        fields.append(pa.field(name, typ))

    add("eval_scores", [json.dumps(s, ensure_ascii=False) for s in score_objs], pa.string())
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def iter_parquet_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(path.glob("*.parquet"))


def process_parquet(args, api_key):
    input_path = Path(args.input)
    output_path = Path(args.output)
    files = iter_parquet_files(input_path)
    if not files:
        raise RuntimeError(f"No parquet files found under {input_path}")
    if len(files) > 1 or input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    global_row = 0
    written = 0
    for file_path in files:
        table = pq.read_table(file_path)
        rows = table.to_pylist()
        if args.max_rows is not None:
            remaining = args.max_rows - written
            if remaining <= 0:
                break
            rows = rows[:remaining]
            table = table.slice(0, len(rows))

        score_objs = []
        for row in rows:
            score_objs.append(score_row(row, global_row, args, api_key))
            global_row += 1
            written += 1
            if written % args.progress_every == 0:
                print(f"scored {written} rows", flush=True)

        out_table = append_score_columns(table, score_objs)
        out_file = output_path / file_path.name if output_path.is_dir() else output_path
        pq.write_table(out_table, out_file)
    print(f"done: scored {written} rows -> {output_path}", flush=True)


def process_jsonl(args, api_key):
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if args.max_rows is not None and written >= args.max_rows:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            score = score_row(row, written, args, api_key)
            row["eval_scores"] = json.dumps(score, ensure_ascii=False)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written % args.progress_every == 0:
                print(f"scored {written} rows", flush=True)
    print(f"done: scored {written} rows -> {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Score Agent-STITCH-S rows and append eval columns.")
    parser.add_argument("--input", required=True, help="Input JSONL, parquet file, or parquet directory.")
    parser.add_argument("--output", required=True, help="Output JSONL, parquet file, or parquet directory.")
    parser.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL"))
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true", help="Write heuristic placeholder scores without calling a judge.")
    args = parser.parse_args()

    if not args.dry_run and (not args.judge_base_url or not args.judge_model):
        raise SystemExit("--judge-base-url and --judge-model are required unless --dry-run is set")

    api_key = os.environ.get(args.api_key_env)
    suffix = Path(args.input).suffix.lower()
    if suffix == ".jsonl":
        process_jsonl(args, api_key)
    else:
        process_parquet(args, api_key)


if __name__ == "__main__":
    main()
