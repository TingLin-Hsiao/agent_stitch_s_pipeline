# Eval Pipeline

This package scores assembled Agent-STITCH-S rows.

## Files

```text
pipeline.py    CLI wrapper; chooses local or API backend
local.py       local Ray/vLLM judge backend
api.py         OpenAI-compatible API judge backend
scoring.py     shared payload building, event extraction, parsing, and score normalization
```

## Logic

Eval reads assembled rows with `msg` and `input`.

For each row, `scoring.py` builds a judge payload containing:

```text
user_utterance
translated_user
target_events
target_sequence_excerpt
available_tools
tool_steps
reference_answer
deterministic_signals
```

The judge returns a JSON object with 9 score categories, total score, `keep`, and `bucket`.

`scoring.py` compacts large target/tool payloads before judging while preserving event order and user-audible SAY text.

## Score Schema

Scores are 0 to 2 each, total 18:

```text
speech_first
tool_waiting_safety
temporal_causality
incremental_update
stitch_markers
silence_gap
grounding
customer_service_quality
tool_validity
```

Hard-fail categories are:

```text
speech_first
tool_waiting_safety
temporal_causality
grounding
tool_validity
```

If a hard-fail category is 0, `keep` becomes false.

Bucket thresholds:

```text
keep    total >= 17 and no hard fail / critical errors
review  total >= 14 and no hard fail / critical errors
reject  total < 14 or any hard fail / critical errors
```

Local eval estimates judge prompt length before inference. Rows over the local model context budget are scored as `prompt_too_long` with `keep=false` and `bucket=reject`. API eval sends requests directly; request or parse failures are converted to reject-style error scores.

## Local Backend

Local backend uses Ray Data and vLLM.

```bash
python3 eval/pipeline.py \
  --backend local \
  --input ./out_full/runs/<run_id>/success \
  --out ./out_eval
```

Useful environment variables:

```text
MODEL_ID
MAX_MODEL_LEN
MAX_OUTPUT_TOKENS
PROMPT_TOKEN_SAFETY_MARGIN
CHAT_TEMPLATE_TOKEN_OVERHEAD
TENSOR_PARALLEL_SIZE
VLLM_CONCURRENCY
VLLM_BATCH_SIZE
MAX_CONCURRENT_BATCHES
RAW_INPUT_BLOCKS
LLM_INPUT_BLOCKS
PREVIEW_ROWS
```

## API Backend

API backend uses OpenAI-compatible `/chat/completions`.
`--api-key` or `API_KEY` may contain one key or comma-separated keys. Multiple keys are used round-robin across requests and retries.

```bash
python3 eval/pipeline.py \
  --backend api \
  --input ./out_full/runs/<run_id>/success \
  --out ./out_eval_api \
  --api-key "$API_KEY" \
  --model google/gemma-4-26B-A4B-it \
  --concurrency 4
```

For custom endpoints:

```bash
python3 eval/pipeline.py \
  --backend api \
  --input /path/to/success \
  --out ./out_eval_api \
  --api-key "$API_KEY" \
  --base-url https://your-endpoint/v1 \
  --model your-model
```

## Output

Eval writes parquet rows with an added `eval_scores` JSON string:

```text
<out>/runs/<run_id>/scored
<out>/runs/<run_id>/keep
<out>/runs/<run_id>/review
<out>/runs/<run_id>/preview/scored.jsonl
<out>/runs/<run_id>/preview/keep.jsonl
<out>/runs/<run_id>/preview/review.jsonl
```

`scored` contains all rows. `keep` and `review` are filtered parquet views based on `eval_scores.bucket`.

`eval_scores` includes:

```text
scores
total
score_total
percentage
keep
bucket
max_silence_gap_sec
timing_estimated
critical_errors
minor_errors
deduction_reasons
comment
```

## Standalone Scorer

`scoring.py` can also score JSONL, a parquet file, or a parquet directory directly:

```bash
python3 eval/scoring.py \
  --input /path/to/input \
  --output /path/to/output \
  --judge-base-url https://your-endpoint/v1 \
  --judge-model your-model
```

Use `--dry-run` to write heuristic placeholder scores without calling a judge.
