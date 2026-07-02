# Generate Pipeline

This package creates assembled Agent-STITCH-S SFT rows.

## Files

```text
pipeline.py    CLI wrapper; chooses local or API backend
local.py       local Ray/vLLM backend
api.py         OpenAI-compatible API backend
common.py      shared canonicalization and deterministic assembly logic
```

## Logic

Generation has two separate responsibilities:

1. Ask a model to write only the STITCH-S patch:
   `translated_user`, `first_say`, per-step private state, per-step bridge SAY, final private state, and final SAY.
2. Deterministically restore the original tool calls and tool results from the source data.

The model is not allowed to rewrite tool calls or tool results. That is enforced by `common.py` during assembly.

Before model inference, both backends filter out rows that are not useful for this dataset:

```text
must have at least one tool call/tool result pair
must have a non-empty reference answer
```

Filtered rows are written to the `failed` output with drop reasons:

```text
missing_tool_call_tool_result
missing_reference_answer
missing_tool_call_tool_result+missing_reference_answer
```

For prompt efficiency, generation sends compact tool-step summaries to the model:

```text
tool_call_brief is capped to 300 characters
long tool observations keep the beginning and end with a TRUNCATED marker
full tool calls and tool results are restored during deterministic assembly
```

Local generation estimates prompt length and drops over-budget rows as `prompt_too_long`. API generation sends the request directly; provider context-limit errors are captured as `api_generation_failed`.

## Local Backend

Local backend uses Ray Data and vLLM.

```bash
python3 generate/pipeline.py \
  --backend local \
  --data voidful/agent-sft \
  --out ./out_full
```

Useful environment variables:

```text
MODEL_ID
MAX_MODEL_LEN
MAX_OUTPUT_TOKENS
MIN_OUTPUT_TOKENS
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

Use `--only-ids` to run a text, CSV, or JSONL list of row ids. For CSV/text, the first column is treated as the id.

## API Backend

API backend uses OpenAI-compatible `/chat/completions`.
`--api-key` or `API_KEY` may contain one key or comma-separated keys. Multiple keys are used round-robin across requests and retries.

```bash
python3 generate/pipeline.py \
  --backend api \
  --data voidful/agent-sft \
  --out ./out_full_api \
  --api-key "$API_KEY" \
  --model google/gemma-4-26B-A4B-it \
  --concurrency 4 \
  --max-rows 100
```

For non-Google OpenAI-compatible endpoints:

```bash
python3 generate/pipeline.py \
  --backend api \
  --data /path/to/input.parquet \
  --out ./out_full_api \
  --api-key "$API_KEY" \
  --base-url https://your-endpoint/v1 \
  --model your-model
```

## Input

Input can be:

```text
voidful/agent-sft
hf://datasets/<repo>
/path/to/parquet_file
/path/to/parquet_directory
```

## Output

```text
<out>/runs/<run_id>/success
<out>/runs/<run_id>/failed
<out>/runs/<run_id>/preview/success.jsonl
<out>/runs/<run_id>/preview/failed.jsonl
```

`success` rows contain:

```text
id
source
user
msg
input
```

`msg` is the assembled STITCH-S trajectory.
