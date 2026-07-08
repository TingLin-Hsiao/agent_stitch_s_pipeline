# Prompts

This directory stores prompts outside Python code so the pipeline logic stays clean.

## Files

```text
generate_stitch_s.txt
eval_stitch_s.txt
```

## generate_stitch_s.txt

Used by both generation backends:

```text
generate/local.py
generate/api.py
```

The model must output only a patch JSON:

```text
translated_user
first_say
steps[*].pre_tool_private_state
steps[*].post_tool_say
final_private_state
final_say
drop_reason
```

The model must not output tool calls or tool results. Those are restored later by deterministic assembly.

Important rules:

```text
SAY chunks are Traditional Chinese.
Private states are English.
SAY chunks must be spoken, TTS-friendly, non-markdown text.
Bridge SAY chunks must not reveal pending tool results.
Final SAY must be grounded in tool results and reference answer.
```

TTS guardrails:

```text
Do not put raw URLs, emails, file paths, command lines, code, JSON, Markdown,
tables, placeholder tokens, long IDs, or full generated drafts inside SAY.
Rewrite them as short spoken summaries and leave exact structured text to the
visible non-spoken output.
```

## eval_stitch_s.txt

Used by both eval backends:

```text
eval/local.py
eval/api.py
```

It defines:

```text
9 scoring categories
18 point total score
hard-fail conditions
keep/review/reject thresholds
spoken SAY quality checks
tool validity checks
grounding and temporal causality rules
```

When editing the eval prompt, keep its JSON output fields aligned with `eval/scoring.py`.
The eval prompt also hard-fails SAY chunks that leak raw structured content or
other text that is useful on screen but unsuitable for direct TTS playback.
