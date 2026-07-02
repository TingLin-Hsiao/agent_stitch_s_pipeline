# Slurm Compatibility Scripts

This directory keeps small wrapper scripts for old workflows.

## Files

```text
run_generate.sh
run_eval.sh
```

Both wrappers call the root `run_pipeline.sh`.

## run_generate.sh

Equivalent to:

```bash
MODE=generate bash ../run_pipeline.sh
```

Submit:

```bash
sbatch --export=ALL,HF_TOKEN scripts/run_generate.sh
```

API generate:

```bash
sbatch --export=ALL,MODE=generate,GENERATE_BACKEND=api,API_KEY=... scripts/run_generate.sh
```

Multiple API keys are supported with comma separation, for example `API_KEY=key1,key2,key3`.

## run_eval.sh

Equivalent to:

```bash
MODE=eval bash ../run_pipeline.sh
```

Submit local eval:

```bash
sbatch --export=ALL,HF_TOKEN,EVAL_INPUT=/path/to/success scripts/run_eval.sh
```

Submit API eval:

```bash
sbatch --export=ALL,EVAL_BACKEND=api,API_KEY=...,EVAL_INPUT=/path/to/success scripts/run_eval.sh
```

Multiple API keys are supported with comma separation, for example `API_KEY=key1,key2,key3`.

For new workflows, prefer the root `run_pipeline.sh` directly.
