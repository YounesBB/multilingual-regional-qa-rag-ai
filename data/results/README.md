# Result Data

This directory contains small CSV artifacts from the CUS-QA native-text RAG
experiments. Large generated JSONL outputs, model caches, retrieval indexes,
and SLURM logs are intentionally excluded from Git.

- `tables/`: cleaned result and diagnostic tables used for reporting.
- `run_summaries/`: small CSV summaries copied from generated run outputs.

The full pipeline writes large intermediate files under `data/runs/` or
`$CUSQA_WORK`; those files are reproducible from the scripts and SLURM wrappers.
