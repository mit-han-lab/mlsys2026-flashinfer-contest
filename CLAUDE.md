# Agent Instructions

This repository is the Agents prompt and workflow release artifact for MLSys 2026 FlashInfer NVIDIA track kernel submissions.

## Repository Rules

- Use English for all repository-facing files, comments, documentation, prompts, and commit messages.
- During prompt-driven workflow reproduction, MUST NOT clone, inspect, copy from, or otherwise use the released submissions repository to obtain implementation answers.
- Do not copy final submission code into a fresh agent workspace when reproducing the search workflow.
- Use `data/flashinfer-trace` as the default dataset location. Respect `FIB_DATASET_PATH` when set.
- Generated outputs belong in `runs/`, `outputs/`, or `profile/`; these paths are ignored by git.
- Use Python 3.12 or 3.13 for the uv environment. Python 3.14 is not compatible with all required CUDA wheels.

## Expected Agent Workflow

For a new kernel optimization task:

1. Set up the environment from `README.md` or `docs/reproduction.md`.
2. Download the FlashInfer workload dataset into `data/flashinfer-trace`, or set `FIB_DATASET_PATH`.
3. Create or enter a separate task implementation workspace from the official FlashInfer starter kit for the target kernel.
4. Read the task prompt under `prompts/`.
5. Consult the official FlashInfer FAQ when contest rules or dependencies are unclear.
6. Use KernelWiki for Blackwell, CUDA, CuTe DSL, Triton, and prior kernel implementation research.
7. Use ncu-report-skill for Nsight Compute profiling and bottleneck analysis.
8. Draft the implementation plan in `docs/draft.md` inside the active agent workspace.
9. Record every performance-related commit in `benchmark.csv`.
10. Record every candidate in `solutions.jsonl` and maintain candidate parent links as a DAG.
11. Keep NCU profiling records for each major optimization direction.

## External Skills

Install these separately:

- `humanize`: `git@github.com:PolyArch/humanize.git`
- `KernelWiki`: `git@github.com:DongyunZou/KernelWiki.git`
- `ncu-report-skill`: `git@github.com:DongyunZou/ncu-report-skill.git`
