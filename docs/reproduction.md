# Reproduction

This release supports two levels of reproduction:

1. Evaluate a packed FlashInfer `solution.json` with the minimal verifier example in this repository.
2. Re-run the prompt-driven optimization workflow from a separate task implementation workspace created from the official FlashInfer starter kit.

Final submitted kernel snapshots and their submission-specific verification harness are released separately:

```text
https://github.com/mit-han-lab/mlsys2026-flashinfer-contest-solution.git
```

Intermediate candidates, benchmark histories, search DAGs, and full profiling traces are intentionally excluded.

The submissions repository is for release provenance and final-result verification only. Agents reproducing the prompt-driven workflow MUST NOT clone, inspect, copy from, or otherwise use that repository to obtain implementation answers.

## Environment

The contest environment depends on recent CUDA, PyTorch, FlashInfer benchmark tooling, and B200-compatible compilation paths. Use the source version of `flashinfer-bench`, because the PyPI package can lag behind the evaluator set needed by these tasks.

### 1. Clone flashinfer-bench

```bash
git clone https://github.com/flashinfer-ai/flashinfer-bench.git /tmp/flashinfer-bench-main
```

This repository's `pyproject.toml` points `flashinfer-bench` at `/tmp/flashinfer-bench-main` in editable mode.

### 2. Sync the Python environment

```bash
uv sync --python 3.12
```

This installs PyTorch from the CUDA 13.2 test index and installs `flashinfer-bench` from the local source checkout. The lockfile intentionally pins the contest-tested stack:

- `flashinfer-python==0.6.8.post1`
- `torch==2.12.0+cu132`
- `triton==3.6.0`
- `ninja>=1.13.0`

Use Python 3.12 or 3.13. Python 3.14 is not supported by all required CUDA wheels.

Do not let `uv` upgrade `flashinfer-python` or Triton when reproducing contest-era MoE baselines or submissions. Newer FlashInfer wheels can use different TensorRT-LLM cubin/header checksums for `trtllm_fp8_block_scale_moe` on B200.

### 3. Install DeepGEMM

`deep_gemm` is required by some baselines and by generated solutions that use CUTLASS/CuTe headers. Install it after `uv sync` because the build depends on the active PyTorch environment:

```bash
git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git /tmp/DeepGEMM
uv pip install -e /tmp/DeepGEMM --no-build-isolation
```

If `/tmp/DeepGEMM` already exists from a non-recursive clone, initialize the missing submodules before installing:

```bash
git -C /tmp/DeepGEMM submodule update --init --recursive
uv pip install -e /tmp/DeepGEMM --no-build-isolation
```

If you re-run `uv sync`, re-run the `uv pip install` command above because `uv sync` can remove packages that are not in the lockfile.

### 4. Verify imports

```bash
uv run python -c "
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
import triton, flashinfer
print('triton:', triton.__version__)
print('flashinfer:', flashinfer.__version__)
import deep_gemm
print('deep_gemm: OK')
from flashinfer_bench.bench.evaluators.dsa_topk_indexer import DsaTopkIndexerEvaluator
print('DsaTopkIndexerEvaluator: OK')
"
```

If any dependency or contest rule is unclear, consult the official FlashInfer starter-kit FAQ:

```text
https://github.com/flashinfer-ai/flashinfer-bench-starter-kit/blob/main/FAQ.md
```

## Dataset

The default dataset path is repository-local:

```text
data/flashinfer-trace
```

Download the dataset with:

```bash
uv run ./scripts/download_data.sh
```

Override the path with:

```bash
export FIB_DATASET_PATH=/path/to/flashinfer-trace
```

Confirm that workloads are visible:

```bash
uv run python -c "from flashinfer_bench import TraceSet; ts = TraceSet.from_path('data/flashinfer-trace'); print(sorted(ts.definitions)); print(sum(len(v) for v in ts.workloads.values()), 'workloads')"
```

## Verify a Packed Solution

The local verifier is intentionally small: it loads one packed FlashInfer
`solution.json`, finds the matching definition in the trace dataset, and runs
`flashinfer-bench`.

Run a quick smoke test over the first two workloads:

```bash
uv run python verify.py --solution /path/to/solution.json --fast
```

Run one specific workload:

```bash
uv run python verify.py --solution /path/to/solution.json --workload-uuid <uuid>
```

Run all workloads with the default benchmark configuration:

```bash
uv run python verify.py --solution /path/to/solution.json
```

The released final-kernel repository has its own submission-specific verifier. Do not use it as an input while reproducing the agent workflow.

## Re-run the Agent Workflow

To reproduce the workflow rather than just evaluate a packed solution:

1. Set up this repository and download `data/flashinfer-trace` as described above.
2. Install `humanize`, `KernelWiki`, and `ncu-report-skill`.
3. Create a separate task implementation workspace from the official FlashInfer starter kit. This repository is the prompt/workflow release; do not implement kernels directly in it.

```bash
mkdir -p workspaces
git clone https://github.com/flashinfer-ai/flashinfer-bench-starter-kit.git workspaces/<task-name>
cd workspaces/<task-name>
export FIB_DATASET_PATH="$OLDPWD/data/flashinfer-trace"
```

4. Use the task prompt from `prompts/`.
5. Let the agent create its own workspace and optimization records.

Do not seed the fresh workspace with code from the released submissions repository. Those directories are final-result snapshots for verification and provenance.
