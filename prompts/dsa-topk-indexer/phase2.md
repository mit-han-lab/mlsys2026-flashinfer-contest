# DSA TopK Indexer Phase 2 Prompt

Develop a kernel that minimizes latency while preserving numerical correctness. The target machine is NVIDIA B200, and the software environment is CUDA 13.2. This task does not restrict the implementation language: CUDA C++, CuTe DSL, Triton, Python-wrapped CUDA extensions, or any other contest-allowed approach may be used.

## Kernel Information

- Definition name: `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`
- Baseline solution name: `flashinfer_deepgemm_wrapper_2ba145`
- Operation type: `dsa_paged`
- Workload count: 128
- Constant axes:
  - `num_index_heads = 64`
  - `index_head_dim = 128`
  - `page_size = 64`
  - `topk = 2048`
  - `kv_cache_num_heads = 1`
  - `head_dim_with_scale = 132`
- Variable axes:
  - `batch_size`
  - `max_num_pages`
  - `num_pages`

The kernel receives FP8 query vectors, an FP8 paged KV index cache in DeepGEMM layout, per-head weights, sequence lengths, and a block table. It must return `topk_indices` with shape `[batch_size, 2048]` and dtype `int32`. Padding entries must be `-1`.

The reference computation is:

1. Interpret the KV cache bytes as FP8 values plus one FP32 scale per token.
2. For each batch element, gather valid KV tokens through `block_table`.
3. Compute `scores = q @ K.T` for 64 heads.
4. Apply ReLU to the scores.
5. Multiply by the learned per-head `weights` and sum over heads.
6. Select top-2048 token indices and convert them to global paged-cache token indices.

## Official Acceptance

The solution must pass the official FlashInfer benchmark correctness checks for `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`. Use the official FlashInfer benchmark/starter-kit evaluator and consult the FAQ when dependency or rule questions are unclear:

```text
https://github.com/flashinfer-ai/flashinfer-bench-starter-kit/blob/main/FAQ.md
```

When working in this release repository, a full validation run can be launched with:

```bash
uv run python verify.py --solution /path/to/solution.json --fast
```

During development, use eight representative workloads before running all 128 workloads:

| UUID | batch_size | max_num_pages | num_pages |
|---|---:|---:|---:|
| `30cecff1-7ea4-474b-90fc-7f4a87206d8e` | 1 | 1 | 11923 |
| `02fa7f90-88b6-4fa5-ba21-29b803c23309` | 2 | 5 | 11923 |
| `05775386-52aa-4abb-982d-0aee72dd2ff6` | 4 | 18 | 11923 |
| `67216408-b49b-4e43-bf07-4136968b3b82` | 8 | 32 | 11923 |
| `ee603b53-f9d6-4210-a83a-caf6219a8a8b` | 15 | 43 | 11923 |
| `e4ecb462-e2be-41b3-8efc-795f49ef07bf` | 15 | 82 | 11923 |
| `a52c09bc-2ee5-4366-be02-457932a80631` | 31 | 43 | 11923 |
| `5db1b172-eda8-4714-9981-a069dc33d7e9` | 30 | 91 | 11923 |

After any major performance improvement, run the full 128-workload evaluation.

## Workflow Requirements

- Record every performance-related commit in `benchmark.csv`.
- Record every candidate in `solutions.jsonl` and maintain parent links as a DAG.
- Keep NCU profiling records for each major optimization direction.
- Actively evaluate and use as many relevant B200 and CUDA 13.2 features as possible, including TMA, TMEM, `tcgen05`, warp specialization, persistent scheduling, wide vectorized memory operations, and coalesced memory access when they fit the kernel.
- Use KernelWiki for research on Blackwell/B200, CUDA 13.2, CuTe DSL, Triton, DeepGEMM, sparse attention indexers, FP8, TMA, TMEM, and `tcgen05`.
- Use ncu-report-skill when profiling or interpreting Nsight Compute reports.
- Do not copy final released submission code into the starting workspace.

## Phase 2 Goal

Start from the best correct Phase 1 implementation. Phase 2 is an exploration phase: use NCU profiling, KernelWiki, and public documentation to identify as many plausible optimization directions as possible, then explore them systematically.

The draft must list the candidate optimization directions, rank them by expected benefit and implementation risk, and split each direction into concrete subtasks. Consider scoring throughput, KV cache load efficiency, page-table indirection, top-k selection cost, occupancy, register pressure, memory stalls, tail effects, and opportunities to use B200-specific features such as TMA, TMEM, and `tcgen05`.

Explore each optimization direction for at most five iterations. If a direction cannot be implemented cleanly, fails correctness, or does not show a credible path to improvement after those iterations, record the evidence and move to the next ranked direction.

For each explored direction, collect before/after benchmark results and enough NCU evidence to justify whether the change should be kept, revised, or rejected.

Before implementing, write an implementation-plan draft and save it to:

```text
docs/draft.md
```

Prepare to run `/humanize:gen-plan` on that draft to generate the detailed implementation plan.
