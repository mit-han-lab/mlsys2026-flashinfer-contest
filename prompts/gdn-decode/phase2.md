# GDN Decode Phase 2 Prompt

Develop a kernel that minimizes latency while preserving numerical correctness. The target machine is NVIDIA B200, and the software environment is CUDA 13.2. This task does not restrict the implementation language: CUDA C++, CuTe DSL, Triton, Python-wrapped CUDA extensions, or any other contest-allowed approach may be used.

## Kernel Information

- Definition name: `gdn_decode_qk4_v8_d128_k_last`
- Baseline solution name: `flashinfer_wrapper_9b7f1e`
- Operation type: `gdn`
- Workload count: 54
- Constant axes:
  - `seq_len = 1`
  - `num_q_heads = 4`
  - `num_k_heads = 4`
  - `num_v_heads = 8`
  - `head_size = 128`
- Variable axes:
  - `batch_size`

The kernel receives single-token Gated Delta Net decode inputs in k-last state layout. It must return:

- `output` with shape `[batch_size, 1, 8, 128]` and dtype `bfloat16`
- `new_state` with shape `[batch_size, 8, 128, 128]` and dtype `float32`

The reference computation is:

1. Compute `g = exp(-exp(A_log) * softplus(a + dt_bias))`.
2. Compute `beta = sigmoid(b)`.
3. Expand 4 query/key heads to 8 value heads by grouped-value attention mapping.
4. Update the recurrent state in k-last layout `[B, H, V, K]`.
5. Compute `output = scale * q @ new_state`.

## Official Acceptance

The solution must pass the official FlashInfer benchmark correctness checks for `gdn_decode_qk4_v8_d128_k_last`. Use the official FlashInfer benchmark/starter-kit evaluator and consult the FAQ when dependency or rule questions are unclear:

```text
https://github.com/flashinfer-ai/flashinfer-bench-starter-kit/blob/main/FAQ.md
```

When working in this release repository, a full validation run can be launched with:

```bash
uv run python verify.py --solution /path/to/solution.json --fast
```

During development, use eight representative workloads before running all 54 workloads:

| UUID | batch_size |
|---|---:|
| `901e5104-dccb-4c3f-ae13-ef4d31a4d456` | 1 |
| `a5714b69-525c-4b95-bb7a-a0f9770c2f48` | 1 |
| `ec9d2340-6d13-40e4-a6fe-4483a1cacd0d` | 4 |
| `2640f1e3-4b02-4041-bdc1-59a28e0b9954` | 8 |
| `76eec66d-0ada-4c35-bf21-a07247ad7f05` | 16 |
| `a8c8beff-e414-4580-b2f7-e5b8f13bc269` | 32 |
| `53385c7f-393d-41db-aec8-5b9eb5bf35d1` | 48 |
| `eaf0a285-447c-4432-8e68-d287acc3cb08` | 64 |

After any major performance improvement, run the full 54-workload evaluation.

## Workflow Requirements

- Record every performance-related commit in `benchmark.csv`.
- Record every candidate in `solutions.jsonl` and maintain parent links as a DAG.
- Keep NCU profiling records for each major optimization direction.
- Actively evaluate and use as many relevant B200 and CUDA 13.2 features as possible, including TMA, TMEM, `tcgen05`, warp specialization, persistent scheduling, wide vectorized memory operations, and coalesced memory access when they fit the kernel.
- Use KernelWiki for research on Blackwell/B200, CUDA 13.2, CuTe DSL, Triton, Gated Delta Net decode, recurrent state updates, BF16 vector loads, memory coalescing, small-kernel launch overhead, and B200-specific features such as TMA, TMEM, warp specialization, and `tcgen05`.
- Use ncu-report-skill when profiling or interpreting Nsight Compute reports.
- Do not copy final released submission code into the starting workspace.

## Phase 2 Goal

Start from the best correct Phase 1 implementation. Phase 2 is an exploration phase: use NCU profiling, KernelWiki, and public documentation to identify as many plausible optimization directions as possible, then explore them systematically.

This is an extremely small decode kernel, with latency on the order of only a few microseconds. Treat memory access details as first-class optimization targets. Pay special attention to using the widest safe load/store instructions, preserving coalesced and contiguous memory access, avoiding unnecessary global-memory traffic, reducing instruction overhead, and keeping register pressure low enough for good occupancy. Small inefficiencies can dominate at this scale.

The draft must list the candidate optimization directions, rank them by expected benefit and implementation risk, and split each direction into concrete subtasks. Consider BF16 vectorized loads for `q`, `k`, and `v`, vectorized or cache-hinted loads for `state`, contiguous row access in k-last layout, split strategy over state rows, warp-level reductions, approximations for gate math that preserve correctness, launch overhead, and batch-size-specific behavior.

Explore each optimization direction for at most five iterations. If a direction cannot be implemented cleanly, fails correctness, or does not show a credible path to improvement after those iterations, record the evidence and move to the next ranked direction.

For each explored direction, collect before/after benchmark results on the eight representative workloads and enough NCU evidence to justify whether the change should be kept, revised, or rejected.

Before implementing, write an implementation-plan draft and save it to:

```text
docs/draft.md
```

Prepare to run `/humanize:gen-plan` on that draft to generate the detailed implementation plan.
