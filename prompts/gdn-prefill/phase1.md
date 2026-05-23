# GDN Prefill Phase 1 Prompt

Develop a kernel that minimizes latency while preserving numerical correctness. The target machine is NVIDIA B200, and the software environment is CUDA 13.2. This task does not restrict the implementation language: CUDA C++, CuTe DSL, Triton, Python-wrapped CUDA extensions, or any other contest-allowed approach may be used.

## Kernel Information

- Definition name: `gdn_prefill_qk4_v8_d128_k_last`
- Baseline solution name: `flashinfer_wrapper_123ca6`
- Operation type: `gdn`
- Workload count: 100
- Constant axes:
  - `num_q_heads = 4`
  - `num_k_heads = 4`
  - `num_v_heads = 8`
  - `head_size = 128`
- Variable axes:
  - `total_seq_len`
  - `num_seqs`
  - `len_cu_seqlens`

The kernel receives variable-length Gated Delta Net prefill inputs in k-last state layout. It must return:

- `output` with shape `[total_seq_len, 8, 128]` and dtype `bfloat16`
- `new_state` with shape `[num_seqs, 8, 128, 128]` and dtype `float32`

The reference computation is:

1. Compute `g = exp(-exp(A_log) * softplus(a + dt_bias))`.
2. Compute `beta = sigmoid(b)`.
3. Expand 4 query/key heads to 8 value heads by grouped-value attention mapping.
4. For each sequence defined by `cu_seqlens`, scan tokens in order and update the recurrent state.
5. Compute `output = scale * q @ state` at each token.
6. Return the final state for every sequence in k-last layout `[N, H, V, K]`.

## Official Acceptance

The solution must pass the official FlashInfer benchmark correctness checks for `gdn_prefill_qk4_v8_d128_k_last`. Use the official FlashInfer benchmark/starter-kit evaluator and consult the FAQ when dependency or rule questions are unclear:

```text
https://github.com/flashinfer-ai/flashinfer-bench-starter-kit/blob/main/FAQ.md
```

When working in this release repository, a full validation run can be launched with:

```bash
uv run python verify.py --solution /path/to/solution.json --fast
```

During development, use eight representative workloads before running all 100 workloads:

| UUID | total_seq_len | num_seqs | len_cu_seqlens |
|---|---:|---:|---:|
| `77daf91d-0660-4c4b-8c32-336a69281cd9` | 6 | 1 | 2 |
| `d8f4a9ae-d391-4f70-8069-f5deb510a2d1` | 35 | 1 | 2 |
| `ba08a83e-e151-4e16-bc70-abee6851604c` | 134 | 1 | 2 |
| `ce832e76-c4c2-421f-ad75-ebda2032c401` | 461 | 2 | 3 |
| `c5257f65-c411-4dbb-9dc1-ad4abcf00254` | 983 | 2 | 3 |
| `aaa378be-7365-4381-b2d3-35951ef88c7a` | 1800 | 3 | 4 |
| `4b6143dd-0e5f-499f-93cb-076d9635bcd0` | 4124 | 15 | 16 |
| `d49df0b2-7838-4865-9a51-7ec7877fe27f` | 8192 | 48 | 49 |

After any major performance improvement, run the full 100-workload evaluation.

## Workflow Requirements

- Record every performance-related commit in `benchmark.csv`.
- Record every candidate in `solutions.jsonl` and maintain parent links as a DAG.
- Keep NCU profiling records for each major optimization direction.
- Actively evaluate and use as many relevant B200 and CUDA 13.2 features as possible, including TMA, TMEM, `tcgen05`, warp specialization, persistent scheduling, wide vectorized memory operations, and coalesced memory access when they fit the kernel.
- Use KernelWiki for research on Blackwell/B200, CUDA 13.2, CuTe DSL, Triton, Gated Delta Net prefill, recurrent state scans, BF16 math, variable-length sequence batching, warp/block scheduling, TMA, TMEM, and `tcgen05`.
- Use ncu-report-skill when profiling or interpreting Nsight Compute reports.
- Do not copy final released submission code into the starting workspace.

## Phase 1 Goal

Research existing GDN prefill and linear-attention implementations and produce the first correct B200 implementation. Focus on understanding the recurrence, k-last state layout, variable-length batching through `cu_seqlens`, numerical requirements, and a simple correct implementation strategy. Performance matters, but correctness and a clean baseline design are the priority for this phase.

Before implementing, write an implementation-plan draft and save it to:

```text
docs/draft.md
```

Prepare to run `/humanize:gen-plan` on that draft to generate the detailed implementation plan.
