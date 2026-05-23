# MoE FP8 Phase 2 Prompt

Develop a CUDA kernel that minimizes latency while preserving numerical correctness. The target machine is NVIDIA B200, and the software environment is CUDA 13.2. For this post-competition MoE prompt, implement the solution in CUDA C++ or CuTe DSL.

## Kernel Information

- Definition name: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
- Baseline solution name: `flashinfer_wrapper_9sdjf3`
- Operation type: `moe`
- Workload count: 19
- Variable axes:
  - `seq_len`
- Constant axes:
  - `num_experts = 256`
  - `num_local_experts = 32`
  - `hidden_size = 7168`
  - `intermediate_size = 2048`
  - `gemm1_out_size = 4096`
  - `num_hidden_blocks = 56`
  - `num_intermediate_blocks = 16`
  - `num_gemm1_out_blocks = 32`

The kernel receives routing logits and bias, FP8 block-scaled hidden states, FP8 block-scaled local expert weights for the gate/up and down projections, a local expert offset, and a routed scaling factor. It must return `output` with shape `[seq_len, 7168]` and dtype `bfloat16`.

The reference computation is:

1. Dequantize FP8 block-scaled hidden states and local expert weights.
2. Run DeepSeek-V3 no-aux routing:
   - `s = sigmoid(routing_logits)`
   - add routing bias only for expert selection
   - group 256 experts into 8 groups of 32 experts
   - score each group by the sum of its top-2 biased scores
   - keep the top-4 groups
   - select top-8 experts from the kept groups
   - compute normalized routing weights from the unbiased sigmoid scores and apply `routed_scaling_factor`
3. For selected local experts, run GEMM1, SwiGLU, GEMM2, and weighted accumulation into the output.

## Official Acceptance

The MoE task uses a looser official correctness threshold than the other released kernels. The solution must pass the official FlashInfer benchmark checks with:

```text
--atol 1 --rtol 0.3 --required-matched-ratio 0.9
```

Do not use the stricter default tolerance when validating this task. Consult the official FlashInfer FAQ when dependency or rule questions are unclear:

```text
https://github.com/flashinfer-ai/flashinfer-bench-starter-kit/blob/main/FAQ.md
```

When working in this release repository, a full validation run can be launched with:

```bash
uv run python verify.py --solution /path/to/solution.json --fast
```

During development, use eight representative workloads before running all 19 workloads:

| UUID | seq_len |
|---|---:|
| `e05c6c03-5603-4a1c-b34c-dcce0ecaeea4` | 1 |
| `b8f4f012-a32e-4356-b4e1-7665b3d598af` | 7 |
| `a7c2bcfd-a2f4-479e-8d32-200115df89cf` | 16 |
| `6230e838-67ca-41dd-a9d6-6f36b7676c6b` | 32 |
| `fc378037-e8fa-4305-b00f-4af47933fd53` | 53 |
| `8f1ff9f1-6747-41d1-a1d8-2868cdacf893` | 80 |
| `1a4c6ba1-3cd2-4d7d-b716-84f2d52b69fc` | 901 |
| `5e8dc11c-f2a9-42d5-8dce-9419cbf34d5d` | 14107 |

After any major performance improvement, run the full 19-workload evaluation.

## Workflow Requirements

- Record every performance-related commit in `benchmark.csv`.
- Record every candidate in `solutions.jsonl` and maintain parent links as a DAG.
- Keep NCU profiling records for each major optimization direction.
- Actively evaluate and use as many relevant B200, CUDA 13.2, CUDA C++, and CuTe DSL features as possible. In particular, look for features that improve FP8 block-scale GEMM, scheduling, memory movement, CUDA Graph capture, shape-specialized dispatch, and `tcgen05` usage where applicable.
- Use KernelWiki for research on Blackwell/B200, CUDA 13.2, CUDA C++, CuTe DSL, FP8 block scaling, MoE routing, grouped GEMM, SwiGLU, expert load imbalance, CUDA Graphs, shape-specialized dispatch, and B200-specific features such as TMA, TMEM, warp specialization, and `tcgen05`.
- Use ncu-report-skill when profiling or interpreting Nsight Compute reports.
- Do not copy final released submission code into the starting workspace.

## Phase 2 Goal

Start from the best correct Phase 1 implementation. Phase 2 is an exploration phase: use NCU profiling, KernelWiki, and public documentation to identify as many plausible CUDA optimization directions as possible, then explore them systematically.

The draft must list the candidate optimization directions, rank them by expected benefit and implementation risk, and split each direction into concrete subtasks. Consider routing overhead, expert compaction, grouped GEMM scheduling, FP8 block-scale application, GEMM1/SwiGLU fusion, GEMM2 cost, weighted accumulation, expert load imbalance, tiny-`seq_len` overhead, large-prefill throughput, CUDA Graph capture, CuTe tiling, and shape-specific dispatch.

Explore each optimization direction for at most five iterations. If a direction cannot be implemented cleanly, fails correctness under the MoE acceptance thresholds, or does not show a credible path to improvement after those iterations, record the evidence and move to the next ranked direction.

For each explored direction, collect before/after benchmark results on the eight representative workloads and enough NCU evidence to justify whether the change should be kept, revised, or rejected.

Before implementing, write an implementation-plan draft and save it to:

```text
docs/draft.md
```

Prepare to run `/humanize:gen-plan` on that draft to generate the detailed implementation plan.
