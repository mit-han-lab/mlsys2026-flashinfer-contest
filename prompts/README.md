# Prompt Release Notes

This directory stores the released prompts for reproducing the agent workflow.

Prompts are organized by kernel and by phase:

```text
prompts/
  moe-fp8/
    phase1.md
    phase2.md
    phase3.md
  gdn-prefill/
    phase1.md
    phase2.md
    phase3.md
  ...
```

## Common Workflow

Each phase follows the same high-level interaction pattern:

1. Start from a separate task implementation workspace created from the [official FlashInfer starter kit](https://github.com/flashinfer-ai/flashinfer-bench-starter-kit.git).
2. Paste the phase prompt from this directory into the agent session.
3. Ask the agent to investigate the repository, workload metadata, profiling evidence, KernelWiki, and relevant public documentation.
4. Require the agent to write its plan draft to `docs/draft.md`.
5. Run `/humanize:gen-plan` to convert `docs/draft.md` into a detailed implementation plan.
6. Run `/humanize:start-rlcr-loop` to start the implementation and review loop.
7. Record performance commits, candidate relationships, benchmark results, and NCU evidence inside the active experiment workspace.

Because LLM responses, search order, profiling noise, and implementation choices are not deterministic, this prompt release cannot guarantee that a future run will reproduce the exact same kernels or optimization path that we submitted. The goal of this release is to document the workflow that produced the submissions.

## Iterative Prompt Refinement

The prompts in this directory are starting points, not fixed scripts. Future users can progressively add stronger requirements as the search matures. In particular, we encourage users to run multiple rounds of Phase 2 and Phase 3 for any kernel. Between rounds, the user can re-invoke the relevant phase prompt with a gradually higher target speedup, stronger full-workload validation requirements, or stricter promotion rules. The target speedup is set by the user, and the agent's task is to keep optimizing until it reaches that user-specified target or produces concrete benchmark and profiling evidence explaining why the current round cannot reach it. The agent should not silently redefine the target or replace the intended baseline.

Human guidance is also useful and should be written directly into the prompt when available. Examples include:

- Which hardware features the agent should try to exploit, such as TMA, TMEM, `tcgen05`, warp specialization, persistent scheduling, or CUDA Graphs.
- Which bottlenecks are likely based on human experience with this kernel family.
- Which implementation directions are known to be risky, too complex, or unlikely to beat the baseline.
- Which workload shapes matter most for the current phase.

These extra constraints and hints can be given when invoking `/humanize:gen-plan` from the CLI or Claude Code session. The key is to make the human intent explicit before the detailed plan is generated.


## Shared Requirements

All task prompts should preserve these requirements unless a specific phase overrides them:

- The kernel must pass the official FlashInfer correctness checks.
- Optimize latency on NVIDIA B200.
- CUDA C++, CuTe DSL, Triton, and other contest-allowed implementation languages are permitted.
- The agent should actively look for ways to exploit B200 and CUDA 13.2 features whenever they are relevant, including TMA, TMEM, `tcgen05`, warp specialization, persistent scheduling, and wide/coalesced memory operations.
- Consult the official FlashInfer FAQ when dependency or rule questions are unclear:

```text
https://github.com/flashinfer-ai/flashinfer-bench-starter-kit/blob/main/FAQ.md
```

- Use KernelWiki for Blackwell, CUDA 13.2, CuTe DSL, Triton, and prior implementation research.
- Use ncu-report-skill when NCU profiling evidence is available or required.
- Keep final submission code separate from the fresh agent workspace.

## Phase Semantics

Phase 1 focuses on research and producing a correct B200 kernel.

Phase 2 focuses on profiling-guided bottleneck analysis and iterative performance optimization. Users may repeat Phase 2 multiple times with progressively higher explicit speedup targets.

Phase 3 focuses on workload-shape analysis and specializing the best kernels for different shape groups. Users may also repeat Phase 3 after each successful optimization round, again raising the target speedup or validation bar explicitly in the next user prompt.
