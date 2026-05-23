#!/usr/bin/env python3
"""Minimal FlashInfer solution verification example.
It loads one packed FlashInfer `solution.json`, selects workloads from a
FlashInfer trace dataset, and evaluates the solution with the official
`flashinfer_bench` entrypoint.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "data" / "flashinfer-trace"


def dataset_path(explicit: str | None) -> Path:
    raw = explicit or os.environ.get("FIB_DATASET_PATH")
    return Path(raw).expanduser().resolve() if raw else DEFAULT_DATASET


def import_flashinfer_bench():
    try:
        from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "flashinfer_bench is not installed. Run `uv sync`, or install the official "
            "FlashInfer benchmark package following docs/reproduction.md."
        ) from exc
    return Benchmark, BenchmarkConfig, Solution, TraceSet


def benchmark_config(solution_definition: str, args: argparse.Namespace):
    _, BenchmarkConfig, _, _ = import_flashinfer_bench()
    kwargs = dict(
        warmup_runs=args.warmup_runs,
        iterations=args.iterations,
        num_trials=args.num_trials,
        use_isolated_runner=args.isolated_runner,
    )
    if solution_definition.startswith("moe_fp8"):
        kwargs.update(atol=1.0, rtol=0.3, required_matched_ratio=0.9)
    return BenchmarkConfig(**kwargs)


def select_workloads(workloads: list, args: argparse.Namespace) -> list:
    if args.workload_uuid:
        wanted = set(args.workload_uuid)
        workloads = [w for w in workloads if w.workload.uuid in wanted]
    if args.limit is not None:
        workloads = workloads[: max(0, args.limit)]
    return workloads


def run(args: argparse.Namespace) -> bool:
    Benchmark, _, Solution, TraceSet = import_flashinfer_bench()

    solution_path = Path(args.solution).expanduser().resolve()
    solution = Solution.model_validate_json(solution_path.read_text())

    dataset = dataset_path(args.dataset)
    if not dataset.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset}\n"
            "Run ./scripts/download_data.sh or set FIB_DATASET_PATH."
        )
    trace_set = TraceSet.from_path(str(dataset))

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition {solution.definition!r} not found in {dataset}")

    definition = trace_set.definitions[solution.definition]
    workloads = select_workloads(list(trace_set.workloads.get(solution.definition, [])), args)
    if not workloads:
        raise ValueError(f"No workloads selected for {solution.definition}")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    result_trace_set = Benchmark(
        bench_trace_set,
        benchmark_config(solution.definition, args),
    ).run_all(dump_traces=False)

    traces = list(result_trace_set.traces.get(definition.name, []))
    passed = 0
    latencies = []
    speedups = []
    for trace in traces:
        evaluation = getattr(trace, "evaluation", None)
        status = "NO_EVAL"
        if evaluation is not None:
            status = getattr(evaluation.status, "value", str(evaluation.status))
            if str(status).lower() == "passed":
                passed += 1
            perf = getattr(evaluation, "performance", None)
            if perf is not None:
                if perf.latency_ms is not None:
                    latencies.append(float(perf.latency_ms))
                if perf.speedup_factor is not None:
                    speedups.append(float(perf.speedup_factor))
        print(f"{trace.workload.uuid}: status={status}")

    print(f"\nsolution:   {solution.name}")
    print(f"definition: {solution.definition}")
    print(f"passed:     {passed}/{len(workloads)}")
    if latencies:
        print(f"latency:    {sum(latencies) / len(latencies):.6f} ms mean")
    if speedups:
        print(f"speedup:    {sum(speedups) / len(speedups):.4f}x mean")
    return passed == len(workloads)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", required=True, help="Path to a packed FlashInfer solution.json.")
    parser.add_argument("--dataset", default=None, help="Path to flashinfer-trace dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected workloads.")
    parser.add_argument("--workload-uuid", action="append", default=None, help="Run one specific workload UUID.")
    parser.add_argument("--fast", action="store_true", help="Use a short benchmark configuration.")
    parser.add_argument("--isolated-runner", action="store_true", help="Use flashinfer-bench isolated runner.")
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--num-trials", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fast:
        args.warmup_runs = 1 if args.warmup_runs is None else args.warmup_runs
        args.iterations = 5 if args.iterations is None else args.iterations
        args.num_trials = 1 if args.num_trials is None else args.num_trials
        args.limit = 2 if args.limit is None and not args.workload_uuid else args.limit
    else:
        args.warmup_runs = 3 if args.warmup_runs is None else args.warmup_runs
        args.iterations = 50 if args.iterations is None else args.iterations
        args.num_trials = 3 if args.num_trials is None else args.num_trials

    return 0 if run(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
