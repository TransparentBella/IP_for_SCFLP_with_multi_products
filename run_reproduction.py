from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scflp_multilevel import (  # noqa: E402
    GeneratorConfig,
    MAATConfig,
    ModelKind,
    TeacherAdapterConfig,
    generate_random_instance,
    load_teacher_dataset_as_model_a,
    solve_model_a_exact,
    solve_model_b_exact,
    solve_with_maat,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce the multilevel two-stage facility location paper.")
    parser.add_argument("--source", choices=["random", "teacher"], default="random")
    parser.add_argument("--teacher-file", default=str(Path("..") / "data_100200400.xlsx"))
    parser.add_argument("--model", choices=["a", "b", "both"], default="both")
    parser.add_argument("--run-maat", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plants", type=int, default=5)
    parser.add_argument("--depots", type=int, default=20)
    parser.add_argument("--customers", type=int, default=40)
    parser.add_argument("--products", type=int, default=5)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--exact-time-limit", type=float, default=60.0)
    parser.add_argument("--exact-mip-gap", type=float, default=0.0001)
    parser.add_argument("--maat-iterations", type=int, default=10)
    parser.add_argument("--maat-beta", type=float, default=None)
    parser.add_argument("--maat-aggregated-time-limit", type=float, default=150.0)
    parser.add_argument("--maat-aggregated-mip-gap", type=float, default=0.005)
    parser.add_argument("--maat-final-time-limit", type=float, default=108.0)
    parser.add_argument("--maat-local-search-time-limit", type=float, default=None)
    parser.add_argument("--maat-gamma", type=float, default=2.5)
    parser.add_argument("--output-flag", type=int, default=0)
    return parser


def run_case(args: argparse.Namespace) -> Path:
    if args.source == "random":
        instance = generate_random_instance(
            GeneratorConfig(
                name=f"rand_l{args.plants}_m{args.depots}_n{args.customers}_p{args.products}",
                n_plants=args.plants,
                n_depots=args.depots,
                n_customers=args.customers,
                n_products=args.products,
                seed=args.seed,
            )
        )
    else:
        instance = load_teacher_dataset_as_model_a(Path(args.teacher_file), TeacherAdapterConfig(seed=args.seed))

    output_dir = Path(args.output_dir)
    output_name = args.output_name or f"{instance.name}_{args.source}_{args.model}.json"
    out_path = output_dir / output_name
    summary: dict[str, Any] = {
        "status": "running",
        "current_phase": "initialised",
        "instance_name": instance.name,
        "source": args.source,
        "metadata": instance.metadata,
        "params": {
            "model": args.model,
            "run_maat": args.run_maat,
            "seed": args.seed,
            "exact_time_limit": args.exact_time_limit,
            "exact_mip_gap": args.exact_mip_gap,
            "maat_iterations": args.maat_iterations,
            "maat_beta": args.maat_beta,
            "maat_aggregated_time_limit": args.maat_aggregated_time_limit,
            "maat_aggregated_mip_gap": args.maat_aggregated_mip_gap,
            "maat_final_time_limit": args.maat_final_time_limit,
            "maat_local_search_time_limit": args.maat_local_search_time_limit,
            "maat_gamma": args.maat_gamma,
        },
        "size": {
            "plants": instance.n_plants,
            "depots": instance.n_depots,
            "customers": instance.n_customers,
            "products": instance.n_products,
        },
    }
    _write_json(out_path, summary)

    if args.model in {"a", "both"}:
        summary["current_phase"] = "model_a_exact"
        _write_json(out_path, summary)
        exact_a = solve_model_a_exact(
            instance,
            time_limit=args.exact_time_limit,
            mip_gap=args.exact_mip_gap,
            output_flag=args.output_flag,
        )
        summary["model_a_exact"] = asdict(exact_a)
        _write_json(out_path, summary)
        if args.run_maat:
            summary["current_phase"] = "model_a_maat"
            _write_json(out_path, summary)
            maat_beta = args.maat_beta if args.maat_beta is not None else 1.5
            maat_a = solve_with_maat(
                instance,
                ModelKind.MODEL_A,
                config=MAATConfig(
                    seed=args.seed,
                    beta=maat_beta,
                    iterations=args.maat_iterations,
                    aggregated_time_limit=args.maat_aggregated_time_limit,
                    aggregated_mip_gap=args.maat_aggregated_mip_gap,
                    final_time_limit=args.maat_final_time_limit,
                    local_search_time_limit=args.maat_local_search_time_limit,
                    gamma=args.maat_gamma,
                    output_flag=args.output_flag,
                ),
            )
            summary["model_a_maat"] = {
                "model_kind": maat_a.model_kind,
                "objective_value": maat_a.objective_value,
                "runtime": maat_a.runtime,
                "stage1_best_depots": maat_a.stage1_best_depots,
                "stage2_best_depots": maat_a.stage2_best_depots,
                "rho": maat_a.rho,
                "mu": maat_a.mu,
                "iterations": maat_a.iterations,
                "metadata": maat_a.metadata,
                "final_exact": asdict(maat_a.exact_result),
            }
            _write_json(out_path, summary)

    if args.model in {"b", "both"}:
        summary["current_phase"] = "model_b_exact"
        _write_json(out_path, summary)
        exact_b = solve_model_b_exact(
            instance,
            time_limit=args.exact_time_limit,
            mip_gap=args.exact_mip_gap,
            output_flag=args.output_flag,
        )
        summary["model_b_exact"] = asdict(exact_b)
        _write_json(out_path, summary)
        if args.run_maat:
            summary["current_phase"] = "model_b_maat"
            _write_json(out_path, summary)
            maat_beta = args.maat_beta
            if maat_beta is None:
                maat_beta = 2.0 if instance.n_products == 10 else 3.0
            maat_b = solve_with_maat(
                instance,
                ModelKind.MODEL_B,
                config=MAATConfig(
                    seed=args.seed,
                    beta=maat_beta,
                    iterations=args.maat_iterations,
                    aggregated_time_limit=args.maat_aggregated_time_limit,
                    aggregated_mip_gap=args.maat_aggregated_mip_gap,
                    final_time_limit=args.maat_final_time_limit,
                    local_search_time_limit=args.maat_local_search_time_limit,
                    gamma=args.maat_gamma,
                    output_flag=args.output_flag,
                ),
            )
            summary["model_b_maat"] = {
                "model_kind": maat_b.model_kind,
                "objective_value": maat_b.objective_value,
                "runtime": maat_b.runtime,
                "stage1_best_depots": maat_b.stage1_best_depots,
                "stage2_best_depots": maat_b.stage2_best_depots,
                "rho": maat_b.rho,
                "mu": maat_b.mu,
                "iterations": maat_b.iterations,
                "metadata": maat_b.metadata,
                "final_exact": asdict(maat_b.exact_result),
            }
            _write_json(out_path, summary)

    summary["status"] = "completed"
    summary["current_phase"] = "completed"
    _write_json(out_path, summary)
    return out_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    out_path = run_case(args)
    print(out_path)


if __name__ == "__main__":
    main()
