from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from run_reproduction import ROOT, build_parser, run_case


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_output_path(output_dir: Path, task: dict[str, Any]) -> Path:
    return output_dir / str(task["output_name"])


def _is_completed(output_path: Path) -> bool:
    if not output_path.exists():
        return False
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("status") == "completed"


def _flatten_result(data: dict[str, Any], model_key: str, mode: str) -> dict[str, Any]:
    def active_open_depots(payload: dict[str, Any]) -> int:
        if "depot_level_choice" in payload and isinstance(payload["depot_level_choice"], dict):
            count = 0
            for value in payload["depot_level_choice"].values():
                if isinstance(value, dict):
                    if any(v is not None for v in value.values()):
                        count += 1
                elif value is not None:
                    count += 1
            return count
        return len(payload.get("open_depots", []))

    record: dict[str, Any] = {
        "instance_name": data.get("instance_name"),
        "source": data.get("source"),
        "mode": mode,
        "model": model_key,
        "status": data.get("status"),
        "current_phase": data.get("current_phase"),
        "plants": data.get("size", {}).get("plants"),
        "depots": data.get("size", {}).get("depots"),
        "customers": data.get("size", {}).get("customers"),
        "products": data.get("size", {}).get("products"),
    }
    payload = data.get(model_key, {})
    if mode == "exact":
        record.update(
            {
                "objective_value": payload.get("objective_value"),
                "best_bound": payload.get("best_bound"),
                "mip_gap": payload.get("mip_gap"),
                "runtime": payload.get("runtime"),
                "open_depots": active_open_depots(payload),
            }
        )
    else:
        final_exact = payload.get("final_exact", {})
        record.update(
            {
                "objective_value": payload.get("objective_value"),
                "best_bound": final_exact.get("best_bound"),
                "mip_gap": final_exact.get("mip_gap"),
                "runtime": payload.get("runtime"),
                "open_depots": active_open_depots(final_exact),
                "rho": payload.get("rho"),
                "mu": payload.get("mu"),
                "iterations": payload.get("iterations"),
            }
        )
    return record


def _write_summary(output_dir: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for result_file in sorted(output_dir.glob("*.json")):
        data = json.loads(result_file.read_text(encoding="utf-8"))
        if "model_a_exact" in data:
            rows.append(_flatten_result(data, "model_a_exact", "exact"))
        if "model_a_maat" in data:
            rows.append(_flatten_result(data, "model_a_maat", "maat"))
        if "model_b_exact" in data:
            rows.append(_flatten_result(data, "model_b_exact", "exact"))
        if "model_b_maat" in data:
            rows.append(_flatten_result(data, "model_b_maat", "maat"))

    summary_path = output_dir / "batch_summary.csv"
    fieldnames = [
        "instance_name",
        "source",
        "mode",
        "model",
        "status",
        "current_phase",
        "plants",
        "depots",
        "customers",
        "products",
        "objective_value",
        "best_bound",
        "mip_gap",
        "runtime",
        "open_depots",
        "rho",
        "mu",
        "iterations",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible batch experiments with resume support.")
    parser.add_argument("--task-file", default=str(ROOT / "batch_tasks.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        print(_write_summary(output_dir))
        return

    parser_defaults = vars(build_parser().parse_args([]))
    tasks = _load_tasks(Path(args.task_file))

    for task in tasks:
        merged = dict(parser_defaults)
        merged.update(task)
        merged["output_dir"] = str(output_dir)
        output_path = _task_output_path(output_dir, merged)
        if not args.force_rerun and _is_completed(output_path):
            print(f"skip completed: {output_path.name}")
            continue
        print(f"run: {output_path.name}")
        run_case(SimpleNamespace(**merged))
        summary_path = _write_summary(output_dir)
        print(f"summary updated: {summary_path.name}")

    summary_path = _write_summary(output_dir)
    print(summary_path)


if __name__ == "__main__":
    main()
