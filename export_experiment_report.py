from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_results(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "completed":
            continue
        rows.append(data)
    return rows


def _active_open_depots(payload: dict[str, Any]) -> int:
    levels = payload.get("depot_level_choice", {})
    if isinstance(levels, dict):
        count = 0
        for value in levels.values():
            if isinstance(value, dict):
                if any(v is not None for v in value.values()):
                    count += 1
            elif value is not None:
                count += 1
        if count > 0:
            return count
    return len(payload.get("open_depots", []))


def _model_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mappings = [
        ("model_a_exact", "Model A", "Exact"),
        ("model_a_maat", "Model A", "MAAT"),
        ("model_b_exact", "Model B", "Exact"),
        ("model_b_maat", "Model B", "MAAT"),
    ]
    for key, model_name, method in mappings:
        if key not in result:
            continue
        payload = result[key]
        exact_payload = payload if method == "Exact" else payload["final_exact"]
        rows.append(
            {
                "instance_name": result["instance_name"],
                "source": result["source"],
                "model": model_name,
                "method": method,
                "plants": result["size"]["plants"],
                "depots": result["size"]["depots"],
                "customers": result["size"]["customers"],
                "products": result["size"]["products"],
                "objective_value": payload["objective_value"] if method == "MAAT" else payload["objective_value"],
                "best_bound": exact_payload.get("best_bound"),
                "mip_gap": exact_payload.get("mip_gap"),
                "runtime": payload["runtime"] if method == "MAAT" else payload["runtime"],
                "open_depots": _active_open_depots(exact_payload),
                "opening_cost": exact_payload.get("opening_cost"),
                "transport_cost": exact_payload.get("transport_cost"),
                "rho": payload.get("rho") if method == "MAAT" else None,
                "mu": payload.get("mu") if method == "MAAT" else None,
                "iterations": payload.get("iterations") if method == "MAAT" else None,
                "deviation_note": result.get("metadata", {}).get("deviation"),
            }
        )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "instance_name",
        "source",
        "model",
        "method",
        "plants",
        "depots",
        "customers",
        "products",
        "objective_value",
        "best_bound",
        "mip_gap",
        "runtime",
        "open_depots",
        "opening_cost",
        "transport_cost",
        "rho",
        "mu",
        "iterations",
        "deviation_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    teacher_rows = [r for r in rows if r["source"] == "teacher"]
    random_rows = [r for r in rows if r["source"] == "random"]

    lines: list[str] = []
    lines.append("# 实验报告")
    lines.append("")
    lines.append("## 数据结构")
    lines.append("")
    lines.append("- 教师数据：`I/J/K/C/D` 五个 sheet，分别对应工厂、候选仓库、客户、一级运输成本、二级运输成本。")
    lines.append("- 论文输入要求：多产品、多层容量、坐标、体积容量。")
    lines.append("- 教师数据适配方式：保留原始单产品流与成本矩阵，补齐 3 层产品容量、3 层体积容量、随机坐标与单位产品体积。")
    lines.append("")
    lines.append("## 代码结构")
    lines.append("")
    lines.append("- `src/scflp_multilevel/exact_models.py`：Model A / Model B 的 Gurobi 精确模型。")
    lines.append("- `src/scflp_multilevel/maat.py`：MAAT 三阶段流程、局部搜索、MPTP 子问题。")
    lines.append("- `src/scflp_multilevel/teacher_adapter.py`：教师数据适配。")
    lines.append("- `run_reproduction.py`：单实验入口。")
    lines.append("- `run_batch_resume.py`：批量实验、断点续跑、汇总表刷新。")
    lines.append("")
    lines.append("## 结果")
    lines.append("")
    lines.append("| Instance | Source | Model | Method | l | m | n | |P| | Obj | Bound | Gap | Runtime(s) | Open | Opening | Transport | rho | mu |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['instance_name']} | {row['source']} | {row['model']} | {row['method']} | "
            f"{row['plants']} | {row['depots']} | {row['customers']} | {row['products']} | "
            f"{_fmt(row['objective_value'])} | {_fmt(row['best_bound'])} | {_fmt(row['mip_gap'])} | "
            f"{_fmt(row['runtime'])} | {_fmt(row['open_depots'])} | {_fmt(row['opening_cost'])} | "
            f"{_fmt(row['transport_cost'])} | {_fmt(row['rho'])} | {_fmt(row['mu'])} |"
        )
    lines.append("")
    lines.append("## 分析")
    lines.append("")
    if teacher_rows:
        lines.append("### 教师数据")
        lines.append("")
        for row in teacher_rows:
            lines.append(
                f"- `{row['instance_name']} / {row['model']} / {row['method']}`: "
                f"目标值 `{_fmt(row['objective_value'])}`，有效开仓 `{_fmt(row['open_depots'])}`，"
                f"固定成本 `{_fmt(row['opening_cost'])}`，运输成本 `{_fmt(row['transport_cost'])}`。"
            )
        lines.append("")
        lines.append("- 教师数据是单产品单容量实例，Model B 的结果是基于适配后的体积容量版本，不应视为论文原始 dataset 的原生 Model B。")
        lines.append("")
    if random_rows:
        lines.append("### 随机实例")
        lines.append("")
        for row in random_rows:
            lines.append(
                f"- `{row['instance_name']} / {row['model']} / {row['method']}`: "
                f"目标值 `{_fmt(row['objective_value'])}`，Gap `{_fmt(row['mip_gap'])}`，运行时间 `{_fmt(row['runtime'])}`。"
            )
        lines.append("")
    lines.append("- MAAT 结果用于验证论文算法流程可运行；在教师适配数据上，精确法与 MAAT 的优劣不应直接类比论文原表。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-style experiment report from result JSON files.")
    parser.add_argument("--output-dir", default="IP_for_SCFLP_with_multi_products/outputs")
    parser.add_argument("--report-md", default="IP_for_SCFLP_with_multi_products/outputs/experiment_report.md")
    parser.add_argument("--report-csv", default="IP_for_SCFLP_with_multi_products/outputs/experiment_report.csv")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results = _load_results(output_dir)
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.extend(_model_rows(result))

    _write_csv(rows, Path(args.report_csv))
    _write_markdown(rows, Path(args.report_md))
    print(Path(args.report_md))
    print(Path(args.report_csv))


if __name__ == "__main__":
    main()
