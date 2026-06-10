from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .exact_models import ExactResult, solve_model_a_exact, solve_model_b_exact
from .instance import ModelKind, MultiLevelInstance


@dataclass(frozen=True)
class MAATConfig:
    seed: int = 42
    beta: float = 1.5
    iterations: int = 10
    aggregated_time_limit: float = 150.0
    aggregated_mip_gap: float = 0.005
    final_time_limit: float = 108.0
    local_search_time_limit: float | None = None
    gamma: float = 2.5
    output_flag: int = 0


@dataclass(frozen=True)
class MAATResult:
    model_kind: str
    objective_value: float
    runtime: float
    exact_result: ExactResult
    stage1_best_depots: tuple[str, ...]
    stage2_best_depots: tuple[str, ...]
    rho: int
    mu: int
    iterations: int
    metadata: dict[str, object]


def _compute_rho(instance: MultiLevelInstance, model_kind: ModelKind) -> int:
    if model_kind == ModelKind.MODEL_A:
        denom = instance.product_capacity_levels.min(axis=2).min(axis=0)
        rho = int(np.max(np.ceil(instance.total_demand_by_product / denom)))
    else:
        total_volume = (instance.demand * instance.product_volumes[None, :]).sum(axis=0)
        min_volume_capacity = instance.volume_capacity_levels.min()
        rho = int(np.max(np.ceil(total_volume / min_volume_capacity)))
    return max(rho, 1)


def _evaluate_transshipment(
    instance: MultiLevelInstance,
    open_indices: list[int],
    model_kind: ModelKind,
    output_flag: int = 0,
) -> float:
    model = gp.Model("mptp")
    model.Params.OutputFlag = output_flag
    I = range(instance.n_plants)
    S = range(len(open_indices))
    K = range(instance.n_customers)
    P = range(instance.n_products)

    x = model.addVars(I, S, P, vtype=GRB.CONTINUOUS, lb=0.0, name="x")
    xhat = model.addVars(S, K, P, vtype=GRB.CONTINUOUS, lb=0.0, name="xhat")

    if model_kind == ModelKind.MODEL_A:
        fixed_cost = sum(instance.depot_opening_cost[j] + instance.largest_product_capacity_costs[j].sum() for j in open_indices)
    else:
        fixed_cost = sum(instance.largest_volume_capacity_costs[j] for j in open_indices)

    transport_cost = gp.quicksum(
        instance.transport_plant_depot[i, open_indices[s], p] * x[i, s, p]
        for i in I for s in S for p in P
    ) + gp.quicksum(
        instance.transport_depot_customer[open_indices[s], k, p] * xhat[s, k, p]
        for s in S for k in K for p in P
    )
    model.setObjective(fixed_cost + transport_cost, GRB.MINIMIZE)

    for i in I:
        for p in P:
            model.addConstr(gp.quicksum(x[i, s, p] for s in S) <= instance.plant_capacity[i, p])
    if model_kind == ModelKind.MODEL_A:
        for s, j in enumerate(open_indices):
            for p in P:
                model.addConstr(gp.quicksum(x[i, s, p] for i in I) <= instance.largest_product_capacities[j, p])
    else:
        for s, j in enumerate(open_indices):
            model.addConstr(
                gp.quicksum(instance.product_volumes[p] * x[i, s, p] for i in I for p in P) <= instance.largest_volume_capacities[j]
            )
    for k in K:
        for p in P:
            model.addConstr(gp.quicksum(xhat[s, k, p] for s in S) == instance.demand[k, p])
    for s in S:
        for p in P:
            model.addConstr(gp.quicksum(x[i, s, p] for i in I) == gp.quicksum(xhat[s, k, p] for k in K))

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        return float("inf")
    return float(model.ObjVal)


def _local_search(
    instance: MultiLevelInstance,
    initial_open: list[int],
    model_kind: ModelKind,
    config: MAATConfig,
) -> tuple[list[int], float]:
    time_limit = config.local_search_time_limit or (0.25 * instance.n_customers * instance.n_products)
    start = perf_counter()
    incumbent = sorted(initial_open)
    incumbent_value = _evaluate_transshipment(instance, incumbent, model_kind, config.output_flag)

    while perf_counter() - start < time_limit:
        theta = 0.0
        non_open = [j for j in range(instance.n_depots) if j not in incumbent]
        if not non_open:
            break

        nearest_open: dict[int, int] = {}
        nearest_distance: dict[int, float] = {}
        for j in non_open:
            distances = [float(np.linalg.norm(instance.depot_coords[j] - instance.depot_coords[s])) for s in incumbent]
            best_idx = int(np.argmin(distances))
            nearest_open[j] = incumbent[best_idx]
            nearest_distance[j] = distances[best_idx]

        ds = {}
        for s in incumbent:
            assigned = [nearest_distance[j] for j in non_open if nearest_open[j] == s]
            ds[s] = max(assigned) if assigned else 0.0

        improvement_found = False
        for j in non_open:
            for s in list(incumbent):
                distance_js = float(np.linalg.norm(instance.depot_coords[j] - instance.depot_coords[s]))
                if distance_js >= config.gamma * ds.get(s, 0.0):
                    continue
                candidate = sorted([node for node in incumbent if node != s] + [j])
                z_prime = _evaluate_transshipment(instance, candidate, model_kind, config.output_flag)
                theta = incumbent_value - z_prime
                if theta > 1e-6:
                    incumbent = candidate
                    incumbent_value = z_prime
                    improvement_found = True
                    break
            if improvement_found or perf_counter() - start >= time_limit:
                break
        if not improvement_found or theta <= 0:
            break
    return incumbent, incumbent_value


def solve_with_maat(
    instance: MultiLevelInstance,
    model_kind: ModelKind,
    *,
    config: MAATConfig | None = None,
) -> MAATResult:
    cfg = config or MAATConfig()
    rng = np.random.default_rng(cfg.seed)
    start = perf_counter()

    rho = _compute_rho(instance, model_kind)
    mu = min(instance.n_depots, max(rho, int(np.ceil(cfg.beta * rho))))

    best_stage1_open: list[int] | None = None
    best_stage1_value = float("inf")

    for t in range(cfg.iterations):
        chosen = set(best_stage1_open or [])
        remaining = [j for j in range(instance.n_depots) if j not in chosen]
        need = max(0, mu - len(chosen))
        if need > 0:
            sampled = rng.choice(remaining, size=need, replace=False)
            chosen.update(int(v) for v in sampled)
        reduced = instance.subset_depots(sorted(chosen), suffix=f"agg_{t + 1}")
        if model_kind == ModelKind.MODEL_A:
            reduced_result = solve_model_a_exact(
                reduced,
                time_limit=cfg.aggregated_time_limit,
                mip_gap=cfg.aggregated_mip_gap,
                output_flag=cfg.output_flag,
            )
        else:
            reduced_result = solve_model_b_exact(
                reduced,
                time_limit=cfg.aggregated_time_limit,
                mip_gap=cfg.aggregated_mip_gap,
                output_flag=cfg.output_flag,
            )
        opened = [sorted(chosen)[reduced.depot_ids.index(did)] for did in reduced_result.open_depots]
        local_open, local_value = _local_search(instance, opened, model_kind, cfg)
        if local_value < best_stage1_value:
            best_stage1_value = local_value
            best_stage1_open = local_open

    assert best_stage1_open is not None
    stage2_open, _ = _local_search(instance, best_stage1_open, model_kind, cfg)
    restricted = instance.subset_depots(stage2_open, suffix="final")
    if model_kind == ModelKind.MODEL_A:
        exact = solve_model_a_exact(restricted, time_limit=cfg.final_time_limit, mip_gap=0.0001, output_flag=cfg.output_flag)
    else:
        exact = solve_model_b_exact(restricted, time_limit=cfg.final_time_limit, mip_gap=0.0001, output_flag=cfg.output_flag)

    runtime = perf_counter() - start
    return MAATResult(
        model_kind=model_kind.value,
        objective_value=exact.objective_value,
        runtime=runtime,
        exact_result=exact,
        stage1_best_depots=tuple(instance.depot_ids[j] for j in best_stage1_open),
        stage2_best_depots=tuple(instance.depot_ids[j] for j in stage2_open),
        rho=rho,
        mu=mu,
        iterations=cfg.iterations,
        metadata={
            "beta": cfg.beta,
            "tau": cfg.aggregated_time_limit,
            "epsilon": cfg.aggregated_mip_gap,
            "tau_prime": cfg.final_time_limit,
            "tau_hat": cfg.local_search_time_limit or (0.25 * instance.n_customers * instance.n_products),
            "gamma": cfg.gamma,
        },
    )
