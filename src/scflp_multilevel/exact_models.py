from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .instance import MultiLevelInstance


@dataclass(frozen=True)
class ExactResult:
    model_name: str
    objective_value: float
    best_bound: float
    mip_gap: float
    runtime: float
    status: int
    open_depots: tuple[str, ...]
    depot_level_choice: dict[str, object]
    opening_cost: float
    transport_cost: float


def _configure(model: gp.Model, time_limit: float | None, mip_gap: float | None, output_flag: int) -> None:
    model.Params.OutputFlag = output_flag
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap


def solve_model_a_exact(
    instance: MultiLevelInstance,
    *,
    time_limit: float | None = None,
    mip_gap: float | None = None,
    output_flag: int = 0,
) -> ExactResult:
    start = perf_counter()
    model = gp.Model(f"model_a_{instance.name}")
    _configure(model, time_limit, mip_gap, output_flag)

    I = range(instance.n_plants)
    J = range(instance.n_depots)
    K = range(instance.n_customers)
    P = range(instance.n_products)
    D = range(instance.n_levels)

    x = model.addVars(I, J, P, vtype=GRB.INTEGER, lb=0.0, name="x")
    xhat = model.addVars(J, K, P, vtype=GRB.INTEGER, lb=0.0, name="xhat")
    y = model.addVars(J, P, D, vtype=GRB.BINARY, name="y")
    q = model.addVars(J, vtype=GRB.BINARY, name="q")

    model.setObjective(
        gp.quicksum(instance.depot_opening_cost[j] * q[j] for j in J)
        + gp.quicksum(instance.product_capacity_fixed_costs[j, p, d] * y[j, p, d] for j in J for p in P for d in D)
        + gp.quicksum(instance.transport_plant_depot[i, j, p] * x[i, j, p] for i in I for j in J for p in P)
        + gp.quicksum(instance.transport_depot_customer[j, k, p] * xhat[j, k, p] for j in J for k in K for p in P),
        GRB.MINIMIZE,
    )

    for i in I:
        for p in P:
            model.addConstr(gp.quicksum(x[i, j, p] for j in J) <= instance.plant_capacity[i, p], name=f"plant_cap_{i}_{p}")
    for j in J:
        for p in P:
            model.addConstr(
                gp.quicksum(x[i, j, p] for i in I)
                <= gp.quicksum(instance.product_capacity_levels[j, p, d] * y[j, p, d] for d in D),
                name=f"depot_cap_{j}_{p}",
            )
            model.addConstr(gp.quicksum(y[j, p, d] for d in D) <= q[j], name=f"one_level_{j}_{p}")
    for k in K:
        for p in P:
            model.addConstr(gp.quicksum(xhat[j, k, p] for j in J) == instance.demand[k, p], name=f"demand_{k}_{p}")
    for j in J:
        for p in P:
            model.addConstr(
                gp.quicksum(x[i, j, p] for i in I) == gp.quicksum(xhat[j, k, p] for k in K),
                name=f"flow_{j}_{p}",
            )

    model.optimize()
    runtime = perf_counter() - start

    active_depots = [j for j in J if any(y[j, p, d].X > 0.5 for p in P for d in D)]
    open_depots = tuple(instance.depot_ids[j] for j in active_depots)
    depot_level_choice = {
        instance.depot_ids[j]: {
            instance.product_ids[p]: next(
                (d for d in D if y[j, p, d].X > 0.5),
                None,
            )
            for p in P
        }
        for j in J
        if j in active_depots
    }
    opening_cost = sum(instance.depot_opening_cost[j] * q[j].X for j in J) + sum(
        instance.product_capacity_fixed_costs[j, p, d] * y[j, p, d].X for j in J for p in P for d in D
    )
    transport_cost = sum(
        instance.transport_plant_depot[i, j, p] * x[i, j, p].X for i in I for j in J for p in P
    ) + sum(
        instance.transport_depot_customer[j, k, p] * xhat[j, k, p].X for j in J for k in K for p in P
    )
    return ExactResult(
        model_name="Model A",
        objective_value=float(model.ObjVal),
        best_bound=float(model.ObjBound),
        mip_gap=float(model.MIPGap) if model.SolCount else float("inf"),
        runtime=runtime,
        status=model.Status,
        open_depots=open_depots,
        depot_level_choice=depot_level_choice,
        opening_cost=float(opening_cost),
        transport_cost=float(transport_cost),
    )


def solve_model_b_exact(
    instance: MultiLevelInstance,
    *,
    time_limit: float | None = None,
    mip_gap: float | None = None,
    output_flag: int = 0,
) -> ExactResult:
    start = perf_counter()
    model = gp.Model(f"model_b_{instance.name}")
    _configure(model, time_limit, mip_gap, output_flag)

    I = range(instance.n_plants)
    J = range(instance.n_depots)
    K = range(instance.n_customers)
    P = range(instance.n_products)
    D = range(instance.n_levels)

    x = model.addVars(I, J, P, vtype=GRB.INTEGER, lb=0.0, name="x")
    xhat = model.addVars(J, K, P, vtype=GRB.INTEGER, lb=0.0, name="xhat")
    y = model.addVars(J, D, vtype=GRB.BINARY, name="y")

    model.setObjective(
        gp.quicksum(instance.volume_fixed_costs[j, d] * y[j, d] for j in J for d in D)
        + gp.quicksum(instance.transport_plant_depot[i, j, p] * x[i, j, p] for i in I for j in J for p in P)
        + gp.quicksum(instance.transport_depot_customer[j, k, p] * xhat[j, k, p] for j in J for k in K for p in P),
        GRB.MINIMIZE,
    )

    for i in I:
        for p in P:
            model.addConstr(gp.quicksum(x[i, j, p] for j in J) <= instance.plant_capacity[i, p], name=f"plant_cap_{i}_{p}")
    for j in J:
        model.addConstr(
            gp.quicksum(instance.product_volumes[p] * x[i, j, p] for i in I for p in P)
            <= gp.quicksum(instance.volume_capacity_levels[j, d] * y[j, d] for d in D),
            name=f"volume_cap_{j}",
        )
        model.addConstr(gp.quicksum(y[j, d] for d in D) <= 1, name=f"one_level_{j}")
    for k in K:
        for p in P:
            model.addConstr(gp.quicksum(xhat[j, k, p] for j in J) == instance.demand[k, p], name=f"demand_{k}_{p}")
    for j in J:
        for p in P:
            model.addConstr(
                gp.quicksum(x[i, j, p] for i in I) == gp.quicksum(xhat[j, k, p] for k in K),
                name=f"flow_{j}_{p}",
            )

    model.optimize()
    runtime = perf_counter() - start

    open_depots = tuple(instance.depot_ids[j] for j in J if sum(y[j, d].X for d in D) > 0.5)
    depot_level_choice = {
        instance.depot_ids[j]: next((d for d in D if y[j, d].X > 0.5), None)
        for j in J
        if sum(y[j, d].X for d in D) > 0.5
    }
    opening_cost = sum(instance.volume_fixed_costs[j, d] * y[j, d].X for j in J for d in D)
    transport_cost = sum(
        instance.transport_plant_depot[i, j, p] * x[i, j, p].X for i in I for j in J for p in P
    ) + sum(
        instance.transport_depot_customer[j, k, p] * xhat[j, k, p].X for j in J for k in K for p in P
    )
    return ExactResult(
        model_name="Model B",
        objective_value=float(model.ObjVal),
        best_bound=float(model.ObjBound),
        mip_gap=float(model.MIPGap) if model.SolCount else float("inf"),
        runtime=runtime,
        status=model.Status,
        open_depots=open_depots,
        depot_level_choice=depot_level_choice,
        opening_cost=float(opening_cost),
        transport_cost=float(transport_cost),
    )
