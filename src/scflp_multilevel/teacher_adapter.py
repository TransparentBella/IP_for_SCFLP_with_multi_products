from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .instance import MultiLevelInstance


def _normalize_named_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all").reset_index(drop=True)
    header = df.iloc[0].tolist()
    body = df.iloc[1:].copy()
    body.columns = header
    return body.reset_index(drop=True)


def _normalize_cost_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all").reset_index(drop=True)
    header = df.iloc[0].tolist()
    header[0] = "id"
    body = df.iloc[1:].copy()
    body.columns = header
    body = body.reset_index(drop=True)
    return body


@dataclass(frozen=True)
class TeacherAdapterConfig:
    seed: int = 42
    capacity_level_multipliers: tuple[float, float, float] = (0.8, 1.0, 1.25)
    fixed_cost_multipliers: tuple[float, float, float] = (0.85, 1.0, 1.2)
    volume_level_multipliers: tuple[float, float, float] = (0.8, 1.0, 1.3)
    product_volume: float = 1.0


def load_teacher_dataset_as_model_a(path: str | Path, config: TeacherAdapterConfig | None = None) -> MultiLevelInstance:
    cfg = config or TeacherAdapterConfig()
    rng = np.random.default_rng(cfg.seed)
    xls = pd.ExcelFile(path)

    plants = _normalize_named_table(pd.read_excel(xls, "I", header=None))
    depots = _normalize_named_table(pd.read_excel(xls, "J", header=None))
    customers = _normalize_named_table(pd.read_excel(xls, "K", header=None))
    c_df = _normalize_cost_table(pd.read_excel(xls, "C", header=None))
    d_df = _normalize_cost_table(pd.read_excel(xls, "D", header=None))

    plant_ids = tuple(plants["Plant_ID"].astype(str))
    depot_ids = tuple(depots["Depot_ID"].astype(str))
    customer_ids = tuple(customers["Customer_ID"].astype(str))

    plant_capacity = np.rint(plants["Capacity_b"].astype(float).to_numpy())[:, None]
    demand = np.rint(customers["Demand_q"].astype(float).to_numpy())[:, None]
    depot_capacity = np.rint(depots["Capacity_p"].astype(float).to_numpy())
    depot_fixed_cost = depots["FixedCost_g"].astype(float).to_numpy()

    c_df = c_df.set_index("id").loc[list(plant_ids), list(depot_ids)]
    d_df = d_df.set_index("id").loc[list(depot_ids), list(customer_ids)]
    transport_plant_depot = c_df.to_numpy(dtype=float)[:, :, None]
    transport_depot_customer = d_df.to_numpy(dtype=float)[:, :, None]

    n_depots = len(depot_ids)
    product_capacity_levels = np.zeros((n_depots, 1, len(cfg.capacity_level_multipliers)))
    product_capacity_fixed_costs = np.zeros_like(product_capacity_levels)
    for i, mul in enumerate(cfg.capacity_level_multipliers):
        product_capacity_levels[:, 0, i] = np.rint(depot_capacity * mul)
    for i, mul in enumerate(cfg.fixed_cost_multipliers):
        product_capacity_fixed_costs[:, 0, i] = np.round(depot_fixed_cost * mul, 2)

    volume_capacity_levels = np.zeros((n_depots, len(cfg.volume_level_multipliers)))
    volume_fixed_costs = np.zeros_like(volume_capacity_levels)
    for i, mul in enumerate(cfg.volume_level_multipliers):
        volume_capacity_levels[:, i] = np.rint(depot_capacity * cfg.product_volume * mul)
        volume_fixed_costs[:, i] = np.round(depot_fixed_cost * (0.9 + 0.2 * i), 2)

    plant_coords = rng.integers(0, len(customer_ids) + 1, size=(len(plant_ids), 2)).astype(float)
    depot_coords = rng.integers(0, len(customer_ids) + 1, size=(len(depot_ids), 2)).astype(float)
    customer_coords = rng.integers(0, len(customer_ids) + 1, size=(len(customer_ids), 2)).astype(float)

    return MultiLevelInstance(
        name=Path(path).stem,
        plant_ids=plant_ids,
        depot_ids=depot_ids,
        customer_ids=customer_ids,
        product_ids=("P1",),
        plant_coords=plant_coords,
        depot_coords=depot_coords,
        customer_coords=customer_coords,
        plant_capacity=plant_capacity,
        demand=demand,
        product_capacity_levels=product_capacity_levels,
        product_capacity_fixed_costs=product_capacity_fixed_costs,
        depot_opening_cost=np.zeros(n_depots, dtype=float),
        transport_plant_depot=transport_plant_depot,
        transport_depot_customer=transport_depot_customer,
        volume_capacity_levels=volume_capacity_levels,
        volume_fixed_costs=volume_fixed_costs,
        product_volumes=np.asarray([cfg.product_volume], dtype=float),
        metadata={
            "source": "teacher_dataset",
            "paper_exact": False,
            "deviation": "The teacher file is single-product and single-capacity with no coordinates. Multi-level capacities and coordinates are synthetic and documented.",
        },
    )
