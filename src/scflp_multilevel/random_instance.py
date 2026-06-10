from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instance import MultiLevelInstance


@dataclass(frozen=True)
class GeneratorConfig:
    name: str = "paper_style"
    n_plants: int = 5
    n_depots: int = 30
    n_customers: int = 60
    n_products: int = 5
    n_levels: int = 3
    seed: int = 42
    coord_upper: int | None = None
    demand_low: int = 1
    demand_high: int = 5
    model_a_capacity_levels: tuple[int, int, int] = (30, 50, 70)
    volume_capacity_levels: tuple[int, int, int] = (180, 280, 400)
    product_volume_low: int = 1
    product_volume_high: int = 4
    plant_capacity_slack: float = 1.08


def _euclidean_cost(a: np.ndarray, b: np.ndarray, rate: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return dist[:, :, None] * rate[None, None, :]


def generate_random_instance(config: GeneratorConfig) -> MultiLevelInstance:
    rng = np.random.default_rng(config.seed)
    coord_upper = config.coord_upper or config.n_customers

    plant_coords = rng.integers(0, coord_upper + 1, size=(config.n_plants, 2))
    depot_coords = rng.integers(0, coord_upper + 1, size=(config.n_depots, 2))
    customer_coords = rng.integers(0, coord_upper + 1, size=(config.n_customers, 2))

    demand = rng.integers(
        config.demand_low,
        config.demand_high + 1,
        size=(config.n_customers, config.n_products),
    ).astype(float)
    total_demand_by_product = demand.sum(axis=0)

    base_capacity = np.ceil(config.plant_capacity_slack * total_demand_by_product / config.n_plants)
    plant_capacity = np.vstack(
        [
            np.maximum(
                1.0,
                np.round(base_capacity * rng.uniform(0.95, 1.08, size=config.n_products)),
            )
            for _ in range(config.n_plants)
        ]
    )
    scaling = total_demand_by_product / np.maximum(plant_capacity.sum(axis=0), 1.0)
    if np.any(scaling > 1.0):
        plant_capacity = np.ceil(plant_capacity * scaling[None, :] * 1.02)

    product_capacity_levels = np.tile(
        np.asarray(config.model_a_capacity_levels, dtype=float),
        (config.n_depots, config.n_products, 1),
    )
    product_volumes = rng.integers(
        config.product_volume_low,
        config.product_volume_high + 1,
        size=config.n_products,
    ).astype(float)
    volume_capacity_levels = np.tile(
        np.asarray(config.volume_capacity_levels, dtype=float),
        (config.n_depots, 1),
    )

    plant_depot_rate = rng.uniform(4.5, 7.0, size=config.n_products)
    depot_customer_rate = rng.uniform(5.5, 8.0, size=config.n_products)
    transport_plant_depot = _euclidean_cost(plant_coords, depot_coords, plant_depot_rate)
    transport_depot_customer = _euclidean_cost(depot_coords, customer_coords, depot_customer_rate)

    avg_pd_cost = transport_plant_depot.mean(axis=0).mean(axis=1)
    avg_dc_cost = transport_depot_customer.mean(axis=2).mean(axis=1)
    fixed_anchor = (avg_pd_cost + avg_dc_cost) * max(config.n_customers / 4.0, 8.0)

    depot_opening_cost = np.round(fixed_anchor * rng.uniform(0.45, 0.7, size=config.n_depots), 2)
    product_capacity_fixed_costs = np.zeros_like(product_capacity_levels)
    for d in range(config.n_levels):
        scale = 0.45 + 0.25 * d
        product_capacity_fixed_costs[:, :, d] = np.round(
            (fixed_anchor[:, None] / max(config.n_products, 1)) * scale * rng.uniform(0.95, 1.05, size=(config.n_depots, config.n_products)),
            2,
        )
    volume_fixed_costs = np.zeros_like(volume_capacity_levels)
    for d in range(config.n_levels):
        scale = 0.9 + 0.35 * d
        volume_fixed_costs[:, d] = np.round(fixed_anchor * scale * rng.uniform(0.95, 1.05, size=config.n_depots), 2)

    return MultiLevelInstance(
        name=config.name,
        plant_ids=tuple(f"I{i + 1}" for i in range(config.n_plants)),
        depot_ids=tuple(f"J{j + 1}" for j in range(config.n_depots)),
        customer_ids=tuple(f"K{k + 1}" for k in range(config.n_customers)),
        product_ids=tuple(f"P{p + 1}" for p in range(config.n_products)),
        plant_coords=plant_coords.astype(float),
        depot_coords=depot_coords.astype(float),
        customer_coords=customer_coords.astype(float),
        plant_capacity=plant_capacity.astype(float),
        demand=demand,
        product_capacity_levels=product_capacity_levels,
        product_capacity_fixed_costs=product_capacity_fixed_costs,
        depot_opening_cost=depot_opening_cost,
        transport_plant_depot=transport_plant_depot,
        transport_depot_customer=transport_depot_customer,
        volume_capacity_levels=volume_capacity_levels,
        volume_fixed_costs=volume_fixed_costs,
        product_volumes=product_volumes,
        metadata={
            "generator": "paper_style_partial",
            "paper_exact_data_generator": False,
            "note": "The paper fully specifies the algorithm, but not all random data formulas. The chosen rules are documented in README.",
        },
    )
