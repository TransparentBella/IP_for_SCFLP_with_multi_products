from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ModelKind(str, Enum):
    MODEL_A = "model_a"
    MODEL_B = "model_b"


@dataclass(frozen=True)
class MultiLevelInstance:
    name: str
    plant_ids: tuple[str, ...]
    depot_ids: tuple[str, ...]
    customer_ids: tuple[str, ...]
    product_ids: tuple[str, ...]
    plant_coords: np.ndarray
    depot_coords: np.ndarray
    customer_coords: np.ndarray
    plant_capacity: np.ndarray
    demand: np.ndarray
    product_capacity_levels: np.ndarray
    product_capacity_fixed_costs: np.ndarray
    depot_opening_cost: np.ndarray
    transport_plant_depot: np.ndarray
    transport_depot_customer: np.ndarray
    volume_capacity_levels: np.ndarray
    volume_fixed_costs: np.ndarray
    product_volumes: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def n_plants(self) -> int:
        return len(self.plant_ids)

    @property
    def n_depots(self) -> int:
        return len(self.depot_ids)

    @property
    def n_customers(self) -> int:
        return len(self.customer_ids)

    @property
    def n_products(self) -> int:
        return len(self.product_ids)

    @property
    def n_levels(self) -> int:
        return int(self.product_capacity_levels.shape[2])

    @property
    def total_demand_by_product(self) -> np.ndarray:
        return self.demand.sum(axis=0)

    @property
    def largest_product_capacities(self) -> np.ndarray:
        return self.product_capacity_levels[:, :, -1]

    @property
    def largest_product_capacity_costs(self) -> np.ndarray:
        return self.product_capacity_fixed_costs[:, :, -1]

    @property
    def largest_volume_capacities(self) -> np.ndarray:
        return self.volume_capacity_levels[:, -1]

    @property
    def largest_volume_capacity_costs(self) -> np.ndarray:
        return self.volume_fixed_costs[:, -1]

    def subset_depots(self, depot_indices: list[int] | np.ndarray, suffix: str) -> "MultiLevelInstance":
        idx = np.asarray(depot_indices, dtype=int)
        return MultiLevelInstance(
            name=f"{self.name}_{suffix}",
            plant_ids=self.plant_ids,
            depot_ids=tuple(self.depot_ids[i] for i in idx),
            customer_ids=self.customer_ids,
            product_ids=self.product_ids,
            plant_coords=self.plant_coords.copy(),
            depot_coords=self.depot_coords[idx].copy(),
            customer_coords=self.customer_coords.copy(),
            plant_capacity=self.plant_capacity.copy(),
            demand=self.demand.copy(),
            product_capacity_levels=self.product_capacity_levels[idx].copy(),
            product_capacity_fixed_costs=self.product_capacity_fixed_costs[idx].copy(),
            depot_opening_cost=self.depot_opening_cost[idx].copy(),
            transport_plant_depot=self.transport_plant_depot[:, idx, :].copy(),
            transport_depot_customer=self.transport_depot_customer[idx].copy(),
            volume_capacity_levels=self.volume_capacity_levels[idx].copy(),
            volume_fixed_costs=self.volume_fixed_costs[idx].copy(),
            product_volumes=self.product_volumes.copy(),
            metadata=dict(self.metadata),
        )
