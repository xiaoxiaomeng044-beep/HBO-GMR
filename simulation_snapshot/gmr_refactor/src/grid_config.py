from dataclasses import dataclass
from pathlib import Path

from .common import load_json


@dataclass
class GridConfig:
    name: str
    nx: int
    ny: int
    nz: int
    N: int
    n_observations: int
    n_timesteps: int
    n_assim: int
    producers: list
    parameter_bounds: dict

    @property
    def total_cells(self):
        return self.nx * self.ny * self.nz

    @property
    def n_producers(self):
        return len(self.producers)

    def get_producer_ij_locations(self):
        return [(p["i"], p["j"]) for p in self.producers]


def load_grid_config(path):
    p = Path(path)
    if not p.is_file():
        candidate = Path(__file__).resolve().parents[1] / path
        if candidate.is_file():
            p = candidate
        elif str(path).replace("\\", "/").startswith("gmr_refactor/"):
            p = Path(__file__).resolve().parents[2] / path
    data = load_json(str(p))
    return GridConfig(
        name=data["name"],
        nx=int(data["nx"]),
        ny=int(data["ny"]),
        nz=int(data["nz"]),
        N=int(data["N"]),
        n_observations=int(data["n_observations"]),
        n_timesteps=int(data["n_timesteps"]),
        n_assim=int(data.get("n_assim", 4)),
        producers=list(data.get("producers", [])),
        parameter_bounds={k: tuple(v) for k, v in data.get("parameter_bounds", {}).items()},
    )
