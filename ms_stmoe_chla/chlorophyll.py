from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ms_stmoe_chla.moe_layers import Expert, MoE, RMSNorm, SparseMoEBlock


DATASET_REGISTRY: Dict[str, str] = {
    "bohai": "bohai_300.csv",
    "nanhai": "nanhai_265.csv",
}

BASELINE_DATASET = "bohai"
BASELINE_DATA_ROOT = "."
BASELINE_INPUT_WINDOW = 30
BASELINE_HORIZON = 15
BASELINE_BATCH_SIZE = 16
BASELINE_EPOCHS = 20
BASELINE_LR = 1e-3
BASELINE_WEIGHT_DECAY = 1e-4
BASELINE_DIM = 64
BASELINE_DEPTH = 2
BASELINE_MOE_DEPTH = 1
BASELINE_NUM_EXPERTS = 8
BASELINE_GATING_TOP_N = 2
BASELINE_ROUTER_NOISE_MULT = 1.0
BASELINE_BALANCE_LOSS_COEF = 1e-2
BASELINE_ROUTER_Z_LOSS_COEF = 1e-3
BASELINE_MOE_RESIDUAL_ALPHA = 1.0
BASELINE_MOE_WARMUP_EPOCHS = 0
BASELINE_MOE_WARMUP_START_SCALE = 0.0
BASELINE_MOE_WARMUP_DISABLE_ROUTER_NOISE = False
BASELINE_DROPOUT = 0.0
BASELINE_GRAPH_K_NEIGHBORS = 8
BASELINE_SPATIAL_HOPS = 1
BASELINE_TEMPORAL_KERNEL_SIZE = (3, 5, 7)
BASELINE_TEMPORAL_DILATIONS = (1, 2, 3)
BASELINE_TEMPORAL_SCALE_FUSION = "adaptive"
BASELINE_ST_AWARE_ROUTER = True
BASELINE_NUM_WORKERS = 0
BASELINE_CHECKPOINT_DIR = "checkpoints"
BASELINE_PREDICTION_DIR = "prediction_results"
BASELINE_PATIENCE = 20
BASELINE_SEED = 42
BASELINE_VALUE_TRANSFORM = "log1p"
BASELINE_SEASONAL_FEATURES = "dayofyear"
BASELINE_LOSS = "huber"
BASELINE_HUBER_DELTA = 1.0
BASELINE_HORIZON_LOSS_END_WEIGHT = 1.0

BASELINE_CHLOROPHYLL_CONFIG: Dict[str, object] = {
    "dataset": BASELINE_DATASET,
    "data_root": BASELINE_DATA_ROOT,
    "input_window": BASELINE_INPUT_WINDOW,
    "horizon": BASELINE_HORIZON,
    "batch_size": BASELINE_BATCH_SIZE,
    "epochs": BASELINE_EPOCHS,
    "lr": BASELINE_LR,
    "weight_decay": BASELINE_WEIGHT_DECAY,
    "dim": BASELINE_DIM,
    "depth": BASELINE_DEPTH,
    "moe_depth": BASELINE_MOE_DEPTH,
    "num_experts": BASELINE_NUM_EXPERTS,
    "gating_top_n": BASELINE_GATING_TOP_N,
    "router_noise_mult": BASELINE_ROUTER_NOISE_MULT,
    "balance_loss_coef": BASELINE_BALANCE_LOSS_COEF,
    "router_z_loss_coef": BASELINE_ROUTER_Z_LOSS_COEF,
    "moe_residual_alpha": BASELINE_MOE_RESIDUAL_ALPHA,
    "moe_warmup_epochs": BASELINE_MOE_WARMUP_EPOCHS,
    "moe_warmup_start_scale": BASELINE_MOE_WARMUP_START_SCALE,
    "moe_warmup_disable_router_noise": BASELINE_MOE_WARMUP_DISABLE_ROUTER_NOISE,
    "dropout": BASELINE_DROPOUT,
    "graph_k_neighbors": BASELINE_GRAPH_K_NEIGHBORS,
    "spatial_hops": BASELINE_SPATIAL_HOPS,
    "temporal_kernel_size": BASELINE_TEMPORAL_KERNEL_SIZE,
    "temporal_dilations": BASELINE_TEMPORAL_DILATIONS,
    "temporal_scale_fusion": BASELINE_TEMPORAL_SCALE_FUSION,
    "no_st_aware_router": not BASELINE_ST_AWARE_ROUTER,
    "num_workers": BASELINE_NUM_WORKERS,
    "checkpoint_dir": BASELINE_CHECKPOINT_DIR,
    "prediction_dir": BASELINE_PREDICTION_DIR,
    "no_save": False,
    "no_save_predictions": False,
    "smoke_test": False,
    "patience": BASELINE_PATIENCE,
    "seed": BASELINE_SEED,
    "value_transform": BASELINE_VALUE_TRANSFORM,
    "seasonal_features": BASELINE_SEASONAL_FEATURES,
    "loss": BASELINE_LOSS,
    "huber_delta": BASELINE_HUBER_DELTA,
    "horizon_loss_end_weight": BASELINE_HORIZON_LOSS_END_WEIGHT,
    "no_moe": False,
    "no_graph_conv": False,
    "no_seasonal_encoding": False,
    "no_multiscale_tcn": False,
}


@dataclass(frozen=True)
class ChlorophyllData:
    name: str
    path: Path
    values: np.ndarray
    node_ids: np.ndarray
    coordinates: np.ndarray
    time_labels: Tuple[str, ...]

    @property
    def num_timesteps(self) -> int:
        return self.values.shape[0]

    @property
    def num_nodes(self) -> int:
        return self.values.shape[1]


@dataclass(frozen=True)
class ChlorophyllSplits:
    train: Dataset
    val: Dataset
    test: Dataset
    mean: float
    std: float
    value_transform: str
    num_nodes: int
    num_timesteps: int
    coordinates: np.ndarray
    static_features: np.ndarray
    adjacency_matrix: np.ndarray
    input_feature_dim: int
    seasonal_features: str


def resolve_dataset_path(dataset: Union[str, Path], root: Union[str, Path] = ".") -> Tuple[str, Path]:
    dataset_path = Path(dataset)

    if dataset_path.suffix:
        path = dataset_path if dataset_path.is_absolute() else Path(root) / dataset_path
        return dataset_path.stem, path

    name = str(dataset).lower()
    if name not in DATASET_REGISTRY:
        choices = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset '{dataset}'. Use one of: {choices}, or pass a CSV path.")

    return name, Path(root) / DATASET_REGISTRY[name]


def load_chlorophyll_csv(dataset: Union[str, Path], root: Union[str, Path] = ".") -> ChlorophyllData:
    name, path = resolve_dataset_path(dataset, root)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    frame = pd.read_csv(path)
    required_columns = {"date", "lat", "lon"}
    missing = required_columns.difference(frame.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"{path.name} is missing required metadata columns: {missing_cols}")

    value_columns = [column for column in frame.columns if column not in required_columns]
    if not value_columns:
        raise ValueError(f"{path.name} does not contain any time-step value columns")

    values = frame[value_columns].to_numpy(dtype=np.float32).T
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} contains NaN or infinite values")

    return ChlorophyllData(
        name=name,
        path=path,
        values=values,
        node_ids=frame["date"].to_numpy(),
        coordinates=frame[["lat", "lon"]].to_numpy(dtype=np.float32),
        time_labels=tuple(str(column) for column in value_columns),
    )


def transform_values(values: np.ndarray, value_transform: str) -> np.ndarray:
    if value_transform == "none":
        return values
    if value_transform == "log1p":
        if values.min() < -1.0:
            raise ValueError("log1p transform requires all values to be greater than or equal to -1")
        return np.log1p(values)
    raise ValueError(f"Unknown value_transform '{value_transform}'. Use 'log1p' or 'none'.")


def haversine_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coordinates must have shape [num_nodes, 2] with latitude and longitude")

    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_static_features(coordinates: np.ndarray) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.float32)
    mean = coords.mean(axis=0, keepdims=True)
    std = coords.std(axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    return ((coords - mean) / std).astype(np.float32)


def build_adjacency_matrix(
    coordinates: np.ndarray,
    *,
    k_neighbors: int = 8,
    distance_sigma: Optional[float] = None,
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=np.float32)
    num_nodes = coords.shape[0]
    if num_nodes <= 0:
        raise ValueError("coordinates must contain at least one node")
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")

    distances = haversine_distance_matrix(coords)
    effective_k = min(k_neighbors, max(1, num_nodes - 1))
    neighbor_distances = []
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float64)

    for node in range(num_nodes):
        order = np.argsort(distances[node])
        neighbors = [index for index in order if index != node][:effective_k]
        neighbor_distances.extend(distances[node, neighbors].tolist())

    if distance_sigma is None:
        positive = np.asarray([distance for distance in neighbor_distances if distance > 0.0], dtype=np.float64)
        distance_sigma = float(np.median(positive)) if positive.size else 1.0
    distance_sigma = max(float(distance_sigma), 1e-6)

    for node in range(num_nodes):
        order = np.argsort(distances[node])
        neighbors = [index for index in order if index != node][:effective_k]
        weights = np.exp(-np.square(distances[node, neighbors]) / np.square(distance_sigma))
        adjacency[node, neighbors] = weights

    adjacency = np.maximum(adjacency, adjacency.T)
    adjacency = adjacency + np.eye(num_nodes, dtype=np.float64)
    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = np.power(np.clip(degree, 1e-12, None), -0.5)
    normalized = inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    return normalized.astype(np.float32)


def build_dayofyear_features(time_labels: Tuple[str, ...]) -> np.ndarray:
    numeric_labels = pd.to_numeric(pd.Series(time_labels), errors="coerce")
    if numeric_labels.notna().all():
        day_of_year = np.mod(numeric_labels.to_numpy(dtype=np.float32) - 1.0, 365.0) + 1.0
        period = np.full_like(day_of_year, 365.0)
        angle = 2.0 * np.pi * day_of_year / period
        return np.stack([np.sin(angle), np.cos(angle)], axis=-1).astype(np.float32)

    parsed = pd.to_datetime(list(time_labels), errors="coerce")
    if parsed.notna().all():
        day_of_year = parsed.dayofyear.to_numpy(dtype=np.float32)
        period = np.where(parsed.is_leap_year.to_numpy(), 366.0, 365.0).astype(np.float32)
    else:
        step_index = np.arange(len(time_labels), dtype=np.float32)
        day_of_year = np.mod(step_index, 365.0) + 1.0
        period = np.full_like(day_of_year, 365.0)

    angle = 2.0 * np.pi * day_of_year / period
    return np.stack([np.sin(angle), np.cos(angle)], axis=-1).astype(np.float32)


def build_temporal_features(time_labels: Tuple[str, ...], seasonal_features: str) -> Optional[np.ndarray]:
    if seasonal_features == "none":
        return None
    if seasonal_features == "dayofyear":
        return build_dayofyear_features(time_labels)
    raise ValueError(f"Unknown seasonal_features '{seasonal_features}'. Use 'none' or 'dayofyear'.")


class ChlorophyllWindowDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        starts: Iterable[int],
        input_window: int,
        horizon: int,
        mean: float,
        std: float,
        temporal_features: Optional[np.ndarray] = None,
    ):
        self.values = values
        self.starts = np.asarray(list(starts), dtype=np.int64)
        self.input_window = input_window
        self.horizon = horizon
        self.mean = mean
        self.std = std
        self.temporal_features = temporal_features
        self.input_feature_dim = 1 if temporal_features is None else 1 + temporal_features.shape[-1]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        mid = start + self.input_window
        end = mid + self.horizon

        x = (self.values[start:mid] - self.mean) / self.std
        y = (self.values[mid:end] - self.mean) / self.std

        if self.temporal_features is not None:
            seasonal = self.temporal_features[start:mid]
            seasonal = np.broadcast_to(seasonal[:, None, :], (*x.shape, seasonal.shape[-1]))
            x = np.concatenate([x[..., None], seasonal], axis=-1)

        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean


def make_chlorophyll_splits(
    dataset: Union[str, Path],
    *,
    root: Union[str, Path] = ".",
    input_window: int = BASELINE_INPUT_WINDOW,
    horizon: int = BASELINE_HORIZON,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    value_transform: str = BASELINE_VALUE_TRANSFORM,
    graph_k_neighbors: int = BASELINE_GRAPH_K_NEIGHBORS,
    seasonal_features: str = BASELINE_SEASONAL_FEATURES,
) -> ChlorophyllSplits:
    if input_window <= 0 or horizon <= 0:
        raise ValueError("input_window and horizon must be positive")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    chlorophyll = load_chlorophyll_csv(dataset, root)
    values = transform_values(chlorophyll.values, value_transform).astype(np.float32, copy=False)
    temporal_features = build_temporal_features(chlorophyll.time_labels, seasonal_features)
    num_windows = chlorophyll.num_timesteps - input_window - horizon + 1
    if num_windows <= 0:
        raise ValueError(
            f"{chlorophyll.path.name} has {chlorophyll.num_timesteps} time steps, which is too short "
            f"for input_window={input_window} and horizon={horizon}"
        )

    train_end = max(1, int(num_windows * train_ratio))
    val_end = max(train_end + 1, int(num_windows * (train_ratio + val_ratio)))
    val_end = min(val_end, num_windows)

    starts = np.arange(num_windows)
    train_starts = starts[:train_end]
    val_starts = starts[train_end:val_end]
    test_starts = starts[val_end:]

    train_values_end = train_starts[-1] + input_window + horizon
    train_values = values[:train_values_end]
    mean = float(train_values.mean())
    std = float(train_values.std())
    if std == 0.0:
        std = 1.0

    return ChlorophyllSplits(
        train=ChlorophyllWindowDataset(values, train_starts, input_window, horizon, mean, std, temporal_features),
        val=ChlorophyllWindowDataset(values, val_starts, input_window, horizon, mean, std, temporal_features),
        test=ChlorophyllWindowDataset(values, test_starts, input_window, horizon, mean, std, temporal_features),
        mean=mean,
        std=std,
        value_transform=value_transform,
        num_nodes=chlorophyll.num_nodes,
        num_timesteps=chlorophyll.num_timesteps,
        coordinates=chlorophyll.coordinates,
        static_features=build_static_features(chlorophyll.coordinates),
        adjacency_matrix=build_adjacency_matrix(chlorophyll.coordinates, k_neighbors=graph_k_neighbors),
        input_feature_dim=1 if temporal_features is None else 1 + temporal_features.shape[-1],
        seasonal_features=seasonal_features,
    )


def make_chlorophyll_dataloaders(
    dataset: Union[str, Path],
    *,
    root: Union[str, Path] = ".",
    input_window: int = BASELINE_INPUT_WINDOW,
    horizon: int = BASELINE_HORIZON,
    batch_size: int = BASELINE_BATCH_SIZE,
    num_workers: int = BASELINE_NUM_WORKERS,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    value_transform: str = BASELINE_VALUE_TRANSFORM,
    graph_k_neighbors: int = BASELINE_GRAPH_K_NEIGHBORS,
    seasonal_features: str = BASELINE_SEASONAL_FEATURES,
) -> Tuple[DataLoader, DataLoader, DataLoader, ChlorophyllSplits]:
    splits = make_chlorophyll_splits(
        dataset,
        root=root,
        input_window=input_window,
        horizon=horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        value_transform=value_transform,
        graph_k_neighbors=graph_k_neighbors,
        seasonal_features=seasonal_features,
    )

    train_loader = DataLoader(
        splits.train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(splits.val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(splits.test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, splits


class SpatialGraphConvolution(nn.Module):
    def __init__(self, dim: int, hops: int = 1):
        super().__init__()
        if hops <= 0:
            raise ValueError("hops must be positive")
        self.hops = hops
        self.transforms = nn.ModuleList(nn.Linear(dim, dim) for _ in range(hops + 1))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        out = self.transforms[0](x)
        h = x
        for hop in range(1, self.hops + 1):
            h = torch.einsum("ij,btjd->btid", adjacency, h)
            out = out + self.transforms[hop](h)
        return out / float(self.hops + 1)


class TemporalDependencyConvolution(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.conv = nn.Conv2d(
            dim,
            dim * 2,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
        )
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = rearrange(x, "b t n d -> b d t n")
        h = self.conv(h)
        h, gate = h.chunk(2, dim=1)
        h = h * torch.sigmoid(gate)
        h = rearrange(h, "b d t n -> b t n d")
        return self.proj(h)


class MultiScaleDilatedTemporalConvolution(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        dilations: Sequence[int] = (1, 2, 3),
        scale_fusion: str = BASELINE_TEMPORAL_SCALE_FUSION,
    ):
        super().__init__()
        if len(kernel_sizes) == 0:
            raise ValueError("kernel_sizes must contain at least one value")
        if len(kernel_sizes) != len(dilations):
            raise ValueError("kernel_sizes and dilations must have the same length")
        if scale_fusion not in {"concat", "adaptive"}:
            raise ValueError("scale_fusion must be 'concat' or 'adaptive'")

        self.scale_fusion = scale_fusion
        self.branches = nn.ModuleList()
        for kernel_size, dilation in zip(kernel_sizes, dilations):
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError("all temporal kernel sizes must be positive odd integers")
            if dilation <= 0:
                raise ValueError("all temporal dilations must be positive")

            padding = ((kernel_size - 1) * dilation) // 2
            self.branches.append(
                nn.Conv2d(
                    dim,
                    dim * 2,
                    kernel_size=(kernel_size, 1),
                    dilation=(dilation, 1),
                    padding=(padding, 0),
                )
            )

        if scale_fusion == "concat":
            self.mix = nn.Sequential(
                nn.Linear(dim * len(self.branches), dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            self.scale_gate = None
            self.output_proj = None
        else:
            self.mix = None
            self.scale_gate = nn.Sequential(
                nn.LayerNorm(dim * 2),
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.Linear(dim, len(self.branches)),
            )
            self.output_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = rearrange(x, "b t n d -> b d t n")
        branch_outputs = []

        for branch in self.branches:
            branch_out = branch(h)
            branch_out, gate = branch_out.chunk(2, dim=1)
            branch_out = branch_out * torch.sigmoid(gate)
            branch_outputs.append(rearrange(branch_out, "b d t n -> b t n d"))

        if self.scale_fusion == "concat":
            return self.mix(torch.cat(branch_outputs, dim=-1))

        scale_features = torch.cat([x[:, -1], x.mean(dim=1)], dim=-1)
        scale_weights = self.scale_gate(scale_features).softmax(dim=-1)
        stacked = torch.stack(branch_outputs, dim=-2)
        fused = (stacked * scale_weights[:, None, :, :, None]).sum(dim=-2)
        return self.output_proj(fused)


class StaticFeatureModeling(nn.Module):
    def __init__(self, static_dim: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(static_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, static_features: torch.Tensor) -> torch.Tensor:
        return self.net(static_features)


class SpatioTemporalRepresentationLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        spatial_hops: int = BASELINE_SPATIAL_HOPS,
        temporal_kernel_size: Union[int, Sequence[int]] = BASELINE_TEMPORAL_KERNEL_SIZE,
        temporal_dilations: Sequence[int] = BASELINE_TEMPORAL_DILATIONS,
        temporal_scale_fusion: str = BASELINE_TEMPORAL_SCALE_FUSION,
        dropout: float = BASELINE_DROPOUT,
        use_graph_conv: bool = True,
        use_multiscale_tcn: bool = True,
    ):
        super().__init__()
        self.use_graph_conv = use_graph_conv
        self.spatial = SpatialGraphConvolution(dim, hops=spatial_hops) if use_graph_conv else None

        if not use_multiscale_tcn and not isinstance(temporal_kernel_size, int):
            temporal_kernel_size = temporal_kernel_size[0]

        if isinstance(temporal_kernel_size, int):
            self.temporal = TemporalDependencyConvolution(dim, kernel_size=temporal_kernel_size)
        else:
            self.temporal = MultiScaleDilatedTemporalConvolution(
                dim,
                kernel_sizes=temporal_kernel_size,
                dilations=temporal_dilations,
                scale_fusion=temporal_scale_fusion,
            )
        self.spatial_norm = nn.LayerNorm(dim)
        self.temporal_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if self.spatial is not None:
            spatial = F.gelu(self.spatial(x, adjacency))
            x = self.spatial_norm(x + self.dropout(spatial))
        temporal = F.gelu(self.temporal(x))
        return self.temporal_norm(x + self.dropout(temporal))


class DenseFeedForwardBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        expert_hidden_mult: int = 4,
        add_ff_before: bool = True,
        add_ff_after: bool = True,
    ):
        super().__init__()
        self.ff_before = Expert(dim, prenorm=True) if add_ff_before else None
        self.ff_prenorm = RMSNorm(dim)
        self.ff = Expert(dim=dim, hidden_mult=expert_hidden_mult)
        self.ff_after = Expert(dim, prenorm=True) if add_ff_after else None

    def forward(
        self,
        x: torch.Tensor,
        noise_gates: bool = False,
        noise_mult: float = 1.0,
        router_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.ff_before is not None:
            x = self.ff_before(x) + x

        residual = x
        x = self.ff(self.ff_prenorm(x)) + residual

        if self.ff_after is not None:
            x = self.ff_after(x) + x

        zero = x.new_zeros(())
        return x, zero, zero, zero


class MSSTMoEForecaster(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        input_window: int,
        horizon: int,
        dim: int = BASELINE_DIM,
        depth: int = BASELINE_DEPTH,
        num_experts: int = BASELINE_NUM_EXPERTS,
        gating_top_n: int = BASELINE_GATING_TOP_N,
        expert_hidden_mult: int = 4,
        dropout: float = BASELINE_DROPOUT,
        input_feature_dim: int = 1,
        adjacency_matrix: Optional[Union[np.ndarray, torch.Tensor]] = None,
        static_features: Optional[Union[np.ndarray, torch.Tensor]] = None,
        spatial_hops: int = BASELINE_SPATIAL_HOPS,
        temporal_kernel_size: Union[int, Sequence[int]] = BASELINE_TEMPORAL_KERNEL_SIZE,
        temporal_dilations: Sequence[int] = BASELINE_TEMPORAL_DILATIONS,
        temporal_scale_fusion: str = BASELINE_TEMPORAL_SCALE_FUSION,
        moe_depth: int = BASELINE_MOE_DEPTH,
        router_noise_mult: float = BASELINE_ROUTER_NOISE_MULT,
        balance_loss_coef: float = BASELINE_BALANCE_LOSS_COEF,
        router_z_loss_coef: float = BASELINE_ROUTER_Z_LOSS_COEF,
        moe_residual_alpha: float = BASELINE_MOE_RESIDUAL_ALPHA,
        use_st_aware_router: bool = BASELINE_ST_AWARE_ROUTER,
        use_moe: bool = True,
        use_graph_conv: bool = True,
        use_multiscale_tcn: bool = True,
    ):
        super().__init__()

        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if input_window <= 0 or horizon <= 0:
            raise ValueError("input_window and horizon must be positive")
        if use_moe and gating_top_n < 2:
            raise ValueError("gating_top_n must be at least 2 for this MoE implementation")
        if use_moe and num_experts < gating_top_n:
            raise ValueError("num_experts must be greater than or equal to gating_top_n")
        if router_noise_mult < 0.0:
            raise ValueError("router_noise_mult must be non-negative")
        if balance_loss_coef < 0.0:
            raise ValueError("balance_loss_coef must be non-negative")
        if router_z_loss_coef < 0.0:
            raise ValueError("router_z_loss_coef must be non-negative")
        if moe_residual_alpha < 0.0:
            raise ValueError("moe_residual_alpha must be non-negative")
        if temporal_scale_fusion not in {"concat", "adaptive"}:
            raise ValueError("temporal_scale_fusion must be 'concat' or 'adaptive'")

        self.num_nodes = num_nodes
        self.input_window = input_window
        self.horizon = horizon
        self.input_feature_dim = input_feature_dim
        self.use_moe = use_moe
        self.use_graph_conv = use_graph_conv
        self.use_multiscale_tcn = use_multiscale_tcn
        self.use_st_aware_router = use_st_aware_router
        self.router_noise_mult = router_noise_mult

        if input_feature_dim <= 0:
            raise ValueError("input_feature_dim must be positive")

        if adjacency_matrix is None:
            adjacency = torch.eye(num_nodes, dtype=torch.float32)
        else:
            adjacency = torch.as_tensor(adjacency_matrix, dtype=torch.float32)
        if adjacency.shape != (num_nodes, num_nodes):
            raise ValueError(f"adjacency_matrix must have shape [{num_nodes}, {num_nodes}]")
        self.register_buffer("adjacency_matrix", adjacency)

        static_tensor = None
        if static_features is not None:
            static_tensor = torch.as_tensor(static_features, dtype=torch.float32)
            if static_tensor.ndim != 2 or static_tensor.shape[0] != num_nodes:
                raise ValueError("static_features must have shape [num_nodes, static_feature_dim]")
            self.register_buffer("static_features", static_tensor)
            self.static_modeling = StaticFeatureModeling(static_tensor.shape[-1], dim)
        else:
            self.register_buffer("static_features", torch.empty(num_nodes, 0))
            self.static_modeling = None

        self.input_projection = nn.Linear(input_feature_dim, dim)
        self.node_embedding = nn.Embedding(num_nodes, dim)
        self.dropout = nn.Dropout(dropout)

        self.representation_layers = nn.ModuleList(
            [
                SpatioTemporalRepresentationLayer(
                    dim,
                    spatial_hops=spatial_hops,
                    temporal_kernel_size=temporal_kernel_size,
                    temporal_dilations=temporal_dilations,
                    temporal_scale_fusion=temporal_scale_fusion,
                    dropout=dropout,
                    use_graph_conv=use_graph_conv,
                    use_multiscale_tcn=use_multiscale_tcn,
                )
                for _ in range(depth)
            ]
        )
        self.representation_norm = nn.LayerNorm(dim)

        if use_moe:
            expert_blocks = [
                SparseMoEBlock(
                    MoE(
                        dim=dim,
                        num_experts=num_experts,
                        gating_top_n=gating_top_n,
                        expert_hidden_mult=expert_hidden_mult,
                        balance_loss_coef=balance_loss_coef,
                        router_z_loss_coef=router_z_loss_coef,
                    ),
                    add_ff_before=True,
                    add_ff_after=True,
                    moe_residual_alpha_init=moe_residual_alpha,
                )
                for _ in range(moe_depth if use_moe else 0)
            ]
        else:
            expert_blocks = [
                DenseFeedForwardBlock(
                    dim=dim,
                    expert_hidden_mult=expert_hidden_mult,
                    add_ff_before=True,
                    add_ff_after=True,
                )
                for _ in range(moe_depth)
            ]

        self.expert_blocks = nn.ModuleList(expert_blocks)
        self.router_context = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        ) if use_moe and use_st_aware_router else None
        self.router_context_norm = nn.LayerNorm(dim) if use_moe and use_st_aware_router else None
        self.norm = nn.LayerNorm(dim)
        self.output_head = nn.Linear(dim, horizon)

    def _summarize_temporal_sequence(
        self,
        h: torch.Tensor,
        norm: nn.LayerNorm,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        recent_repr = h[:, -1]
        context_repr = h.mean(dim=1)
        change_repr = h[:, -1] - h[:, 0]
        node_repr = norm(recent_repr + context_repr)
        return node_repr, recent_repr, context_repr, change_repr

    def _apply_expert_blocks(
        self,
        h: torch.Tensor,
        *,
        recent_repr: torch.Tensor,
        context_repr: torch.Tensor,
        change_repr: torch.Tensor,
        node_prior: torch.Tensor,
        moe_residual_scale: float,
        router_noise_enabled: bool,
        include_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        router_context = None
        if self.router_context is not None and self.router_context_norm is not None:
            router_features = torch.cat(
                [
                    recent_repr,
                    context_repr,
                    change_repr,
                    node_prior.expand(h.shape[0], -1, -1),
                ],
                dim=-1,
            )
            router_context = self.router_context(router_features)

        aux_loss = h.new_zeros(())
        for block in self.expert_blocks:
            use_router_noise = self.training and router_noise_enabled and self.router_noise_mult > 0.0
            router_input = None
            if router_context is not None and self.router_context_norm is not None:
                router_input = self.router_context_norm(h + router_context)
            if self.use_moe:
                h, block_aux_loss, _, _ = block(
                    h,
                    noise_gates=use_router_noise,
                    noise_mult=self.router_noise_mult,
                    router_input=router_input,
                    moe_residual_scale=moe_residual_scale,
                )
            else:
                h, block_aux_loss, _, _ = block(
                    h,
                    noise_gates=use_router_noise,
                    noise_mult=self.router_noise_mult,
                    router_input=router_input,
                )
            if include_aux_loss:
                aux_loss = aux_loss + block_aux_loss

        return h, aux_loss

    def forward(
        self,
        x: torch.Tensor,
        *,
        moe_residual_scale: float = 1.0,
        router_noise_enabled: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, input_window, num_nodes] or [batch, input_window, num_nodes, features]")
        if moe_residual_scale < 0.0:
            raise ValueError("moe_residual_scale must be non-negative")
        if x.shape[1] != self.input_window or x.shape[2] != self.num_nodes or x.shape[3] != self.input_feature_dim:
            raise ValueError(
                f"Expected x shape [batch, {self.input_window}, {self.num_nodes}, {self.input_feature_dim}], "
                f"got {tuple(x.shape)}"
            )

        _, _, nodes, _ = x.shape
        device = x.device

        node_ids = torch.arange(nodes, device=device)

        node_prior = self.node_embedding(node_ids)[None, :, :]

        h = self.input_projection(x)
        h = h + node_prior[:, None, :, :]
        static_repr = None
        if self.static_modeling is not None:
            static_repr = self.static_modeling(self.static_features)
            node_prior = node_prior + static_repr[None, :, :]
            h = h + static_repr[None, None, :, :]
        h = self.dropout(h)

        for layer in self.representation_layers:
            h = layer(h, self.adjacency_matrix)

        h, recent_repr, context_repr, change_repr = self._summarize_temporal_sequence(h, self.representation_norm)
        h, aux_loss = self._apply_expert_blocks(
            h,
            recent_repr=recent_repr,
            context_repr=context_repr,
            change_repr=change_repr,
            node_prior=node_prior,
            moe_residual_scale=moe_residual_scale,
            router_noise_enabled=router_noise_enabled,
            include_aux_loss=True,
        )

        h = self.norm(h)
        prediction = self.output_head(h)
        return rearrange(prediction, "b n h -> b h n"), aux_loss
