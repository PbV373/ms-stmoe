from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ms_stmoe_chla.chlorophyll import (
    BASELINE_CHLOROPHYLL_CONFIG,
    MSSTMoEForecaster,
    make_chlorophyll_dataloaders,
)


def parse_int_list(value: str) -> Union[int, Tuple[int, ...]]:
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)):
        if not value:
            raise argparse.ArgumentTypeError("value must contain at least one integer")
        try:
            values = tuple(int(part) for part in value)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError("value must be an integer or comma-separated integers") from exc
        return values[0] if len(values) == 1 else values

    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("value must contain at least one integer")
    try:
        values = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer or comma-separated integers") from exc
    return values[0] if len(values) == 1 else values


def as_tuple(value: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def load_config_defaults(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")

    return {str(key).replace("-", "_"): value for key, value in config.items()}


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="Path to a JSON training config. Command-line flags override config values.")
    config_args, remaining_args = config_parser.parse_known_args()
    config_defaults = load_config_defaults(config_args.config)

    def cfg(name: str, default: Any) -> Any:
        return config_defaults.get(name, default)

    parser = argparse.ArgumentParser(
        description="Train MS-STMoE on Bohai or Nanhai chlorophyll CSV data.",
        parents=[config_parser],
    )
    parser.add_argument("--dataset", default=cfg("dataset", BASELINE_CHLOROPHYLL_CONFIG["dataset"]), help="Dataset name: bohai, nanhai, or a CSV path.")
    parser.add_argument("--data-root", default=cfg("data_root", BASELINE_CHLOROPHYLL_CONFIG["data_root"]), help="Directory containing bohai_300.csv and nanhai_265.csv.")
    parser.add_argument("--input-window", type=int, default=cfg("input_window", BASELINE_CHLOROPHYLL_CONFIG["input_window"]), help="Number of historical time steps.")
    parser.add_argument("--horizon", type=int, default=cfg("horizon", BASELINE_CHLOROPHYLL_CONFIG["horizon"]), help="Number of future time steps to predict.")
    parser.add_argument("--batch-size", type=int, default=cfg("batch_size", BASELINE_CHLOROPHYLL_CONFIG["batch_size"]))
    parser.add_argument("--epochs", type=int, default=cfg("epochs", BASELINE_CHLOROPHYLL_CONFIG["epochs"]))
    parser.add_argument("--lr", type=float, default=cfg("lr", BASELINE_CHLOROPHYLL_CONFIG["lr"]))
    parser.add_argument("--weight-decay", type=float, default=cfg("weight_decay", BASELINE_CHLOROPHYLL_CONFIG["weight_decay"]))
    parser.add_argument("--dim", type=int, default=cfg("dim", BASELINE_CHLOROPHYLL_CONFIG["dim"]))
    parser.add_argument("--depth", type=int, default=cfg("depth", BASELINE_CHLOROPHYLL_CONFIG["depth"]))
    parser.add_argument("--moe-depth", type=int, default=cfg("moe_depth", BASELINE_CHLOROPHYLL_CONFIG["moe_depth"]))
    parser.add_argument("--num-experts", type=int, default=cfg("num_experts", BASELINE_CHLOROPHYLL_CONFIG["num_experts"]))
    parser.add_argument("--gating-top-n", type=int, default=cfg("gating_top_n", BASELINE_CHLOROPHYLL_CONFIG["gating_top_n"]))
    parser.add_argument("--router-noise-mult", type=float, default=cfg("router_noise_mult", BASELINE_CHLOROPHYLL_CONFIG["router_noise_mult"]), help="Gumbel noise multiplier for MoE routing during training. Use 0 to disable router noise.")
    parser.add_argument("--balance-loss-coef", type=float, default=cfg("balance_loss_coef", BASELINE_CHLOROPHYLL_CONFIG["balance_loss_coef"]), help="Auxiliary loss weight for balancing expert usage.")
    parser.add_argument("--router-z-loss-coef", type=float, default=cfg("router_z_loss_coef", BASELINE_CHLOROPHYLL_CONFIG["router_z_loss_coef"]), help="Auxiliary router z-loss weight for stabilizing gate logits.")
    parser.add_argument("--moe-residual-alpha", type=float, default=cfg("moe_residual_alpha", BASELINE_CHLOROPHYLL_CONFIG["moe_residual_alpha"]), help="Initial learnable scale applied to the MoE residual branch.")
    parser.add_argument("--moe-warmup-epochs", type=int, default=cfg("moe_warmup_epochs", BASELINE_CHLOROPHYLL_CONFIG["moe_warmup_epochs"]), help="Linearly increase the effective MoE residual scale over this many epochs. Use 0 to disable.")
    parser.add_argument("--moe-warmup-start-scale", type=float, default=cfg("moe_warmup_start_scale", BASELINE_CHLOROPHYLL_CONFIG["moe_warmup_start_scale"]), help="Starting multiplier for the MoE residual branch during warm-up.")
    parser.add_argument("--moe-warmup-disable-router-noise", action="store_true", default=cfg("moe_warmup_disable_router_noise", BASELINE_CHLOROPHYLL_CONFIG["moe_warmup_disable_router_noise"]), help="Disable router noise while MoE residual warm-up is active.")
    parser.add_argument("--dropout", type=float, default=cfg("dropout", BASELINE_CHLOROPHYLL_CONFIG["dropout"]))
    parser.add_argument("--graph-k-neighbors", type=int, default=cfg("graph_k_neighbors", BASELINE_CHLOROPHYLL_CONFIG["graph_k_neighbors"]))
    parser.add_argument("--spatial-hops", type=int, default=cfg("spatial_hops", BASELINE_CHLOROPHYLL_CONFIG["spatial_hops"]))
    parser.add_argument("--temporal-kernel-size", type=parse_int_list, default=parse_int_list(cfg("temporal_kernel_size", BASELINE_CHLOROPHYLL_CONFIG["temporal_kernel_size"])))
    parser.add_argument("--temporal-dilations", type=parse_int_list, default=parse_int_list(cfg("temporal_dilations", BASELINE_CHLOROPHYLL_CONFIG["temporal_dilations"])))
    parser.add_argument("--temporal-scale-fusion", choices=("adaptive", "concat"), default=cfg("temporal_scale_fusion", BASELINE_CHLOROPHYLL_CONFIG["temporal_scale_fusion"]), help="Fusion strategy for multi-scale temporal convolution branches.")
    parser.add_argument("--no-st-aware-router", action="store_true", default=cfg("no_st_aware_router", BASELINE_CHLOROPHYLL_CONFIG["no_st_aware_router"]), help="Ablation: use the original MoE router without explicit spatio-temporal context.")
    parser.add_argument("--num-workers", type=int, default=cfg("num_workers", BASELINE_CHLOROPHYLL_CONFIG["num_workers"]))
    parser.add_argument("--device", default=cfg("device", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--checkpoint-dir", default=cfg("checkpoint_dir", BASELINE_CHLOROPHYLL_CONFIG["checkpoint_dir"]))
    parser.add_argument("--no-save", action="store_true", default=cfg("no_save", BASELINE_CHLOROPHYLL_CONFIG["no_save"]), help="Do not write a best checkpoint.")
    parser.add_argument("--prediction-dir", default=cfg("prediction_dir", BASELINE_CHLOROPHYLL_CONFIG["prediction_dir"]), help="Directory for saved test predictions.")
    parser.add_argument("--no-save-predictions", action="store_true", default=cfg("no_save_predictions", BASELINE_CHLOROPHYLL_CONFIG["no_save_predictions"]), help="Do not save original-scale test predictions.")
    parser.add_argument("--smoke-test", action="store_true", default=cfg("smoke_test", BASELINE_CHLOROPHYLL_CONFIG["smoke_test"]), help="Run one train batch and one eval batch.")
    parser.add_argument("--patience", type=int, default=cfg("patience", BASELINE_CHLOROPHYLL_CONFIG["patience"]), help="Early stop after this many epochs without val loss improvement.")
    parser.add_argument("--seed", type=int, default=cfg("seed", BASELINE_CHLOROPHYLL_CONFIG["seed"]), help="Random seed for reproducible training.")
    parser.add_argument("--value-transform", choices=("log1p", "none"), default=cfg("value_transform", BASELINE_CHLOROPHYLL_CONFIG["value_transform"]))
    parser.add_argument("--seasonal-features", choices=("dayofyear", "none"), default=cfg("seasonal_features", BASELINE_CHLOROPHYLL_CONFIG["seasonal_features"]))
    parser.add_argument("--loss", choices=("huber", "mse", "mae"), default=cfg("loss", BASELINE_CHLOROPHYLL_CONFIG["loss"]))
    parser.add_argument("--huber-delta", type=float, default=cfg("huber_delta", BASELINE_CHLOROPHYLL_CONFIG["huber_delta"]))
    parser.add_argument("--horizon-loss-end-weight", type=float, default=cfg("horizon_loss_end_weight", BASELINE_CHLOROPHYLL_CONFIG["horizon_loss_end_weight"]), help="Final horizon loss weight for linear horizon-weighted loss. Use 1 for uniform loss.")
    parser.add_argument("--no-moe", action="store_true", default=cfg("no_moe", BASELINE_CHLOROPHYLL_CONFIG["no_moe"]), help="Ablation: replace MoE expert blocks with dense FFN blocks.")
    parser.add_argument("--no-graph-conv", action="store_true", default=cfg("no_graph_conv", BASELINE_CHLOROPHYLL_CONFIG["no_graph_conv"]), help="Ablation: remove spatial graph convolution.")
    parser.add_argument("--no-seasonal-encoding", action="store_true", default=cfg("no_seasonal_encoding", BASELINE_CHLOROPHYLL_CONFIG["no_seasonal_encoding"]), help="Ablation: remove seasonal/day-of-year input encoding.")
    parser.add_argument("--no-multiscale-tcn", action="store_true", default=cfg("no_multiscale_tcn", BASELINE_CHLOROPHYLL_CONFIG["no_multiscale_tcn"]), help="Ablation: replace multi-scale TCN with a single-scale temporal convolution.")

    known_dests = {action.dest for action in parser._actions}
    unknown_config_keys = sorted(set(config_defaults).difference(known_dests))
    if unknown_config_keys:
        unknown = ", ".join(unknown_config_keys)
        raise ValueError(f"Unknown config option(s) in {config_args.config}: {unknown}")

    args = parser.parse_args(remaining_args)
    args.config = config_args.config
    return args


def dataset_label(dataset: str) -> str:
    path = Path(dataset)
    if path.suffix:
        return path.stem
    return dataset.lower()


def ablation_label(args: argparse.Namespace) -> str:
    ablations = []
    if args.no_moe:
        ablations.append("no_moe")
    if args.no_st_aware_router and not args.no_moe:
        ablations.append("no_st_aware_router")
    if args.no_graph_conv:
        ablations.append("no_graph_conv")
    if args.no_seasonal_encoding:
        ablations.append("no_seasonal")
    if args.no_multiscale_tcn:
        ablations.append("no_multiscale_tcn")
    return "_".join(ablations) if ablations else "full"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clone_state_dict(model: torch.nn.Module) -> dict:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def filesystem_path(path: Path) -> Path:
    resolved = Path(os.path.abspath(os.fspath(path)))
    if os.name != "nt":
        return resolved

    text = str(resolved)
    if text.startswith("\\\\?\\"):
        return Path(text)
    if text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + text[2:])
    return Path("\\\\?\\" + text)


def save_checkpoint(payload: dict, path: Path) -> None:
    filesystem_target = filesystem_path(path)
    filesystem_target.parent.mkdir(parents=True, exist_ok=True)
    with filesystem_target.open("wb") as checkpoint_file:
        torch.save(payload, checkpoint_file)


def save_prediction_archive(
    path: Path,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: dict,
) -> None:
    filesystem_target = filesystem_path(path)
    filesystem_target.parent.mkdir(parents=True, exist_ok=True)
    with filesystem_target.open("wb") as prediction_file:
        np.savez_compressed(
            prediction_file,
            y_true=y_true,
            y_pred=y_pred,
            **metadata,
        )


def compute_main_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_name: str,
    huber_delta: float,
    horizon_loss_end_weight: float,
) -> torch.Tensor:
    if horizon_loss_end_weight <= 0.0:
        raise ValueError("horizon_loss_end_weight must be positive")

    reduction = "none" if horizon_loss_end_weight != 1.0 else "mean"
    if loss_name == "huber":
        loss = F.huber_loss(prediction, target, delta=huber_delta, reduction=reduction)
    elif loss_name == "mse":
        loss = F.mse_loss(prediction, target, reduction=reduction)
    elif loss_name == "mae":
        loss = F.l1_loss(prediction, target, reduction=reduction)
    else:
        raise ValueError(f"Unknown loss '{loss_name}'")

    if horizon_loss_end_weight == 1.0:
        return loss

    weights = torch.linspace(
        1.0,
        horizon_loss_end_weight,
        steps=prediction.shape[1],
        device=prediction.device,
        dtype=prediction.dtype,
    )
    return (loss * weights[None, :, None]).mean()


def invert_value_transform(tensor: torch.Tensor, value_transform: str) -> torch.Tensor:
    if value_transform == "none":
        return tensor
    if value_transform == "log1p":
        return torch.expm1(tensor).clamp_min(0.0)
    raise ValueError(f"Unknown value_transform '{value_transform}'")


def run_epoch(
    model: MSSTMoEForecaster,
    loader: torch.utils.data.DataLoader,
    *,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    loss_name: str,
    huber_delta: float,
    horizon_loss_end_weight: float,
    moe_residual_scale: float = 1.0,
    router_noise_enabled: bool = True,
    max_batches: Optional[int] = None,
) -> Tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_count = 0

    for batch_index, (x, y) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)

        with torch.set_grad_enabled(is_train):
            prediction, aux_loss = model(
                x,
                moe_residual_scale=moe_residual_scale,
                router_noise_enabled=router_noise_enabled,
            )
            main_loss = compute_main_loss(prediction, y, loss_name, huber_delta, horizon_loss_end_weight)
            mse = F.mse_loss(prediction.detach(), y)
            loss = main_loss + aux_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_mae += float(F.l1_loss(prediction.detach(), y).detach()) * batch_size
        total_rmse += float(torch.sqrt(mse.detach())) * batch_size
        total_count += batch_size

    return total_loss / total_count, total_mae / total_count, total_rmse / total_count


def evaluate_by_horizon(
    model: MSSTMoEForecaster,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    mean: float,
    std: float,
    value_transform: str,
    moe_residual_scale: float = 1.0,
    max_batches: Optional[int] = None,
    prediction_save_path: Optional[Path] = None,
    prediction_metadata: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()

    abs_error_sum = None
    squared_error_sum = None
    total_count = 0
    prediction_chunks = []
    target_chunks = []
    should_save_predictions = prediction_save_path is not None

    with torch.no_grad():
        for batch_index, (x, y) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)

            prediction, _ = model(x, moe_residual_scale=moe_residual_scale, router_noise_enabled=False)
            prediction = prediction * std + mean
            y = y * std + mean
            prediction = invert_value_transform(prediction, value_transform)
            y = invert_value_transform(y, value_transform)

            if should_save_predictions:
                prediction_chunks.append(prediction.detach().cpu().numpy().astype(np.float32))
                target_chunks.append(y.detach().cpu().numpy().astype(np.float32))

            abs_error = (prediction - y).abs().sum(dim=(0, 2))
            squared_error = (prediction - y).square().sum(dim=(0, 2))

            if abs_error_sum is None:
                abs_error_sum = abs_error
                squared_error_sum = squared_error
            else:
                abs_error_sum = abs_error_sum + abs_error
                squared_error_sum = squared_error_sum + squared_error

            total_count += prediction.shape[0] * prediction.shape[2]

    if abs_error_sum is None or squared_error_sum is None or total_count == 0:
        raise ValueError("Cannot evaluate an empty dataloader")

    mse = squared_error_sum / total_count
    mae = abs_error_sum / total_count
    rmse = torch.sqrt(mse)

    if should_save_predictions:
        save_prediction_archive(
            prediction_save_path,
            y_true=np.concatenate(target_chunks, axis=0),
            y_pred=np.concatenate(prediction_chunks, axis=0),
            metadata=prediction_metadata or {},
        )

    return mae.cpu(), rmse.cpu(), mse.cpu()


def print_horizon_metrics(mae: torch.Tensor, rmse: torch.Tensor, mse: torch.Tensor) -> None:
    print("per_horizon_metrics_original_scale")
    print("step,mae,rmse,mse")
    for step, (step_mae, step_rmse, step_mse) in enumerate(zip(mae, rmse, mse), start=1):
        print(f"{step},{step_mae:.6f},{step_rmse:.6f},{step_mse:.6f}")
    print(f"avg,{mae.mean():.6f},{rmse.mean():.6f},{mse.mean():.6f}")


def moe_warmup_scale(epoch: int, warmup_epochs: int, start_scale: float) -> float:
    if warmup_epochs <= 0:
        return 1.0
    if warmup_epochs == 1:
        return 1.0
    if epoch >= warmup_epochs:
        return 1.0

    progress = float(epoch - 1) / float(warmup_epochs - 1)
    return start_scale + (1.0 - start_scale) * progress


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if args.router_noise_mult < 0.0:
        raise ValueError("--router-noise-mult must be non-negative")
    if args.balance_loss_coef < 0.0:
        raise ValueError("--balance-loss-coef must be non-negative")
    if args.router_z_loss_coef < 0.0:
        raise ValueError("--router-z-loss-coef must be non-negative")
    if args.moe_residual_alpha < 0.0:
        raise ValueError("--moe-residual-alpha must be non-negative")
    if args.moe_warmup_epochs < 0:
        raise ValueError("--moe-warmup-epochs must be non-negative")
    if not 0.0 <= args.moe_warmup_start_scale <= 1.0:
        raise ValueError("--moe-warmup-start-scale must be between 0 and 1")
    if args.horizon_loss_end_weight <= 0.0:
        raise ValueError("--horizon-loss-end-weight must be positive")
    if args.no_seasonal_encoding:
        args.seasonal_features = "none"

    temporal_kernel_sizes = as_tuple(args.temporal_kernel_size)
    temporal_dilations = as_tuple(args.temporal_dilations)
    if args.no_multiscale_tcn:
        temporal_kernel_size_for_model: Union[int, Tuple[int, ...]] = temporal_kernel_sizes[0]
        temporal_dilations_for_model = (temporal_dilations[0],)
    else:
        temporal_kernel_size_for_model = temporal_kernel_sizes[0] if len(temporal_kernel_sizes) == 1 else temporal_kernel_sizes
        temporal_dilations_for_model = temporal_dilations

    if not args.no_multiscale_tcn and len(temporal_kernel_sizes) != len(temporal_dilations):
        raise ValueError("--temporal-kernel-size and --temporal-dilations must have the same number of values")

    train_loader, val_loader, test_loader, splits = make_chlorophyll_dataloaders(
        args.dataset,
        root=args.data_root,
        input_window=args.input_window,
        horizon=args.horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        value_transform=args.value_transform,
        graph_k_neighbors=args.graph_k_neighbors,
        seasonal_features=args.seasonal_features,
    )

    model = MSSTMoEForecaster(
        num_nodes=splits.num_nodes,
        input_window=args.input_window,
        horizon=args.horizon,
        dim=args.dim,
        depth=args.depth,
        num_experts=args.num_experts,
        gating_top_n=args.gating_top_n,
        dropout=args.dropout,
        input_feature_dim=splits.input_feature_dim,
        adjacency_matrix=splits.adjacency_matrix,
        static_features=splits.static_features,
        spatial_hops=args.spatial_hops,
        temporal_kernel_size=temporal_kernel_size_for_model,
        temporal_dilations=temporal_dilations_for_model,
        temporal_scale_fusion=args.temporal_scale_fusion,
        moe_depth=args.moe_depth,
        router_noise_mult=args.router_noise_mult,
        balance_loss_coef=args.balance_loss_coef,
        router_z_loss_coef=args.router_z_loss_coef,
        moe_residual_alpha=args.moe_residual_alpha,
        use_st_aware_router=not args.no_st_aware_router,
        use_moe=not args.no_moe,
        use_graph_conv=not args.no_graph_conv,
        use_multiscale_tcn=not args.no_multiscale_tcn,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"dataset={dataset_label(args.dataset)} nodes={splits.num_nodes} "
        f"input_window={args.input_window} horizon={args.horizon} "
        f"graph_k_neighbors={args.graph_k_neighbors} spatial_hops={args.spatial_hops} "
        f"temporal_kernel_size={as_tuple(temporal_kernel_size_for_model)} temporal_dilations={temporal_dilations_for_model} "
        f"temporal_scale_fusion={'single' if args.no_multiscale_tcn else args.temporal_scale_fusion} "
        f"moe_depth={0 if args.no_moe else args.moe_depth} "
        f"st_aware_router={not args.no_st_aware_router and not args.no_moe} "
        f"router_noise_mult={0.0 if args.no_moe else args.router_noise_mult} "
        f"balance_loss_coef={0.0 if args.no_moe else args.balance_loss_coef} "
        f"router_z_loss_coef={0.0 if args.no_moe else args.router_z_loss_coef} "
        f"moe_residual_alpha={0.0 if args.no_moe else args.moe_residual_alpha} "
        f"moe_warmup_epochs={0 if args.no_moe else args.moe_warmup_epochs} "
        f"moe_warmup_start_scale={0.0 if args.no_moe else args.moe_warmup_start_scale} "
        f"moe_warmup_disable_router_noise={False if args.no_moe else args.moe_warmup_disable_router_noise} "
        f"dense_ffn_depth={args.moe_depth if args.no_moe else 0} "
        f"ablations={ablation_label(args)} "
        f"input_features={splits.input_feature_dim} seasonal_features={splits.seasonal_features} "
        f"time_steps={splits.num_timesteps} train/val/test="
        f"{len(splits.train)}/{len(splits.val)}/{len(splits.test)}"
    )
    print(f"normalization mean={splits.mean:.6f} std={splits.std:.6f}")
    print(f"value_transform={splits.value_transform} loss={args.loss} horizon_loss_end_weight={args.horizon_loss_end_weight}")

    checkpoint_path = Path(args.checkpoint_dir) / f"{dataset_label(args.dataset)}_ms_stmoe_{ablation_label(args)}.pt"
    prediction_path = (
        None
        if args.no_save_predictions
        else Path(args.prediction_dir) / dataset_label(args.dataset) / f"ms_stmoe_{ablation_label(args)}_predictions.npz"
    )
    best_val = float("inf")
    best_epoch = 0
    best_model_state = None
    best_moe_residual_scale = 1.0
    epochs_without_improvement = 0

    epochs = 1 if args.smoke_test else args.epochs
    for epoch in range(1, epochs + 1):
        max_batches = 1 if args.smoke_test else None
        current_moe_residual_scale = 1.0 if args.no_moe else moe_warmup_scale(
            epoch,
            args.moe_warmup_epochs,
            args.moe_warmup_start_scale,
        )
        warmup_active = (
            not args.no_moe
            and args.moe_warmup_epochs > 0
            and epoch <= args.moe_warmup_epochs
        )
        router_noise_enabled = not (warmup_active and args.moe_warmup_disable_router_noise)
        train_loss, train_mae, train_rmse = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            loss_name=args.loss,
            huber_delta=args.huber_delta,
            horizon_loss_end_weight=args.horizon_loss_end_weight,
            moe_residual_scale=current_moe_residual_scale,
            router_noise_enabled=router_noise_enabled,
            max_batches=max_batches,
        )
        val_loss, val_mae, val_rmse = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            loss_name=args.loss,
            huber_delta=args.huber_delta,
            horizon_loss_end_weight=args.horizon_loss_end_weight,
            moe_residual_scale=current_moe_residual_scale,
            router_noise_enabled=False,
            max_batches=max_batches,
        )

        print(
            f"epoch={epoch:03d} "
            f"moe_residual_scale={current_moe_residual_scale:.4f} "
            f"router_noise_enabled={router_noise_enabled} "
            f"train_loss={train_loss:.6f} train_mae={train_mae:.6f} train_rmse={train_rmse:.6f} "
            f"val_loss={val_loss:.6f} val_mae={val_mae:.6f} val_rmse={val_rmse:.6f}"
        )

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_model_state = clone_state_dict(model)
            best_moe_residual_scale = current_moe_residual_scale
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if improved and not args.no_save:
            save_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "num_nodes": splits.num_nodes,
                    "mean": splits.mean,
                    "std": splits.std,
                    "value_transform": splits.value_transform,
                    "loss": args.loss,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                    "best_moe_residual_scale": best_moe_residual_scale,
                },
                checkpoint_path,
            )

        if args.smoke_test:
            break

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"early_stop epoch={epoch:03d} best_val_loss={best_val:.6f} "
                f"patience={args.patience}"
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        print(
            f"loaded_best_model epoch={best_epoch:03d} best_val_loss={best_val:.6f} "
            f"moe_residual_scale={best_moe_residual_scale:.4f}"
        )

    test_loss, test_mae, test_rmse = run_epoch(
        model,
        test_loader,
        optimizer=None,
        device=device,
        loss_name=args.loss,
        huber_delta=args.huber_delta,
        horizon_loss_end_weight=args.horizon_loss_end_weight,
        moe_residual_scale=best_moe_residual_scale,
        router_noise_enabled=False,
        max_batches=1 if args.smoke_test else None,
    )
    print(f"test_loss={test_loss:.6f} test_mae={test_mae:.6f} test_rmse={test_rmse:.6f}")

    horizon_mae, horizon_rmse, horizon_mse = evaluate_by_horizon(
        model,
        test_loader,
        device=device,
        mean=splits.mean,
        std=splits.std,
        value_transform=splits.value_transform,
        moe_residual_scale=best_moe_residual_scale,
        max_batches=1 if args.smoke_test else None,
        prediction_save_path=prediction_path,
        prediction_metadata={
            "model_name": np.array("MS-STMoE"),
            "dataset_name": np.array(dataset_label(args.dataset)),
            "ablation": np.array(ablation_label(args)),
            "steps": np.arange(1, args.horizon + 1, dtype=np.int64),
            "input_window": np.array(args.input_window, dtype=np.int64),
            "horizon": np.array(args.horizon, dtype=np.int64),
            "num_nodes": np.array(splits.num_nodes, dtype=np.int64),
            "mean": np.array(splits.mean, dtype=np.float32),
            "std": np.array(splits.std, dtype=np.float32),
            "value_transform": np.array(splits.value_transform),
            "checkpoint_path": np.array(str(checkpoint_path)),
            "best_epoch": np.array(best_epoch, dtype=np.int64),
            "best_val_loss": np.array(best_val, dtype=np.float32),
            "best_moe_residual_scale": np.array(best_moe_residual_scale, dtype=np.float32),
        },
    )
    print_horizon_metrics(horizon_mae, horizon_rmse, horizon_mse)
    if prediction_path is not None:
        print(f"saved predictions: {prediction_path}")

    if not args.no_save:
        print(f"best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
