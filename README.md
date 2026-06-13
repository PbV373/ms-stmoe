# MS-STMoE for Chlorophyll-a Forecasting

PyTorch implementation of **MS-STMoE**, a multi-scale spatio-temporal mixture-of-experts model for multi-step marine chlorophyll-a (Chl-a) forecasting.

The model combines:

- geographic graph construction from node latitude/longitude,
- graph convolution for spatial dependencies,
- multi-scale temporal convolution for short- and medium-term dynamics,
- sparse mixture-of-experts routing for heterogeneous marine regions.

## Project Layout

```text
ms_stmoe_chla/
  chlorophyll.py       # Chl-a dataset loading, graph construction, MS-STMoE forecaster
  moe_layers.py        # sparse MoE layers
  distributed.py       # distributed helper utilities for MoE layers
configs/
  bohai.json           # Bohai Sea training configuration
  nanhai.json          # South China Sea training configuration
train_chlorophyll.py   # training and evaluation entry point
test_moe_layers.py     # distributed MoE consistency check
```

## Environment Installation

Create and activate a Python environment, then install the project in editable mode:

```bash
conda create -n ms-stmoe-chla python=3.10
conda activate ms-stmoe-chla
pip install -e .
```

If you already have a suitable Python environment, only run:

```bash
pip install -e .
```

Core dependencies are listed in `setup.py`. The training script uses CUDA automatically when PyTorch can access a GPU; otherwise pass `--device cpu`.

## Data Placement

The repository expects Chl-a CSV files with metadata columns followed by time-step columns:

- `date`: node identifier
- `lat`: node latitude
- `lon`: node longitude
- remaining columns: ordered Chl-a observations over time

Place the CSV files in the project root directory, next to `train_chlorophyll.py`:

```text
ms-stmoe-chla/
  bohai_300.csv
  nanhai_265.csv
  train_chlorophyll.py
```

Bundled dataset aliases:

- `bohai`: `bohai_300.csv`
- `nanhai`: `nanhai_265.csv`

You can also pass a custom CSV path with the same format:

```bash
python train_chlorophyll.py --dataset path/to/custom_chla.csv
```

## Training Commands

Use the paper-aligned configs:

```bash
python train_chlorophyll.py --config configs/bohai.json
python train_chlorophyll.py --config configs/nanhai.json
```

Or override options from the command line:

```bash
python train_chlorophyll.py --dataset bohai --epochs 20 --batch-size 16
python train_chlorophyll.py --dataset nanhai --epochs 20 --batch-size 16
```

By default, training saves the best checkpoint under `checkpoints/` and test predictions under `prediction_results/`.

## Testing Commands

Run a quick CPU smoke test to verify that the environment, data loading, model forward pass, and evaluation loop work:

```bash
python train_chlorophyll.py --dataset bohai --smoke-test --no-save --no-save-predictions --device cpu
python train_chlorophyll.py --dataset nanhai --smoke-test --no-save --no-save-predictions --device cpu
```

Run a short test-oriented training/evaluation pass without writing files:

```bash
python train_chlorophyll.py --dataset bohai --epochs 1 --no-save --no-save-predictions
python train_chlorophyll.py --dataset nanhai --epochs 1 --no-save --no-save-predictions
```

The script evaluates the best validation model on the test split at the end of each run and prints per-horizon MAE, RMSE, and MSE on the original Chl-a scale.

## Ablation

```bash
python train_chlorophyll.py --dataset bohai --no-moe
python train_chlorophyll.py --dataset bohai --no-graph-conv
python train_chlorophyll.py --dataset bohai --no-seasonal-encoding
python train_chlorophyll.py --dataset bohai --no-multiscale-tcn
```

Use `--dataset nanhai` for the South China Sea dataset.

## Outputs

By default, the script saves:

- checkpoints under `checkpoints/`
- original-scale prediction archives under `prediction_results/`

Use `--no-save` and `--no-save-predictions` to disable these outputs.

The training script reports aggregate validation/test losses and per-horizon MAE, RMSE, and MSE on the original Chl-a scale.

## Acknowledgement

The sparse MoE layer implementation is adapted from the open-source ST-MoE PyTorch implementation under the MIT license. The forecasting wrapper, Chl-a data pipeline, geographic graph construction, and MS-STMoE training workflow are adapted for marine water quality forecasting.
