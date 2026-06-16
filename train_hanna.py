"""HANNA training & evaluation — base, motif-ablation, and active-learning recovery.

Three regimes share one script:

  1. Base training         (no --split_name): train/val from --train_data, eval on --test_data.
────────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────────

  # Base training
  python train_hanna.py
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from utils.Evaluation import Evaluation
from utils.Own_Scaler import CustomScaler
from utils.Train import HANNA, Smooth_L1_Loss, train_model
from utils.preprocess import preprocess_input, prepare_gamma_data, split_and_reshape_input

ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="HANNA training with optional motif-ablation splits and "
                "active-learning anti-forgetting strategies (see module docstring).",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--train_data", default="data/data_train.pkl")
parser.add_argument("--test_data",  default="data/data_test.pkl")
parser.add_argument("--suffix",     default="HANNA")
parser.add_argument("--results_dir",default="Results")
parser.add_argument("--lr",           type=float, default=0.0005)
parser.add_argument("--batch_size",   type=int,   default=64)
parser.add_argument("--n_epochs",     type=int,   default=200)
parser.add_argument("--patience",     type=int,   default=10)
parser.add_argument("--patience_early", type=int, default=30)
parser.add_argument("--weight_decay", type=float, default=1e-6)
parser.add_argument("--delta",        type=float, default=0.25)
parser.add_argument("--device",       default="auto", choices=["auto", "cpu", "mps", "cuda"],
                    help="Device for training. 'auto' picks cuda > mps > cpu.")
parser.add_argument("--split_name",   default=None,
                    help="If set, load data/splits/<name>/{train,val,test_clean,test_motif}.pkl "
                         "(built by `scripts/pipeline.py build-splits`) instead of "
                         "--train_data/--test_data. Evaluates on both test_clean and "
                         "test_motif and reports the gap.")
parser.add_argument("--init_from",    default=None,
                    help="Path to an existing best_model_<X>.pth to warm-start from. Used by "
                         "the active-learning recovery sweep — fine-tunes the motif-ablated "
                         "checkpoint on a split that adds back some held-out motif data, "
                         "rather than training from scratch. Required for the resistance "
                         "techniques (they only make sense relative to a baseline).")
parser.add_argument("--hidden",       type=int, default=96,
                    help="Hidden width of HANNA's three MLPs (theta/alpha/phi). Default 96 "
                         "matches the paper. Param count scales as ~4*hidden^2 + 389*hidden.")

args = parser.parse_args()


def pick_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_sysid_tensor(arr, device):
    if isinstance(arr, torch.Tensor):
        return arr.to(device)
    return torch.tensor(np.asarray(arr), dtype=torch.int64, device=device)


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _rel(path) -> str:
    """Path relative to ROOT for logging, or absolute if it lives outside the repo."""
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


device = pick_device(args.device)
print("Device in use:", device)
device_str = str(device)

Embedding_BERT = 384

# ── Load data ─────────────────────────────────────────────────────────────────

if args.split_name:
    split_dir = os.path.join("data", "splits", args.split_name)
    print(f"Loading ablation split from {split_dir} ...")
    train_df      = _load_pkl(os.path.join(split_dir, "train.pkl"))
    val_df        = _load_pkl(os.path.join(split_dir, "val.pkl"))
    test_clean_df = _load_pkl(os.path.join(split_dir, "test_clean.pkl"))
    test_motif_df = _load_pkl(os.path.join(split_dir, "test_motif.pkl"))

    train      = prepare_gamma_data(train_df,      mode="full", device=device_str, verbose=True)
    val        = prepare_gamma_data(val_df,        mode="full", device=device_str, verbose=True)
    test_clean = prepare_gamma_data(test_clean_df, mode="full", device=device_str, verbose=True)
    test_motif = prepare_gamma_data(test_motif_df, mode="full", device=device_str, verbose=True)

    X_train, y_train = train["X"], train["y"]
    X_val,   y_val   = val["X"],   val["y"]
    train_systems_ID = _to_sysid_tensor(train["systems_ID"], device)
    val_systems_ID   = _to_sysid_tensor(val["systems_ID"], device)

    # Two test sets: "test_clean" (motif-free held-out) and "test_motif" (extrapolation set).
    test_sets = {"test_clean": test_clean, "test_motif": test_motif}

else:
    train_df = _load_pkl(args.train_data)
    test_df  = _load_pkl(args.test_data)

    train_val = prepare_gamma_data(train_df, mode="split", device=device_str, verbose=True)
    test      = prepare_gamma_data(test_df,  mode="full",  device=device_str, verbose=True)

    X_train = train_val["X_train"]
    y_train = train_val["y_train"]
    X_val   = train_val["X_val"]
    y_val   = train_val["y_val"]
    train_systems_ID = train_val["train_systems_ID"]
    val_systems_ID   = train_val["val_systems_ID"]

    test_sets = {"test": test}

# ── Reshape to component-wise format ─────────────────────────────────────────

X_train = preprocess_input(X_train, Embedding_BERT=Embedding_BERT, device=device_str)
X_val   = preprocess_input(X_val,   Embedding_BERT=Embedding_BERT, device=device_str)
for name, tdata in test_sets.items():
    tdata["X"] = preprocess_input(tdata["X"], Embedding_BERT=Embedding_BERT, device=device_str)

# ── Scale (fit on train, apply to val/test) ───────────────────────────────────
# By default the scaler is fit on this split's train. For no-replay recovery
# runs that train is motif-only, which shifts the normalisation away from the
# distribution the warm-started baseline checkpoint expects. --baseline-scaler-
# source overrides the fit to the baseline distribution so test_clean reflects
# real forgetting, not a scaler artefact.

scaler = CustomScaler(Embedding_BERT=Embedding_BERT)

X_train_scaled = scaler.fit_transform(X_train.cpu().numpy())
X_val_scaled   = scaler.transform(X_val.cpu().numpy())
X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
X_val_scaled   = torch.tensor(X_val_scaled,   dtype=torch.float32, device=device)

for name, tdata in test_sets.items():
    scaled = scaler.transform(tdata["X"].cpu().numpy())
    tdata["X_scaled"] = torch.tensor(scaled, dtype=torch.float32, device=device)

# ── Split into model inputs ───────────────────────────────────────────────────

T_train, x_train, FP_train = split_and_reshape_input(X_train_scaled)
T_val,   x_val,   FP_val   = split_and_reshape_input(X_val_scaled)
for name, tdata in test_sets.items():
    tdata["T"], tdata["x"], tdata["FP"] = split_and_reshape_input(tdata["X_scaled"])

# ── Model ─────────────────────────────────────────────────────────────────────
# Resistance is active if any anti-forgetting knob is set. When inactive the
# path below is byte-identical to the plain HANNA + train_model pipeline.

baseline_state = None
model = HANNA(Embedding_ChemBERT=Embedding_BERT, nodes=args.hidden).to(device)

if args.init_from:
    if not os.path.exists(args.init_from):
        raise FileNotFoundError(f"--init_from checkpoint not found: {args.init_from}")
    baseline_state = torch.load(args.init_from, map_location=device)
    model.load_state_dict(baseline_state)
    print(f"Warm-started model from {_rel(args.init_from)}", flush=True)

loss_fn   = Smooth_L1_Loss(delta=args.delta, use_simple_loss=True).to(device)

optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=args.lr, weight_decay=args.weight_decay,
)
scheduler = ReduceLROnPlateau(optimizer, "min", patience=args.patience, factor=0.1)

# ── Train ─────────────────────────────────────────────────────────────────────
train_model(
    model,
    T_train, x_train, FP_train, y_train,
    T_val,   x_val,   FP_val,   y_val,
    loss_fn, optimizer, scheduler, device,
    train_systems_ID, val_systems_ID,
    n_epochs=args.n_epochs,
    batch_size=args.batch_size,
    patience=args.patience_early,
    suffix=args.suffix,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
# train_model trains `model` in place and saves the best (lowest-val-loss) state
# to models/best_model_<suffix>.pth. Reload that checkpoint so evaluation uses
# the best model rather than the last-epoch weights.

trained_model = model
best_ckpt = os.path.join("models", f"best_model_{args.suffix}.pth")
if os.path.exists(best_ckpt):
    trained_model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"Loaded best checkpoint for evaluation: {best_ckpt}")

results_dir = os.path.join(args.results_dir, "evaluation")
os.makedirs(results_dir, exist_ok=True)

per_split_summary: dict[str, dict] = {}
for name, tdata in test_sets.items():
    eval_suffix = f"{args.suffix}_{name}" if args.split_name else args.suffix
    print(f"\n── Evaluating {name} ──")
    sysid = tdata["systems_ID"]
    mae, mse, systemwise_mae, systemwise_mse = Evaluation.evaluate(
        T_Test=tdata["T"],
        x_Test=tdata["x"],
        FP_Test=tdata["FP"],
        y_test=tdata["y"],
        test_systems_ID=sysid,
        trained_model=trained_model,
        results_dir=results_dir,
        suffix=eval_suffix,
    )
    median_mae = float(np.median(systemwise_mae))
    median_mse = float(np.median(systemwise_mse))
    per_split_summary[name] = {
        "MAE": float(mae), "MSE": float(mse),
        "Median_MAE": median_mae, "Median_MSE": median_mse,
        "n_systems": len(systemwise_mae),
    }
    print(
        f"Results [{eval_suffix}]: "
        f"MAE={mae:.6f}  MSE={mse:.6f}  "
        f"Median MAE={median_mae:.6f}  Median MSE={median_mse:.6f}  "
        f"({len(systemwise_mae):,} systems)"
    )