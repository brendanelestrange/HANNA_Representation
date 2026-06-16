"""HANNA representation experiment: ChemBERTa +/- RDKit descriptors.

This trains the *unmodified* HANNA architecture, but changes the per-component
feature vector that feeds it. HANNA normally uses a 384-dim ChemBERTa-2 embedding
per component. Here we optionally concatenate RDKit physico-chemical descriptors
(see experiments/descriptors.py), growing the per-component vector to 384+D, or
replace the embedding with descriptors entirely (a control).

Configs (--descriptor-set):
  none          : 384-dim ChemBERTa only                    (baseline)
  curated       : ChemBERTa(384) + curated descriptors(18)   -> 402
  full          : ChemBERTa(384) + full descriptors(~217)    -> ~601
  curated_only  : curated descriptors(18) only               (no ChemBERTa, control)
  full_only     : full descriptors(~217) only                (no ChemBERTa, control)

Everything downstream of the feature vector is identical to train_hanna.py's base
path (same model, same CustomScaler that standardises every non-mole-fraction
column, same Smooth-L1 loss, same train/val split via random_state=10). The val
split is fixed across configs so comparisons are apples-to-apples; --seed only
controls weight init / batch shuffling.

Outputs:
  models/best_model_<suffix>.pth          (via train_model)
  Results/losses/losses_<suffix>.{png,csv}(via train_model)
  reports/figures/<suffix>/...            (parity / residual / MAE-hist plots)
  reports/metrics/<suffix>.json           (consolidated metrics for the report)
"""

import argparse
import csv
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.descriptors import (  # noqa: E402
    DescriptorSanitizer,
    compute_raw_descriptors,
    get_descriptor_names,
)
from utils.Evaluation import Evaluation  # noqa: E402
from utils.Own_Scaler import CustomScaler  # noqa: E402
from utils.preprocess import preprocess_input, split_and_reshape_input  # noqa: E402
# Architecture is imported from a FROZEN snapshot (experiments/train_frozen.py), not the
# live utils/Train.py, so concurrent edits to utils/Train.py cannot change the model
# mid-sweep (this exact bug confounded the first run). The snapshot is the "with-theta"
# HANNA: a learned per-component projection theta(E_i) -> R^nodes whose output also feeds
# the cosine-distance gate, so the similarity gate lives in a fixed 96-dim space for every
# representation (making configs of different input dim directly comparable).
from experiments.train_frozen import HANNA, Smooth_L1_Loss, train_model  # noqa: E402

BERT_DIM = 384
ARCH_TAG = "hanna_theta_frozen"  # see import note above


def pick_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_df(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def component_id_to_smiles(*dfs):
    cid2smi = {}
    for d in dfs:
        for cid, smi in zip(d["component_1_ID"], d["SMILES1"]):
            cid2smi.setdefault(int(cid), smi)
        for cid, smi in zip(d["component_2_ID"], d["SMILES2"]):
            cid2smi.setdefault(int(cid), smi)
    return cid2smi


def build_component_descriptors(train_df, test_df, set_name):
    """Return (comp_id -> sanitised descriptor vector, descriptor_names)."""
    cid2smi = component_id_to_smiles(train_df, test_df)
    all_cids = sorted(cid2smi)
    train_cids = set(map(int, train_df["component_1_ID"])) | set(
        map(int, train_df["component_2_ID"])
    )
    names = get_descriptor_names(set_name)
    raw = compute_raw_descriptors([cid2smi[c] for c in all_cids], names)
    train_rows = [i for i, c in enumerate(all_cids) if c in train_cids]
    san = DescriptorSanitizer().fit(raw[train_rows])
    clean = san.transform(raw)
    comp_desc = {c: clean[i] for i, c in enumerate(all_cids)}
    return comp_desc, names


def stack_bert(df, col):
    return torch.stack(list(df[col]), dim=0).to(torch.float32).cpu().numpy()


def stack_desc(df, col, comp_desc):
    return np.stack([comp_desc[int(c)] for c in df[col]], axis=0).astype(np.float32)


def build_feature_matrix(df, comp_desc, use_bert, use_desc):
    """Build X = [T, x1, feat_1, feat_2] where feat_i is the per-component vector."""
    base = df[["T", "x1"]].values.astype(np.float32)
    feats = []
    for ci, di in ((1, 1), (2, 2)):
        parts = []
        if use_bert:
            parts.append(stack_bert(df, f"BERT_component_{ci}_ID"))
        if use_desc:
            parts.append(stack_desc(df, f"component_{di}_ID", comp_desc))
        feats.append(np.concatenate(parts, axis=1))
    X = np.concatenate([base, feats[0], feats[1]], axis=1)
    return X


def targets_and_ids(df):
    y = df[["ln_gamma_1", "ln_gamma_2"]].values.astype(np.float32)
    sysid = df["system_ID"].values.astype(np.int64)
    return y, sysid


def read_overall_metrics(csv_path):
    out = {}
    if not os.path.exists(csv_path):
        return out
    with open(csv_path) as f:
        for row in csv.reader(f):
            if len(row) == 2 and row[0] != "Metric":
                try:
                    out[row[0]] = float(row[1])
                except ValueError:
                    pass
    return out


def read_loss_history(csv_path):
    epochs, val = [], []
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["Epoch"]))
            val.append(float(row["Validation Total Loss"]))
    if not epochs:
        return None
    best_idx = int(np.argmin(val))
    return {
        "epochs_trained": epochs[-1],
        "best_epoch": epochs[best_idx],
        "best_val_loss": val[best_idx],
        "final_val_loss": val[-1],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--descriptor-set", required=True,
                   choices=["none", "curated", "full", "curated_only", "full_only"])
    p.add_argument("--suffix", required=True)
    p.add_argument("--train_data", default="data/data_train.pkl")
    p.add_argument("--test_data", default="data/data_test.pkl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)          # scheduler patience
    p.add_argument("--patience_early", type=int, default=30)    # early-stop patience
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--delta", type=float, default=0.25)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--subsample_systems", type=int, default=0,
                   help="If >0, keep only this many train+val systems (smoke test).")
    p.add_argument("--report_dir", default="reports")
    args = p.parse_args()

    t_start = time.time()
    set_seed(args.seed)
    device = pick_device(args.device)
    device_str = str(device)
    print(f"Device: {device}  |  config: {args.descriptor_set}  |  suffix: {args.suffix}  |  seed: {args.seed}",
          flush=True)

    use_bert = args.descriptor_set in ("none", "curated", "full")
    use_desc = args.descriptor_set in ("curated", "full", "curated_only", "full_only")
    desc_set_for_names = {"curated": "curated", "full": "full",
                          "curated_only": "curated", "full_only": "full"}.get(args.descriptor_set)

    train_df = load_df(args.train_data)
    test_df = load_df(args.test_data)

    if args.subsample_systems > 0:
        keep = train_df["system_ID"].drop_duplicates().iloc[: args.subsample_systems]
        train_df = train_df[train_df["system_ID"].isin(keep)].reset_index(drop=True)
        print(f"[smoke] subsampled to {train_df['system_ID'].nunique()} systems / {len(train_df)} rows",
              flush=True)

    comp_desc, desc_names = (None, [])
    if use_desc:
        print(f"Computing '{desc_set_for_names}' descriptors ...", flush=True)
        comp_desc, desc_names = build_component_descriptors(train_df, test_df, desc_set_for_names)
        print(f"  D = {len(desc_names)} descriptors per component", flush=True)

    emb_dim = (BERT_DIM if use_bert else 0) + (len(desc_names) if use_desc else 0)
    print(f"Per-component feature dim (Embedding_BERT) = {emb_dim}", flush=True)

    # ── Build feature matrices ───────────────────────────────────────────────
    X_all = build_feature_matrix(train_df, comp_desc, use_bert, use_desc)
    y_all, sysid_all = targets_and_ids(train_df)
    X_test = build_feature_matrix(test_df, comp_desc, use_bert, use_desc)
    y_test, sysid_test = targets_and_ids(test_df)

    # ── Train/val split by system (identical to train_hanna.py: random_state=10)
    train_ids, val_ids = train_test_split(
        train_df["system_ID"].unique(), test_size=0.2, random_state=10
    )
    tr_mask = np.isin(sysid_all, train_ids)
    va_mask = np.isin(sysid_all, val_ids)
    X_train, y_train, sys_train = X_all[tr_mask], y_all[tr_mask], sysid_all[tr_mask]
    X_val, y_val, sys_val = X_all[va_mask], y_all[va_mask], sysid_all[va_mask]
    print(f"train rows={len(X_train)}  val rows={len(X_val)}  test rows={len(X_test)}", flush=True)

    # ── Reshape to [N, 2, emb_dim+2] ─────────────────────────────────────────
    Xtr = preprocess_input(torch.tensor(X_train, device=device_str), Embedding_BERT=emb_dim, device=device_str)
    Xva = preprocess_input(torch.tensor(X_val, device=device_str), Embedding_BERT=emb_dim, device=device_str)
    Xte = preprocess_input(torch.tensor(X_test, device=device_str), Embedding_BERT=emb_dim, device=device_str)

    # ── Scale (fit on train, apply to val/test) ──────────────────────────────
    scaler = CustomScaler(Embedding_BERT=emb_dim)
    Xtr_s = torch.tensor(scaler.fit_transform(Xtr.cpu().numpy()), dtype=torch.float32, device=device)
    Xva_s = torch.tensor(scaler.transform(Xva.cpu().numpy()), dtype=torch.float32, device=device)
    Xte_s = torch.tensor(scaler.transform(Xte.cpu().numpy()), dtype=torch.float32, device=device)

    T_tr, x_tr, FP_tr = split_and_reshape_input(Xtr_s)
    T_va, x_va, FP_va = split_and_reshape_input(Xva_s)
    T_te, x_te, FP_te = split_and_reshape_input(Xte_s)

    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)
    sys_train_t = torch.tensor(sys_train, dtype=torch.int64, device=device)
    sys_val_t = torch.tensor(sys_val, dtype=torch.int64, device=device)

    # ── Model / loss / optim ─────────────────────────────────────────────────
    model = HANNA(Embedding_ChemBERT=emb_dim, nodes=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss_fn = Smooth_L1_Loss(delta=args.delta, use_simple_loss=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=args.patience, factor=0.1)

    train_model(
        model,
        T_tr, x_tr, FP_tr, y_train_t,
        T_va, x_va, FP_va, y_val_t,
        loss_fn, optimizer, scheduler, device,
        sys_train_t, sys_val_t,
        n_epochs=args.n_epochs, batch_size=args.batch_size,
        patience=args.patience_early, suffix=args.suffix,
    )

    # ── Reload best checkpoint and evaluate ──────────────────────────────────
    best_ckpt = os.path.join("models", f"best_model_{args.suffix}.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print(f"Loaded best checkpoint: {best_ckpt}", flush=True)

    fig_dir = os.path.join(args.report_dir, "figures", args.suffix)
    os.makedirs(fig_dir, exist_ok=True)
    mae, mse, sysmae, sysmse = Evaluation.evaluate(
        T_Test=T_te, x_Test=x_te, FP_Test=FP_te, y_test=y_test_t,
        test_systems_ID=torch.tensor(sysid_test, dtype=torch.int64, device=device),
        trained_model=model, results_dir=fig_dir, suffix=args.suffix,
    )
    median_mae = float(np.median(sysmae))
    median_mse = float(np.median(sysmse))

    overall = read_overall_metrics(os.path.join(fig_dir, f"avg_metrics_{args.suffix}.csv"))
    hist = read_loss_history(os.path.join("Results", "losses", f"losses_{args.suffix}.csv"))

    metrics = {
        "suffix": args.suffix,
        "architecture": ARCH_TAG,
        "descriptor_set": args.descriptor_set,
        "use_bert": use_bert,
        "use_desc": use_desc,
        "embedding_dim": emb_dim,
        "n_descriptors": len(desc_names),
        "descriptor_names": desc_names,
        "n_params": n_params,
        "seed": args.seed,
        "hidden": args.hidden,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "n_train_rows": int(len(X_train)),
        "n_val_rows": int(len(X_val)),
        "n_test_rows": int(len(X_test)),
        "n_test_systems": int(len(sysmae)),
        # headline = system-wise (macro) metrics, matching the repo convention
        "test_mean_system_MAE": float(mae),
        "test_mean_system_MSE": float(mse),
        "test_median_system_MAE": median_mae,
        "test_median_system_MSE": median_mse,
        # micro (per-point) metrics from evaluate's avg_metrics csv
        "test_overall_MAE": overall.get("Overall MAE - Trained Model"),
        "test_overall_MSE": overall.get("Overall MSE - Trained Model"),
        "loss_history": hist,
        "wall_time_sec": round(time.time() - t_start, 1),
        "device": device_str,
        "subsample_systems": args.subsample_systems,
    }
    os.makedirs(os.path.join(args.report_dir, "metrics"), exist_ok=True)
    out_json = os.path.join(args.report_dir, "metrics", f"{args.suffix}.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print(f"DONE [{args.suffix}]  config={args.descriptor_set}  emb_dim={emb_dim}  params={n_params:,}")
    print(f"  test mean-system MAE = {mae:.5f}   median-system MAE = {median_mae:.5f}")
    print(f"  test mean-system MSE = {mse:.5f}   median-system MSE = {median_mse:.5f}")
    print(f"  test overall (micro) MAE = {metrics['test_overall_MAE']}")
    if hist:
        print(f"  best epoch {hist['best_epoch']} (val {hist['best_val_loss']:.5f}), "
              f"trained {hist['epochs_trained']} epochs")
    print(f"  wall time {metrics['wall_time_sec']}s  ->  {out_json}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
