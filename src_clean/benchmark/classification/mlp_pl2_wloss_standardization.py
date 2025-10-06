# lightning_sweep.py
# --------------------------------------------------
# Hyperparameter Sweep (Lightning version) + StandardScaler
# --------------------------------------------------
import os, csv, random, argparse, pathlib, json, time, pickle, sys
from functools import partial
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from tqdm.auto import tqdm

# Add src_clean to path for importing benchmark.utils
script_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.dirname(script_dir)
src_clean_dir = os.path.dirname(benchmark_dir)
sys.path.append(src_clean_dir)

from benchmark.utils import remove_outliers, add_label_indices, compute_embedding


# ---------- 0) helpers --------------------------------------------------------
class ScenarioDataset(Dataset):
    """Wrap HF dataset to use pre-computed combined_embedding for ultra-fast training."""
    def __init__(self, hf_ds, scenario="avg", label_key="label_idx", scaler: Optional[StandardScaler] = None):
        assert scenario in ("concat", "avg", "g", "r")
        self.ds, self.scenario, self.lkey = hf_ds, scenario, label_key
        self.scaler = scaler

    def __len__(self):  return len(self.ds)

    def __getitem__(self, idx):
        rec = self.ds[idx]
        try:
            # Use pre-computed combined_embedding for maximum speed
            if "combined_embedding" in rec:
                x_np = np.array(rec["combined_embedding"], dtype=np.float32)
                if self.scaler is not None:
                    x_np = self.scaler.transform(x_np.reshape(1, -1)).flatten()
            else:
                # Fallback to compute_embedding if combined_embedding not available
                x_np = compute_embedding(rec, band_combination=self.scenario, hand_crafted=False, 
                                       scaler=self.scaler, return_format="combined")
        except:
            print(f"Error in __getitem__: {idx}")
            print(f"rec: {rec}")
            raise

        return torch.from_numpy(x_np.astype(np.float32)), torch.tensor(rec[self.lkey], dtype=torch.long)


class MLP(nn.Module):
    """Simple feed-forward classifier."""
    def __init__(self, in_dim, hidden_dims, num_classes, dropout=0.0):
        super().__init__()
        layers, last = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            if dropout: layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  return self.net(x)


# ---------- 1) Lightning modules ---------------------------------------------
class LitMLP(pl.LightningModule):
    def __init__(self, in_dim, hidden_dims, num_classes, lr, dropout=0.0, class_weights=None):
        super().__init__()
        self.save_hyperparameters()             # -> logs all hparams automatically
        self.model     = MLP(in_dim, hidden_dims, num_classes, dropout)
        if class_weights is not None:
            # keep on model so it moves with .to(device)
            self.register_buffer("class_weights", class_weights)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    # — training / validation / test steps —
    def _shared_step(self, batch):
        x, y   = batch
        logits = self(x)
        loss   = self.criterion(logits, y)
        acc    = (logits.argmax(1) == y).float().mean()
        return loss, acc

    def training_step(self, batch, _):
        loss, acc = self._shared_step(batch)
        # log *epoch*-level metrics so the CSV has train_loss_epoch / train_acc_epoch
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_acc_epoch",  acc,  on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        loss, acc = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc",  acc,  prog_bar=True, sync_dist=True)

    def test_step(self, batch, _):
        loss, acc = self._shared_step(batch)
        self.log_dict({"test_loss": loss, "test_acc": acc}, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class ScenarioDataModule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, test_ds, batch_size, scenario,
                 num_workers=8, val_batch=128, scaler: Optional[StandardScaler] = None):
        super().__init__()
        self.train_ds, self.val_ds, self.test_ds = train_ds, val_ds, test_ds
        self.batch_size, self.val_batch = batch_size, val_batch
        self.scenario, self.num_workers = scenario, num_workers
        self.scaler = scaler

    def train_dataloader(self):
        return DataLoader(
            ScenarioDataset(self.train_ds, self.scenario, scaler=self.scaler),
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            ScenarioDataset(self.val_ds, self.scenario, scaler=self.scaler),
            batch_size=self.val_batch, shuffle=False,
            num_workers=self.num_workers//2 or 2, pin_memory=True, persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            ScenarioDataset(self.test_ds, self.scenario, scaler=self.scaler),
            batch_size=self.val_batch, shuffle=False,
            num_workers=self.num_workers//2 or 2, pin_memory=True, persistent_workers=True
        )


# ---------- 2) Standardization util ------------------------------------------
def fit_standardizer(train_hf_ds, scenario: str, batch_size: int = 8192, max_samples: Optional[int] = None) -> StandardScaler:
    """
    Fit a StandardScaler on training features only (for the chosen scenario).
    Iterates in batches to avoid loading everything into memory.
    """
    scaler = StandardScaler()
    loader = DataLoader(
        ScenarioDataset(train_hf_ds, scenario),  # no scaler during fitting
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    n_seen = 0
    for xb, _ in tqdm(loader, desc="Fitting StandardScaler (partial_fit)"):
        x = xb.numpy()
        if max_samples is not None:
            remain = max_samples - n_seen
            if remain <= 0:
                break
            x = x[:remain]
        scaler.partial_fit(x)
        n_seen += len(x)

    return scaler


# ---------- 3) Main -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="avg")
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="sweep_results")
    parser.add_argument("--input_embs", type=str, default="/projects/b1094/StarEmbed/embeddings/csdr1_raw4_catflags_filtered_embs_hand_crafted_trn_val_tst_bandgr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hand_crafted", type=int, default=0)
    parser.add_argument("--standardize", type=int, default=1, help="Use StandardScaler (fit on train only)")
    parser.add_argument("--std_max_samples", type=int, default=0, help="Cap samples when fitting scaler (0=all)")
    args = parser.parse_args()

    # print all args' values
    print(f"scenario: {args.scenario}")
    print(f"hidden_layers: {args.hidden_layers}")
    print(f"batch_size: {args.batch_size}")
    print(f"lr: {args.lr}")
    print(f"dropout: {args.dropout}")
    print(f"patience: {args.patience}")
    print(f"epochs: {args.epochs}")
    print(f"seed: {args.seed}")
    print(f"input_embs: {args.input_embs}")
    print(f"out_dir: {args.out_dir}")
    print(f"standardize: {args.standardize}")

    # ---------- Globals & reproducibility ----------
    SEED = args.seed
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

    DATASET_PATH = args.input_embs
    ds           = remove_outliers(load_from_disk(DATASET_PATH), hand_crafted=bool(args.hand_crafted))
    train_ds, val_ds, test_ds = ds["train"], ds["validation"], ds["test"]

    # Check for pre-computed combined_embedding
    if len(train_ds) > 0 and "combined_embedding" in train_ds[0]:
        print("✓ Found pre-computed combined_embedding - using ULTRA FAST mode!")
    else:
        print(f"⚠ No combined_embedding found - using slower compute_embedding with scenario '{args.scenario}'")
        print("  → Consider running compute_avg_embeddings.py first for much better performance")

    # Add label indices using unified utility
    ds, label2idx, text_labels = add_label_indices(ds, num_proc=4, sort_labels=True)
    train_ds, val_ds, test_ds = ds["train"], ds["validation"], ds["test"]
    num_classes = len(text_labels)
    print(f"label2idx: {label2idx}")
    print(f"text_labels: {text_labels}")

    print("Train class distribution →", Counter(train_ds["label_idx"]))

    # Inverse-frequency class weights (normalized to mean=1)
    cls_counts  = np.bincount(train_ds["label_idx"], minlength=num_classes)
    inv_freq    = 1.0 / cls_counts
    weights     = torch.tensor(inv_freq / inv_freq.mean(), dtype=torch.float32)

    # ---------- Sweep definition ----------
    SCENARIO, PATIENCE, EPOCHS = args.scenario, args.patience, args.epochs
    if args.hidden_layers == 3:
        HIDDEN_DIMS = [1024, 512, 256]
    elif args.hidden_layers == 2:
        HIDDEN_DIMS = [512, 256]
    else:
        raise ValueError(f"Invalid number of hidden layers: {args.hidden_layers}")

    base_name = pathlib.Path(args.input_embs).stem
    experiments = [
        {"name": f"{base_name}_bs{args.batch_size}_lr{args.lr}_do{args.dropout}_{args.scenario}_s{args.seed}",
         "batch_size": args.batch_size, "lr": args.lr, "dropout": args.dropout},
    ]

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Run sweep ----------
    for exp in experiments:
        name = exp["name"]
        print(f"\n=== Experiment: {name} ===")

        # Infer input-dim from first sample
        first_x, _ = ScenarioDataset(train_ds, SCENARIO)[0]

        # Fit StandardScaler on train (scenario-specific)
        scaler = None
        if args.standardize:
            max_samples = args.std_max_samples if args.std_max_samples > 0 else None
            scaler = fit_standardizer(train_ds, SCENARIO, batch_size=8192, max_samples=max_samples)

        # DataModule with shared scaler
        dm = ScenarioDataModule(
            train_ds, val_ds, test_ds,
            batch_size=exp["batch_size"], scenario=SCENARIO,
            num_workers=8, val_batch=128, scaler=scaler
        )

        # Lightning model
        lit_model = LitMLP(
            in_dim       = first_x.numel(),
            hidden_dims  = HIDDEN_DIMS,
            num_classes  = num_classes,
            lr           = exp["lr"],
            dropout      = exp["dropout"],
            class_weights= weights
        )

        # Callbacks / logger
        logger   = CSVLogger(out_dir, name=name, flush_logs_every_n_steps=50)
        ckpt_cb  = ModelCheckpoint(monitor="val_loss", mode="min",
                                   filename="{epoch}-{val_loss:.4f}", save_weights_only=True)
        early_cb = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)

        trainer  = pl.Trainer(
            max_epochs=EPOCHS,
            callbacks=[ckpt_cb, early_cb],
            logger=logger,
            deterministic=True,            # full reproducibility
            log_every_n_steps=50,
            accelerator="auto", devices="auto",
        )

        # Save scaler for reproducibility (rank-0 only)
        if args.standardize and trainer.is_global_zero:
            os.makedirs(logger.log_dir, exist_ok=True)
            with open(os.path.join(logger.log_dir, "standard_scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)

        trainer.fit(lit_model, dm)

        best_ckpt = ckpt_cb.best_model_path
        single_gpu_tester = pl.Trainer(accelerator="gpu", devices=1, logger=False)
        single_gpu_tester.test(lit_model, datamodule=dm, ckpt_path=best_ckpt)

        # ---------- post-processing metrics/plots ----------
        log_csv = os.path.join(logger.log_dir, "metrics.csv")
        if trainer.is_global_zero:
            df = pd.read_csv(log_csv)

            # Smooth per-step logs to per-epoch curves
            loss_curves = df.pivot_table(index="epoch", values=["train_loss_epoch","val_loss"])
            acc_curves  = df.pivot_table(index="epoch", values=["train_acc_epoch","val_acc"])

            plt.figure(figsize=(10,4)); plt.plot(loss_curves); plt.title(f"Loss ({name})")
            plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(loss_curves.columns); plt.tight_layout()
            plt.savefig(os.path.join(logger.log_dir, "loss.png")); plt.close()

            plt.figure(figsize=(10,4)); plt.plot(acc_curves); plt.title(f"Accuracy ({name})")
            plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(acc_curves.columns); plt.tight_layout()
            plt.savefig(os.path.join(logger.log_dir, "acc.png")); plt.close()

            # Confusion matrix on test set
            lit_model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for x, y in dm.test_dataloader():
                    logits = lit_model(x.to(lit_model.device))
                    preds.extend(logits.argmax(1).cpu().numpy()); labels.extend(y.numpy())

            pd.DataFrame(classification_report(labels, preds, target_names=text_labels,
                                               output_dict=True)).T.to_csv(
                os.path.join(logger.log_dir, "metrics_report.csv"))

            cm  = confusion_matrix(labels, preds, labels=list(range(num_classes)))
            cmn = cm.astype(float) / cm.sum(1, keepdims=True)
            with open(os.path.join(logger.log_dir, "confusion_data.pkl"), "wb") as f:
                pickle.dump(cmn, f)
            plt.figure(figsize=(8,6))
            sns.heatmap(cmn, annot=True, fmt=".3f", cmap="viridis",
                        xticklabels=text_labels, yticklabels=text_labels)
            plt.xlabel("Pred"); plt.ylabel("True")
            plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(logger.log_dir, "confusion.pdf"))
            plt.close()

    print("\nSweep complete. See results in", out_dir)
