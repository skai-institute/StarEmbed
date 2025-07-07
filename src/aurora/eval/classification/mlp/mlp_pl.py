# lightning_sweep.py
# --------------------------------------------------
# Hyperparameter Sweep (Lightning version)
# --------------------------------------------------
import os, csv, random
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from sklearn.metrics import confusion_matrix, classification_report

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from tqdm.auto import tqdm
import argparse
import pathlib


# ---------- 0) helpers --------------------------------------------------------
def remove_outlier(dataset):
    bad_idx_trn, bad_idx_val, bad_idx_tst = 23082, 473, 7880
    dataset["train"]      = dataset["train"].select( list(range(bad_idx_trn)) +
                                                     list(range(bad_idx_trn+1,len(dataset["train"]))) )
    dataset["validation"] = dataset["validation"].select( list(range(bad_idx_val)) +
                                                         list(range(bad_idx_val+1,len(dataset["validation"]))) )
    dataset["test"]       = dataset["test"].select( list(range(bad_idx_tst)) +
                                                    list(range(bad_idx_tst+1,len(dataset["test"]))) )
    return dataset


class ScenarioDataset(Dataset):
    """Wrap HF dataset to compute per‑sample features for band scenarios."""
    def __init__(self, hf_ds, scenario="avg", label_key="label_idx"):
        assert scenario in ("concat", "avg", "g", "r")
        self.ds, self.scenario, self.lkey = hf_ds, scenario, label_key

    def __len__(self):  return len(self.ds)

    def __getitem__(self, idx):
        rec   = self.ds[idx]
        emb_g = np.squeeze(np.array(rec["embeddings_g"], dtype=np.float32))
        emb_r = np.squeeze(np.array(rec["embeddings_r"], dtype=np.float32))
        avg_g, avg_r = emb_g.mean(0), emb_r.mean(0)
        if   self.scenario == "concat": x_np = np.concatenate([avg_g, avg_r], 0)
        elif self.scenario == "avg":    x_np = 0.5 * (avg_g + avg_r)
        elif self.scenario == "g":      x_np = avg_g
        else:                           x_np = avg_r
        return torch.from_numpy(x_np), torch.tensor(rec[self.lkey], dtype=torch.long)


class MLP(nn.Module):
    """Simple feed‑forward classifier."""
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
    def __init__(self, in_dim, hidden_dims, num_classes, lr, dropout=0.0):
        super().__init__()
        self.save_hyperparameters()             # -> logs all hparams automatically
        self.model     = MLP(in_dim, hidden_dims, num_classes, dropout)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):                return self.model(x)

    # — training / validation / test steps —
    def _shared_step(self, batch):
        x, y   = batch
        logits = self(x)
        loss   = self.criterion(logits, y)
        acc    = (logits.argmax(1) == y).float().mean()
        return loss, acc

    def training_step(self, batch, _):
        loss, acc = self._shared_step(batch)
        # log *epoch*‑level metrics so the CSV has train_loss_epoch / train_acc_epoch
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
                 num_workers=8, val_batch=128):
        super().__init__()
        self.train_ds, self.val_ds, self.test_ds = train_ds, val_ds, test_ds
        self.batch_size, self.val_batch = batch_size, val_batch
        self.scenario, self.num_workers = scenario, num_workers

    # Lightning will call these automatically
    def train_dataloader(self):
        return DataLoader(ScenarioDataset(self.train_ds, self.scenario),
                          batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(ScenarioDataset(self.val_ds, self.scenario),
                          batch_size=self.val_batch, shuffle=False,
                          num_workers=self.num_workers//2 or 2, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(ScenarioDataset(self.test_ds, self.scenario),
                          batch_size=self.val_batch, shuffle=False,
                          num_workers=self.num_workers//2 or 2, pin_memory=True)


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
    parser.add_argument("--input_embs", type=str, default="/projects/p32795/weijian/embs/csdr1_raw4_catflags_filtered_embs_chronos_bolt_tiny_trn_val_tst_ctx200_bandgr")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---------- 2) Globals & reproducibility -------------------------------------
    SEED = args.seed
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

    DATASET_PATH   = args.input_embs
    ds             = remove_outlier(load_from_disk(DATASET_PATH))
    train_ds, val_ds, test_ds = ds["train"], ds["validation"], ds["test"]

    # label remap
    orig_labels   = sorted(set(train_ds["class_str"]))
    label2idx     = {lab: i for i, lab in enumerate(orig_labels)}
    train_ds      = train_ds.map(lambda e: {"label_idx": label2idx[e["class_str"]]}, num_proc=4)
    val_ds        = val_ds.map  (lambda e: {"label_idx": label2idx[e["class_str"]]}, num_proc=2)
    test_ds       = test_ds.map (lambda e: {"label_idx": label2idx[e["class_str"]]}, num_proc=2)
    num_classes   = len(orig_labels)

    print("Train class distribution →", Counter(train_ds["label_idx"]))


    # ---------- 3) Sweep definition ----------------------------------------------
    SCENARIO, PATIENCE, EPOCHS = args.scenario, args.patience, args.epochs
    if args.hidden_layers == 3:
        HIDDEN_DIMS                = [1024, 512, 256]
    elif args.hidden_layers == 2:
        HIDDEN_DIMS                = [512, 256]
    else:
        raise ValueError(f"Invalid number of hidden layers: {args.hidden_layers}")
    base_name = pathlib.Path(args.input_embs).stem
    experiments = [
        {"name": f"{base_name}_bs{args.batch_size}_lr{args.lr}_do{args.dropout}", "batch_size": args.batch_size, "lr": args.lr, "dropout": args.dropout},
    ]

    # Output dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)


    # ---------- 4) Run sweep ------------------------------------------------------
    for exp in experiments:
        name = exp["name"]
        print(f"\n=== Experiment: {name} ===")

        # DataModule (uses first sample to infer input‑dim)
        first_x, _  = ScenarioDataset(train_ds, SCENARIO)[0]
        dm          = ScenarioDataModule(train_ds, val_ds, test_ds,
                                        batch_size=exp["batch_size"], scenario=SCENARIO)

        # Lightning model
        lit_model   = LitMLP(in_dim=first_x.numel(),
                            hidden_dims=HIDDEN_DIMS,
                            num_classes=num_classes,
                            lr=exp["lr"],
                            dropout=exp["dropout"])

        # Callbacks / logger
        logger      = CSVLogger(out_dir, name=name, flush_logs_every_n_steps=50)
        ckpt_cb     = ModelCheckpoint(monitor="val_loss", mode="min",
                                    filename="{epoch}-{val_loss:.4f}", save_weights_only=True)
        early_cb    = EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)

        trainer     = pl.Trainer(
            max_epochs=EPOCHS,
            callbacks=[ckpt_cb, early_cb],
            logger=logger,
            deterministic=True,            # full reproducibility
            log_every_n_steps=50,
            accelerator="auto", devices="auto",
        )

        trainer.fit(lit_model, dm)

        best_ckpt = ckpt_cb.best_model_path           # path from rank‑0
        single_gpu_tester = pl.Trainer(
            accelerator="gpu", devices=1, logger=False
        )
        single_gpu_tester.test(lit_model, datamodule=dm, ckpt_path=best_ckpt)
        # trainer.test(lit_model, datamodule=dm)

        # ---------- 5) post‑processing metrics/plots -----------------------------
        # Lightning’s CSVLogger writes metrics into:
        
        log_csv = os.path.join(logger.log_dir, "metrics.csv")
        if trainer.is_global_zero:
            df      = pd.read_csv(log_csv)

            # Smooth per‑step logs to per‑epoch curves
            loss_curves = df.pivot_table(index="epoch", values=["train_loss_epoch","val_loss"])
            acc_curves  = df.pivot_table(index="epoch", values=["train_acc_epoch","val_acc"])

            plt.figure(figsize=(10,4)); plt.plot(loss_curves); plt.title(f"Loss ({name})")
            plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(loss_curves.columns); plt.tight_layout()
            plt.savefig(os.path.join(logger.log_dir, "loss.png")); plt.close()

            plt.figure(figsize=(10,4)); plt.plot(acc_curves); plt.title(f"Accuracy ({name})")
            plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(acc_curves.columns); plt.tight_layout()
            plt.savefig(os.path.join(logger.log_dir, "acc.png")); plt.close()

            # Confusion matrix on test set (best ckpt already loaded by trainer.test)
            lit_model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for x, y in dm.test_dataloader():
                    logits = lit_model(x.to(lit_model.device))
                    preds.extend(logits.argmax(1).cpu().numpy()); labels.extend(y.numpy())

            pd.DataFrame(classification_report(labels, preds, target_names=orig_labels,
                                            output_dict=True)).T.to_csv(
                os.path.join(logger.log_dir, "metrics_report.csv"))

            cm  = confusion_matrix(labels, preds, labels=list(range(num_classes)))
            cmn = cm.astype(float) / cm.sum(1, keepdims=True)
            plt.figure(figsize=(8,6))
            sns.heatmap(cmn, annot=True, fmt=".3f", cmap="viridis",
                        xticklabels=orig_labels, yticklabels=orig_labels)
            plt.title(f"Confusion Matrix ({name})")
            plt.xlabel("Pred"); plt.ylabel("True")
            plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(logger.log_dir, "confusion.png"))
            plt.close()

    print("\nSweep complete. See results in", out_dir)
