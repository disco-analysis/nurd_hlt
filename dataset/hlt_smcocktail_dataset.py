"""
HLT SM Cocktail dataset for NURD training.

The nuisance variable z is the **binned AE reconstruction loss**.
NURD exact weights w(y,z) = p(y)*p(z)/p(y,z) are pre-computed on load
so that train_exact.py can look them up with dataset.weights[(y,z)].

Dataset returns (pf_features, label, nuisance_bin) per event.
"""
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
from sklearn.model_selection import train_test_split


def _make_nurd_weights(labels, nuisances):
    """
    Exact NURD weights: w(y,z) = 1/p(y,z), normalized.
    Equivalent to p(y)*p(z)/p(y,z) up to a constant.
    """
    group_counts = Counter()
    for y, z in zip(labels.tolist(), nuisances.tolist()):
        group_counts[(int(y), int(z))] += 1
    total = len(labels)
    weights_unnorm = {k: total / v for k, v in group_counts.items()}
    total_w = sum(weights_unnorm.values())
    return {k: v / total_w for k, v in weights_unnorm.items()}


class HLTSmCocktailDataset(Dataset):
    """
    Args:
        pf_data:    [N, max_cands, n_feats]  PF candidate features
        obj_data:   [N, obj_feat_dim]        pre-normalised object-level AE inputs
        labels:     [N] long
        ae_model:   frozen pre-trained Autoencoder (eval mode)
        n_bins:     number of quantile bins for the AE reco nuisance
        split:      "train" | "val"
        val_split:  fraction held out for validation
        seed:       random seed for the train/val split
    """
    def __init__(self, pf_data, obj_data, labels, ae_model, n_bins=10,
                 split="train", val_split=0.1, seed=42, bin_edges=None):
        super().__init__()

        # ── compute AE reco loss per event ────────────────────────────────────
        ae_model.eval()
        device = next(ae_model.parameters()).device
        mse = torch.nn.MSELoss(reduction='none')
        ae_reco_all = []
        bs = 4096
        with torch.no_grad():
            for i in range(0, obj_data.shape[0], bs):
                batch = obj_data[i:i+bs].to(device)
                recon, _ = ae_model(batch)
                ae_reco_all.append(mse(recon, batch).mean(dim=1).cpu())
        ae_reco_all = torch.cat(ae_reco_all)

        # ── quantile binning of the nuisance ─────────────────────────────────
        if bin_edges is None:
            quantiles = torch.linspace(0, 1, n_bins + 1)
            bin_edges = torch.quantile(ae_reco_all, quantiles)
        self.bin_edges = bin_edges
        nuisances_all = torch.bucketize(ae_reco_all, bin_edges[1:-1]).long()

        # ── train / val split ─────────────────────────────────────────────────
        idx_all = np.arange(len(labels))
        idx_tr, idx_val = train_test_split(
            idx_all, test_size=val_split, random_state=seed,
            stratify=labels.cpu().numpy()
        )
        idx = idx_tr if split == "train" else idx_val
        idx = torch.tensor(idx, dtype=torch.long)

        self.features  = pf_data[idx]
        self.obj       = obj_data[idx]
        self.labels    = labels[idx].float()
        self.nuisances = nuisances_all[idx].float()
        self.split     = split

        # ── NURD exact weights ────────────────────────────────────────────────
        self.weights = _make_nurd_weights(labels[idx], nuisances_all[idx])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.nuisances[idx]

    def get_label_prior(self):
        total = len(self.labels)
        counts = Counter(int(y) for y in self.labels.tolist())
        return {k: v / total for k, v in counts.items()}

    def get_nuisance_prior(self):
        total = len(self.nuisances)
        counts = Counter(int(z) for z in self.nuisances.tolist())
        return {k: v / total for k, v in counts.items()}


def build_hlt_datasets(pt_path, ae_model, n_bins=10, val_split=0.1, seed=42, max_events=-1):
    """
    Load the HLT .pt file, pre-normalise obj features, and return
    (train_dataset, val_dataset).  Call once; pass the same bin_edges
    to both splits so nuisance definitions are consistent.
    """
    raw = torch.load(pt_path, map_location="cpu")
    pf     = raw["pf"]
    labels = raw["label"].long()
    obj    = raw["obj"]
    if max_events > 0:
        pf, labels, obj = pf[:max_events], labels[:max_events], obj[:max_events]
    pf = torch.nan_to_num(pf, nan=0.0, posinf=0.0, neginf=0.0)

    # flatten + z-score normalise obj features (first 4 features per cand)
    obj_flat = obj[:, :, :4].reshape(obj.shape[0], -1).float().numpy()
    mu  = obj_flat.mean(axis=0).astype(np.float32)
    std = obj_flat.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std)
    obj_norm = torch.from_numpy((obj_flat - mu) / std)
    obj_scaler = {"mu": torch.from_numpy(mu), "std": torch.from_numpy(std)}

    # build train split first to get bin_edges from training data
    ds_train = HLTSmCocktailDataset(pf, obj_norm, labels, ae_model,
                                    n_bins=n_bins, split="train",
                                    val_split=val_split, seed=seed)
    ds_val   = HLTSmCocktailDataset(pf, obj_norm, labels, ae_model,
                                    n_bins=n_bins, split="val",
                                    val_split=val_split, seed=seed,
                                    bin_edges=ds_train.bin_edges)
    return ds_train, ds_val, obj_scaler
