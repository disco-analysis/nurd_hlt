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


def _make_nurd_weights(labels, nuisances, max_weight_ratio=10.0):
    """
    Exact NURD weights: w(y,z) = p(y)*p(z)/p(y,z) = n_y*n_z / (N*n_yz).
    Under this weighting, y and z are marginally independent.
    Normalized so that the per-sample mean weight equals 1, then clipped at
    max_weight_ratio × mean to prevent extreme weights from destabilising training.
    """
    N = len(labels)
    labels_list    = [int(y) for y in labels.tolist()]
    nuisances_list = [int(z) for z in nuisances.tolist()]

    group_counts    = Counter(zip(labels_list, nuisances_list))
    label_counts    = Counter(labels_list)
    nuisance_counts = Counter(nuisances_list)

    weights_raw = {
        (y, z): (label_counts[y] * nuisance_counts[z]) / (N * n_yz)
        for (y, z), n_yz in group_counts.items()
    }
    # normalize so E[w] = 1 over all training samples
    mean_w = sum(weights_raw[k] * v for k, v in group_counts.items()) / N
    weights_norm = {k: v / mean_w for k, v in weights_raw.items()}
    # clip to max_weight_ratio × 1.0 (since mean is now 1) to reduce variance
    cap = max_weight_ratio
    return {k: min(v, cap) for k, v in weights_norm.items()}


class HLTSmCocktailDataset(Dataset):
    """
    Args:
        pf_data:      [N, max_cands, n_feats]  PF candidate features
        obj_data:     [N, obj_feat_dim]        pre-normalised object-level AE inputs
        labels:       [N] long
        ae_model:     frozen pre-trained Autoencoder (eval mode)
        n_bins:       number of quantile bins for the AE reco nuisance
        split:        "train" | "val"
        val_split:    fraction held out for validation
        seed:         random seed for the train/val split
        gen_weights:  [N] float per-event physics weights (genWeight × scale for QCD, 1.0 otherwise)
    """
    def __init__(self, pf_data, obj_data, labels, ae_model, n_bins=10,
                 split="train", val_split=0.1, seed=42, bin_edges=None, gen_weights=None):
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

        self.features     = pf_data[idx]
        self.obj          = obj_data[idx]
        self.labels       = labels[idx].float()
        self.nuisances    = nuisances_all[idx].float()
        self.ae_reco      = ae_reco_all[idx].float()
        self.gen_weights  = gen_weights[idx].float() if gen_weights is not None else None
        self.split        = split

        # ── NURD exact weights ────────────────────────────────────────────────
        self.weights = _make_nurd_weights(labels[idx], nuisances_all[idx])
        _w = list(self.weights.values())
        import statistics
        _w_mean = sum(_w) / len(_w)
        _w_std  = statistics.stdev(_w)
        print(f"[{split}] NURD weight groups={len(_w)}  mean={_w_mean:.3f}  "
              f"std={_w_std:.3f}  min={min(_w):.3f}  max={max(_w):.3f}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        out = (self.features[idx], self.labels[idx], self.nuisances[idx], self.ae_reco[idx])
        if self.gen_weights is not None:
            out += (self.gen_weights[idx],)
        return out

    def get_label_prior(self):
        total = len(self.labels)
        counts = Counter(int(y) for y in self.labels.tolist())
        return {k: v / total for k, v in counts.items()}

    def get_nuisance_prior(self):
        total = len(self.nuisances)
        counts = Counter(int(z) for z in self.nuisances.tolist())
        return {k: v / total for k, v in counts.items()}


def build_hlt_datasets(pt_path, ae_model, n_bins=10, val_split=0.1, seed=42, max_events=-1, exclude_labels=None, gen_weight_path=None, gen_weight_clip=None, qcd_label=1):
    """
    Load the HLT .pt file, pre-normalise obj features, and return
    (train_dataset, val_dataset).  Call once; pass the same bin_edges
    to both splits so nuisance definitions are consistent.

    exclude_labels: list of integer labels to drop before training (e.g. [2] to drop TTBar).
    Remaining labels are remapped to be contiguous starting from 0.
    """
    raw = torch.load(pt_path, map_location="cpu")
    pf     = raw["pf"]
    labels = raw["label"].long()
    obj    = raw["obj"]
    if max_events > 0:
        pf, labels, obj = pf[:max_events], labels[:max_events], obj[:max_events]
    pf = torch.nan_to_num(pf, nan=0.0, posinf=0.0, neginf=0.0)

    gen_weights = None
    if gen_weight_path is not None:
        gen_weights = torch.load(gen_weight_path, map_location="cpu").float()
        if max_events > 0:
            gen_weights = gen_weights[:max_events]
        if gen_weight_clip is not None:
            qcd_mask = (labels == qcd_label)
            qcd_mean = gen_weights[qcd_mask].mean()
            gen_weights = gen_weights / qcd_mean
            gen_weights = torch.clamp(gen_weights, max=gen_weight_clip)
            print(f"[gen_weight_clip={gen_weight_clip}] QCD mean before clip: {qcd_mean:.3e}  "
                  f"After clip: min={gen_weights[qcd_mask].min():.3e} "
                  f"max={gen_weights[qcd_mask].max():.3e} "
                  f"mean={gen_weights[qcd_mask].mean():.3f}")

    if exclude_labels:
        mask = torch.ones(len(labels), dtype=torch.bool)
        for lbl in exclude_labels:
            mask &= (labels != lbl)
        pf, labels, obj = pf[mask], labels[mask], obj[mask]
        if gen_weights is not None:
            gen_weights = gen_weights[mask]
        unique_lbls = sorted(labels.unique().tolist())
        remap = {old: new for new, old in enumerate(unique_lbls)}
        labels = torch.tensor([remap[l.item()] for l in labels], dtype=torch.long)
        print(f"[exclude_labels={exclude_labels}] Remapped labels: {remap}. Remaining events: {len(labels)}")

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
                                    val_split=val_split, seed=seed,
                                    gen_weights=gen_weights)
    ds_val   = HLTSmCocktailDataset(pf, obj_norm, labels, ae_model,
                                    n_bins=n_bins, split="val",
                                    val_split=val_split, seed=seed,
                                    bin_edges=ds_train.bin_edges,
                                    gen_weights=gen_weights)
    return ds_train, ds_val, obj_scaler
