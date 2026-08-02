"""
make_datacard_ttbar.py

Produces a CMS Combine ABCD datacard for a ttbar signal search using the
HLT NURD contrastive model.

Two discriminating axes:
  Axis 1 (x): AE reconstruction loss  — measures how "surprising" an event is to the AE
  Axis 2 (y): Mahalanobis distance in PCA-whitened NURD latent space

ABCD regions (defined by thresholds t1, t2):

         axis2 (MD)
         ^
  high   |   C     |   A (signal region)
         |---------|----------
  low    |   D     |   B
         +---------|---------> axis1 (AE loss)
                  t1

Background prediction in A: N_bkg_A_hat = N_bkg_B * N_bkg_C / N_bkg_D
Signal = TT events (label == 2 in the SM cocktail dataset)
Background = DY (0) + QCD (1) + WJets (3)

Thresholds are optimised on QCD-only events (same as eval_abcd_nurd.py).

Usage:
------
python make_datacard_ttbar.py \\
    --ckpt    /path/to/checkpoint_main.pth.tar \\
    --ae_ckpt /path/to/checkpoint_ae.pth \\
    --test_pt /path/to/hlt_smcocktail_test.pt \\
    [--n_pca 6] \\
    [--p1 0.90 --p2 0.90]  # explicit thresholds instead of scanning \\
    [--outdir outputs_datacard] \\
    [--datacard_name datacard_ttbar.txt]

The script writes:
  <outdir>/datacard_ttbar.txt   — CMS Combine ABCD datacard
  <outdir>/abcd_summary.json    — all ABCD counts and thresholds for bookkeeping
"""

import os
import gc
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.hlt_con import HLTContrastiveModel
from models.hlt_autoencoder import HLTAutoencoder


# ---------------------------------------------------------------------------
# Model loading  (identical to eval_abcd_nurd.py)
# ---------------------------------------------------------------------------

def load_nurd_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]
    sd   = ckpt["state_dict_model"]

    e_proj_key = next((k for k in sd if k.endswith(".attn.e.weight")), None)
    num_tokens  = sd[e_proj_key].shape[1] - 1 if e_proj_key else cfg.get("linear_dim", 100)
    num_classes = sd["classifier.weight"].shape[0]

    model = HLTContrastiveModel(
        num_classes=num_classes,
        embed_size=cfg["embed_size"],
        latent_dim=cfg["latent_dim"],
        proj_dim=cfg["proj_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dim_ff=cfg["dim_ff"],
        linear_dim=cfg["linear_dim"],
        num_tokens=num_tokens,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded NURD model: latent_dim={cfg['latent_dim']}, n_layers={cfg['num_layers']}", flush=True)
    return model, ckpt


def load_ae(ae_ckpt_path, ae_scaler, device):
    ae_ckpt = torch.load(ae_ckpt_path, map_location=device)
    ae_cfg  = ae_ckpt.get("ae_config", {
        "features": None, "latent_dim": 16,
        "encoder_config": {"nodes": [512, 256]},
        "decoder_config": {"nodes": [256, 512, None]},
        "alpha": 1.0,
    })
    if ae_cfg["features"] is None:
        first_w = ae_ckpt["ae"][next(iter(ae_ckpt["ae"]))]
        ae_cfg["features"] = first_w.shape[1]
    ae = HLTAutoencoder(ae_cfg).to(device)
    ae.load_state_dict(ae_ckpt["ae"])
    ae.eval()
    print(f"Loaded AE: features={ae_cfg['features']}, latent={ae_cfg['latent_dim']}", flush=True)
    return ae


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def compute_ae_scores(ae, ae_scaler, pt_path_or_dict, device, batch_size=4096):
    """Returns (ae_scores [N], labels [N])."""
    mu  = ae_scaler["mu"].cpu().numpy()
    std = ae_scaler["std"].cpu().numpy()

    raw = torch.load(pt_path_or_dict, map_location="cpu") if isinstance(pt_path_or_dict, str) else pt_path_or_dict
    obj = raw["obj"][:, :, :4].reshape(raw["obj"].shape[0], -1).float().numpy()
    obj_norm = torch.from_numpy(((obj - mu) / (std + 1e-8)).astype(np.float32))
    labels = raw["label"].numpy()
    N = obj_norm.shape[0]
    print(f"  AE inference on {N} events...", flush=True)

    scores = []
    with torch.no_grad():
        for i0 in range(0, N, batch_size):
            xb = obj_norm[i0:i0 + batch_size].to(device)
            recon, _ = ae(xb)
            mse = ((recon - xb) ** 2).mean(dim=1)
            scores.append(mse.cpu())
    return torch.cat(scores).numpy().astype(np.float32), labels


def embed_pf(model, pt_path_or_dict, device, batch_size=512, return_logits=False):
    """Returns (latents [N, D], labels [N]) or (latents, logits [N, C], labels) if return_logits."""
    raw = torch.load(pt_path_or_dict, map_location="cpu") if isinstance(pt_path_or_dict, str) else pt_path_or_dict
    pf     = torch.nan_to_num(raw["pf"], nan=0.0, posinf=0.0, neginf=0.0)
    labels = raw["label"].numpy()
    N = pf.shape[0]
    print(f"  Encoder inference on {N} events...", flush=True)

    latents, logits_list = [], []
    with torch.no_grad():
        for i0 in range(0, N, batch_size):
            xb = pf[i0:i0 + batch_size].to(device)
            latent, logit = model(xb)
            latents.append(latent.cpu())
            if return_logits:
                logits_list.append(logit.cpu())
    if return_logits:
        return (torch.cat(latents, dim=0).numpy(),
                torch.cat(logits_list, dim=0).numpy(),
                labels)
    return torch.cat(latents, dim=0).numpy(), labels


def compute_logit_axis2(logits, qcd_label=1):
    """1 - P(QCD) from classifier logits — higher means more anomalous."""
    import torch.nn.functional as F_
    probs = F_.softmax(torch.from_numpy(logits.astype(np.float32)), dim=1).numpy()
    return (1.0 - probs[:, qcd_label]).astype(np.float32)


def compute_md_scores(latents, labels, n_pca=None, bkg_labels=None):
    """
    Fit PCA-whitened Gaussian to bkg_labels events, compute MD for all events.
    Returns md [N].
    """
    if bkg_labels is None:
        bkg_labels = [1]  # QCD by default

    _CLASS_NAMES = {0: "DY", 1: "QCD", 2: "TT", 3: "WJets"}
    class_transforms = []
    for cls in bkg_labels:
        mask = labels == cls
        if mask.sum() < 10:
            print(f"  WARNING: class {cls} has only {mask.sum()} events — skipping", flush=True)
            continue
        ref = latents[mask]
        mu  = ref.mean(axis=0)
        centered = ref - mu
        cov = (centered.T @ centered) / ref.shape[0]
        L, V = np.linalg.eigh(cov)
        if n_pca is not None:
            V = V[:, -n_pca:]
            L = L[-n_pca:]
        L = np.clip(L, 1e-6, None)
        W = V / np.sqrt(L)
        class_transforms.append((cls, mu, W))
        print(f"  Fit MD transform on {mask.sum()} {_CLASS_NAMES.get(cls, str(cls))} events", flush=True)

    md_per_class = []
    for cls, mu_c, W_c in class_transforms:
        z_c = (latents - mu_c) @ W_c
        md_per_class.append((z_c * z_c).sum(axis=1))
    return np.stack(md_per_class, axis=0).min(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Cross-section weights: xsec (pb) * lumi (pb^-1) / n_gen
#
# Lumi: 172,398 pb^-1 (full Run 3, 2022-2024)
# xsec: GenXSecAnalyzer on Run3Winter25 samples (QCD/TT confirmed directly;
#        DY/WJet from PDG/CMS central values)
# n_gen: total events GENERATED before any HLT/scouting filter, confirmed via
#        DAS summary + independent per-file sum:
#   QCD   — /QCD_Bin-Pt-15to7000_.../Run3Winter25.../MINIAODSIM  (dedicated 100k campaign)
#   TT    — /TT_TuneCP5_.../Run3Winter25.../GEN-SIM-RAW          (dedicated 100k campaign)
#   DY    — /DYto2L-4Jets_.../Run3Winter25.../GEN-SIM-RAW        (72M-event full production)
#   WJet  — /WJetsToLNu_.../Run3Winter25.../GEN-SIM-RAW          (29M-event full production)
# ---------------------------------------------------------------------------
LUMI_PB = 172_398.0  # full Run 3, pb^-1
_XSEC   = {0: 5_469.0, 1: 1.436e9, 2: 814.5, 3: 56_120.0}   # pb
_N_GEN  = {0: 72_101_775, 1: 100_000, 2: 100_000, 3: 29_163_730}
CLASS_WEIGHTS = {cls: _XSEC[cls] * LUMI_PB / _N_GEN[cls] for cls in _XSEC}
# Resulting per-event weights: QCD ~2.48e9, TT ~1404, DY ~13.1, WJet ~332
# QCD weight is large by construction — only 100k events represent a 1.4e9 pb process.


def make_event_weights(labels, use_weights=True):
    if not use_weights:
        return np.ones(len(labels), dtype=np.float64)
    return np.array([CLASS_WEIGHTS.get(int(l), 1.0) for l in labels], dtype=np.float64)


# ---------------------------------------------------------------------------
# ABCD helpers
# ---------------------------------------------------------------------------

def abcd_counts(ax1, ax2, t1, t2, weights=None):
    """Returns dict {A, B, C, D} with (weighted) event counts."""
    if weights is None:
        weights = np.ones(len(ax1), dtype=np.float64)
    return {
        "A": float(weights[(ax1 > t1)  & (ax2 > t2)].sum()),
        "B": float(weights[(ax1 > t1)  & (ax2 <= t2)].sum()),
        "C": float(weights[(ax1 <= t1) & (ax2 > t2)].sum()),
        "D": float(weights[(ax1 <= t1) & (ax2 <= t2)].sum()),
    }


def find_best_abcd_wp(ae_qcd, md_qcd, min_A=10, min_D=100, n_scan=48):
    """Scan percentile pairs; return (best, nc_grid, percent) — nc_grid[i,j] = nonclosure."""
    percent = np.linspace(0.50, 0.98, n_scan)
    best = {"nonclosure": np.inf}
    nc_grid = np.full((len(percent), len(percent)), np.nan)
    for i, p1 in enumerate(percent):
        for j, p2 in enumerate(percent):
            t1_ = float(np.quantile(ae_qcd, p1))
            t2_ = float(np.quantile(md_qcd, p2))
            A = int(((ae_qcd > t1_) & (md_qcd > t2_)).sum())
            B = int(((ae_qcd > t1_) & (md_qcd <= t2_)).sum())
            C = int(((ae_qcd <= t1_) & (md_qcd > t2_)).sum())
            D = int(((ae_qcd <= t1_) & (md_qcd <= t2_)).sum())
            if A < min_A or D < min_D:
                continue
            A_hat = (B * C) / max(D, 1e-8)
            nc    = (A - A_hat) / max(A_hat, 1e-8)
            nc_grid[i, j] = nc
            if np.isfinite(nc) and abs(nc) < abs(best["nonclosure"]):
                best.update({"nonclosure": nc, "t1": t1_, "t2": t2_,
                             "p1": p1, "p2": p2, "A": A, "B": B, "C": C, "D": D})
    if "t1" not in best:
        raise RuntimeError("No ABCD working point found. Try lowering --min_A / --min_D.")
    return best, nc_grid, percent


# ---------------------------------------------------------------------------
# Closure plots
# ---------------------------------------------------------------------------

def plot_closure_scans(ae_qcd, md_qcd, best, nc_grid, percent, outdir,
                       axis2_label="MD"):
    """Save 2D and 1D ABCD closure scan plots."""
    # 2D: full (p1, p2) grid coloured by |non-closure|
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    pct_abs = np.clip(np.abs(nc_grid) * 100.0, 0.0, 100.0)
    vmax = float(np.nanpercentile(pct_abs, 95)) if np.any(np.isfinite(pct_abs)) else 100.0
    mesh = ax.pcolormesh(percent, percent, pct_abs.T, cmap="viridis_r",
                         vmin=0.0, vmax=vmax, shading="auto")
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("|Non-closure| (%)", fontsize=14)
    ax.scatter([best["p1"]], [best["p2"]], marker="*", s=400, color="red",
               edgecolor="black", linewidth=1.0, zorder=5,
               label=f"Best: p1={best['p1']:.3f}, p2={best['p2']:.3f}\n"
                     f"|NC|={100.0*abs(best['nonclosure']):.2f}%")
    ax.set_xlabel("Percentile threshold — axis 1 (AE reco loss)", fontsize=13)
    ax.set_ylabel(f"Percentile threshold — axis 2 ({axis2_label})", fontsize=13)
    ax.set_title("ABCD closure scan (QCD)", fontsize=14)
    ax.legend(loc="lower left", fontsize=11, framealpha=0.9)
    fig.tight_layout()
    out2d = os.path.join(outdir, "closure_scan_2d.png")
    fig.savefig(out2d, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Closure 2D scan saved: {out2d}")

    # 1D: equal-percentile cut (p1=p2=p) swept from loose to tight
    pvals = np.linspace(0.10, 0.98, 60)
    Ntot  = float(len(ae_qcd))
    effs, ratios, uncs = [], [], []
    for p in pvals:
        t1_ = float(np.quantile(ae_qcd, p))
        t2_ = float(np.quantile(md_qcd, p))
        A = int(((ae_qcd > t1_) & (md_qcd > t2_)).sum())
        B = int(((ae_qcd > t1_) & (md_qcd <= t2_)).sum())
        C = int(((ae_qcd <= t1_) & (md_qcd > t2_)).sum())
        D = int(((ae_qcd <= t1_) & (md_qcd <= t2_)).sum())
        A_hat = (B * C) / max(D, 1e-8)
        ratio = A_hat / max(A, 1e-8)
        sigma = abs(ratio) * np.sqrt(
            (0.0 if A == 0 else 1.0 / A) + (0.0 if B == 0 else 1.0 / B) +
            (0.0 if C == 0 else 1.0 / C) + (0.0 if D == 0 else 1.0 / D)
        )
        effs.append(A / max(Ntot, 1.0))
        ratios.append(ratio)
        uncs.append(sigma)

    effs   = np.array(effs);  ratios = np.array(ratios);  uncs = np.array(uncs)
    order  = np.argsort(effs)
    effs   = effs[order];   ratios = ratios[order];   uncs = uncs[order]
    A_hat_opt = best["B"] * best["C"] / max(best["D"], 1e-8)
    eff_opt   = best["A"] / max(Ntot, 1.0)
    ratio_opt = A_hat_opt / max(best["A"], 1e-8)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(effs, ratios, c="steelblue", label="Predicted / true (equal-pct cut)")
    ax.fill_between(effs, ratios - uncs, ratios + uncs,
                    facecolor="steelblue", alpha=0.35, interpolate=True)
    ax.axhline(1.00, color="black", linestyle="-",  linewidth=1.0)
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(1.05, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.scatter([eff_opt], [ratio_opt], marker="*", s=250, c="red", zorder=5,
               label=f"Optimized WP ({100.0*abs(best['nonclosure']):.1f}% NC)")
    ax.set_xlabel("Selection efficiency (QCD in A / total QCD)", fontsize=14)
    ax.set_ylabel("Predicted bkg. / true bkg.", fontsize=14)
    ax.set_ylim(0.0, 1.5)
    ax.set_xscale("log")
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out1d = os.path.join(outdir, "closure_scan_1d.png")
    fig.savefig(out1d, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Closure 1D scan saved: {out1d}")


# ---------------------------------------------------------------------------
# Datacard writer
# ---------------------------------------------------------------------------

def write_datacard(
    path,
    bins_sig,
    bins_bkg,
    bins_obs,
    nonclosure=None,
    bkg_process_desc=None,
    signal_name="tt",
    bkg_name="bkg",
    sig_unc=None,
    lumi_unc=None,
):
    N_bkg_B = max(bins_bkg["B"], 1e-3)
    N_bkg_C = max(bins_bkg["C"], 1e-3)
    N_bkg_D = max(bins_bkg["D"], 1e-3)

    def fmt(x): return f"{x:.4f}"

    obs_A = int(round(bins_obs["A"]))
    obs_B = int(round(bins_obs["B"]))
    obs_C = int(round(bins_obs["C"]))
    obs_D = int(round(bins_obs["D"]))

    lines = [
        "imax 4", "jmax 1", "kmax 0", "",
        60 * "-", "",
        "bin          A        B        C        D",
        f"observation  {obs_A:<8d} {obs_B:<8d} {obs_C:<8d} {obs_D:<8d}",
        "", 60 * "-", "",
        f"bin      {'A':<10} {'B':<10} {'C':<10} {'D':<10} {'A':<10} {'B':<10} {'C':<10} {'D':<10}",
        f"process  {signal_name:<10} {signal_name:<10} {signal_name:<10} {signal_name:<10} "
        f"{bkg_name:<10} {bkg_name:<10} {bkg_name:<10} {bkg_name:<10}",
        f"process  {'0':<10} {'0':<10} {'0':<10} {'0':<10} "
        f"{'1':<10} {'1':<10} {'1':<10} {'1':<10}",
        f"rate     {fmt(bins_sig['A']):<10} {fmt(bins_sig['B']):<10} "
        f"{fmt(bins_sig['C']):<10} {fmt(bins_sig['D']):<10} "
        f"{'1':<10} {'1':<10} {'1':<10} {'1':<10}",
        "", 60 * "-", "",
        f"r_B   rateParam  B  {bkg_name}  {fmt(N_bkg_B)}",
        f"r_C   rateParam  C  {bkg_name}  {fmt(N_bkg_C)}",
        f"r_D   rateParam  D  {bkg_name}  {fmt(N_bkg_D)}",
        f"r_A   rateParam  A  {bkg_name}  @0*@1/@2  r_B,r_C,r_D",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Datacard written to: {path}")


def write_datacard_cnt(
    path,
    sig_A,
    bkg_A,
    obs_A,
    signal_name="tt",
    bkg_name="bkg",
):
    """
    Write a single-bin cut-and-count datacard for the signal region A only.
    Background rate comes directly from MC — no rateParam transfer factor.
    """
    def fmt(x): return f"{x:.4f}"

    lines = [
        "imax 1",
        "jmax 1",
        "kmax 0",
        "",
        60 * "-",
        "",
        f"bin          A",
        f"observation  {int(round(obs_A))}",
        "",
        60 * "-",
        "",
        f"bin      {'A':<10} {'A':<10}",
        f"process  {signal_name:<10} {bkg_name:<10}",
        f"process  {'0':<10} {'1':<10}",
        f"rate     {fmt(sig_A):<10} {fmt(bkg_A):<10}",
        "",
        60 * "-",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Cut-and-count datacard written to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Make CMS Combine ABCD datacard for ttbar search")
    parser.add_argument("--ckpt",       required=True, help="NURD main checkpoint (.pth.tar)")
    parser.add_argument("--ae_ckpt",    required=True, help="AE checkpoint (.pth)")
    parser.add_argument("--test_pt",    required=True, help="SM cocktail test file (.pt)")
    parser.add_argument("--eval_json",  default=None,
                        help="Path to abcd_thresholds.json from eval_abcd_nurd.py. "
                             "If provided, t1/t2/n_pca are read from this file and the scan is skipped.")
    parser.add_argument("--n_pca",      type=int,   default=None,  help="PCA dims for MD (default: all, overridden by --eval_json)")
    parser.add_argument("--p1",         type=float, default=None,  help="AE-loss percentile threshold (skips scan, ignored if --eval_json set)")
    parser.add_argument("--p2",         type=float, default=None,  help="MD percentile threshold (skips scan, ignored if --eval_json set)")
    parser.add_argument("--tt_label",          type=int,   default=2,     help="Class label for TT (default: 2)")
    parser.add_argument("--bkg_process_labels", type=int, nargs="+", default=[1],
                        help="Which labels count as background in ABCD (default: 1=QCD only). "
                             "Events with other labels are excluded from the analysis entirely. "
                             "Example: --bkg_process_labels 0 1 3  includes DY+QCD+WJets.")
    parser.add_argument("--bkg_labels", type=int, nargs="+", default=[1],
                        help="Labels to use when fitting the MD Gaussian (default: 1=QCD). "
                             "Should be a subset of --bkg_process_labels.")
    parser.add_argument("--min_md",     action="store_true",
                        help="Use min-MD across bkg_labels instead of QCD-only")
    parser.add_argument("--axis2_logit", action="store_true",
                        help="Use 1-P(QCD) classifier score as axis2 instead of Mahalanobis distance")
    parser.add_argument("--min_A",      type=int,   default=10,    help="Min QCD events in A for WP scan")
    parser.add_argument("--min_D",      type=int,   default=100,   help="Min QCD events in D for WP scan")
    parser.add_argument("--outdir",     default="outputs_datacard", help="Output directory")
    parser.add_argument("--datacard_name", default="datacard_ttbar.txt")
    parser.add_argument("--sig_unc",    type=float, default=0.05,  help="Signal systematic (flat fraction)")
    parser.add_argument("--lumi_unc",   type=float, default=0.016, help="Lumi uncertainty (fraction)")
    parser.add_argument("--no_xsec_weights", action="store_true",
                        help="Disable xsec*lumi/nevents weighting (use raw event counts)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── load thresholds from eval JSON if provided ────────────────────────────
    if args.eval_json is not None:
        with open(args.eval_json) as f:
            eval_thresholds = json.load(f)
        args.p1   = None  # not used — we have absolute thresholds
        args.p2   = None
        _t1_fixed = eval_thresholds["t1"]
        _t2_fixed = eval_thresholds["t2"]
        if args.n_pca is None:
            args.n_pca = eval_thresholds.get("n_pca", None)
        print(f"Loaded thresholds from {args.eval_json}: t1={_t1_fixed:.4g}, t2={_t2_fixed:.4g}, n_pca={args.n_pca}")
    else:
        _t1_fixed = _t2_fixed = None

    # ── load models ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading models...")
    model, ckpt = load_nurd_model(args.ckpt, device)
    ae_scaler   = ckpt["ae_scaler"]
    ae          = load_ae(args.ae_ckpt, ae_scaler, device)

    # ── AE scores ─────────────────────────────────────────────────────────────
    print("\n[2/4] Computing AE scores...")
    ae_scores, labels = compute_ae_scores(ae, ae_scaler, args.test_pt, device)
    del ae; gc.collect()
    if device == "cuda": torch.cuda.empty_cache()

    # ── axis2 scores (MD or logit) ────────────────────────────────────────────
    axis2_label = "1-P(QCD)" if args.axis2_logit else "MD"
    print(f"\n[3/4] Computing axis2 scores ({axis2_label})...")
    bkg_labels = None
    if args.axis2_logit:
        _, logits_arr, _ = embed_pf(model, args.test_pt, device, return_logits=True)
        axis2_scores = compute_logit_axis2(logits_arr, qcd_label=1)
    else:
        latents, _  = embed_pf(model, args.test_pt, device)
        bkg_labels  = ([0, 1, 3] if args.min_md else args.bkg_labels)
        axis2_scores = compute_md_scores(latents, labels, n_pca=args.n_pca, bkg_labels=bkg_labels)

    # ── valid event mask ──────────────────────────────────────────────────────
    CLASS_NAMES = {0: "DY", 1: "QCD", 2: "TT", 3: "WJets"}
    keep_labels = set(args.bkg_process_labels) | {args.tt_label}
    process_mask = np.isin(labels, list(keep_labels))
    mask = np.isfinite(ae_scores) & np.isfinite(axis2_scores) & (ae_scores > 0) & process_mask
    ae_scores    = ae_scores[mask]
    axis2_scores = axis2_scores[mask]
    labels       = labels[mask]
    print(f"Valid events after masking (keeping labels {sorted(keep_labels)}): {mask.sum()}")

    # ── split by process ──────────────────────────────────────────────────────
    sig_mask = labels == args.tt_label
    bkg_mask = np.isin(labels, args.bkg_process_labels)
    bkg_process_desc = " + ".join(
        f"{CLASS_NAMES.get(l, str(l))} (label={l})"
        for l in sorted(args.bkg_process_labels)
    )
    print(f"\nProcess counts used in analysis:")
    for cls in sorted(keep_labels):
        name = CLASS_NAMES.get(cls, str(cls))
        role = "SIGNAL" if cls == args.tt_label else "background"
        print(f"  {name} (label={cls}): {(labels==cls).sum()}  [{role}]")

    ae_sig, md_sig = ae_scores[sig_mask], axis2_scores[sig_mask]
    ae_bkg, md_bkg = ae_scores[bkg_mask], axis2_scores[bkg_mask]

    use_weights = not args.no_xsec_weights
    event_weights = make_event_weights(labels, use_weights)
    if use_weights:
        print(f"\nxsec weights: { {k: f'{v:.3g}' for k,v in CLASS_WEIGHTS.items()} }")
    sig_weights = event_weights[sig_mask]
    bkg_weights = event_weights[bkg_mask]

    # ── find ABCD thresholds ──────────────────────────────────────────────────
    print("\n[4/4] Finding ABCD working point...")
    qcd_mask_all = labels == 1
    if qcd_mask_all.sum() == 0:
        raise RuntimeError("No QCD events found after filtering. Check --bkg_process_labels.")

    if _t1_fixed is not None:
        # Use thresholds loaded from eval_abcd_nurd.py JSON — skip scan entirely
        t1, t2 = _t1_fixed, _t2_fixed
        nonclosure_qcd = None
        print(f"Using thresholds from eval JSON: t1={t1:.4g}, t2={t2:.4g}")
    elif args.p1 is not None and args.p2 is not None:
        t1 = float(np.quantile(ae_scores[qcd_mask_all], args.p1))
        t2 = float(np.quantile(axis2_scores[qcd_mask_all], args.p2))
        nonclosure_qcd = None
        print(f"Using specified percentiles: p1={args.p1}, p2={args.p2}")
        print(f"Thresholds: t1={t1:.4g}, t2={t2:.4g}")
    else:
        # Scan on QCD-only (same as eval_abcd_nurd.py)
        ae_qcd = ae_scores[qcd_mask_all]
        md_qcd = axis2_scores[qcd_mask_all]
        wp, nc_grid, percent = find_best_abcd_wp(ae_qcd, md_qcd, min_A=args.min_A, min_D=args.min_D)
        t1, t2 = wp["t1"], wp["t2"]
        nonclosure_qcd = wp["nonclosure"]
        print(f"Best WP: p1={wp['p1']:.3f}, p2={wp['p2']:.3f}")
        print(f"Thresholds: t1={t1:.4g}, t2={t2:.4g}")
        print(f"QCD ABCD: A={wp['A']}  B={wp['B']}  C={wp['C']}  D={wp['D']}")
        print(f"QCD nonclosure: {100*nonclosure_qcd:.2f}%")
        plot_closure_scans(ae_qcd, md_qcd, wp, nc_grid, percent,
                           outdir=args.outdir, axis2_label="NURD contrastive MD")

    # ── ABCD counts ──────────────────────────────────────────────────────────
    bins_sig = abcd_counts(ae_sig, md_sig, t1, t2, weights=sig_weights)
    bins_bkg = abcd_counts(ae_bkg, md_bkg, t1, t2, weights=bkg_weights)
    bins_obs = abcd_counts(ae_scores, axis2_scores, t1, t2, weights=event_weights)

    bkg_A_hat    = bins_bkg["B"] * bins_bkg["C"] / max(bins_bkg["D"], 1e-8)
    nonclosure_bkg = (bins_bkg["A"] - bkg_A_hat) / max(bkg_A_hat, 1e-8)

    print(f"\n{'='*55}")
    print(f"{'Region':<10} {'Signal(TT)':<14} {'Background':<14} {'Observed':<12}")
    print(f"{'-'*55}")
    for reg in ["A", "B", "C", "D"]:
        print(f"  {reg:<8} {bins_sig[reg]:<14.1f} {bins_bkg[reg]:<14.1f} {bins_obs[reg]:<12.0f}")
    print(f"{'='*55}")
    print(f"Background A predicted (B*C/D) : {bkg_A_hat:.2f}")
    print(f"Background A true (MC)         : {bins_bkg['A']:.2f}")
    print(f"Nonclosure (all bkg)           : {100*nonclosure_bkg:.2f}%")
    if nonclosure_qcd is not None:
        print(f"Nonclosure (QCD-only, at WP)   : {100*nonclosure_qcd:.2f}%")
    print(f"Signal (TT) in A               : {bins_sig['A']:.1f}")
    print(f"Signal / background in A       : {bins_sig['A']/max(bkg_A_hat,1e-8):.3f}")
    print()

    # Use the larger of the two nonclosure estimates as the systematic
    nc_for_card = max(abs(nonclosure_bkg), abs(nonclosure_qcd) if nonclosure_qcd else 0)

    # ── write datacards ───────────────────────────────────────────────────────
    card_path = os.path.join(args.outdir, args.datacard_name)
    write_datacard(
        card_path,
        bins_sig=bins_sig,
        bins_bkg=bins_bkg,
        bins_obs=bins_obs,
        nonclosure=nc_for_card,
        bkg_process_desc=bkg_process_desc,
        sig_unc=args.sig_unc,
        lumi_unc=args.lumi_unc,
    )

    cnt_name = args.datacard_name.replace(".txt", "_cnt.txt")
    write_datacard_cnt(
        os.path.join(args.outdir, cnt_name),
        sig_A=bins_sig["A"],
        bkg_A=bins_bkg["A"],
        obs_A=bins_obs["A"],
    )

    # ── save summary JSON ─────────────────────────────────────────────────────
    summary = {
        "ckpt": args.ckpt,
        "ae_ckpt": args.ae_ckpt,
        "test_pt": args.test_pt,
        "thresholds": {"t1": float(t1), "t2": float(t2)},
        "n_pca": args.n_pca,
        "axis2_mode": "logit" if args.axis2_logit else "md",
        "bkg_process_labels": args.bkg_process_labels,
        "bkg_labels_for_md": (None if args.axis2_logit else bkg_labels),
        "tt_label": args.tt_label,
        "bins_signal": bins_sig,
        "bins_background": bins_bkg,
        "bins_observed": bins_obs,
        "bkg_A_predicted": float(bkg_A_hat),
        "nonclosure_bkg": float(nonclosure_bkg),
        "nonclosure_qcd": float(nonclosure_qcd) if nonclosure_qcd is not None else None,
        "S_over_B_in_A": float(bins_sig["A"] / max(bkg_A_hat, 1e-8)),
    }
    json_path = os.path.join(args.outdir, "abcd_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {json_path}")


if __name__ == "__main__":
    main()
