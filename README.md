# NURD HLT Anomaly Detection

An anomaly detection analysis using a two-axis ABCD background estimation method. Axis 1 is the AE reconstruction loss (how anomalous an event looks) and axis 2 is a contrastive model score (what type of event it is). NURD is used to decorrelate the two axes so the ABCD method is valid.

---

## How it works

### Axis 1 — Autoencoder (density estimation)

An MLP autoencoder trained on object-level features (pT, η, φ). Its reconstruction loss is the nuisance variable — events that reconstruct poorly are flagged as anomalous.

### Axis 2 — Contrastive encoder (clustering)

A Linformer-based Transformer encodes PF candidates into a low-dimensional latent vector, trained with SupCon + cross-entropy to cluster events by class.

### NURD decorrelation

NURD enforces independence between the two axes via two mechanisms:

- **Reweighting** — computes sample weights `w(y,z) = p(y)·p(z)/p(y,z)` so that under the weighted distribution the class label `y` and the binned AE reco loss `z` are statistically independent. Applied to the CE loss each batch.
- **Critic** — a small MLP trained to predict the nuisance bin `z` from `(latent, y)`. The encoder is penalised when the critic succeeds, pushing it to drop information about `z`. The critic is retrained for `--critic_epochs` epochs at the start of each encoder epoch.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

Training is two steps — the AE must be trained first since its checkpoint is required by the main training script.

### Step 1 — Train the AE

```bash
python train_ae.py \
    --data /eos/user/e/escheull/smcocktail_1M_noZB/hlt_smcocktail_train.pt
```

The checkpoint is saved to `checkpoints/hlt/ae/checkpoint_ae.pth`.

### Step 2 — Train the NURD contrastive model

```bash
python train_hlt.py \
    --data    /eos/user/e/escheull/smcocktail_1M_noZB/hlt_smcocktail_train.pt \
    --ae_ckpt checkpoints/hlt/ae/checkpoint_ae.pth \
    --reweight 1 \
    --joint_indep 1 \
    --critic_epochs 2 \
    --_lambda 1.0
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--reweight` | 1 | Enable NURD sample reweighting |
| `--joint_indep` | 1 | Enable adversarial critic |
| `--critic_epochs` | 2 | Critic training steps per encoder epoch |
| `--_lambda` | 0.01 | Weight on the critic loss |
| `--n_bins` | 10 | Number of quantile bins for the AE reco nuisance |
| `--contrast_weight` | 0.05 | Balance between contrastive and CE loss |

Checkpoints are saved to `checkpoints/hlt/<project_name>/<exp_name>/`.

---

## Evaluation

```bash
python eval_abcd_nurd.py \
    --ckpt       checkpoints/hlt/hlt/hlt_nurd_run/checkpoint_main.pth.tar \
    --ae_ckpt    checkpoints/hlt/ae/checkpoint_ae.pth \
    --test_pt    /eos/user/e/escheull/smcocktail_1M_noZB/hlt_smcocktail_test.pt \
    --signal_pt  /eos/user/e/escheull/smcocktail_1M_noZB/hlt_signal_TpTp.pt \
    --outdir     outputs_abcd/<run> \
    --n_pca      6
```

This also saves `<outdir>/abcd_thresholds.json` with the optimised t1/t2 thresholds for use with the datacard script.

---

## Datacard for CMS Combine

```bash
python make_datacard_ttbar.py \
    --ckpt       checkpoints/hlt/hlt/hlt_nurd_run/checkpoint_main.pth.tar \
    --ae_ckpt    checkpoints/hlt/ae/checkpoint_ae.pth \
    --test_pt    /eos/user/e/escheull/smcocktail_1M_noZB/hlt_smcocktail_test.pt \
    --outdir     outputs_datacard/<run> \
    --n_pca      6
```

To skip the threshold scan and reuse thresholds from a previous eval run:

```bash
python make_datacard_ttbar.py \
    --ckpt      checkpoints/hlt/hlt/hlt_nurd_run/checkpoint_main.pth.tar \
    --ae_ckpt   checkpoints/hlt/ae/checkpoint_ae.pth \
    --test_pt   /eos/user/e/escheull/smcocktail_1M_noZB/hlt_smcocktail_test.pt \
    --eval_json outputs_abcd/<run>/abcd_thresholds.json \
    --outdir    outputs_datacard/<run>
```

To run the datacard with Combine:

```bash
combine -M Significance datacard_ttbar.txt -t -1 --expectSignal 1
combine -M AsymptoticLimits datacard_ttbar.txt -t -1
```
