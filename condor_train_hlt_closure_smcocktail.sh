#!/bin/bash
set -e

echo "==== HLT NURD contrastive training (closure loss, smcocktail) started: $(date) ===="
echo "Host: $(hostname)"

source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
unset PYTHONHOME PYTHONPATH
source /eos/user/e/escheull/con_env/bin/activate
PYTHON=/eos/user/e/escheull/con_env/bin/python3
echo "Python: $PYTHON ($($PYTHON --version))"
echo "Torch CUDA: $($PYTHON -c 'import torch; print(torch.version.cuda, "| CUDA available:", torch.cuda.is_available())')"

cd /afs/cern.ch/user/e/escheull/nobackup/hlt_nurd_con
git checkout closure_loss

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EOS_CHECKPOINTS="/eos/user/e/escheull/ssl_checkpoints"
mkdir -p "$EOS_CHECKPOINTS"
if [ -d checkpoints ] && [ ! -L checkpoints ]; then
    cp -r checkpoints/* "$EOS_CHECKPOINTS/" 2>/dev/null || true
    rm -rf checkpoints
fi
if [ ! -L checkpoints ]; then
    ln -s "$EOS_CHECKPOINTS" checkpoints
fi

export WANDB_API_KEY=$(cat ~/.wandb_api_key)

AE_CKPT="/eos/user/e/escheull/ssl_checkpoints/hlt/hlt/ae_pretrain/checkpoint_ae.pth"

$PYTHON train_hlt.py \
    --data         /eos/user/e/escheull/smcocktail_1M_noZB/hlt_smcocktail_train.pt \
    --ae_ckpt      "$AE_CKPT" \
    --epochs       100 \
    --batch_size   4096 \
    --lr           1e-4 \
    --weight_decay 5e-3 \
    --n_bins       10 \
    --reweight     1 \
    --joint_indep  1 \
    --contrast_weight 0.2 \
    --contrast_temp   0.05 \
    --embed_size   128 \
    --latent_dim   6 \
    --proj_dim     6 \
    --num_heads    8 \
    --num_layers   4 \
    --dim_ff       512 \
    --linear_dim   16 \
    --critic_schedule          warmup \
    --critic_warmup_epochs     7 \
    --critic_ramp_epochs       10 \
    --critic_train_frac        0.2 \
    --critic_lr_multiplier     10.0 \
    --n_critic_steps_per_batch 3 \
    --_lambda      0.01 \
    --no_mi_norm \
    --closure_weight 0.1 \
    --qcd_label      1 \
    --exp_name     hlt_nurd_closure_smcocktail \
    --project_name hlt \
    --manualSeed   42

echo "==== HLT NURD contrastive training (closure loss, smcocktail) finished: $(date) ===="
