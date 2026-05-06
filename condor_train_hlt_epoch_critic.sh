#!/bin/bash
set -e

echo "==== HLT NURD contrastive training (per-epoch critic) started: $(date) ===="
echo "Host: $(hostname)"

source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
unset PYTHONHOME PYTHONPATH
source /eos/user/e/escheull/con_env/bin/activate
PYTHON=/eos/user/e/escheull/con_env/bin/python3
echo "Python: $PYTHON ($($PYTHON --version))"
echo "Torch CUDA: $($PYTHON -c 'import torch; print(torch.version.cuda, "| CUDA available:", torch.cuda.is_available())')"

cd /afs/cern.ch/user/e/escheull/nobackup/hlt_nurd_con

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Symlink checkpoints to EOS so they survive job eviction
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
    --batch_size   1024 \
    --lr           1e-4 \
    --weight_decay 5e-3 \
    --n_bins       10 \
    --reweight     1 \
    --joint_indep  1 \
    --critic_epochs 2 \
    --contrast_weight 0.05 \
    --contrast_temp   0.05 \
    --embed_size   128 \
    --latent_dim   6 \
    --proj_dim     6 \
    --num_heads    8 \
    --num_layers   4 \
    --dim_ff       512 \
    --linear_dim   16 \
    --critic_schedule per_epoch \
    --_lambda      0.01 \
    --exp_name     hlt_nurd_run_epoch_critic \
    --project_name hlt \
    --manualSeed   42

echo "==== HLT NURD contrastive training (per-epoch critic) finished: $(date) ===="
