#!/bin/bash
set -e

echo "==== AE pre-training started: $(date) ===="
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

$PYTHON train_ae.py \
    --data        /eos/user/e/escheull/smcocktail_1M_noZB/hlt_smcocktail_train.pt \
    --epochs      100 \
    --batch_size  2048 \
    --lr          1e-3 \
    --latent_dim  16 \
    --enc_nodes   512 256 \
    --dec_nodes   256 512 \
    --exp_name    ae_pretrain \
    --project_name hlt

echo "==== AE pre-training finished: $(date) ===="
