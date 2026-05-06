#!/bin/bash
set -e

echo "==== NURD ABCD eval started: $(date) ===="
echo "Host: $(hostname)"

source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
unset PYTHONHOME PYTHONPATH
source /eos/user/e/escheull/con_env/bin/activate
PYTHON=/eos/user/e/escheull/con_env/bin/python3
echo "Python: $PYTHON ($($PYTHON --version))"
echo "Torch CUDA: $($PYTHON -c 'import torch; print(torch.version.cuda, "| CUDA available:", torch.cuda.is_available())')"

cd /afs/cern.ch/user/e/escheull/nobackup/hlt_nurd_con

export WANDB_API_KEY=$(cat ~/.wandb_api_key)

CKPT=$(ls -t /eos/user/e/escheull/ssl_checkpoints/hlt/hlt/hlt_nurd_run_epoch_critic/checkpoint_main_*.pth.tar 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    # fall back to old un-timestamped name if present
    CKPT="/eos/user/e/escheull/ssl_checkpoints/hlt/hlt/hlt_nurd_run_epoch_critic/checkpoint_main.pth.tar"
fi
AE_CKPT="/eos/user/e/escheull/ssl_checkpoints/hlt/hlt/ae_pretrain/checkpoint_ae.pth"

$PYTHON eval_abcd_nurd.py \
    --ckpt          "$CKPT" \
    --ae_ckpt       "$AE_CKPT" \
    --test_pt       /eos/user/e/escheull/smcocktail_1M_noZB/hlt_smcocktail_test.pt \
    --signal_pt     /eos/user/e/escheull/signal_pt/hlt_signal_TpTp.pt \
    --outdir        /eos/user/e/escheull/abcd_outputs/nurd_epoch_critic \
    --wandb_run_name nurd_epoch_critic_abcd \
    --n_pca         6

echo "==== NURD ABCD eval finished: $(date) ===="
