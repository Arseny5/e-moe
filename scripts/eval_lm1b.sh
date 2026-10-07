#!/bin/bash
# Generative perplexity (GPT-2-large) and sample entropy over NFE, 2000 samples per NFE.

# Usage: ALGO=emoe CKPT=outputs/lm1b_emoe/checkpoints/last.ckpt bash scripts/eval_lm1b.sh
# ALGO in {mdlm, sedd, vadd, emoe, mdlm_moe}

set -e
cd "$(dirname "$0")/.."
DATA_CACHE=${DATA_CACHE:-./data_cache}
PRED=ancestral_cache; EXTRA=""; MAIN_ALGO=$ALGO
[ "$ALGO" = "sedd" ] && PRED=analytic
[ "$ALGO" = "vadd" ] && EXTRA="+decoder=small decoder.length=128"
[ "$ALGO" = "mdlm_moe" ] && MAIN_ALGO=emoe && EXTRA="algo.det_moe=True"
for NFE in 1 2 4 8 16 32 64 128; do
  D=outputs/eval_${ALGO}/nfe$NFE; mkdir -p $D
  python -u -m main mode=sample_eval algo=$MAIN_ALGO model=small model.length=128 $EXTRA \
    data=lm1b data.wrap=True data.cache_dir=$DATA_CACHE \
    loader.eval_batch_size=8 loader.eval_global_batch_size=8 trainer.devices=1 \
    eval.checkpoint_path=$CKPT eval.generated_samples_path=$D/samples.json \
    eval.compute_generative_perplexity=True eval.perplexity_batch_size=4 \
    sampling.predictor=$PRED sampling.steps=$NFE sampling.num_sample_batches=250 \
    wandb.disabled=True hydra.run.dir=$D > $D/log 2>&1
  mkdir -p outputs/mauve_samples && cp $D/samples.json outputs/mauve_samples/${ALGO}_nfe${NFE}.json
  echo "NFE=$NFE $(grep -ao 'Generative perplexity: [0-9.]*' $D/log | tail -1) $(grep -ao 'Sample entropy: [0-9.]*' $D/log | tail -1)"
done
