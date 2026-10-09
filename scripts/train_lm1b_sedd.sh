#!/bin/bash
# Train SEDD on LM1B: small DiT, sequence length 128, sentence packing,
# 1M steps at global batch 512. Set DATA_CACHE for the tokenized dataset and OUT for outputs.

set -e
cd "$(dirname "$0")/.."
DATA_CACHE=$(realpath -m "${DATA_CACHE:-./data_cache}")  # absolute: Hydra runs inside the output dir
OUT=${OUT:-outputs/lm1b_sedd}
python -u -m main \
  algo=sedd model=small model.length=128 \
  data=lm1b data.wrap=True data.cache_dir=$DATA_CACHE \
  loader.global_batch_size=512 loader.batch_size=256 \
  loader.eval_global_batch_size=64 loader.eval_batch_size=16 loader.num_workers=8 \
  trainer.devices=2 trainer.max_steps=1000000 trainer.num_sanity_val_steps=0 \
  trainer.val_check_interval=1000000 trainer.limit_val_batches=0.0 sampling.num_sample_batches=0 eval.compute_generative_perplexity=False \
  sampling.predictor=analytic sampling.steps=100 \
  eval.perplexity_batch_size=4 wandb.disabled=True hydra.run.dir=$OUT \
  ~callbacks.checkpoint_monitor callbacks.checkpoint_every_n_steps.every_n_train_steps=20000
