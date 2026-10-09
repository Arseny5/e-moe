import os, sys, json, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np, torch, mauve
from hydra import compose, initialize
import main as _resolvers
import dataloader

SAMP = os.environ.get("SAMPLES", "outputs/mauve_samples")
OUT  = "results_mauve.txt"
N    = 2000
SEEDS = (0, 1, 2)


with initialize(version_base=None, config_path="../configs"):
    cfg = compose(config_name="config", overrides=[
        "model=small", "data=lm1b", "algo=mdlm", "model.length=128", "data.wrap=True",
        "data.cache_dir=" + os.environ.get('DATA_CACHE', './data_cache'),
        "loader.global_batch_size=64", "loader.batch_size=64",
        "loader.eval_global_batch_size=64", "loader.eval_batch_size=64",
        "loader.num_workers=2", "trainer.devices=1", "wandb.disabled=True"])
tok = dataloader.get_tokenizer(cfg)
_, valid_dl = dataloader.get_dataloaders(cfg, tok, skip_train=True, valid_seed=1)
real = []
for b in valid_dl:
    real += tok.batch_decode(b["input_ids"], skip_special_tokens=False)
    if len(real) >= N: break
real = real[:N]
print(f"reference texts: {len(real)}", flush=True)

def score(q_text):
    vals = []
    for sd in SEEDS:
        out = mauve.compute_mauve(
            p_text=real, q_text=q_text,
            featurize_model_name="gpt2-large",
            device_id=0, max_text_length=128,
            verbose=False, seed=sd)
        vals.append(out.mauve)
    return float(np.mean(vals)), float(np.std(vals))

rows = []
files = sorted(glob.glob(f"{SAMP}/*.json"),
               key=lambda p: (os.path.basename(p).split("_")[0],
                              int(os.path.basename(p).split("nfe")[1].split(".")[0])))
for f in files:
    tag = os.path.basename(f)[:-5]
    q = json.load(open(f))["generated_seqs"][:N]
    m, s = score(q)
    line = f"{tag:16s} n={len(q):5d}  MAUVE = {m:.4f} +- {s:.4f}"
    rows.append(line); print(line, flush=True)
    open(OUT, "w").write("\n".join(rows) + "\n")

open(OUT, "w").write("\n".join(rows) + "\nDONE_MAUVE\n")
print("saved:", OUT)
