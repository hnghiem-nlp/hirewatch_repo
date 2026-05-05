# HireWatch — Code Submission

This package contains the code accompanying the HireWatch paper. It has
two parts:

1. **Inference pipeline** (`src/experiments/`) — produces a single LLM
   trial JSON given a model, a job, a tier, a safety condition, and a
   seed.
2. **Analysis notebooks** (`notebooks/`) — load trial-level JSON files
   from `results/` and produce the paper's figures and tables.

## Setup

```bash
pip install -r requirements.txt
```

## Reproducing a single trial

```bash
# 1. Provide API keys
cp config/secrets.example config/secrets
# (edit config/secrets and fill in your OpenAI / Anthropic / OpenRouter keys)

# 2. Run one trial (Tier 2, no safety, persona B, one seed, one job)
PYTHONPATH=src python src/experiments/run_interview_onet.py \
    --provider openai --model gpt-4o-mini \
    --personas B --seeds 42 --reps 1 --safe none \
    --jobs 13-1071
```

The trial JSON lands at
`results/interview_v4/onet_baseline/gpt-4o-mini/temp_0.0/personaB/13-1071_00-personaB-K25-seed42-rep1.json`.

The per-trial CLI shown above is the same code path that produced the
full sweep reported in the paper; the full sweep was orchestrated on a
slurm cluster and is not bundled here.

## Reproducing the analyses

```bash
# 1. Download the trial-results bundle (v4_results.tar.gz, ~281 MB)
#    from the Harvard Dataverse anonymous preview URL:
#
#      https://dataverse.harvard.edu/previewurl.xhtml?token=715d8129-ec2f-4ac3-a2ff-28ab534b208e
#
#    Verify integrity (optional):
#      sha256sum v4_results.tar.gz
#      # Expected:
#      # af4da7f182795e0d646e0503d005ff1b94c1177a821f9a1331de933d2e2097f4

# 2. Extract into this directory. The bundle's layout already matches
#    results/{interview_v4,thinking_ablation}/...
cd <path/to/this/code/dir>
tar -xzf /path/to/v4_results.tar.gz

# 3. Tell the notebooks where the project root is
export HIREWATCH_ROOT=$(pwd)

# 4. Launch Jupyter and open any notebook under notebooks/
jupyter notebook
```

Each notebook produces the figures and tables referenced in the paper.
Outputs land under `outputs/figures/` and `outputs/tables/`.

> Note on extraction time: the bundle contains ~217k small JSON files;
> on a local SSD it unpacks in 3--5 minutes, but on shared/network
> storage it can take 20--30 minutes. Plan accordingly.

## Folder map

```
src/                  # Python source (importable as `experiments.*`, `query.*`, etc.)
data/                 # Question bank (50 unlawful + 150 lawful items) and 20 O*NET jobs
config/               # Model roster, run settings, paper labels, secrets template
notebooks/            # One subdir per paper section / appendix
results/              # Empty landing zone for the v4 results bundle
outputs/              # Figures and tables written by the notebooks
```
