# Experiment 8: run CoDiT and the proposed detector on GAIT

The runner invokes the downloaded upstream CoDiT GAIT evaluator and runs the
proposed randomized-observer detector on the same machine, the same native
12-feature GAIT records, and the same healthy and severe-patient subjects. The
notebook downloads the GAIT archive at the start of its first run, extracts the
records into CoDiT's data directory, and skips the download when the complete
record set is already there. It does not require manually prepared feature or
CoDiT score CSV files.

## Upstream resources

- [Official CoDiT repository](https://github.com/kaustubhsridhar/time-series-OOD)
- CoDiT's GAIT README: `ours/gait/README.md` in that repository.
- [GAIT dataset archive used by CoDiT](https://drive.google.com/drive/folders/1Z-3YnlhcCxI_KlFF6FF7tMp5MSEZURRH?usp=sharing)
- [CoDiT's released GAIT checkpoints](https://drive.google.com/drive/folders/1p0F2D3oTUgB3QKq_0uLu1F9eKRzRibml?usp=sharing)
- [PhysioNet GAITND 1.0.0 dataset page](https://physionet.org/content/gaitndd/1.0.0/)

CoDiT's own GAIT instructions specify the Google Drive archive and the three
checkpoint files `gait_16.pt`, `gait_18.pt`, and `gait_20.pt`. The notebook
downloads the data archive using `gdown` (included in the benchmark extra); you
must download the three model checkpoints from their link below. The PhysioNet
page is the public dataset record. The archive must contain numeric
`controlN.ts`, `parkN.ts`, `alsN.ts`, and `huntN.ts` files with a time column
followed by 12 measurements.

## Setup

Place the CoDiT checkout at `../time-series-OOD-main` next to this toolbox
directory, as in the paper workspace, or clone it into
`benchmarks/external/time-series-OOD`. The setup helper accepts either:

```powershell
powershell -ExecutionPolicy Bypass -File benchmarks/experiment_8/setup_sources.ps1
```

The notebook downloads and extracts the archive into this documented location
if the required records are missing:

```text
<CoDiT checkout>/ours/gait/data/gait-in-neurodegenerative-disease-database-1.0.0/
```

Place the released model files here:

```text
<CoDiT checkout>/ours/gait/saved_models/gait_16.pt
<CoDiT checkout>/ours/gait/saved_models/gait_18.pt
<CoDiT checkout>/ours/gait/saved_models/gait_20.pt
```

Create CoDiT's isolated Python environment by following `ours/gait/README.md`.
The original pinned dependencies are old (Python 3.6.13 and PyTorch 1.9.0).
Keep that environment separate from the toolbox environment. Install this
toolbox and its benchmark dependencies in the environment used to launch the
runner. The runner accepts `--codit-python` to point to the Python executable
for CoDiT.

## Run the comparison

From the toolbox repository root:

```powershell
python -m benchmarks.run_experiment_8 `
  --codit-root ..\time-series-OOD-main `
  --codit-python C:\path\to\conda\envs\gait\python.exe
```

The notebook explicitly calls the idempotent data setup before starting the
comparison. The command line runner does the same automatically. It checks for
all 43 records used by the protocol and prints that it is reusing them when
they are already present; incomplete data trigger a fresh archive download.

By default, the runner uses 100 CoDiT randomizations, all three released
window lengths, and CoDiT's published healthy split and severe patient subset.
It runs upstream `check_OOD_gait.py` separately for each window length, reads
the official Fisher-combined per-window output, and then fits and scores the
proposed detector over the same records. CoDiT selects CPU or CUDA according
to its own environment. Use `--data`, `--checkpoint-dir`, `--n-randomizations`,
and `--output` to override paths or run settings.

Outputs are written to `benchmarks/outputs/experiment_8/`: CoDiT's raw outputs,
matched window scores, a summary JSON, an AUROC PDF, and a LaTeX table. CoDiT's
Fisher statistic is high for in-distribution records; the output CSV reverses
its sign so all stored anomaly scores increase with OOD evidence. Matching uses
the raw zero-based window start. The proposed method uses the full preceding
observer history and filter warmup, so comparison windows start after that
required context is available.

The comparison is descriptive. It uses repeated windows from a small number of
subjects and does not claim the paper's large-sample conformal guarantee.
