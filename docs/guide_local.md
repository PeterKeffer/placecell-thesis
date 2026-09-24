# Your own computer

Linux or macOS, with or without a GPU. The smoke test runs on any of them. The full reproduction
needs about 750 GPU hours for the models and 700 hours for navigation
([Compute](../README.md#compute)), so on one computer run only a few conditions, or use a cluster
([hpc3](guide_hpc3.md), [any SLURM cluster](guide_slurm.md)). [README.md](../README.md) explains
what each run computes.

## 1. Set up

```bash
git clone <this repository> placecell-thesis && cd placecell-thesis
bash scripts/setup_env.sh
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ~/miniforge3/envs/placecell
pc doctor
```

`setup_env.sh` uses your conda, or installs Miniforge into `--prefix` (default `~/miniforge3`)
when it finds none, and creates the environment in `<prefix>/envs/placecell`
([what it installs](../README.md#install)). It ends by printing the line that activates the
environment; the third line above is that line for the default prefix. `--dry-run` prints every
step and runs none.

| Computer | PyTorch and JAX | MiniWorld (WallGap) renders through |
|---|---|---|
| Mac | PyTorch on the Apple GPU (MPS), JAX on the CPU | the window system; the display must be awake |
| Linux with an NVIDIA GPU | CUDA builds | the NVIDIA EGL driver (system `libEGL.so.1`) |
| Linux without a GPU | CPU builds | a software Mesa build with EGL |

`--gpu cpu` or `--gpu cuda` overrides the guess. On a Mac, `caffeinate -u -t 5` wakes the display
and `caffeinate -d -i <command>` keeps it awake while MiniWorld runs. On Linux without a GPU, set
the variables that the CPU branch of `scripts/slurm/env_miniworld.sh` sets for a software Mesa
build (`__EGL_VENDOR_LIBRARY_FILENAMES`, `LIBGL_DRIVERS_PATH`, `LD_LIBRARY_PATH`). The museum
environment (JAXenstein) renders in JAX and needs neither.

## 2. Smoke test

```bash
pc reproduce --profile local --smoke --only baseline
pytest tests/
```

The first line runs the baseline through every stage at toy size: data, visual encoder, model,
stored forward pass, measures, one navigation policy and the summaries. It writes to `smoke/`. The
numbers mean nothing; the run shows that everything works. Without `--only` the smoke runs all 154
jobs of the plan, about 90 minutes and 1.7 GB on an M-series Mac.

## 3. Reproduce

```bash
pc reproduce --profile local --dry-run
pc reproduce --profile local --only baseline,no_competition --seeds 42
pc reproduce --profile local
```

The first line prints the plan. The second runs two conditions with one seed; the runs they depend
on are added. The third runs the whole plan, one job after the other. If a job fails, the command
stops; fix the cause and run it again, and finished jobs are skipped.

## 4. Results

`measures/summary.csv` holds one row per condition (mean and SD over seeds), and
`measures/<condition>__seed<seed>__<model>.csv` the measures of one model. Navigation results are in
`navigation/summary.csv` and `navigation/policies.csv`. Models and reports are in `artifacts/`, run
records in `runs/by_id/`. [README.md](../README.md#reproducing-the-thesis) maps the columns to the
thesis tables.
