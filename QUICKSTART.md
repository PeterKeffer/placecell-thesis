# Quickstart

Three ways to run the thesis: on your own computer, on the lab cluster the thesis ran on, and on
any other SLURM cluster. Each starts the same way: set up the software, activate it, check it.
[README.md](README.md) explains what each run computes.

## 1. Your own computer (macOS or Linux)

```bash
git clone <this repository> placecell-thesis && cd placecell-thesis
bash scripts/setup_env.sh
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ~/miniforge3/envs/placecell
pc doctor
pc reproduce --profile local --smoke
```

`setup_env.sh` creates the conda environment `<prefix>/envs/placecell` (`--prefix`, default
`~/miniforge3`) with Python 3.12, PyTorch, JAX, JAXenstein, MiniWorld and this package, at the
tested versions in `constraints.txt`. It uses your conda, or installs Miniforge into `--prefix`
when it finds none. On a Mac it installs the default PyTorch (Apple GPU through MPS) and JAX on the
CPU. On Linux it installs CUDA builds when it finds a GPU or SLURM, CPU builds otherwise;
`--gpu cpu` or `--gpu cuda` overrides the guess. `--dry-run` prints every step and runs none. It
ends with `pc doctor`, which checks Python, the torch and JAX devices, one rendered MiniWorld and
museum frame, and write access, and then prints the line that activates the environment; the
third line above is that line for the default prefix.

`--smoke` runs every stage of every thesis condition on 16 short episodes and small models (154
jobs, about 90 minutes and 1.7 GB on an M-series Mac) and writes to `smoke/`. It shows that
everything runs; the numbers mean nothing. `--only baseline,no_competition` restricts the plan to
some conditions (runs they depend on are added), `--seeds 42` to one seed, `--dry-run` prints the
plan. MiniWorld needs an awake display on macOS: run `caffeinate -u -t 5` to wake it, then
`caffeinate -d -i pc reproduce --profile local --smoke` to keep it awake.

Without `--smoke` the same command runs the full reproduction, every job in order on this machine:
about 750 GPU hours for the models and 700 hours of navigation (see Compute), so use a cluster
(sections 2 and 3) unless you run only a few conditions with `--only`. If it stops, run the command
again: finished jobs are skipped.

## 2. The lab cluster (hpc3, partition klab-gpu)

On the login node:

```bash
git clone <this repository> placecell-thesis && cd placecell-thesis
bash scripts/setup_env.sh --prefix <a folder on the lab share>/miniforge3
# run the activation line that setup_env.sh prints at the end
pc doctor --no-render
pc reproduce --profile hpc3 --dry-run
pc reproduce --profile hpc3 --smoke --only baseline
pc reproduce --profile hpc3
```

Keep the environment (about 8 GB) and the checkout on the lab share, not in your home folder;
the environment goes to `--prefix` even when conda comes from a module. The login node has no GPU,
so `pc doctor --no-render` skips rendering; every job checks CUDA, EGL and the MiniWorld textures
itself before it starts. `configs/launcher/hpc3.yaml` holds the cluster's settings:

- GPU jobs run on `klab-gpu` and request a full H100 (`--gres=gpu:H100.80gb:1`). A request that
  could land on a 10 GB MIG slice is refused: MIG slices have no graphics API for MiniWorld and too
  little memory for the model.
- `klab-7` is excluded, because its CUDA fails silently on some days. Add
  `-o launcher.gpu_type=B200` to run on the B200 node klab-8 instead.
- Each GPU job asks for 200 GB, 16 CPUs and 48 h, the envelope of the thesis runs. `pc measures`
  runs as CPU jobs on `klab-cpu` (8 CPUs, 64 GB, 4 threads for its forward pass).
- At most 2 GPU jobs and 7 CPU jobs of the reproduction run at once. Raise this with
  `-o launcher.max_concurrent_gpu_jobs=N` only when the lab agrees.
- MiniWorld renders through the NVIDIA EGL driver; `libGLU` comes from spack (`mesa-glu`), see
  `scripts/slurm/site/hpc3.sh`.

Jobs activate the environment that ran `pc reproduce`. The profile holds no personal values; if
your account needs an account or QOS, write them into `~/.config/placecell/user.yaml` on the login
node:

```yaml
launcher:
  account: <account>
  qos: <qos>
```

From a laptop the same works over SSH. The laptop's `~/.config/placecell/user.yaml`:

```yaml
remote:
  host: <ssh alias of the login node>
  repo_root: <checkout on the lab share>
  python: <the python path printed at the end of setup_env.sh on the login node>
```

`pc reproduce --profile hpc3 --remote` then copies the checkout to `repo_root`, freezes the code
in `.code_snapshots/` so that later copies cannot change jobs that are queued or running, and
submits from the login node. `pc hpc -c configs/thesis/<condition>.yaml` does the same for a single
run and streams its log; `pc hpc-logs --job-id <id>` reattaches.

Watching and stopping jobs:

```bash
squeue --me -o '%.10i %.45j %.8T %.10M %R'
tail -f runs/slurm_logs/pc.train.baseline.s42_<jobid>.out
scancel <jobid>
squeue --me -h -o '%i %j' | awk '$2 ~ /^pc\./ {print $1}' | xargs -r scancel
```

Reproduction jobs are named `pc.<step>`, for example `pc.train.baseline.s42`. `scancel <jobid>`
stops one job; jobs that wait for it are cancelled with it. The last line stops the whole
reproduction and nothing else. Cancel only your own jobs, by id or with that filter. Running
`pc reproduce` again submits only jobs that are neither finished nor queued, and queues them behind
those still waiting.

Results: `measures/summary.csv` (one row per condition, mean and SD over seeds),
`measures/<condition>__seed<seed>__<model>.csv`, `navigation/summary.csv` and
`navigation/policies.csv`. Models and reports are in `artifacts/`, run records in `runs/by_id/`,
logs in `runs/slurm_logs/`.

## 3. Any other SLURM cluster

Use `--profile slurm`. Personal values go into `~/.config/placecell/user.yaml` (or the variables
`PLACECELL_SLURM_PARTITION`, `PLACECELL_SLURM_ACCOUNT`, `PLACECELL_SLURM_QOS`):

```yaml
launcher:
  partition: <GPU partition>
  account: <account, if required>
  qos: <qos, if required>
```

Values that hold for everyone on the cluster go into a copy of `configs/launcher/slurm.yaml`, for
example `configs/launcher/mycluster.yaml`, run as `--profile mycluster`:

- `gpu_type`: the GPU type in `--gres`. MiniWorld needs a full GPU with the NVIDIA driver.
- `mig_gpu_types`: MIG slice types of the cluster, if any; requests that could land on one are
  refused.
- `exclude_nodes`, `constraint`: nodes to avoid, node features to require.
- `job_resources.measures.partition` and `job_resources.summary.partition`: a CPU partition, if
  GPU partitions refuse jobs without a GPU.
- `max_concurrent_gpu_jobs`, `max_concurrent_cpu_jobs`: how many reproduction jobs run at once.
- `site_env_script`: a script under `scripts/slurm/` that every job sources before the environment
  checks, for `module load` or spack lines (the lab's is `site/hpc3.sh`).

GPU nodes need the system `libEGL.so.1` (libglvnd) and the NVIDIA EGL vendor file, which the
NVIDIA driver installs. Set `PLACECELL_EGL_LIBRARY` if `libEGL.so.1` is not in `/usr/lib64` or
`/usr/lib/x86_64-linux-gnu`. MiniWorld jobs on CPU nodes need a software Mesa build; set
`PLACECELL_MESA_PREFIX` to it (under `exports:` in the user file).

## How jobs protect the nodes

- Every job requests memory, CPUs and time, caps BLAS and OpenMP threads, and gets 120 s of
  warning before its time runs out. On a timeout or `scancel` the batch script forwards TERM to
  the whole process group, waits, kills what is left, and deletes its temporary folder.
- Each job has its own temporary folder and caches, reached through a short `TMPDIR` link.
- GPU jobs run a CUDA check first and stop within a minute on a broken GPU.
- Collection runs in spawned worker processes, one environment per worker, never more workers
  than the job has CPUs. A watchdog stops workers that make no progress for 300 s, a memory
  watchdog stops them above 90% of the job's memory, and previews render in their own process
  with a 30 s timeout.
- A job whose input failed is cancelled instead of waiting forever, and dependency lanes keep the
  number of running reproduction jobs at the configured limit.

## Compute

Figures from the thesis runs, per training seed at full size:

| Job | Resources | Time |
|---|---|---|
| Data: collection, split, visual encoder, encoding (3 chains: WallGap, museum, objects removed) | 1 GPU | about 5 h each |
| Place-cell model (128 epochs), `pc evaluate`, `pc analyze` on all test episodes | 1 H100-class GPU, up to 200 GB RAM | about 9 h |
| Stored forward pass (`pc collect-representations`) | 1 GPU | minutes |
| `pc measures` | CPU; holds the codes of all test episodes (about 10 GB for 512 units) | not timed at full size |
| Navigation policy, 3 million steps | CPU; the pixel input on a GPU | PPO 6 to 7 h, DQN 9 to 11 h (seven in parallel on one machine) |

The plan has 83 trainings, 84 measure jobs and 84 navigation policies: about 750 GPU hours for the
models, which takes a little over two weeks with two GPUs, and about 700 hours for navigation.
Disk, estimated from the sizes of the thesis artifacts: a raw dataset takes over 10 GB, an encoded
one 2.6 GB, a trained model 1.4 GB, a stored forward pass 3.5 GB with the place code alone and
about 20 GB with all five stages; in total well under 1 TB.
