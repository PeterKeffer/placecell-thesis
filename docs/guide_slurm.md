# Any SLURM cluster

For a SLURM cluster other than the lab's ([hpc3](guide_hpc3.md) has its own guide). The repository
needs three things from you: a site script if the cluster provides tools through `module load` or
spack, a launcher profile with the settings everyone on the cluster shares, and your personal
values. [README.md](../README.md) explains what the jobs compute, how they protect the nodes and
how much compute they need.

## 1. Site script (clusters with modules or spack)

Skip this step if conda is already on your PATH or the cluster provides none; `setup_env.sh` then
uses your conda or installs Miniforge. Otherwise write `scripts/slurm/site/<cluster>.sh` with the
proxy, the module or spack lines and any exports your cluster needs:

```bash
export http_proxy=http://<proxy>:<port> https_proxy=http://<proxy>:<port>
_placecell_shell_options="$(shopt -po nounset pipefail || true)"
case "$-" in *e*) _placecell_shell_options+=$'\nset -e' ;; esac
set +euo pipefail
module load miniconda3            # or: spack load miniconda3
module load git
eval "${_placecell_shell_options}"
unset _placecell_shell_options
```

`setup_env.sh --site <cluster>` sources it before anything else and uses the conda it puts on PATH;
it never installs Miniforge then. Every job sources it before it activates the environment. The
option lines keep the caller's shell settings, since the script runs both in your login shell and
in jobs under `set -euo pipefail`, where module and spack functions often fail. If MiniWorld needs
a `libGLU` from a module, export its `lib` folder as `PLACECELL_EXTRA_LIBRARY_PATH`.
`scripts/slurm/site/hpc3.sh` is a complete example with spack.

## 2. Set up

```bash
git clone <this repository> placecell-thesis && cd placecell-thesis
bash scripts/setup_env.sh --site <cluster> --prefix <a folder on shared storage>
# run the activation line that setup_env.sh prints at the end
pc doctor --no-render
```

Leave out `--site <cluster>` without a site script. The environment (about 8 GB) goes to
`<prefix>/envs/placecell`, also when conda comes from a module; keep it and the checkout on storage
that the compute nodes see. Temporary files and package caches go to `<prefix>/setup_tmp` during
the install and are removed at the end. On a login node with `sbatch` the script installs CUDA
builds of PyTorch and JAX. `pc doctor --no-render` skips rendering on a login node without a GPU;
every job checks CUDA, EGL and the MiniWorld textures itself before it starts.

## 3. Launcher profile and personal values

Copy `configs/launcher/slurm.yaml` to `configs/launcher/<cluster>.yaml` for the settings everyone on
the cluster shares:

- `partition`, `account`, `qos`: when they are the same for everyone.
- `gpu_type`: the GPU type in `--gres`. MiniWorld needs a full GPU with the NVIDIA driver.
- `mig_gpu_types`: MIG slice types of the cluster, if any; requests that could land on one are
  refused.
- `exclude_nodes`, `constraint`: nodes to avoid, node features to require.
- `job_resources.measures.partition` and `job_resources.summary.partition`: a CPU partition, if
  GPU partitions refuse jobs without a GPU.
- `max_concurrent_gpu_jobs`, `max_concurrent_cpu_jobs`: how many reproduction jobs run at once.
- `site_env_script: site/<cluster>.sh`, the site script of step 1.

Personal values go into `~/.config/placecell/user.yaml` (or the variables
`PLACECELL_SLURM_PARTITION`, `PLACECELL_SLURM_ACCOUNT`, `PLACECELL_SLURM_QOS`) and win over the
profile:

```yaml
launcher:
  partition: <GPU partition>
  account: <account, if required>
  qos: <qos, if required>
exports:
  PLACECELL_MESA_PREFIX: <software Mesa build, for MiniWorld on CPU nodes>
```

GPU nodes need the system `libEGL.so.1` (libglvnd) and the NVIDIA EGL vendor file, which the NVIDIA
driver installs. Set `PLACECELL_EGL_LIBRARY` if `libEGL.so.1` is not in `/usr/lib64` or
`/usr/lib/x86_64-linux-gnu`. MiniWorld jobs on CPU nodes need the software Mesa build in
`PLACECELL_MESA_PREFIX`.

## 4. Reproduce

```bash
pc reproduce --profile <cluster> --dry-run
pc reproduce --profile <cluster> --smoke --only baseline
pc reproduce --profile <cluster>
```

The first line prints the plan and writes the batch scripts without submitting; check the
`#SBATCH` lines and the `source` lines there. The second submits the baseline chain at toy size (7
small jobs, output in `smoke/`). The third submits the whole plan as a SLURM dependency chain.
Running it again submits only jobs that are neither finished nor queued. `--only <conditions>` and
`--seeds 42` narrow the plan. Without a profile of your own, `--profile slurm` uses the neutral
defaults and your user file.

## 5. Watch and stop jobs

```bash
squeue --me -o '%.10i %.45j %.8T %.10M %R'
tail -f runs/slurm_logs/pc.train.baseline.s42_<jobid>.out
scancel <jobid>
squeue --me -h -o '%i %j' | awk '$2 ~ /^pc\./ {print $1}' | xargs -r scancel
```

Reproduction jobs are named `pc.<step>`, for example `pc.train.baseline.s42`. `scancel <jobid>`
stops one job and the jobs that wait for it. The last line stops the whole reproduction and nothing
else.

## 6. Results

`measures/summary.csv` holds one row per condition (mean and SD over seeds), and
`measures/<condition>__seed<seed>__<model>.csv` the measures of one model. Navigation results are
in `navigation/summary.csv` and `navigation/policies.csv`. Models and reports are in `artifacts/`,
run records in `runs/by_id/`, logs in `runs/slurm_logs/`.
[README.md](../README.md#reproducing-the-thesis) maps the columns to the thesis tables.

## 7. From a laptop

The laptop's `~/.config/placecell/user.yaml`:

```yaml
remote:
  host: <ssh alias of the login node>
  repo_root: <checkout on the cluster>
  python: <the python path printed at the end of setup_env.sh on the login node>
```

`pc reproduce --profile <cluster> --remote` then copies the checkout and submits from the login
node; `pc hpc -c configs/thesis/<condition>.yaml -o launcher=<cluster>` does the same for one run
and streams its log ([Running on a cluster](../README.md#running-on-a-cluster)).
