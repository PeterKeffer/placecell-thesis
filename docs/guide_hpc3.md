# The lab cluster (hpc3)

The cluster the thesis ran on. Conda, git and `mesa-glu` (libGLU for MiniWorld) come from spack
there, and the internet only through a proxy. `scripts/slurm/site/hpc3.sh` sets up both; the setup
sources it, and so does every job before it activates the environment. [README.md](../README.md)
explains what the jobs compute, how they protect the nodes and how much compute they need.

## 1. Set up (once)

```bash
ssh <user>@hpc3.rz.uos.de -o ServerAliveInterval=60 -o ServerAliveCountMax=10
tmux new -s placecell
export http_proxy=http://rhn-proxy.rz.uos.de:3128 https_proxy=http://rhn-proxy.rz.uos.de:3128
spack load git@2.31.1
cd /share/klab/<user>
git clone <this repository> placecell-thesis && cd placecell-thesis
bash scripts/setup_env.sh --site hpc3 --prefix /share/klab/<user>/<folder>
```

tmux keeps the setup running when the connection drops; `tmux attach -t placecell` returns to it.
The clone needs the proxy and spack's git before the site script exists; after that the site script
sets both. `setup_env.sh --site hpc3` sources the site script and takes conda from
`spack load miniconda3@4.10.3`; the lab loads conda from spack and never installs it. Conda
installs only Python 3.12, and pip installs everything else at the versions in `constraints.txt`.
The environment (about 8 GB) goes to `<folder>/envs/placecell`. Temporary files and package caches
go to `<folder>/setup_tmp` and are removed at the end, so a full `/tmp` cannot stop the install.

Work on `/share/klab/<user>` (students: `/share/klab/labstudents/<your folder>`): it is
persistent, nothing expires, and the cluster has no quota tools. Keep the checkout, the environment
and all outputs there, not in your home folder.

## 2. Activate (every login)

```bash
cd /share/klab/<user>/placecell-thesis
source scripts/slurm/site/hpc3.sh && eval "$(conda shell.bash hook)" && conda activate /share/klab/<user>/<folder>/envs/placecell
pc doctor --no-render
```

`setup_env.sh` prints an equivalent line with full paths at its end. The login node has no GPU, so
`pc doctor --no-render` skips rendering; every job checks CUDA, EGL and the MiniWorld textures
itself before it starts.

## 3. Reproduce

```bash
pc reproduce --profile hpc3 --dry-run
pc reproduce --profile hpc3 --smoke --only baseline
pc reproduce --profile hpc3
```

The first line prints the plan and writes the batch scripts without submitting. The second submits
the baseline chain at toy size (7 small jobs, output in `smoke/`). The third submits the whole plan
as a SLURM dependency chain. Running it again submits only jobs that are neither finished nor
queued. `--only <conditions>` and `--seeds 42` narrow the plan.

`configs/launcher/hpc3.yaml` holds the lab's settings, so you need no user file:

- `account: klab` and `qos: klab`, the Slurm access of every lab member.
- Lab members use only `klab-gpu`, `klab-l40s` and `klab-cpu`. GPU jobs run on `klab-gpu` with a
  full H100 80GB (`--gres=gpu:H100.80gb:1`); `pc measures` and the summaries run on `klab-cpu`.
- A request that could land on a 10 GB MIG slice of an H100 is refused: MIG slices have no graphics
  API for MiniWorld and too little memory for the model.
- `klab-7` (RTX 6000 cards) is excluded, because its CUDA fails on some days. Add
  `-o launcher.gpu_type=B200` to run on `klab-8` (B200) instead; it takes batch jobs only.
- Each GPU job asks for 200 GB, 16 CPUs and 48 h, the cluster's job limit.
- At most 2 GPU jobs and 7 CPU jobs of the reproduction run at once. Raise this with
  `-o launcher.max_concurrent_gpu_jobs=N` only when the lab agrees.

## 4. Watch and stop jobs

```bash
squeue --me -o '%.10i %.45j %.8T %.10M %R'
tail -f runs/slurm_logs/pc.train.baseline.s42_<jobid>.out
scancel <jobid>
squeue --me -h -o '%i %j' | awk '$2 ~ /^pc\./ {print $1}' | xargs -r scancel
```

Reproduction jobs are named `pc.<step>`, for example `pc.train.baseline.s42`. `scancel <jobid>`
stops one job and the jobs that wait for it. The last line stops the whole reproduction and nothing
else. Cancel only your own jobs, by id or with that filter.

## 5. Results

`measures/summary.csv` holds one row per condition (mean and SD over seeds), and
`measures/<condition>__seed<seed>__<model>.csv` the measures of one model. Navigation results are
in `navigation/summary.csv` and `navigation/policies.csv`. Models and reports are in `artifacts/`,
run records in `runs/by_id/`, logs in `runs/slurm_logs/`.
[README.md](../README.md#reproducing-the-thesis) maps the columns to the thesis tables.

## 6. Debugging on a GPU node

```bash
sinfo -h -o "%N %G"
salloc -p klab-gpu -x klab-8 --gres=gpu:H100.80gb:1 -c 8 --mem 64G -t 04:00:00 srun --pty bash
```

The first line lists the GPU types of each node. The second opens a shell on a full H100 for up to
4 hours, the limit of interactive sessions; `-x klab-8` keeps it off the B200 node, which takes no
interactive jobs. Activate the environment there as in step 2.

## 7. From a laptop

The laptop's `~/.config/placecell/user.yaml`:

```yaml
remote:
  host: <ssh alias of hpc3>
  repo_root: /share/klab/<user>/placecell-thesis
  python: /share/klab/<user>/<folder>/envs/placecell/bin/python
```

`pc reproduce --profile hpc3 --remote` then copies the checkout and submits from the login node;
`pc hpc -c configs/thesis/<condition>.yaml -o launcher=hpc3` does the same for one run and streams
its log ([Running on a cluster](../README.md#running-on-a-cluster)).

## Troubleshooting

- `OSError: [Errno 28] No space left on device` during `pip` or `conda`: `/tmp` is full.
  `setup_env.sh` already keeps its files under `--prefix`; for installs of your own, run
  `export TMPDIR=/share/klab/<user>` first.
- `symbol lookup error: /lib64/libk5crypto.so.3: undefined symbol: EVP_KDF_ctrl` from git: run
  `spack load git@2.31.1`. The site script does this.
- A job waits with `ReqNodeNotAvail`: `sinfo -Nl -n <node>` shows the node's state. A node drained
  with "kill task failed" needs the cluster admin; otherwise the scheduler may need about 15 minutes
  to start the job.
