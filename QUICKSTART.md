# Quickstart

Pick the guide for the machine you run on. Each one lists every command from the clone to the
results.

- [Your own computer](docs/guide_local.md): Linux or macOS, with or without a GPU. Good for the
  smoke test and a few conditions.
- [The lab cluster hpc3](docs/guide_hpc3.md): the cluster the thesis ran on, where spack provides
  conda, git and mesa-glu.
- [Any other SLURM cluster](docs/guide_slurm.md): with a site script of your own for `module load`
  or spack lines.

A first run on your own computer:

```bash
git clone <this repository> placecell-thesis && cd placecell-thesis
bash scripts/setup_env.sh
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ~/miniforge3/envs/placecell
pc doctor
pc reproduce --profile local --smoke --only baseline
```

The last line runs the baseline through every stage at toy size and writes to `smoke/`. On a Mac,
MiniWorld needs an awake display: run the last line as `caffeinate -d -i pc reproduce ...`.
[README.md](README.md) explains what each run computes.
