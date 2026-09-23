# Quickstart

From a fresh clone to the measures of one small model. [README.md](README.md) has the full
reproduction.

## 1. Install

```bash
conda create -n placecell python=3.12
conda activate placecell
pip install -e ".[rl,jax,dev]"
pc --help
```

## 2. Smoke test

The museum environment (JAXenstein) runs without a display:

```bash
pc pipeline --config configs/experiment/smoke_jaxenstein.yaml
```

This collects 16 short episodes, trains the visual encoder and a small place-cell model for one
epoch, and evaluates and analyzes it, in about two minutes on a laptop CPU. Outputs go to
`smoke/artifacts/` and `smoke/runs/`. The WallGap version (`smoke_miniworld.yaml`) needs a
display, because MiniWorld renders with OpenGL.

## 3. One thesis condition, shrunk

The thesis configs train on 8,192 episodes of 2,048 steps. These overrides shrink the baseline to
a few minutes and write to `smoke/`:

```bash
SMALL=(
  -o tracking.artifact_root=smoke/artifacts -o tracking.run_root=smoke/runs
  -o environment.episode_length=256 -o collection.episodes=16 -o collection.episode_length=256
  -o splits.train_fraction=0.5 -o splits.validation_fraction=0.25 -o splits.test_fraction=0.25
  -o splits.constraints.minimum_episodes_per_split=1
  -o vision.epochs=1 -o vision.batch_size=16 -o vision.data_loader_num_workers=0
  -o vision.data_loader_persistent_workers=false -o vision.data_loader_pin_memory=false
  -o 'spatial_model.encoder.layer_sizes=[32]' -o 'spatial_model.predictor.layer_sizes=[32]'
  -o spatial_model.training.code_dim=64 -o spatial_model.training.epochs=1
  -o spatial_model.training.batch_size=2 -o spatial_model.training.num_workers=0
  -o evaluation.eval_every_n_epochs=1 -o evaluation.nonlinear_decode_enabled=false
  -o analysis.decode_nonlinear_enabled=false -o analysis.num_bins_x=16 -o analysis.num_bins_y=16
  -o analysis.rate_map_panel_top_k=8 -o analysis.rate_map_grid_top_k=8
)
pc pipeline -c configs/thesis/baseline.yaml "${SMALL[@]}" --place-tag baseline
pc collect-representations -c configs/thesis/baseline.yaml "${SMALL[@]}" -o reuse.place_model_artifact_id=tag:baseline
pc measures -c configs/thesis/baseline.yaml "${SMALL[@]}" -o reuse.place_model_artifact_id=tag:baseline -o measures.output_dir=smoke/measures
```

For the museum, use `configs/thesis/museum.yaml` and add `-o collection.num_parallel_envs=16`.
Any other file in `configs/thesis/` runs the same way; the WallGap ones reuse the data collected
for the baseline.

## 4. Read the result

```bash
pc summarize smoke/measures/*.csv --output smoke/measures/summary.csv
```

`smoke/measures/` holds one CSV row per model (decoding, single units, similarity, traversals,
population measures) and a per-unit table. At this size the numbers mean nothing; they only show
that every stage runs.

## 5. A navigation agent

```bash
pc downstream-train -c configs/thesis/navigation/ppo_place_code_north.yaml \
  -o tracking.artifact_root=smoke/artifacts -o tracking.run_root=smoke/runs \
  -o environment.episode_length=64 -o training.total_timesteps=256 -o training.n_envs=2 \
  -o training.n_steps=64 -o training.batch_size=32 -o training.final_eval_episodes=2
pc navigation-measures smoke/runs/by_id/ppo_place_code_north__* --output smoke/measures/navigation.csv
```

This needs the tagged baseline from step 3 and a display (MiniWorld).

## 6. Clean up

```bash
rm -rf smoke
```
