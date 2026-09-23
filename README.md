# Learning Spatial Representations via Self-Supervised Learning

This repository holds the code and configurations for the master's thesis of the same name. An
agent walks at random through a simulated 3D environment. A convolutional autoencoder compresses
each camera frame to a 64-number latent. A recurrent model reads the sequence of latents and is
trained to predict its own next code. Its place code then carries position, and many of its units
behave like place cells. The repository trains every model variant of the thesis, computes the
measures the thesis reports, and trains the navigation agents that use the frozen codes.

[QUICKSTART.md](QUICKSTART.md) walks through a first run.

## Install

Python 3.11 or newer; the code was developed on 3.12.

```bash
conda create -n placecell python=3.12
conda activate placecell
pip install -e ".[rl,jax,dev]"
```

`rl` adds Stable-Baselines3 for navigation. `jax` adds JAX and the JAXenstein simulator (pinned
commit) for the museum environment. `dev` adds pytest and ruff.

MiniWorld, which renders the WallGap environment, needs an OpenGL context. On macOS it runs only
while the display is awake (for long runs, `caffeinate -d -i pc ...`). JAXenstein needs no display.

## Smoke test

```bash
pc pipeline --config configs/experiment/smoke_jaxenstein.yaml   # museum, no display, about 2 min
pc pipeline --config configs/experiment/smoke_miniworld.yaml    # WallGap, needs a display
pytest tests/
```

Each smoke run executes all seven stages (collect, split, visual encoder, encode, place-cell
model, evaluate, analyze) on a few short episodes and writes to `smoke/`, which git ignores.

## How a run works

`pc pipeline --config <config>` runs the stages in order. Each stage writes an immutable artifact
under `artifacts/` and a run record under `runs/by_id/`. A stage whose inputs and settings match an
existing artifact reuses it (`policies.artifact_reuse: reuse_if_config_match`), so all WallGap
conditions share one collected dataset, split and visual encoder, and only the place-cell model is
trained again. `pc inspect-config --config <config>` prints the resolved configuration, and
`-o key=value` overrides any setting.

`--place-tag <name>` gives the trained model a name that later commands accept as `tag:<name>`.
The commands below tag the seed-42 model of a condition with the condition's name and the other
seeds as `<condition>_seed1` and `<condition>_seed2`.

## Reproducing the thesis

Every condition of the thesis has one file in `configs/thesis/`. It inherits
`configs/thesis/baseline.yaml` (or `museum.yaml`) and sets only the keys in which the condition
differs. For a condition `C`:

```bash
pc pipeline -c configs/thesis/C.yaml --place-tag C
pc collect-representations -c configs/thesis/C.yaml -o reuse.place_model_artifact_id=tag:C
pc measures -c configs/thesis/C.yaml -o reuse.place_model_artifact_id=tag:C
```

The first command trains, evaluates and analyzes the model. The second stores its forward pass
over the first 512 episodes of the training, validation and test splits. The third writes
`measures/<condition>__seed<seed>__<model>.csv` and a per-unit table beside it.

| Thesis part | Configs | Notes |
|---|---|---|
| Baseline model; Results, emergence of localized fields, traversals, population code | `baseline.yaml`, `untrained.yaml` | three seeds |
| Prediction target and regularizers | `no_prediction`, `no_ema`, `next_visual_latent_target`, `same_step_no_predictor`, `same_step_with_predictor`, `reconstruction_target`, `variance_regularizer_off`, `covariance_regularizer_off`, `both_regularizers_off` | three seeds, except next visual latent target and reconstruction target |
| Weight decay | `weight_decay_0`, `weight_decay_1e-6`, `weight_decay_1e-5_retrained`, `weight_decay_1e-4`, `both_regularizers_off_no_weight_decay`, `no_prediction_no_weight_decay`, `reconstruction_target_no_weight_decay`, `no_competition_no_weight_decay` | one seed; see below for the retrained reference |
| Self-motion and history | `actions_only`, `self_motion_only`, `no_motion_input`, `feedforward_encoder` | three seeds, feedforward encoder one |
| Competition | `winners_1`, `winners_5`, `winners_26`, `winners_51`, `winners_128`, `no_competition`, `l1_0.003`, `l1_0.01`, `l1_0.03` | `no_competition` and `l1_0.003` three seeds; the baseline is the 10-winner point |
| Competition after training | `competition_added_after_training`, `retrofitted_competition` | need the trained `no_competition` model; retrofitted three seeds, one parent per seed; see below |
| Encoder and predictor width, cell type, code width | `width_<encoder>_<predictor>` (8 files; 1,024 / 512 is the baseline), `cell_type_gru_gru`, `cell_type_gru_lstm`, `cell_type_lstm_lstm`, `code_width_128`, `code_width_256` | one seed |
| Objects removed | `objects_removed` | collects its own dataset |
| Second environment | `museum`, `museum_untrained` | three seeds |
| Navigation | `configs/thesis/navigation/` (28 files) | see Navigation |

Tables and the columns that hold their values:

| Thesis table | Measures CSV columns |
|---|---|
| Scorecard: decoding and single units (mean ± SD) | `decode_encoder.place_codes_position_ridge_rmse`, `..._ridge_r2`, `..._position_mlp_rmse`, `..._heading_ridge_median_error_degrees`, `silent_fraction`, `information_above_null95_all_units_mean`, `field_area_percent_finite_median`, `_q25`, `_q75` |
| Single-unit measures of every model | `spatial_information_bits_all_units_mean` and the `*_all_units_zero_filled_median` columns of fields, split-half, traversal response and variance explained |
| Population measures of every model | `participation_ratio`, `step_trustworthiness`, `step_continuity`, `partial_spearman_euclidean`, `mean_units_per_visited_location` |
| Decoding at five stages | `decode_<source>_*`, after collecting all five sources (below) |
| Full ablation tables (weight decay, widths, code width, cell types) | the `full_*` columns, `participation_ratio`, `step_trustworthiness` |
| Responses on single traversals | the `traversal_*` columns |
| Navigation success | `pc navigation-measures` (below) |

### Seeds

The default training seed is 42. The thesis reports seeds 42, 1 and 2 where the table says n = 3.
Collection, split and visual encoder keep seed 42 for every run.

```bash
pc pipeline -c configs/thesis/C.yaml -o seed.training_seed=1 --place-tag C_seed1
pc collect-representations -c configs/thesis/C.yaml -o reuse.place_model_artifact_id=tag:C_seed1
pc measures -c configs/thesis/C.yaml -o reuse.place_model_artifact_id=tag:C_seed1
pc summarize measures/*.csv --output measures/summary.csv
```

`pc summarize` groups the rows by condition and writes n, mean and sample SD of every column. A
column is left out for a condition when any of its runs has no value.

### Conditions that depend on another model

- `retrofitted_competition` starts from the weights of the trained `no_competition` model
  (`reuse.place_model_artifact_id: tag:no_competition`), freezes the recurrent encoder, and trains
  the linear head, the predictor and the embeddings for 128 epochs with 10 winners. Train
  `no_competition` with `--place-tag no_competition` first. For seed 1:
  `-o seed.training_seed=1 -o reuse.place_model_artifact_id=tag:no_competition_seed1`.
- `competition_added_after_training` trains nothing. It keeps the 10 largest values of each step
  of the `no_competition` code at read time (`measures.read_time_top_k: 10`):
  `pc measures -c configs/thesis/competition_added_after_training.yaml -o reuse.place_model_artifact_id=tag:no_competition`.
- `weight_decay_1e-5_retrained` is the baseline recipe trained a second time as the matched
  reference of the weight-decay rows. Its config forces a new model and runs only the training,
  evaluation and analysis stages, so pass the WallGap data:
  `pc pipeline -c configs/thesis/weight_decay_1e-5_retrained.yaml --dataset <encoded id> --split <split id>`
  (the folder names in `artifacts/datasets/encoded/` and `artifacts/splits/`).

### Decoding inputs and stages

`pc collect-representations` stores only the place code by default. For the table of decoding at
five stages, collect all five sources; `pc measures` then decodes each of them:

```bash
pc collect-representations -c configs/thesis/C.yaml -o reuse.place_model_artifact_id=tag:C \
  -o 'representation_collection.sources=[encoder.hidden_state,encoder.pre_sparsifier,encoder.place_codes,predictor.hidden_state,predictor.place_codes]'
```

For the feedforward encoder, use `encoder.backbone_output` in place of `encoder.hidden_state`.
`pc measures --inputs` also decodes the visual latent and the stack of the current and 15
preceding latents, from the latents stored with the forward pass.

## Navigation

The navigation agents read the frozen baseline model. Train it first with
`pc pipeline -c configs/thesis/baseline.yaml --place-tag baseline`; the place-code, recurrent-state
and visual-latent configs refer to `tag:baseline`. One config exists per algorithm (`ppo`, `dqn`
with 3-step returns), input and goal (`north` at (0, 18), `south` at (18, -22)). Each sets the
learning rate and the entropy coefficient (PPO) or final exploration rate (DQN) of its input.
Policy seeds are 7, 8 and 9:

```bash
pc downstream-train -c configs/thesis/navigation/ppo_place_code_north.yaml -o seed=7
pc navigation-measures runs/by_id/ppo_* runs/by_id/dqn_* --epsilon 0.05 --output navigation/policies.csv
pc summarize navigation/policies.csv --output navigation/summary.csv
```

Inputs: `place_code` (the k-winners code, L2-normalized per step), `recurrent_state` (top LSTM
layer), `visual_latent`, `pixels` (60 x 80 RGB through a CNN trained with the policy),
`coordinates` (true position / 36), `synthetic_place_cells` and `synthetic_grid_cells` (512 cells
computed from the true position). Every input is joined with the scaled goal position and the
sine and cosine of the heading. Episodes last at most 1,024 steps and start anywhere in the four
rooms. A policy that reaches the goal receives 1 - 0.2 x step / 1,024. The model is loaded from
its checkpoint with the lowest validation decoding error (`models.place_model_checkpoint:
best_primary`); all other measures use the last checkpoint.

`pc navigation-measures` reads each run's final deterministic evaluation (30 episodes from saved
start poses), counts failures whose last 64 poses repeat with period one or two, bins the training
history into 100,000-step bins, and reports the first bin of three in a row with at least 50% and
80% success. With `--epsilon 0.05` it runs the same starts again, taking a random action on 5% of
steps.

## What the measures compute

`pc measures` reads the stored forward pass of one model and its analysis report.

- Single units, first 512 test episodes: rate maps of the positive part of each unit on 60 x 60
  bins, Gaussian smoothing sigma 0.3 bins. Skaggs information against 999 within-episode circular
  shifts (at least 5% of the episode); bits above null is max(information - 95th percentile of
  the shifts, 0), averaged over all units with silent units as zero. A field is every bin above
  20% of the peak, counted as 4-connected components; field area is divided by the number of
  visited bins. Split-half correlation uses the raw signed codes, averaged over the even/odd split
  and 20 random balanced episode splits. Variance explained is 1 - SS_res / SS_tot of an
  unsmoothed rate map from even episodes predicting odd episodes, and back. Traversal response is
  the share of field traversals that reach 30% of the unit's 99.5th-percentile activity, for units
  with at least 5 traversals.
- Decoding: ridge regression with train-set scaling and 10 penalties from 1e-6 to 1e3, chosen on
  512 validation episodes and scored on 512 test episodes; an MLP with two hidden layers of 128
  units (AdamW, at most 200 epochs, early stopping on validation). Steps 15 to the end of each episode. Heading is decoded as sine
  and cosine and scored as the median angular error.
- Similarity: cosine similarity of 8 million pairs from different episodes (first 128 test
  episodes, every 4th step) against distance; the half-distance is where the mean curve falls to
  half its value at contact, for all pairs and for pairs whose headings differ by less than 45 or
  more than 135 degrees.
- Traversals, all test episodes, from a CPU forward pass: 60 x 60 unsmoothed maps, field 20% and
  core 50% of the peak, response 30% of the 99.5th percentile of the unit's activity, eight
  heading sectors with the preferred sector chosen on even episodes and scored on odd ones,
  999 shifts (`measures.traversal_shifts`; the museum config uses 100).
- Population, copied from the analysis report of all test episodes: participation ratio on
  5,000 random steps, trustworthiness and continuity (k = 15) on 4,096 random steps, the partial
  Spearman correlation of code distance (1 - Pearson) and Euclidean distance controlling for
  path distance on 20 x 20 bins, and the mean number of units with a field per visited bin.

## Running on the cluster

## Compute

Rough figures for one condition at full size, measured on the runs behind the thesis:

| Step | Resources | Time |
|---|---|---|
| Collection (8,192 episodes of 2,048 steps) | WallGap: one GPU with graphics (EGL); museum: one GPU | about 3 h |
| Visual encoder | one GPU | about 2 h |
| Place-cell model (128 epochs) | one GPU (H100 class) | about 4 h |
| `pc analyze` on all test episodes | up to 181 GB RAM for the 1,024-unit encoder, 16 CPUs | about 5 h |
| `pc measures` | CPU; the traversal null keeps 999 shifted indices of all 5 million test steps (about 40 GB) | not timed at full size |
| Navigation, 3 million steps per policy | CPU, pixels on a GPU; seven policies in parallel on one machine | PPO about 6 to 7 h, DQN about 9 to 11 h per batch of seven |

A representation set with the place code of 3 x 512 episodes takes about 3.5 GB.

## Repository layout

```
configs/thesis/              one config per thesis condition
configs/thesis/navigation/   one config per navigation algorithm, input and goal
configs/experiment/          WallGap and museum base recipes, smoke configs
configs/<group>/             shared blocks (environment, collection, vision, model, analysis, ...)
configs/downstream/          navigation base configs (PPO, DQN)
src/placecell_research/
  stages/                    one module per pipeline stage
  collection/ envs/          data collection, MiniWorld WallGap and JAXenstein museum
  vision/                    convolutional autoencoder
  spatial_model/ objectives/ training/   place-cell model, losses, training loop
  evaluation/ analysis/      pc evaluate and pc analyze
  measures/                  pc measures, pc summarize, pc navigation-measures
  downstream/                navigation agents
  launch/                    the pc command line
scripts/                     standalone scripts: frozen-code controls, traversal and navigation evaluation, SLURM environment setup
tests/                       pytest suite
```
