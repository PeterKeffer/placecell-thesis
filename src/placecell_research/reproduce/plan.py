"""Dependency graph of the thesis reproduction from configs/thesis and configs/reproduce.yaml."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from placecell_research.config.loader import load_raw_config_payload

DATA_CHAIN_SECTIONS = ("environment", "collection", "dataset", "splits", "vision")
NAVIGATION_TOKEN = "navigation"


@dataclass(frozen=True, slots=True)
class Step:
    """One job: a pc command, the config that sets its launcher, and the steps it waits for."""

    name: str
    kind: str
    config: Path
    arguments: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    tolerant_dependencies: tuple[str, ...] = ()
    environment_kind: str = "common"
    downstream: bool = False
    extra_arguments: tuple[tuple[str, ...], ...] = ()
    resumable: bool = True

    def commands(self) -> list[tuple[str, ...]]:
        return [self.arguments, *self.extra_arguments]


@dataclass(frozen=True, slots=True)
class ReproductionSettings:
    seeds: tuple[int, ...]
    three_seed_conditions: frozenset[str]
    stage_decoding_conditions: frozenset[str]
    stage_decoding_sources: tuple[str, ...]
    input_decoding_conditions: frozenset[str]
    navigation_model_condition: str
    navigation_seeds: tuple[int, ...]
    navigation_epsilon: float
    smoke: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OutputRoots:
    artifact_root: str
    run_root: str
    measures_dir: str
    navigation_dir: str

    def overrides(self) -> list[str]:
        return [
            f"tracking.artifact_root={self.artifact_root}",
            f"tracking.run_root={self.run_root}",
        ]


def load_reproduction_settings(config_root: Path) -> ReproductionSettings:
    payload = yaml.safe_load((config_root / "reproduce.yaml").read_text())
    return ReproductionSettings(
        seeds=tuple(int(seed) for seed in payload["seeds"]),
        three_seed_conditions=frozenset(payload["three_seed_conditions"]),
        stage_decoding_conditions=frozenset(payload["stage_decoding_conditions"]),
        stage_decoding_sources=tuple(payload["stage_decoding_sources"]),
        input_decoding_conditions=frozenset(payload["input_decoding_conditions"]),
        navigation_model_condition=str(payload["navigation_model_condition"]),
        navigation_seeds=tuple(int(seed) for seed in payload["navigation_seeds"]),
        navigation_epsilon=float(payload["navigation_epsilon"]),
        smoke=dict(payload.get("smoke") or {}),
    )


def output_roots(smoke: bool, settings: ReproductionSettings) -> OutputRoots:
    if not smoke:
        return OutputRoots("artifacts", "runs", "measures", "navigation")
    root = str(settings.smoke.get("output_root", "smoke"))
    return OutputRoots(
        f"{root}/artifacts", f"{root}/runs", f"{root}/measures", f"{root}/navigation"
    )


def model_tag(condition: str, seed: int) -> str:
    return condition if seed == 42 else f"{condition}_seed{seed}"


def _nested(payload: dict, dotted: str):
    value = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _data_chain_key(raw: dict) -> str:
    seed = raw.get("seed") or {}
    payload = {section: raw.get(section) for section in DATA_CHAIN_SECTIONS}
    payload["seed"] = {
        key: seed.get(key) for key in ("global_seed", "collection_seed", "split_seed")
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def _smoke_model_overrides(raw: dict, smoke: dict) -> list[str]:
    layer_divisor = int(smoke["layer_size_divisor"])
    overrides = []
    for component in ("encoder", "predictor"):
        sizes = _nested(raw, f"spatial_model.{component}.layer_sizes") or []
        if sizes:
            scaled = ",".join(str(max(8, int(size) // layer_divisor)) for size in sizes)
            overrides.append(f"spatial_model.{component}.layer_sizes=[{scaled}]")
    code_dim = _nested(raw, "spatial_model.training.code_dim")
    if code_dim:
        smoke_code_dim = max(8, int(code_dim) // int(smoke["code_dim_divisor"]))
        overrides.append(f"spatial_model.training.code_dim={smoke_code_dim}")
    for index, _phase in enumerate(_nested(raw, "spatial_model.training.phases") or []):
        overrides.append(f"spatial_model.training.phases.{index}.epochs=1")
    return overrides


@dataclass(slots=True)
class _Condition:
    name: str
    config: Path
    raw: dict
    base_overrides: list[str]
    environment_kind: str
    parent: str = ""
    measures_only_parent: str = ""
    forced_retraining: bool = False
    seeds: tuple[int, ...] = ()


def _condition_names(config_root: Path) -> list[str]:
    return sorted(path.stem for path in (config_root / "thesis").glob("*.yaml"))


def navigation_names(config_root: Path) -> list[str]:
    return sorted(path.stem for path in (config_root / "thesis" / "navigation").glob("*.yaml"))


def _load_condition(
    name: str, config_root: Path, settings: ReproductionSettings, roots: OutputRoots, smoke: bool
) -> _Condition:
    config = config_root / "thesis" / f"{name}.yaml"
    raw_without_smoke = load_raw_config_payload(config, [])
    environment_kind = str(_nested(raw_without_smoke, "environment.kind") or "miniworld")
    overrides = [*roots.overrides(), f"measures.output_dir={roots.measures_dir}"]
    if smoke:
        overrides.extend(settings.smoke.get("experiment_overrides", []))
        overrides.extend(settings.smoke.get("environment_overrides", {}).get(environment_kind, []))
        overrides.extend(_smoke_model_overrides(raw_without_smoke, settings.smoke))
    raw = load_raw_config_payload(config, overrides)
    reuse_reference = str(_nested(raw, "reuse.place_model_artifact_id") or "")
    read_time_top_k = int(_nested(raw, "measures.read_time_top_k") or 0)
    parent = reuse_reference.removeprefix("tag:") if reuse_reference.startswith("tag:") else ""
    condition = _Condition(
        name=name,
        config=config,
        raw=raw,
        base_overrides=overrides,
        environment_kind=environment_kind,
        forced_retraining=_nested(raw, "policies.artifact_reuse") == "force_recompute",
    )
    if read_time_top_k:
        condition.measures_only_parent = _defaults_parent(config)
    else:
        condition.parent = parent
    return condition


def _defaults_parent(config: Path) -> str:
    defaults = yaml.safe_load(config.read_text()).get("defaults") or []
    names = [item for item in defaults if isinstance(item, str) and "/" not in item]
    return names[0] if names else ""


def _stage_decoding_sources(condition: _Condition, settings: ReproductionSettings) -> list[str]:
    sources = list(settings.stage_decoding_sources)
    if _nested(condition.raw, "spatial_model.encoder.family") == "mlp":
        sources = [
            "encoder.backbone_output" if source == "encoder.hidden_state" else source
            for source in sources
        ]
    return sources


def _select(
    only: list[str] | None,
    conditions: dict[str, _Condition],
    navigations: list[str],
    settings: ReproductionSettings,
    include_navigation: bool,
) -> tuple[set[str], list[str]]:
    if not only:
        return set(conditions), list(navigations) if include_navigation else []
    valid = {*conditions, *navigations, NAVIGATION_TOKEN}
    unknown = sorted(set(only) - valid)
    if unknown:
        raise ValueError(
            f"Unknown names in --only: {unknown}. Conditions: {sorted(conditions)}; navigation "
            f"configs: {navigations}; or '{NAVIGATION_TOKEN}' for all of them."
        )
    selected = {name for name in only if name in conditions}
    chosen_navigation = [name for name in navigations if name in only]
    if NAVIGATION_TOKEN in only or (
        include_navigation and settings.navigation_model_condition in selected
    ):
        chosen_navigation = list(navigations)
    return selected, chosen_navigation if include_navigation or chosen_navigation else []


def _close_over_parents(selected: set[str], conditions: dict[str, _Condition]) -> set[str]:
    closed = set(selected)
    pending = list(selected)
    while pending:
        condition = conditions[pending.pop()]
        for parent in (condition.parent, condition.measures_only_parent):
            if parent and parent not in closed:
                if parent not in conditions:
                    raise ValueError(f"{condition.name} depends on unknown condition {parent!r}.")
                closed.add(parent)
                pending.append(parent)
    return closed


def _ordered(names: set[str], conditions: dict[str, _Condition], first: str) -> list[str]:
    ordered: list[str] = []

    def visit(name: str) -> None:
        if name in ordered:
            return
        condition = conditions[name]
        for parent in (condition.parent, condition.measures_only_parent):
            if parent:
                visit(parent)
        ordered.append(name)

    for name in sorted(names, key=lambda item: (item != first, item)):
        visit(name)
    return ordered


def build_plan(
    config_root: Path,
    *,
    only: list[str] | None = None,
    seeds: list[int] | None = None,
    smoke: bool = False,
    include_navigation: bool = True,
) -> list[Step]:
    """Every job of the reproduction in a dependency-respecting order."""
    settings = load_reproduction_settings(config_root)
    roots = output_roots(smoke, settings)
    conditions = {
        name: _load_condition(name, config_root, settings, roots, smoke)
        for name in _condition_names(config_root)
    }
    navigations = navigation_names(config_root)
    if smoke:
        smoke_navigation = set(settings.smoke["navigation_configs"])
        navigations = [name for name in navigations if name in smoke_navigation]
    selected, chosen_navigation = _select(
        only, conditions, navigations, settings, include_navigation
    )
    if chosen_navigation:
        selected.add(settings.navigation_model_condition)
    selected = _close_over_parents(selected, conditions)

    default_seeds = tuple(settings.smoke["seeds"]) if smoke else settings.seeds
    requested_seeds = tuple(seeds) if seeds else default_seeds
    for condition in conditions.values():
        thesis_seeds = settings.seeds if condition.name in settings.three_seed_conditions else (42,)
        if condition.measures_only_parent:
            thesis_seeds = (42,)
        condition.seeds = tuple(seed for seed in thesis_seeds if seed in requested_seeds)
    required_seeds = _required_parent_seeds(selected, conditions, chosen_navigation, settings)

    steps: list[Step] = []
    data_steps: dict[str, str] = {}
    ordered_names = _ordered(selected, conditions, settings.navigation_model_condition)
    all_seeds = sorted(
        {*requested_seeds, *(seed for _, seed in required_seeds)},
        key=lambda seed: (seed != 42, seed),
    )
    measures_steps: list[str] = []
    for seed in all_seeds:
        for name in ordered_names:
            condition = conditions[name]
            if seed not in condition.seeds and (name, seed) not in required_seeds:
                continue
            chain = _data_chain_key(condition.raw)
            if not condition.measures_only_parent and chain not in data_steps:
                data_steps[chain] = f"data.{name}"
                steps.append(_data_step(condition))
            steps.extend(
                _condition_steps(condition, seed, settings, data_steps, conditions, roots)
            )
            measures_steps.append(f"measures.{name}.s{seed}")

    navigation_steps: list[str] = []
    navigation_seeds = (
        tuple(settings.smoke["navigation_seeds"]) if smoke else settings.navigation_seeds
    )
    model_step = f"train.{settings.navigation_model_condition}.s42"
    for navigation in chosen_navigation:
        config = config_root / "thesis" / "navigation" / f"{navigation}.yaml"
        overrides = roots.overrides()
        if smoke:
            overrides.extend(settings.smoke.get("navigation_overrides", []))
        for seed in navigation_seeds:
            step_name = f"navigation.{navigation}.s{seed}"
            navigation_steps.append(step_name)
            steps.append(
                Step(
                    name=step_name,
                    kind="navigation",
                    config=config,
                    arguments=(
                        "downstream-train",
                        "--config",
                        str(config),
                        *_override_arguments([*overrides, f"seed={seed}"]),
                    ),
                    dependencies=(model_step,),
                    environment_kind="miniworld",
                    downstream=True,
                )
            )
    first_condition = conditions[ordered_names[0]]
    summary_arguments = (
        "summarize",
        roots.measures_dir,
        "--output",
        f"{roots.measures_dir}/summary.csv",
    )
    extra_summaries: tuple[tuple[str, ...], ...] = ()
    if navigation_steps:
        policies = f"{roots.navigation_dir}/policies.csv"
        steps.append(
            Step(
                name="navigation.measures",
                kind="navigation",
                config=config_root / "thesis" / "navigation" / f"{chosen_navigation[0]}.yaml",
                arguments=(
                    "navigation-measures",
                    f"{roots.run_root}/by_id",
                    "--epsilon",
                    str(settings.navigation_epsilon),
                    "--output",
                    policies,
                ),
                tolerant_dependencies=tuple(navigation_steps),
                environment_kind="miniworld",
                downstream=True,
            )
        )
        extra_summaries = (
            ("summarize", policies, "--output", f"{roots.navigation_dir}/summary.csv"),
        )
    steps.append(
        Step(
            name="summary",
            kind="summary",
            config=first_condition.config,
            arguments=summary_arguments,
            tolerant_dependencies=tuple(
                [*measures_steps, *(["navigation.measures"] if navigation_steps else [])]
            ),
            extra_arguments=extra_summaries,
            resumable=False,
        )
    )
    _check_order(steps)
    return steps


def _data_step(condition: _Condition) -> Step:
    stages = "[collect_dataset,create_split,train_vision_encoder,encode_dataset]"
    return Step(
        name=f"data.{condition.name}",
        kind="data",
        config=condition.config,
        arguments=(
            "pipeline",
            "--config",
            str(condition.config),
            *_override_arguments(
                [
                    *condition.base_overrides,
                    f"pipeline.stages={stages}",
                    "policies.artifact_reuse=reuse_if_config_match",
                ]
            ),
        ),
        environment_kind=condition.environment_kind,
    )


def _required_parent_seeds(
    selected: set[str],
    conditions: dict[str, _Condition],
    chosen_navigation: list[str],
    settings: ReproductionSettings,
) -> set[tuple[str, int]]:
    required: set[tuple[str, int]] = set()
    if chosen_navigation:
        required.add((settings.navigation_model_condition, 42))
    for name in selected:
        condition = conditions[name]
        for seed in condition.seeds:
            if condition.parent:
                required.add((condition.parent, seed))
            if condition.measures_only_parent:
                required.add((condition.measures_only_parent, seed))
    return required


def _override_arguments(overrides: list[str]) -> list[str]:
    arguments: list[str] = []
    for override in overrides:
        arguments.extend(["-o", override])
    return arguments


def _condition_steps(
    condition: _Condition,
    seed: int,
    settings: ReproductionSettings,
    data_steps: dict[str, str],
    conditions: dict[str, _Condition],
    roots: OutputRoots,
) -> list[Step]:
    config = str(condition.config)
    seed_overrides = [*condition.base_overrides, f"seed.training_seed={seed}"]
    if condition.measures_only_parent:
        parent_tag = model_tag(condition.measures_only_parent, seed)
        return [
            Step(
                name=f"measures.{condition.name}.s{seed}",
                kind="measures",
                config=condition.config,
                arguments=(
                    "measures",
                    "--config",
                    config,
                    *_override_arguments(
                        [*seed_overrides, f"reuse.place_model_artifact_id=tag:{parent_tag}"]
                    ),
                ),
                dependencies=(f"representations.{condition.measures_only_parent}.s{seed}",),
            )
        ]
    tag = model_tag(condition.name, seed)
    train_overrides = list(seed_overrides)
    train_dependencies = [data_steps[_data_chain_key(condition.raw)]]
    if condition.parent:
        train_overrides.append(
            f"reuse.place_model_artifact_id=tag:{model_tag(condition.parent, seed)}"
        )
        train_dependencies.append(f"train.{condition.parent}.s{seed}")
    train_arguments = ["pipeline", "--config", config, *_override_arguments(train_overrides)]
    train_arguments.extend(["--place-tag", tag])
    if condition.forced_retraining:
        train_arguments.extend(["--dataset", "auto"])
    model_overrides = [*seed_overrides, f"reuse.place_model_artifact_id=tag:{tag}"]
    representation_overrides = list(model_overrides)
    if seed == 42 and condition.name in settings.stage_decoding_conditions:
        sources = ",".join(_stage_decoding_sources(condition, settings))
        representation_overrides.append(f"representation_collection.sources=[{sources}]")
    measures_arguments = ["measures", "--config", config, *_override_arguments(model_overrides)]
    if seed == 42 and condition.name in settings.input_decoding_conditions:
        measures_arguments.append("--inputs")
    prefix = f"{condition.name}.s{seed}"
    return [
        Step(
            name=f"train.{prefix}",
            kind="train",
            config=condition.config,
            arguments=tuple(train_arguments),
            dependencies=tuple(train_dependencies),
            environment_kind=condition.environment_kind,
        ),
        Step(
            name=f"representations.{prefix}",
            kind="representations",
            config=condition.config,
            arguments=(
                "collect-representations",
                "--config",
                config,
                *_override_arguments(representation_overrides),
            ),
            dependencies=(f"train.{prefix}",),
            environment_kind=condition.environment_kind,
        ),
        Step(
            name=f"measures.{prefix}",
            kind="measures",
            config=condition.config,
            arguments=tuple(measures_arguments),
            dependencies=(f"representations.{prefix}",),
        ),
    ]


def _check_order(steps: list[Step]) -> None:
    seen: set[str] = set()
    for step in steps:
        missing = [
            name for name in (*step.dependencies, *step.tolerant_dependencies) if name not in seen
        ]
        if missing:
            raise RuntimeError(f"Plan order bug: {step.name} needs {missing} first.")
        if step.name in seen:
            raise RuntimeError(f"Plan has {step.name} twice.")
        seen.add(step.name)
