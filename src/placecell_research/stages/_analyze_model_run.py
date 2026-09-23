"""Coordinator helpers extracted from placecell_research.stages.analyze_model."""

from __future__ import annotations

import gc
import json
import sys
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from placecell_research.analysis.base import AnalysisInput, AnalysisResult
from placecell_research.analysis.registry import (
    run_analysis_modules,
    run_comparative_modules,
)
from placecell_research.analysis.validation_summary import (
    flatten_analysis_results,
    write_summary_csv,
    write_summary_json,
)
from placecell_research.artifacts.compatibility import (
    CompatibilityReference,
    validate_artifact_compatibility,
)
from placecell_research.evaluation.inference import InputBatchCache
from placecell_research.evaluation.runtime import publish_report
from placecell_research.utils.memory_watchdog import read_current_rss_mb

if TYPE_CHECKING:
    from placecell_research.stages.analyze_model import (
        _AnalysisSourceReference,
        _CollectionCacheKey,
        _CollectionCacheValue,
        _CollectionPlan,
        _SingleAnalysisWorkItem,
        _SourceCollectionGroupKey,
    )


def disable_targets_where(
    analysis_config: dict[str, object],
    predicate: Callable[[str, dict[str, object]], bool],
) -> tuple[dict[str, object], list[dict[str, str]]]:
    """Disable every enabled target whose (source_name, payload) matches predicate."""
    targets = analysis_config.get("targets", {})
    if not isinstance(targets, dict):
        return analysis_config, []
    skipped_targets: list[dict[str, str]] = []
    filtered_targets: dict[str, object] = {}
    for target_name, target_payload in targets.items():
        if not isinstance(target_payload, dict):
            filtered_targets[str(target_name)] = target_payload
            continue
        filtered_payload = dict(target_payload)
        source_name = str(filtered_payload.get("source", ""))
        if bool(filtered_payload.get("enabled", True)) and predicate(source_name, filtered_payload):
            filtered_payload["enabled"] = False
            skipped_targets.append({"target": str(target_name), "source": source_name})
        filtered_targets[str(target_name)] = filtered_payload
    if not skipped_targets:
        return analysis_config, []
    return {**analysis_config, "targets": filtered_targets}, skipped_targets


def _release_work_item_memory() -> None:
    """Drop per-work-item plotting and cycle garbage before the next item allocates."""
    plt.close("all")
    gc.collect()


_TrackedCollection = tuple[str, "weakref.ReferenceType[object]", int]


def _track_group_collections(
    tracked_collections: list[_TrackedCollection],
    group_label: str,
    collection_cache: dict[_CollectionCacheKey, _CollectionCacheValue],
) -> None:
    """Record a weak reference to every representation array a source group collected."""
    for representations, _metadata, _position_cache in collection_cache.values():
        for source_name, array in representations.items():
            tracked_collections.append(
                (
                    f"{group_label}:{source_name}",
                    weakref.ref(array),
                    int(getattr(array, "nbytes", 0)),
                )
            )


def _log_group_boundary_memory(
    group_label: str,
    *,
    tracked_collections: list[_TrackedCollection],
    input_batch_cache: InputBatchCache,
    result_count: int,
) -> None:
    """Report what the stage still holds now that a source group is done."""
    survivors = [
        (label, reference, nbytes)
        for label, reference, nbytes in tracked_collections
        if reference() is not None
    ]
    retained_bytes = sum(nbytes for _label, _reference, nbytes in survivors)
    rss_mb = read_current_rss_mb()
    print(
        "[analyze_model] group_boundary "
        f"after={group_label} "
        f"rss_gib={'unavailable' if rss_mb is None else f'{rss_mb / 1024.0:.2f}'} "
        f"input_batch_cache_gib={input_batch_cache.resident_bytes() / 1024**3:.2f} "
        f"results={result_count} "
        f"retained_collections={len(survivors)} "
        f"retained_gib={retained_bytes / 1024**3:.2f}",
        file=sys.stderr,
        flush=True,
    )
    for label, reference, nbytes in survivors:
        retained_array = reference()
        if retained_array is None:
            continue
        referrer_types = sorted(
            {type(referrer).__name__ for referrer in gc.get_referrers(retained_array)}
        )
        print(
            f"[analyze_model] group_boundary RETAINED {label} "
            f"{nbytes / 1024**3:.2f} GiB referrers={referrer_types}",
            file=sys.stderr,
            flush=True,
        )


def build_target_work_items(
    target_config: dict[str, object],
    *,
    coverage_extra_splits: list[str],
    default_model_id: str,
    default_dataset_id: str,
    default_dataset_type: str,
    default_split_id: str,
    default_split_name: str,
    make_reference: Callable[..., _AnalysisSourceReference],
    make_work_item: Callable[..., _SingleAnalysisWorkItem],
) -> list[_SingleAnalysisWorkItem]:
    """Build the single-target work items, expanding dataset_coverage over extra splits."""
    target_work_items: list[_SingleAnalysisWorkItem] = []
    for target_name, target_payload in target_config.items():
        if not target_payload.get("enabled", True):
            continue
        target_module_names = list(target_payload.get("modules", []))
        reference = make_reference(
            label=str(target_name),
            source_name=str(target_payload["source"]),
            model_artifact_id=default_model_id,
            dataset_artifact_id=default_dataset_id,
            dataset_artifact_type=default_dataset_type,
            split_artifact_id=default_split_id,
            split_name=default_split_name,
        )
        target_work_items.append(
            make_work_item(
                progress_label=str(target_name),
                reference=reference,
                module_names=target_module_names,
            )
        )
        if "dataset_coverage" in target_module_names:
            for extra_split_name in coverage_extra_splits:
                extra_reference = make_reference(
                    label=str(target_name),
                    source_name=reference.source_name,
                    model_artifact_id=reference.model_artifact_id,
                    dataset_artifact_id=reference.dataset_artifact_id,
                    dataset_artifact_type=reference.dataset_artifact_type,
                    split_artifact_id=reference.split_artifact_id,
                    split_name=extra_split_name,
                )
                target_work_items.append(
                    make_work_item(
                        progress_label=f"{target_name}:dataset_coverage:{extra_split_name}",
                        reference=extra_reference,
                        module_names=["dataset_coverage"],
                        result_key_overrides={
                            "dataset_coverage": (
                                f"{target_name}.dataset_coverage_{extra_split_name}"
                            )
                        },
                    )
                )
    return target_work_items


def build_comparative_work_items(
    enabled_comparative_items: list[tuple[str, dict[str, object]]],
    *,
    default_model_id: str,
    default_dataset_id: str,
    default_dataset_type: str,
    default_split_id: str,
    default_split_name: str,
    registry,
    resolve_reference: Callable[..., _AnalysisSourceReference],
    reference_from_payload: Callable[..., _AnalysisSourceReference],
    comparative_input_payloads: Callable[[str, dict[str, object]], list[dict[str, object]]],
) -> list[tuple[str, dict[str, object], list[_AnalysisSourceReference]]]:
    """Resolve each enabled comparative analysis to its concrete source references."""
    comparative_work_items: list[
        tuple[str, dict[str, object], list[_AnalysisSourceReference]]
    ] = []
    for analysis_name, analysis_payload in enabled_comparative_items:
        references = [
            resolve_reference(
                reference_from_payload(
                    payload,
                    default_model_artifact_id=default_model_id,
                    default_dataset_artifact_id=default_dataset_id,
                    default_dataset_artifact_type=default_dataset_type,
                    default_split_artifact_id=default_split_id,
                    default_split_name=default_split_name,
                    default_source_name=str(
                        analysis_payload.get("source", "encoder.place_codes")
                    ),
                ),
                registry=registry,
            )
            for payload in comparative_input_payloads(analysis_name, analysis_payload)
        ]
        comparative_work_items.append((analysis_name, analysis_payload, references))
    return comparative_work_items


def execute_single_work_items(
    target_work_items: list[_SingleAnalysisWorkItem],
    *,
    single_results: dict[str, AnalysisResult],
    comparative_results: dict[str, AnalysisResult],
    collection_plans_by_group: dict[_SourceCollectionGroupKey, _CollectionPlan],
    analysis_config: dict[str, object],
    analysis_workspace: Path,
    partial_analysis_dir: Path,
    run_directory,
    registry,
    device,
    model_cache: dict[str, object],
    batch_size: int,
    analysis_max_episodes: int | None,
    used_input_artifact_ids: set,
    progress,
    single_work_items_by_source_group: Callable[
        [list[_SingleAnalysisWorkItem]],
        list[tuple[_SourceCollectionGroupKey, list[_SingleAnalysisWorkItem]]],
    ],
    work_item_needs_model_inference: Callable[[_SingleAnalysisWorkItem], bool],
    build_analysis_input: Callable[..., AnalysisInput],
    build_dataset_coverage_analysis_input: Callable[..., AnalysisInput],
    snapshot_partial_and_refresh: Callable[..., None],
    preserve_workspace_and_refresh: Callable[..., None],
) -> None:
    """Run the per-source-group single-target analysis loop, mutating single_results."""
    input_batch_cache = InputBatchCache()
    tracked_collections: list[_TrackedCollection] = []
    for group_key, group_work_items in single_work_items_by_source_group(target_work_items):
        collection_plan = collection_plans_by_group[group_key]
        single_collection_cache: dict[_CollectionCacheKey, _CollectionCacheValue] = {}
        analysis_input: AnalysisInput | None = None
        for work_item in group_work_items:
            analysis_input = None
            _release_work_item_memory()
            if work_item_needs_model_inference(work_item):
                analysis_input = build_analysis_input(
                    work_item.reference,
                    registry=registry,
                    device=device,
                    model_cache=model_cache,
                    collection_plan=collection_plan,
                    collection_cache=single_collection_cache,
                    batch_size=batch_size,
                    max_episodes=analysis_max_episodes,
                    input_batch_cache=input_batch_cache,
                )
            else:
                analysis_input = build_dataset_coverage_analysis_input(
                    work_item.reference,
                    registry=registry,
                )
            progress.advance(detail=f"prepare {work_item.progress_label}")
            used_input_artifact_ids.update(
                {
                    work_item.reference.model_artifact_id,
                    work_item.reference.dataset_artifact_id,
                    work_item.reference.split_artifact_id,
                }
            )
            target_output_dir = analysis_workspace / work_item.reference.label

            def _advance_single_module(
                module_name: str,
                progress_label: str = work_item.progress_label,
            ) -> None:
                progress.advance(detail=f"{progress_label}:{module_name}")

            try:
                module_results = run_analysis_modules(
                    analysis_input,
                    target_output_dir,
                    analysis_config,
                    work_item.module_names,
                    progress_callback=_advance_single_module,
                )
            except BaseException as exc:
                preserve_workspace_and_refresh(
                    partial_analysis_dir,
                    analysis_workspace,
                    run_directory=run_directory,
                    reason=type(exc).__name__,
                )
                raise
            for module_name, result in module_results.items():
                result_key = work_item.result_key(module_name)
                single_results[result_key] = result
                snapshot_partial_and_refresh(
                    partial_analysis_dir,
                    run_directory=run_directory,
                    single_results=single_results,
                    comparative_results=comparative_results,
                    completed_results={result_key: result},
                )
            del module_results
        analysis_input = None
        _track_group_collections(tracked_collections, str(group_key[-1]), single_collection_cache)
        single_collection_cache.clear()
        _release_work_item_memory()
        _log_group_boundary_memory(
            str(group_key[-1]),
            tracked_collections=tracked_collections,
            input_batch_cache=input_batch_cache,
            result_count=len(single_results),
        )
    input_batch_cache.clear()
    _release_work_item_memory()


def execute_comparative_work_items(
    comparative_work_items: list[tuple[str, dict[str, object], list[_AnalysisSourceReference]]],
    *,
    single_results: dict[str, AnalysisResult],
    comparative_results: dict[str, AnalysisResult],
    analysis_config: dict[str, object],
    analysis_workspace: Path,
    partial_analysis_dir: Path,
    run_directory,
    registry,
    device,
    model_cache: dict[str, object],
    batch_size: int,
    analysis_max_episodes: int | None,
    used_input_artifact_ids: set,
    progress,
    make_collection_plan: Callable[..., _CollectionPlan],
    required_comparative_batch_keys: Callable[[dict[str, object]], list[str]],
    build_analysis_input: Callable[..., AnalysisInput],
    snapshot_partial_and_refresh: Callable[..., None],
    preserve_workspace_and_refresh: Callable[..., None],
) -> None:
    """Run the comparative analysis loop, mutating comparative_results."""
    for analysis_name, analysis_payload, references in comparative_work_items:
        _release_work_item_memory()
        validate_artifact_compatibility(
            registry,
            [
                CompatibilityReference(
                    label=f"comparative:{analysis_name}:{reference.label}",
                    dataset_artifact_id=reference.dataset_artifact_id,
                    dataset_artifact_type=reference.dataset_artifact_type,
                    split_artifact_id=reference.split_artifact_id,
                    model_artifact_id=reference.model_artifact_id,
                )
                for reference in references
            ],
            allow_cross_dataset_encoder_mismatch=bool(
                analysis_payload.get("allow_encoder_mismatch", False)
            ),
            encoder_mismatch_remedy=(
                "Encode both datasets with the same vision encoder, or set "
                f"`analysis.comparative.{analysis_name}.allow_encoder_mismatch: true` to compare "
                "across encoders on purpose."
            ),
        )
        collection_plan = make_collection_plan(
            source_names=tuple(sorted({reference.source_name for reference in references})),
            include_batch_keys=tuple(required_comparative_batch_keys(analysis_payload)),
        )
        collection_cache: dict[_CollectionCacheKey, _CollectionCacheValue] = {}
        analysis_inputs = [
            build_analysis_input(
                reference,
                registry=registry,
                device=device,
                model_cache=model_cache,
                collection_plan=collection_plan,
                collection_cache=collection_cache,
                batch_size=batch_size,
                max_episodes=analysis_max_episodes,
            )
            for reference in references
        ]
        for reference in references:
            used_input_artifact_ids.update(
                {
                    reference.model_artifact_id,
                    reference.dataset_artifact_id,
                    reference.split_artifact_id,
                }
            )
        labels = [analysis_input.label for analysis_input in analysis_inputs]
        comparative_output_dir = analysis_workspace / "comparative" / analysis_name
        try:
            completed_comparative_results = run_comparative_modules(
                analysis_inputs,
                labels,
                comparative_output_dir,
                {
                    "comparative": {analysis_name: analysis_payload},
                    "max_cost_tier": analysis_config.get("max_cost_tier", "heavy"),
                },
                progress_callback=lambda completed_name: progress.advance(
                    detail=f"comparative:{completed_name}"
                ),
            )
        except BaseException as exc:
            preserve_workspace_and_refresh(
                partial_analysis_dir,
                analysis_workspace,
                run_directory=run_directory,
                reason=type(exc).__name__,
            )
            raise
        comparative_results.update(completed_comparative_results)
        snapshot_partial_and_refresh(
            partial_analysis_dir,
            run_directory=run_directory,
            single_results=single_results,
            comparative_results=comparative_results,
            completed_results=completed_comparative_results,
        )
        analysis_inputs.clear()
        collection_cache.clear()


def _active_config(config, analysis_config: dict[str, object]) -> dict[str, object]:
    """Schema-resolved config whose analysis section carries the values this stage ran with."""
    active_config = config.to_dict()
    return {**active_config, "analysis": {**active_config["analysis"], **analysis_config}}


def finalize_report(
    *,
    single_results: dict[str, AnalysisResult],
    comparative_results: dict[str, AnalysisResult],
    raw_config: dict[str, object],
    analysis_config: dict[str, object],
    enabled_comparative_items: list[tuple[str, dict[str, object]]],
    registry,
    run_directory,
    stage_run,
    stage_fingerprint: str,
    used_input_artifact_ids: set,
    default_model_id: str,
    default_dataset_id: str,
    default_dataset_type: str,
    default_split_id: str,
    default_split_name: str,
    stage_log_path: Path,
    runtime,
    analysis_population_coding_metrics: Callable[..., dict[str, float]],
    copy_analysis_outputs: Callable[[Path, dict[str, object]], None],
    write_analysis_report_readme: Callable[..., None],
    write_analysis_browser_links: Callable[..., None],
    replace_partial_analysis_with_report_link: Callable[[object, Path], None],
    refresh_analysis_run_shortcuts: Callable[[object, Path], None],
    augment_stage_result: Callable[[object, dict[str, object]], dict[str, object]],
    emit_metrics_block: Callable[..., None],
) -> dict[str, object]:
    """Flatten results, publish the report artifact, wire shortcuts, and finalize the run."""
    flat_summary = flatten_analysis_results(single_results, comparative_results)
    flat_summary.update(analysis_population_coding_metrics(single_results))
    config_text = json.dumps(raw_config, indent=2, sort_keys=True)

    def _write_report(output_dir: Path, artifact_id: str) -> None:
        del artifact_id
        write_summary_json(output_dir / "summary.json", flat_summary)
        write_summary_csv(output_dir / "metrics.csv", flat_summary)
        copy_analysis_outputs(output_dir, single_results)
        copy_analysis_outputs(output_dir, comparative_results)
        write_analysis_report_readme(
            output_dir=output_dir,
            model_artifact_id=default_model_id,
            dataset_artifact_id=default_dataset_id,
            split_name=default_split_name,
            target_names=sorted(analysis_config.get("targets", {}).keys()),
            comparative_names=sorted(name for name, _payload in enabled_comparative_items),
        )

    report_id, report_path = publish_report(
        registry=registry,
        artifact_type="analysis_report",
        summary_name=f"{default_model_id}_{default_split_name}",
        run_directory=run_directory,
        stage_name="analyze_model",
        config_text=config_text,
        input_artifact_ids=sorted(used_input_artifact_ids),
        files_writer=_write_report,
        config_fingerprint_value=stage_fingerprint,
        raw_config=raw_config,
        active_config=_active_config(runtime.config, analysis_config),
        hyperparameter_sections=[
            "analysis",
            "dataset",
            "splits",
            "seed",
            "policies",
            "reuse",
            "tracking",
        ],
        hyperparameter_context={
            "model_artifact_id": default_model_id,
            "dataset_artifact_id": default_dataset_id,
            "dataset_artifact_type": default_dataset_type,
            "split_artifact_id": default_split_id,
            "split_name": default_split_name,
            "analysis_targets": sorted(analysis_config.get("targets", {}).keys()),
            "comparative_targets": sorted(
                name for name, _payload in enabled_comparative_items
            ),
        },
    )
    write_analysis_browser_links(
        registry_root=registry.root,
        report_path=report_path,
        report_id=report_id,
        run_id=run_directory.identity.run_id,
        model_artifact_id=default_model_id,
        dataset_artifact_id=default_dataset_id,
        split_name=default_split_name,
    )
    run_directory.update_run_manifest(
        {
            "status": "completed",
            "produced_artifact_ids": [report_id],
            "summary": {
                "analysis_report_id": report_id,
                "analysis_report_path": str(report_path),
                **flat_summary,
            },
        }
    )
    run_directory.write_symlink("results/analysis_report", report_path)
    replace_partial_analysis_with_report_link(run_directory, report_path)
    refresh_analysis_run_shortcuts(run_directory, report_path)
    registry.mark_artifact_completed("analysis_report", report_id)
    stage_run.log(flat_summary)
    emit_metrics_block(
        "analyze_model",
        flat_summary,
        metadata={
            "split": default_split_name,
            "targets": len(analysis_config.get("targets", {})),
            "comparative": len(enabled_comparative_items),
            "report_id": report_id,
        },
        log_path=stage_log_path,
    )
    stage_run.finalize(
        status="completed",
        summary={
            "analysis_report_id": report_id,
            "analysis_report_path": str(report_path),
            **flat_summary,
        },
        upload_files=[
            report_path / "manifest.json",
            report_path / "README.md",
            report_path / "resolved_config.yaml",
            report_path / "used_hyperparameters.yaml",
            report_path / "summary.json",
            report_path / "metrics.csv",
        ],
    )
    return augment_stage_result(runtime, {
        "analysis_report_id": report_id,
        "analysis_report_path": str(report_path),
        **flat_summary,
    })
