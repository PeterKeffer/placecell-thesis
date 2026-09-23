"""Readable console and log formatting for stage metrics."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console

_STDOUT_CONSOLE = Console(stderr=False, markup=False, highlight=False, soft_wrap=True)
DEFAULT_COMPACT_METRIC_WIDTH = 144
MIN_COMPACT_METRIC_WIDTH = 108


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        magnitude = abs(value)
        if magnitude == 0.0:
            return "0.0000"
        if magnitude >= 1000 or magnitude < 1e-3:
            return f"{value:.3e}"
        return f"{value:.4f}"
    return str(value)


def _split_metric_key(key: str) -> tuple[str, str]:
    if "/" in key:
        group, name = key.split("/", 1)
        return group, name
    parts = key.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:2]), ".".join(parts[2:])
    if len(parts) == 2:
        return parts[0], parts[1]
    return "summary", key


def _group_rank(group_name: str) -> tuple[int, str]:
    preferred = {
        "train": 0,
        "validation": 1,
        "loss": 2,
        "pipeline": 3,
        "paths": 4,
        "artifacts": 5,
        "details": 6,
        "validation.details": 7,
        "summary": 8,
    }
    return preferred.get(group_name, 10), group_name


def _chunked(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[index:index + chunk_size] for index in range(0, len(items), chunk_size)]


def _preferred_compact_metric_width() -> int:
    if _STDOUT_CONSOLE.is_terminal:
        terminal_width = max(40, _STDOUT_CONSOLE.size.width - 2)
        return max(MIN_COMPACT_METRIC_WIDTH, min(DEFAULT_COMPACT_METRIC_WIDTH, terminal_width))
    env_columns = str(os.environ.get("COLUMNS", "")).strip()
    if env_columns.isdigit():
        return max(MIN_COMPACT_METRIC_WIDTH, min(DEFAULT_COMPACT_METRIC_WIDTH, int(env_columns)))
    return DEFAULT_COMPACT_METRIC_WIDTH


def _render_compact_metric_rows(
    rows: list[tuple[str, Any]],
    *,
    max_width: int | None = None,
) -> list[str]:
    effective_max_width = max_width or _preferred_compact_metric_width()
    cells = [f"{label}={_format_value(value)}" for label, value in rows]
    if not cells:
        return []
    cell_width = max(len(cell) for cell in cells)
    columns = max(1, min(len(cells), effective_max_width // max(cell_width + 2, 1)))
    while columns > 1:
        used_width = columns * cell_width + (columns - 1) * 2
        if used_width <= effective_max_width:
            break
        columns -= 1
    rendered_lines: list[str] = []
    for chunk in _chunked(cells, columns):
        padded = [f"{cell:<{cell_width}}" for cell in chunk]
        rendered_lines.append("  " + "  ".join(padded).rstrip())
    return rendered_lines


def _try_render_inline_group(
    group_name: str,
    rows: list[tuple[str, Any]],
    max_width: int,
) -> str | None:
    if not rows:
        return None
    cells = [f"{label}={_format_value(value)}" for label, value in rows]
    candidate = f"[{group_name}] " + "  ".join(cells)
    if len(candidate) > max_width:
        return None
    return candidate


def _try_render_cross_tab(
    grouped: dict[str, list[tuple[str, Any]]],
    max_width: int,
) -> list[str] | None:
    if len(grouped) < 2:
        return None
    ordered_group_names = [
        group_name
        for group_name in sorted(grouped, key=_group_rank)
        if grouped[group_name]
    ]
    if len(ordered_group_names) < 2:
        return None
    first_group_rows = grouped[ordered_group_names[0]]
    metric_names = [metric_name for metric_name, _value in first_group_rows]
    if not metric_names:
        return None
    metric_name_set = set(metric_names)
    row_payloads: list[tuple[str, dict[str, Any]]] = []
    for group_name in ordered_group_names:
        rows = grouped[group_name]
        if {metric_name for metric_name, _value in rows} != metric_name_set:
            return None
        row_payloads.append((group_name, {metric_name: value for metric_name, value in rows}))

    row_label_width = max(len(group_name) for group_name, _row_payload in row_payloads)
    formatted_columns = {
        metric_name: [
            _format_value(row_payload[metric_name])
            for _group_name, row_payload in row_payloads
        ]
        for metric_name in metric_names
    }
    column_widths = {
        metric_name: max(
            len(metric_name),
            *(len(value) for value in formatted_columns[metric_name]),
        )
        for metric_name in metric_names
    }
    header = "  " + " " * row_label_width + "  " + "  ".join(
        f"{metric_name:>{column_widths[metric_name]}}"
        for metric_name in metric_names
    )
    lines = [header]
    for group_name, row_payload in row_payloads:
        lines.append(
            "  "
            + f"{group_name:<{row_label_width}}"
            + "  "
            + "  ".join(
                f"{_format_value(row_payload[metric_name]):>{column_widths[metric_name]}}"
                for metric_name in metric_names
            )
        )
    if any(len(line) > max_width for line in lines):
        return None
    return lines


def _split_core_and_detail(
    grouped: dict[str, list[tuple[str, Any]]],
) -> tuple[dict[str, list[tuple[str, Any]]], dict[str, list[tuple[str, Any]]]]:
    core_groups: dict[str, list[tuple[str, Any]]] = {}
    detail_groups: dict[str, list[tuple[str, Any]]] = {}
    for group_name, rows in grouped.items():
        if not rows:
            continue
        group_rank, _ = _group_rank(group_name)
        if group_rank < 10 or len(rows) > 2:
            core_groups[group_name] = rows
        else:
            detail_groups[group_name] = rows
    return core_groups, detail_groups


def _merge_detail_groups(
    detail_groups: dict[str, list[tuple[str, Any]]],
) -> dict[str, list[tuple[str, Any]]]:
    merged_groups: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for group_name, rows in sorted(detail_groups.items(), key=lambda item: _group_rank(item[0])):
        for metric_name, value in rows:
            if group_name.startswith("validation."):
                merged_group_name = "validation.details"
                stripped_group_name = group_name.removeprefix("validation.")
            else:
                merged_group_name = "details"
                stripped_group_name = group_name
            merged_metric_name = f"{stripped_group_name}.{metric_name}"
            merged_groups[merged_group_name].append((merged_metric_name, value))
    return {
        group_name: sorted(rows, key=lambda item: item[0])
        for group_name, rows in merged_groups.items()
        if rows
    }


def _render_grouped_block(
    title: str,
    grouped: dict[str, list[tuple[str, Any]]],
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    effective_max_width = _preferred_compact_metric_width()
    lines = [f"=== {title} ==="]
    if metadata:
        metadata_text = " | ".join(
            f"{key}={_format_value(value)}"
            for key, value in metadata.items()
        )
        if metadata_text:
            lines.append(metadata_text)

    cross_tab_lines = _try_render_cross_tab(grouped, effective_max_width)
    if cross_tab_lines is not None:
        lines.extend(cross_tab_lines)
        return "\n".join(lines)

    core_groups, detail_groups = _split_core_and_detail(grouped)
    merged_detail_groups = _merge_detail_groups(detail_groups)
    render_groups = {
        **core_groups,
        **merged_detail_groups,
    }
    for group_name in sorted(render_groups, key=_group_rank):
        rows = render_groups[group_name]
        inline_group = _try_render_inline_group(group_name, rows, effective_max_width)
        if inline_group is not None:
            lines.append(inline_group)
            continue
        lines.append(f"[{group_name}]")
        lines.extend(_render_compact_metric_rows(rows, max_width=effective_max_width))
    return "\n".join(lines)


def render_metrics_block(
    title: str,
    metrics: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    grouped: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for key, value in sorted(metrics.items()):
        if key == "epoch":
            continue
        group_name, metric_name = _split_metric_key(key)
        grouped[group_name].append((metric_name, value))
    return _render_grouped_block(title, grouped, metadata=metadata)


def emit_metrics_block(
    title: str,
    metrics: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
    log_path: Path | None = None,
) -> str:
    block = render_metrics_block(title, metrics, metadata=metadata)
    _STDOUT_CONSOLE.print(block)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(block + "\n\n")
    return block


def render_text_block(
    title: str,
    body: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    lines = [f"=== {title} ==="]
    if metadata:
        metadata_text = " | ".join(
            f"{key}={_format_value(value)}"
            for key, value in metadata.items()
        )
        if metadata_text:
            lines.append(metadata_text)
    normalized_body = body.rstrip()
    if normalized_body:
        lines.append(normalized_body)
    return "\n".join(lines)


def emit_text_block(
    title: str,
    body: str,
    *,
    metadata: dict[str, Any] | None = None,
    log_path: Path | None = None,
) -> str:
    block = render_text_block(title, body, metadata=metadata)
    _STDOUT_CONSOLE.print(block)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(block + "\n\n")
    return block


def _is_metric_result_entry(key: str, value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if "/" in key:
        return True
    if key.startswith(("train.", "validation.", "loss.")):
        return True
    return key.count(".") >= 2


def _split_summary_key(key: str) -> tuple[str, str]:
    if "." in key:
        group_name, field_name = key.split(".", 1)
        return group_name, field_name
    if key.startswith("pipeline_"):
        return "pipeline", key.removeprefix("pipeline_")
    if key == "completed_stages":
        return "pipeline", "completed_stages"
    if key.endswith("_path") or key == "checkpoint_path":
        return "paths", key
    if key.endswith("_id") or key.endswith("artifact_id"):
        return "artifacts", key
    return "summary", key


def _render_pipeline_summary(result: dict[str, Any]) -> str:
    completed_stages = [str(stage_name) for stage_name in result.get("completed_stages", [])]
    stage_chain = " -> ".join(completed_stages) if completed_stages else "none"
    grouped: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    grouped["pipeline"].append(("run_id", result.get("pipeline_run_id", "")))
    grouped["pipeline"].append(("completed_stages", stage_chain))
    grouped["paths"].append(("run_path", result.get("pipeline_run_path", "")))
    grouped["paths"].append(("results_path", result.get("pipeline_results_path", "")))
    grouped["paths"].append(("open_path", result.get("pipeline_open_path", "")))
    if result.get("slurm_log_path"):
        grouped["paths"].append(("slurm_log_path", result.get("slurm_log_path", "")))

    for key in ("evaluation_report_id", "analysis_report_id"):
        if result.get(key):
            grouped["artifacts"].append((key, result[key]))
    for key in ("evaluation_report_path", "analysis_report_path"):
        if result.get(key):
            grouped["paths"].append((key, result[key]))

    block = _render_grouped_block("pipeline", grouped)
    return (
        block
        + "\n\nsummary:\n"
        + "  detailed metrics were already printed during stage execution\n"
        + "  and written to the report artifacts and stage logs"
    )


def render_stage_result_summary(result: dict[str, Any] | None) -> str:
    if not result:
        return ""
    if "pipeline_run_id" in result and "completed_stages" in result:
        return _render_pipeline_summary(result)

    summary_entries: dict[str, Any] = {
        key: value for key, value in result.items() if not _is_metric_result_entry(key, value)
    }
    if not summary_entries:
        return render_metrics_block("result", result)

    grouped: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for key, value in sorted(summary_entries.items()):
        group_name, field_name = _split_summary_key(key)
        grouped[group_name].append((field_name, value))

    block = _render_grouped_block("result", grouped)
    omitted_metric_count = sum(
        1
        for key, value in result.items()
        if _is_metric_result_entry(key, value)
    )
    if omitted_metric_count == 0:
        return block
    return (
        block
        + "\n\nsummary:\n"
        + f"  omitted_metric_values  {omitted_metric_count}\n"
        + "  note                   detailed metrics were already printed above"
    )
