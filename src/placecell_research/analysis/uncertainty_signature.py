"""Does the code carry its own uncertainty signature?"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from placecell_research.evaluation.decode import (
    chunked_ridge_predict,
    episode_level_decode_skip_reason,
    fit_position_ridge_decoder,
)

from .base import AnalysisInput, AnalysisResult
from .figures import apply_publication_style, despine

NUM_PERMUTATIONS = 500
PERMUTATION_SEED = 0
_EPS = 1e-12
NUM_SELECTED_FROM_TESTS = 2
SELECTION_RULE = (
    "reported statistic = larger |rho| of two pre-specified tests (magnitude, entropy); "
    "its p is Bonferroni-corrected for that two-way choice"
)


def bonferroni_corrected(p_value: float, num_tests: int = NUM_SELECTED_FROM_TESTS) -> float:
    """The p a selected test costs once the selection is paid for."""
    if not np.isfinite(p_value):
        return float("nan")
    return float(min(1.0, num_tests * p_value))


def _normalized_entropy(codes: np.ndarray) -> np.ndarray:
    """Entropy of |code| normalized across units, scaled to [0, 1]; NaN for all-zero rows."""
    magnitudes = np.abs(codes.astype(np.float64))
    totals = magnitudes.sum(axis=1, keepdims=True)
    has_activity = totals[:, 0] > _EPS
    probabilities = magnitudes / np.where(totals > _EPS, totals, np.nan)
    entropy_terms = np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0)
    entropy = -entropy_terms.sum(axis=1) / np.log(codes.shape[1])
    return np.where(has_activity, entropy, np.nan)


def _centered_ranks(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values).astype(np.float64)
    return ranks - ranks.mean()


def _spearman_with_block_permutation_p(
    errors: np.ndarray,
    statistic: np.ndarray,
    episode_blocks: list[np.ndarray],
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Spearman rho plus a two-sided p-value from episode-block permutations."""
    finite = np.isfinite(errors) & np.isfinite(statistic)
    if finite.sum() < 3:
        return float("nan"), float("nan")
    kept_positions = np.flatnonzero(finite)
    new_position = np.full(len(errors), -1, dtype=np.int64)
    new_position[kept_positions] = np.arange(len(kept_positions))
    blocks = [new_position[block][new_position[block] >= 0] for block in episode_blocks]
    blocks = [block for block in blocks if block.size > 0]

    error_ranks = _centered_ranks(errors[kept_positions])
    statistic_ranks = _centered_ranks(statistic[kept_positions])
    error_scale = float(np.sqrt((error_ranks**2).sum()))
    statistic_scale = float(np.sqrt((statistic_ranks**2).sum()))
    if error_scale < _EPS or statistic_scale < _EPS:
        return float("nan"), float("nan")
    observed = float(error_ranks @ statistic_ranks / (error_scale * statistic_scale))

    exceed_count = 0
    for _ in range(NUM_PERMUTATIONS):
        order = rng.permutation(len(blocks))
        gather = np.concatenate([blocks[block_index] for block_index in order])
        null = float(error_ranks @ statistic_ranks[gather] / (error_scale * statistic_scale))
        if abs(null) >= abs(observed):
            exceed_count += 1
    p_value = (1.0 + exceed_count) / (1.0 + NUM_PERMUTATIONS)
    return observed, float(p_value)


@dataclass(slots=True)
class UncertaintySignatureModule:
    """Rank correlation between held-out decode error and per-step code statistics."""

    name: str = "uncertainty_signature"
    cost_tier: str = "light"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        representation = analysis_input.representation
        if representation.ndim != 3:
            raise ValueError(f"Expected representation [N, T, D], got {representation.shape}.")
        num_episodes, num_steps, _ = representation.shape
        flat_codes = representation.reshape(-1, representation.shape[-1])
        flat_positions = analysis_input.position_xy.reshape(-1, 2)
        if analysis_input.valid_mask is None:
            valid_rows = np.arange(num_episodes * num_steps, dtype=np.int64)
        else:
            valid_rows = np.flatnonzero(
                analysis_input.valid_mask.reshape(-1).astype(bool, copy=False)
            ).astype(np.int64, copy=False)
        codes = flat_codes[valid_rows].astype(np.float32, copy=False)
        positions = flat_positions[valid_rows].astype(np.float32, copy=False)
        episode_ids = valid_rows // num_steps

        skip_reason = episode_level_decode_skip_reason(episode_ids)
        if skip_reason is not None:
            return AnalysisResult(
                metrics={"uncertainty_signature_skipped": 1.0},
                per_unit_metrics={},
                figures={},
                tables={},
                metadata={"uncertainty_signature_skip_reason": skip_reason},
            )

        decoder = fit_position_ridge_decoder(
            codes,
            positions,
            train_fraction=float(config.get("decode_train_fraction", 0.8)),
            alpha=float(config.get("decode_ridge_alpha", 1e-3)),
            episode_ids=episode_ids,
        )
        episode_order = np.argsort(episode_ids[decoder.validation_indices], kind="stable")
        validation_indices = decoder.validation_indices[episode_order]
        predictions = chunked_ridge_predict(decoder.ridge, codes, validation_indices)
        errors = np.linalg.norm(predictions - positions[validation_indices], axis=1)
        validation_episode_ids = episode_ids[validation_indices]
        _, block_starts = np.unique(validation_episode_ids, return_index=True)
        episode_blocks = np.split(np.arange(len(validation_indices)), np.sort(block_starts)[1:])

        validation_codes = codes[validation_indices]
        magnitude = np.linalg.norm(validation_codes.astype(np.float64), axis=1)
        entropy = _normalized_entropy(validation_codes)

        rng = np.random.default_rng(PERMUTATION_SEED)
        rho_magnitude, p_magnitude = _spearman_with_block_permutation_p(
            errors, magnitude, episode_blocks, rng
        )
        rho_entropy, p_entropy = _spearman_with_block_permutation_p(
            errors, entropy, episode_blocks, rng
        )

        statistics = {
            "magnitude": (magnitude, rho_magnitude, p_magnitude, "Code magnitude (L2 norm)"),
            "entropy": (entropy, rho_entropy, p_entropy, "Normalized activation entropy"),
        }
        best_name = "magnitude"
        if np.isfinite(rho_entropy) and (
            not np.isfinite(rho_magnitude) or abs(rho_entropy) > abs(rho_magnitude)
        ):
            best_name = "entropy"
        best_values, best_rho, best_p, best_label = statistics[best_name]
        selected_p_bonferroni = bonferroni_corrected(best_p)

        module_dir = output_dir / self.name
        figure_path = (
            module_dir / f"uncertainty_signature__{analysis_input.source_name}"
            f"__{analysis_input.split_name}.png"
        )
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        caption = (
            f"Held-out steps, n = {len(errors)}; Spearman rho (error vs magnitude) = "
            f"{rho_magnitude:.3f}, p = {p_magnitude:.4f}; (error vs entropy) = "
            f"{rho_entropy:.3f}, p = {p_entropy:.4f}. Plotted: {best_name}, the larger |rho| "
            f"of the two, Bonferroni p = {selected_p_bonferroni:.4f} "
            f"({NUM_PERMUTATIONS} episode-block permutations)"
        )
        apply_publication_style()
        figure, axis = plt.subplots(figsize=(6.0, 4.4))
        finite = np.isfinite(best_values) & np.isfinite(errors)
        hexbin = axis.hexbin(
            best_values[finite],
            errors[finite],
            gridsize=40,
            cmap="viridis",
            mincnt=1,
            linewidths=0.1,
        )
        figure.colorbar(hexbin, ax=axis, label="held-out steps per bin")
        axis.set_xlabel(best_label)
        axis.set_ylabel("Decode error (arena units)")
        despine(axis)
        figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
        figure.text(0.5, 0.015, caption, ha="center", fontsize=7.5)
        figure.savefig(figure_path, dpi=160)
        plt.close(figure)

        return AnalysisResult(
            metrics={
                "spearman_error_vs_magnitude": rho_magnitude,
                "spearman_error_vs_magnitude_p_value": p_magnitude,
                "spearman_error_vs_entropy": rho_entropy,
                "spearman_error_vs_entropy_p_value": p_entropy,
                "selected_by_larger_abs_rho_spearman": best_rho,
                "selected_by_larger_abs_rho_p_value_bonferroni": selected_p_bonferroni,
            },
            per_unit_metrics={},
            figures={"uncertainty_signature_hexbin": figure_path},
            tables={},
            metadata={
                "uncertainty_signature_caption": caption,
                "uncertainty_signature_selected_statistic": best_name,
                "uncertainty_signature_selection_rule": SELECTION_RULE,
                "uncertainty_signature_num_selected_from_tests": NUM_SELECTED_FROM_TESTS,
                "num_validation_steps": int(len(errors)),
                "num_validation_episodes": int(len(episode_blocks)),
                "num_permutations": NUM_PERMUTATIONS,
                "permutation_seed": PERMUTATION_SEED,
                "top_k_margin_note": (
                    "Pre-sparsifier logits are not part of the per-target module input; "
                    "run this module on an encoder_pre_sparsifier target for a margin analog."
                ),
            },
        )
