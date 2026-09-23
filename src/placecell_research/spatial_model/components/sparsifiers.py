"""Sparsifying activations."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def sparsemax(values: Tensor, dim: int = -1) -> Tensor:
    """Sparsemax activation with exact zeros."""
    sorted_values, _ = torch.sort(values, dim=dim, descending=True)
    cssv = torch.cumsum(sorted_values, dim=dim) - 1
    range_values = torch.arange(
        1,
        values.size(dim) + 1,
        device=values.device,
        dtype=values.dtype,
    )
    view_shape = [1] * values.ndim
    view_shape[dim] = values.size(dim)
    range_values = range_values.view(view_shape)
    support = sorted_values > cssv / range_values
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = cssv.gather(dim, support_size - 1) / support_size
    return torch.clamp(values - tau, min=0.0)


class _EntmaxBisectFunction(torch.autograd.Function):
    """alpha-entmax via bisection with the exact analytic backward."""

    @staticmethod
    def forward(ctx, values: Tensor, alpha: float, dim: int, n_iter: int) -> Tensor:
        ctx.alpha = float(alpha)
        ctx.dim = dim
        alpha_minus_one = ctx.alpha - 1.0
        exponent = 1.0 / alpha_minus_one
        scaled = values * alpha_minus_one
        max_val, _ = scaled.max(dim=dim, keepdim=True)
        tau_lo = max_val - 1.0
        tau_hi = max_val.clone()
        probabilities = torch.clamp(scaled - tau_lo, min=0.0) ** exponent
        for _ in range(n_iter):
            tau = (tau_lo + tau_hi) / 2.0
            probabilities = torch.clamp(scaled - tau, min=0.0) ** exponent
            overshot = probabilities.sum(dim=dim, keepdim=True) < 1.0
            tau_hi = torch.where(overshot, tau, tau_hi)
            tau_lo = torch.where(overshot, tau_lo, tau)
        probabilities = probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(1e-12)
        ctx.save_for_backward(probabilities)
        return probabilities

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (probabilities,) = ctx.saved_tensors
        support_gradient = torch.where(
            probabilities > 0,
            probabilities ** (2.0 - ctx.alpha),
            torch.zeros_like(probabilities),
        )
        grad_input = grad_output * support_gradient
        normalizer = grad_input.sum(dim=ctx.dim, keepdim=True) / support_gradient.sum(
            dim=ctx.dim, keepdim=True
        ).clamp_min(1e-12)
        grad_input = grad_input - normalizer * support_gradient
        return grad_input, None, None, None


def entmax_bisect(values: Tensor, alpha: float = 1.5, dim: int = -1, n_iter: int = 50) -> Tensor:
    """Native, differentiable alpha-entmax with exact zeros for alpha > 1."""
    if alpha <= 1.0:
        return torch.softmax(values, dim=dim)
    return _EntmaxBisectFunction.apply(values, alpha, dim, n_iter)


class IdentitySparsifier(nn.Module):
    def forward(self, values: Tensor) -> Tensor:
        return values


class SparsemaxSparsifier(nn.Module):
    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, values: Tensor) -> Tensor:
        return sparsemax(values / max(self.temperature, 1e-6), dim=-1)


class EntmaxSparsifier(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, values: Tensor) -> Tensor:
        return entmax_bisect(values, alpha=self.alpha, dim=-1)


class KWinnersSparsifier(nn.Module):
    """Top-k sparsifier with optional selection-only anti-collapse controllers."""

    BALANCE_UPPER_FREQUENCY_RATIO = 2.0

    forward_mutable_buffer_names = (
        "duty_cycle",
        "_boost_step",
        "balance_bias",
        "_steps_since_win",
        "_ever_rescued",
        "_noise_step",
        "_k_anneal_step",
    )

    def __init__(
        self,
        k_fraction: float,
        num_units: int,
        boost_strength: float = 0.0,
        boost_update_rate: float = 0.01,
        boost_anneal_steps: int = 0,
        balance_bias_rate: float = 0.0,
        balance_bias_strategy: str = "age_liveness",
        balance_liveness_rate_ratio: float = 0.1,
        balance_liveness_patience_steps: int = 1,
        selection_noise_scale: float = 0.0,
        selection_noise_anneal_steps: int = 0,
        selection_noise_final_scale: float = 0.0,
        balance_bias_clamp: float = 0.0,
        balance_bias_leak: float = 0.0,
        k_anneal_start: int = 0,
        k_anneal_steps: int = 0,
        binarize: bool = False,
    ) -> None:
        super().__init__()
        if num_units <= 0:
            raise ValueError(f"KWinnersSparsifier num_units must be positive; got {num_units}.")
        if k_anneal_start < 0 or k_anneal_steps < 0:
            raise ValueError("k-Winners k annealing values must be nonnegative.")
        if boost_strength > 0.0 and balance_bias_rate > 0.0:
            raise ValueError("Legacy k-winners boost and balance bias cannot both be enabled.")
        if balance_bias_strategy not in {"age_liveness", "load_sign"}:
            raise ValueError(
                "k-Winners balance bias strategy must be 'age_liveness' or 'load_sign'; "
                f"got {balance_bias_strategy!r}."
            )
        if balance_bias_strategy != "age_liveness" and balance_bias_rate == 0.0:
            raise ValueError("k-Winners load-sign balance requires a positive bias rate.")
        if not 0.0 < balance_liveness_rate_ratio <= 1.0:
            raise ValueError("k-Winners balance liveness rate ratio must be in (0, 1].")
        if balance_liveness_patience_steps <= 0:
            raise ValueError("k-Winners balance liveness patience must be positive.")
        if balance_bias_clamp < 0.0:
            raise ValueError("k-Winners balance bias clamp must be nonnegative.")
        if not 0.0 <= balance_bias_leak < 1.0:
            raise ValueError("k-Winners balance bias leak must be in [0, 1).")
        k_anneal_configured = k_anneal_start > 0 or k_anneal_steps > 0
        if (k_anneal_start > 0) != (k_anneal_steps > 0):
            raise ValueError(
                "k-Winners k annealing requires positive k_anneal_start and k_anneal_steps."
            )
        if k_anneal_configured and (
            boost_strength > 0.0
            or balance_bias_rate > 0.0
            or selection_noise_scale > 0.0
            or selection_noise_anneal_steps > 0
            or selection_noise_final_scale > 0.0
        ):
            raise ValueError(
                "k-Winners k annealing cannot be combined with boost, balance bias, or "
                "selection noise."
            )
        self.k_fraction = k_fraction
        self.num_units = int(num_units)
        self.binarize = bool(binarize)
        self.boost_strength = float(boost_strength)
        self.boost_update_rate = float(boost_update_rate)
        self.boost_anneal_steps = int(boost_anneal_steps)
        self.balance_bias_rate = float(balance_bias_rate)
        self.balance_bias_strategy = balance_bias_strategy
        self.balance_liveness_rate_ratio = float(balance_liveness_rate_ratio)
        self.balance_liveness_patience_steps = int(balance_liveness_patience_steps)
        self.selection_noise_scale = float(selection_noise_scale)
        self.selection_noise_anneal_steps = int(selection_noise_anneal_steps)
        self.selection_noise_final_scale = float(selection_noise_final_scale)
        self.balance_bias_clamp = float(balance_bias_clamp)
        self.balance_bias_leak = float(balance_bias_leak)
        self.k_anneal_start = int(k_anneal_start)
        self.k_anneal_steps = int(k_anneal_steps)
        self.last_auxiliary_outputs: dict[str, Tensor] = {}
        self.register_buffer(
            "duty_cycle",
            torch.full((self.num_units,), self._target_duty_cycle()),
            persistent=False,
        )
        self.register_buffer("_boost_step", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("balance_bias", torch.zeros(self.num_units), persistent=True)
        self.register_buffer(
            "_steps_since_win",
            torch.zeros(self.num_units, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "_ever_rescued",
            torch.zeros(self.num_units, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer("_noise_step", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("_k_anneal_step", torch.zeros((), dtype=torch.long), persistent=True)

    def _target_duty_cycle(self) -> float:
        return max(1, int(round(self.num_units * self.k_fraction))) / self.num_units

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-1] != self.num_units:
            raise ValueError(
                f"KWinnersSparsifier expected {self.num_units} units, got {values.shape[-1]}."
            )
        final_k_active = max(1, int(round(self.num_units * self.k_fraction)))
        k_anneal_active = self.k_anneal_start > 0
        k_active = self._step_effective_k(final_k_active)
        boosting = self.boost_strength > 0.0 and self.training
        balance_configured = self.balance_bias_rate > 0.0
        balance_active = balance_configured and self.training
        noise_configured = self.selection_noise_scale > 0.0
        controller_configured = balance_configured or noise_configured
        effective_noise_scale = (
            self._step_selection_noise_scale()
            if self.training and noise_configured
            else 0.0
        )
        normalized_scores = None
        deterministic_selection_scores = None
        if controller_configured:
            detached_values = values.detach()
            score_mean = detached_values.mean(dim=-1, keepdim=True)
            score_spread = detached_values.std(dim=-1, keepdim=True)
            safe_spread = score_spread.clamp_min(torch.finfo(values.dtype).eps)
            normalized_scores = (values - score_mean) / safe_spread
            selection_scores = normalized_scores
            if balance_active:
                selection_scores = selection_scores + self.balance_bias
            if boosting:
                strength = self._step_boost_strength()
                selection_scores = selection_scores + (
                    self._boost_bonus(values, self.num_units, k_active, strength) / safe_spread
                )
            deterministic_selection_scores = selection_scores
            if effective_noise_scale > 0.0:
                selection_scores = (
                    selection_scores + effective_noise_scale * torch.randn_like(values)
                )
        elif boosting:
            strength = self._step_boost_strength()
            selection_scores = values + self._boost_bonus(
                values, self.num_units, k_active, strength
            )
        else:
            selection_scores = values
        _topk_values, topk_indices = torch.topk(selection_scores, k_active, dim=-1)
        mask = torch.zeros_like(values, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)
        if boosting:
            self._update_duty_cycle(mask, self.num_units)
        deterministic_mask = None
        if balance_active:
            balance_mask = mask
            if effective_noise_scale > 0.0:
                if deterministic_selection_scores is None:
                    raise RuntimeError("Balance bias requires deterministic selection scores.")
                deterministic_indices = torch.topk(
                    deterministic_selection_scores,
                    k_active,
                    dim=-1,
                ).indices
                balance_mask = torch.zeros_like(mask).scatter_(
                    -1,
                    deterministic_indices,
                    True,
                )
                deterministic_mask = balance_mask
            self._update_balance_bias(balance_mask, k_active)
        if controller_configured:
            if normalized_scores is None:
                raise RuntimeError("k-winners controller diagnostics require normalized scores.")
            self._record_controller_diagnostics(
                normalized_scores,
                mask,
                k_active,
                effective_noise_scale,
                deterministic_mask=deterministic_mask,
            )
        else:
            self.last_auxiliary_outputs = {}
        if k_anneal_active:
            self.last_auxiliary_outputs.update(
                {
                    "kwinners.effective_k": values.new_tensor(float(k_active)),
                    "kwinners.k_anneal_step": values.new_tensor(float(self._k_anneal_step)),
                }
            )
        masked = torch.where(mask, values, torch.zeros_like(values))
        if self.binarize:
            binary = mask.to(values.dtype)
            return binary + (masked - masked.detach())
        return masked

    def _boost_bonus(
        self, values: Tensor, num_units: int, k_active: int, strength: float
    ) -> Tensor:
        target_duty = k_active / num_units
        coefficient = (strength * (1.0 - self.duty_cycle / target_duty)).clamp(-3.0, 3.0)
        spread = values.detach().std(dim=-1, keepdim=True)
        return coefficient * spread

    def _step_boost_strength(self) -> float:
        if self.boost_anneal_steps <= 0:
            return self.boost_strength
        progress = min(1.0, float(self._boost_step) / self.boost_anneal_steps)
        self._boost_step += 1
        return self.boost_strength * (1.0 - progress)

    def _step_selection_noise_scale(self) -> float:
        if self.selection_noise_anneal_steps <= 0:
            return self.selection_noise_scale
        step = min(int(self._noise_step), self.selection_noise_anneal_steps)
        progress = step / self.selection_noise_anneal_steps
        scale = self.selection_noise_scale + progress * (
            self.selection_noise_final_scale - self.selection_noise_scale
        )
        if step < self.selection_noise_anneal_steps:
            self._noise_step += 1
        return scale

    def _step_effective_k(self, final_k_active: int) -> int:
        if self.k_anneal_start <= 0:
            return final_k_active
        if self.k_anneal_start > self.num_units:
            raise ValueError(
                "k-Winners k_anneal_start must not exceed num_units; "
                f"got {self.k_anneal_start} > {self.num_units}."
            )
        if self.k_anneal_start < final_k_active:
            raise ValueError(
                "k-Winners k_anneal_start must be at least the final k; "
                f"got {self.k_anneal_start} < {final_k_active}."
            )
        step = min(int(self._k_anneal_step), self.k_anneal_steps)
        progress = step / self.k_anneal_steps
        effective_k = round(
            self.k_anneal_start + progress * (final_k_active - self.k_anneal_start)
        )
        if self.training and step < self.k_anneal_steps:
            self._k_anneal_step += 1
        return int(effective_k)

    @torch.no_grad()
    def _update_duty_cycle(self, mask: Tensor, num_units: int) -> None:
        selected_frequency = mask.to(self.duty_cycle.dtype).reshape(-1, num_units).mean(dim=0)
        self.duty_cycle.mul_(1.0 - self.boost_update_rate).add_(
            self.boost_update_rate * selected_frequency
        )

    @torch.no_grad()
    def _update_balance_bias(self, mask: Tensor, k_active: int) -> None:
        win_frequency = mask.reshape(-1, self.num_units).float().mean(dim=0)
        won = win_frequency > 0.0
        self._steps_since_win.add_(1)
        self._steps_since_win.masked_fill_(won, 0)
        target_frequency = k_active / self.num_units
        if self.balance_bias_strategy == "load_sign":
            load_error = target_frequency - win_frequency
            delta = self.balance_bias_rate * torch.sign(load_error)
            self.balance_bias.add_(delta)
            if self.balance_bias_leak > 0.0:
                self.balance_bias.mul_(1.0 - self.balance_bias_leak)
            self.balance_bias.sub_(self.balance_bias.mean())
            if self.balance_bias_clamp > 0.0:
                self.balance_bias.clamp_(-self.balance_bias_clamp, self.balance_bias_clamp)
            self._ever_rescued.logical_or_(delta > 0.0)
            return
        upper_frequency = self.BALANCE_UPPER_FREQUENCY_RATIO * target_frequency
        overactive = win_frequency > upper_frequency
        stale = self._steps_since_win >= self.balance_liveness_patience_steps
        liveness_rate = self.balance_bias_rate * self.balance_liveness_rate_ratio
        delta = torch.zeros_like(self.balance_bias)
        delta.masked_fill_(overactive, -self.balance_bias_rate)
        delta.masked_fill_(stale, liveness_rate)
        self.balance_bias.add_(delta)
        self._ever_rescued.logical_or_(stale)

    @property
    def balance_liveness_rate(self) -> float:
        return self.balance_bias_rate * self.balance_liveness_rate_ratio

    @torch.no_grad()
    def _record_controller_diagnostics(
        self,
        normalized_scores: Tensor,
        mask: Tensor,
        k_active: int,
        effective_noise_scale: float,
        *,
        deterministic_mask: Tensor | None = None,
    ) -> None:
        win_frequency = mask.reshape(-1, self.num_units).float().mean(dim=0)
        win_mass = win_frequency / win_frequency.sum().clamp_min(
            torch.finfo(win_frequency.dtype).eps
        )
        normalized_entropy = -(
            win_mass * win_mass.clamp_min(torch.finfo(win_mass.dtype).eps).log()
        ).sum() / torch.as_tensor(float(self.num_units), device=win_mass.device).log()
        if k_active < self.num_units:
            boundary_scores = torch.topk(normalized_scores, k_active + 1, dim=-1).values
            boundary_margin = (
                boundary_scores[..., k_active - 1] - boundary_scores[..., k_active]
            ).mean()
        else:
            boundary_margin = normalized_scores.new_zeros(())
        target_frequency = normalized_scores.new_tensor(k_active / self.num_units)
        mean_absolute_load_error = (win_frequency - target_frequency).abs().mean()
        max_load_violation = (
            (win_frequency.max() - target_frequency) / target_frequency.clamp_min(1e-12)
        )
        bias_to_boundary_margin = (
            self.balance_bias.abs().mean() / boundary_margin
            if k_active < self.num_units
            else boundary_margin
        )
        diagnostics = {
            "kwinners.balance_bias_min": self.balance_bias.min().detach(),
            "kwinners.balance_bias_max": self.balance_bias.max().detach(),
            "kwinners.balance_bias_mean_abs": self.balance_bias.abs().mean().detach(),
            "kwinners.balance_bias_std": self.balance_bias.std(unbiased=False).detach(),
            "kwinners.liveness_age_max": self._steps_since_win.max()
            .detach()
            .to(dtype=normalized_scores.dtype),
            "kwinners.liveness_age_mean": self._steps_since_win.float().mean().detach(),
            "kwinners.liveness_stale_fraction": (
                self._steps_since_win >= self.balance_liveness_patience_steps
            )
            .float()
            .mean()
            .detach(),
            "kwinners.liveness_ever_rescued_fraction": self._ever_rescued.float()
            .mean()
            .detach(),
            "kwinners.selection_noise_scale": normalized_scores.new_tensor(
                effective_noise_scale
            ),
            "kwinners.normalized_boundary_margin_mean": boundary_margin.detach(),
            "kwinners.balance_bias_to_boundary_margin": bias_to_boundary_margin.detach(),
            "kwinners.recruited_unit_fraction": (win_frequency > 0).float().mean().detach(),
            "kwinners.max_win_frequency": win_frequency.max().detach(),
            "kwinners.mean_absolute_load_error": mean_absolute_load_error.detach(),
            "kwinners.max_load_violation": max_load_violation.detach(),
            "kwinners.win_frequency_entropy": normalized_entropy.detach(),
        }
        if deterministic_mask is not None:
            deterministic_win_frequency = deterministic_mask.reshape(
                -1,
                self.num_units,
            ).float().mean(dim=0)
            winner_agreement = (mask & deterministic_mask).sum(dim=-1).float() / k_active
            diagnostics.update(
                {
                    "kwinners.deterministic_recruited_unit_fraction": (
                        deterministic_win_frequency > 0
                    ).float().mean().detach(),
                    "kwinners.noisy_deterministic_winner_agreement": (
                        winner_agreement.mean().detach()
                    ),
                }
            )
        self.last_auxiliary_outputs = diagnostics


class GroupedKWinnersSparsifier(nn.Module):
    """Two-stage hard top-k: micro-columns of fixed, unlearned membership."""

    forward_mutable_buffer_names = ("group_bias",)

    def __init__(
        self,
        k_fraction: float,
        num_units: int,
        num_groups: int,
        group_bias_rate: float = 0.0,
        group_bias_clamp: float = 0.0,
        group_bias_leak: float = 0.0,
    ) -> None:
        super().__init__()
        if num_units <= 0 or num_groups <= 0:
            raise ValueError("GroupedKWinnersSparsifier needs positive num_units and num_groups.")
        if group_bias_rate < 0.0 or group_bias_clamp < 0.0:
            raise ValueError("Grouped k-winners group bias rate and clamp must be nonnegative.")
        if not 0.0 <= group_bias_leak < 1.0:
            raise ValueError("Grouped k-winners group bias leak must be in [0, 1).")
        if num_units % num_groups != 0:
            raise ValueError(
                f"GroupedKWinnersSparsifier num_groups={num_groups} must divide "
                f"num_units={num_units}."
            )
        self.num_units = int(num_units)
        self.num_groups = int(num_groups)
        self.units_per_group = self.num_units // self.num_groups
        self.k_active = max(1, int(round(self.num_units * k_fraction)))
        if self.k_active > self.num_groups:
            raise ValueError(
                f"GroupedKWinnersSparsifier keeps one winner per group, so k={self.k_active} "
                f"cannot exceed num_groups={self.num_groups}."
            )
        self.group_bias_rate = float(group_bias_rate)
        self.group_bias_clamp = float(group_bias_clamp)
        self.group_bias_leak = float(group_bias_leak)
        self.last_auxiliary_outputs: dict[str, Tensor] = {}
        self.register_buffer("group_bias", torch.zeros(self.num_groups), persistent=True)

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-1] != self.num_units:
            raise ValueError(
                f"GroupedKWinnersSparsifier expected {self.num_units} units, "
                f"got {values.shape[-1]}."
            )
        grouped = values.reshape(*values.shape[:-1], self.num_groups, self.units_per_group)
        group_best, group_argmax = grouped.max(dim=-1)
        group_scores = group_best
        if self.group_bias_rate > 0.0:
            group_spread = group_best.detach().std(dim=-1, keepdim=True).clamp_min(
                torch.finfo(values.dtype).eps
            )
            group_scores = group_best + self.group_bias * group_spread
        _values, active_groups = torch.topk(group_scores, self.k_active, dim=-1)
        group_mask = torch.zeros_like(group_best, dtype=torch.bool)
        group_mask.scatter_(-1, active_groups, True)
        unit_mask = torch.zeros_like(grouped, dtype=torch.bool)
        unit_mask.scatter_(-1, group_argmax.unsqueeze(-1), True)
        mask = (unit_mask & group_mask.unsqueeze(-1)).reshape(values.shape)
        if self.group_bias_rate > 0.0 and self.training:
            self._update_group_bias(group_mask)
        with torch.no_grad():
            unit_win_frequency = mask.reshape(-1, self.num_units).float().mean(dim=0)
            group_win_frequency = group_mask.reshape(-1, self.num_groups).float().mean(dim=0)
            self.last_auxiliary_outputs = {
                "kwinners.effective_k": values.new_tensor(float(self.k_active)),
                "grouped_kwinners.unit_dead_fraction": (unit_win_frequency == 0).float().mean(),
                "grouped_kwinners.group_dead_fraction": (group_win_frequency == 0)
                .float()
                .mean(),
                "grouped_kwinners.group_max_load_ratio": group_win_frequency.max()
                / (self.k_active / self.num_groups),
                "grouped_kwinners.group_bias_max_abs": self.group_bias.abs().max(),
                "grouped_kwinners.group_bias_std": self.group_bias.std(unbiased=False),
            }
        return torch.where(mask, values, torch.zeros_like(values))

    @torch.no_grad()
    def _update_group_bias(self, group_mask: Tensor) -> None:
        """Sign step on the per-group load error, mirroring the per-unit load_sign guard."""
        group_win_frequency = group_mask.reshape(-1, self.num_groups).float().mean(dim=0)
        target_frequency = self.k_active / self.num_groups
        self.group_bias.add_(
            self.group_bias_rate * torch.sign(target_frequency - group_win_frequency)
        )
        if self.group_bias_leak > 0.0:
            self.group_bias.mul_(1.0 - self.group_bias_leak)
        self.group_bias.sub_(self.group_bias.mean())
        if self.group_bias_clamp > 0.0:
            self.group_bias.clamp_(-self.group_bias_clamp, self.group_bias_clamp)


class BlockwiseSparsifier(nn.Module):
    def __init__(self, blocks: list[tuple[int, int, nn.Module]], width: int) -> None:
        super().__init__()
        self.ranges = [(int(start), int(end)) for start, end, _module in blocks]
        self.blocks = nn.ModuleList([module for _start, _end, module in blocks])
        self.width = int(width)
        self._validate_ranges()

    def _validate_ranges(self) -> None:
        expected_start = 0
        for start, end in self.ranges:
            if start != expected_start or end <= start:
                raise ValueError(
                    "BlockwiseSparsifier blocks must tile the last dimension without gaps "
                    f"or overlaps; got {self.ranges}."
                )
            expected_start = end
        if expected_start != self.width:
            raise ValueError(
                "BlockwiseSparsifier blocks must tile the configured width; "
                f"got end={expected_start}, width={self.width}."
            )

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-1] != self.width:
            raise ValueError(
                f"BlockwiseSparsifier expected last dimension {self.width}, "
                f"got {values.shape[-1]}."
            )
        outputs = [
            block(values[..., start:end])
            for (start, end), block in zip(self.ranges, self.blocks, strict=False)
        ]
        return torch.cat(outputs, dim=-1)


class LateralInhibitionSparsifier(nn.Module):
    """Subtract a fraction of the competitor mean before top-k, then select."""

    def __init__(self, k_fraction: float, strength: float, rectify: bool) -> None:
        super().__init__()
        self.k_fraction = k_fraction
        self.strength = strength
        self.rectify = bool(rectify)

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-1] <= 1:
            inhibited = values
        else:
            competitor_mean = (values.sum(dim=-1, keepdim=True) - values) / (values.shape[-1] - 1)
            inhibited = values - self.strength * competitor_mean
        if self.rectify:
            inhibited = inhibited.clamp_min(0.0)
        k_active = max(1, int(round(values.shape[-1] * self.k_fraction)))
        _topk_values, topk_indices = torch.topk(inhibited, k_active, dim=-1)
        mask = torch.zeros_like(values, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)
        return torch.where(mask, inhibited, torch.zeros_like(inhibited))


class SoftWTASparsifier(nn.Module):
    def __init__(self, beta: float, learnable_beta: bool, target_sparsity: float) -> None:
        super().__init__()
        initial_beta = torch.tensor(float(beta))
        self.beta = nn.Parameter(initial_beta) if learnable_beta else initial_beta
        self.target_sparsity = target_sparsity

    def forward(self, values: Tensor) -> Tensor:
        beta = (
            self.beta
            if isinstance(self.beta, Tensor)
            else torch.tensor(self.beta, device=values.device)
        )
        centered = values - values.mean(dim=-1, keepdim=True)
        gates = torch.sigmoid(beta * centered)
        threshold = torch.quantile(
            gates,
            q=max(0.0, min(1.0, 1.0 - self.target_sparsity)),
            dim=-1,
            keepdim=True,
        )
        hard_mask = (gates >= threshold).to(values.dtype)
        return values * hard_mask
