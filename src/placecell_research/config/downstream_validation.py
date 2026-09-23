"""Focused validation helpers for downstream configuration."""

from __future__ import annotations

from .downstream_schema import DownstreamRunConfig


def validate_route_stitching_position_source(config: DownstreamRunConfig) -> None:
    """Require the state prefix consumed by route stitching to be physical XY."""
    online_pcdt = config.training.online_pcdt
    if online_pcdt is None or not online_pcdt.route_stitching:
        return
    feature_sources = list(config.observation.feature_sources)
    if not feature_sources or feature_sources[0] != "current_position_xy":
        raise ValueError(
            "training.online_pcdt.route_stitching requires observation.feature_sources "
            "to start with 'current_position_xy'; route waypoints use the first two state "
            "values as physical XY coordinates."
        )
