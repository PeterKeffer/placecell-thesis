"""Summarize completed policy evaluations with paired outcomes and failure-loop support."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def absorbing_tail(episode, limit):
    if episode["success"] or episode["steps"] != limit:
        return False
    poses = np.asarray(episode["positions_and_heading"][-64:])
    if len(poses) != 64:
        return False
    one = np.allclose(poses, poses[-1], atol=1e-5, rtol=0)
    two = np.allclose(poses[2:], poses[:-2], atol=1e-5, rtol=0)
    return bool(one or two)


def main(directory):
    plan = json.loads((directory / "plan.json").read_text())
    available = {}
    for job in plan:
        path = directory / f"{job['name']}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result.get("complete"):
                available[job["name"]] = result
    rows = []
    for job in plan:
        result = available.get(job["name"])
        if result is None:
            continue
        name = job["name"]
        treatment = name.rsplit("_", 1)[-1]
        baseline = available.get(name.rsplit("_", 1)[0] + "_greedy")
        episodes = result["episodes"]
        success = np.asarray([episode["success"] for episode in episodes])
        row = dict(
            algorithm=job["algorithm"],
            goal=job["goal"],
            representation=job["representation"],
            seed=job["seed"],
            treatment=treatment,
            run_id=job["run_id"],
            successful_episodes=int(success.sum()),
            evaluated_episodes=len(episodes),
            success_rate=float(success.mean()),
            failed_episodes=int((~success).sum()),
            absorbing_failure_episodes=sum(
                absorbing_tail(e, result["episode_limit"]) for e in episodes
            ),
            recovered_vs_greedy="",
            lost_vs_greedy="",
            paired_success_change="",
        )
        if baseline is not None:
            control = np.asarray([episode["success"] for episode in baseline["episodes"]])
            assert len(control) == len(success) == 30
            assert result["starts_sha256"] == baseline["starts_sha256"]
            assert result["policy_sha256"] == baseline["policy_sha256"]
            row.update(
                recovered_vs_greedy=int((success & ~control).sum()),
                lost_vs_greedy=int((~success & control).sum()),
                paired_success_change=float(success.mean() - control.mean()),
            )
        rows.append(row)
    if rows:
        with (directory / "per_policy_summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    status = dict(
        completed_conditions=len(rows),
        planned_conditions=len(plan),
        complete=len(rows) == len(plan),
        loop_definition="Final 64 poses repeat with period 1 or 2 within 1e-5; failed at limit.",
        scope="Paired episode outcomes; policy training seeds are the replication unit.",
    )
    (directory / "summary_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
