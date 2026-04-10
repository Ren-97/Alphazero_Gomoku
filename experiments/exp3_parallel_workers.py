"""
Exp 3: Train from scratch (same seed, no checkpoint) with different self-play worker counts;
compare wall time and periodic AZ vs Pure win rate.

Outputs: experiments/outputs/exp3/(CSV + PNGs).

Run:
    python experiments/exp3_parallel_workers.py
"""

from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from train import TrainPipeline

# --- experiment knobs ---
workers = [1, 4, 8, 16]
iterations = 300
eval_every = 10
eval_games = 50
seed = 123
use_gpu = True
out_dir = Path(__file__).resolve().parent / "outputs" / "exp3"


class _NoMetrics:
    def write(self, row):
        pass

    def close(self):
        pass


class Exp3Train(TrainPipeline):
    def _try_resume_from_checkpoint(self):
        return

    def __init__(self, self_play_workers: int, use_gpu=None):
        super().__init__(use_gpu=use_gpu, self_play_workers=self_play_workers)
        try:
            self._metrics.close()
        except Exception:
            pass
        self._metrics = _NoMetrics()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed % (2**32))


def run_branch(pipe: Exp3Train, workers: int, eval_every: int, eval_games: int) -> tuple[list[dict], float]:
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i in range(pipe.game_batch_num):
        pipe.collect_selfplay_data(pipe.play_batch_size)
        loss_s = ""
        if len(pipe.data_buffer) > pipe.batch_size:
            loss, _, _, _ = pipe.policy_update()
            loss_s = f"{loss:.6f}"
        base_s = ""
        if (i + 1) % eval_every == 0:
            wr, _ = pipe._evaluate_current_vs_pure(eval_games)
            base_s = f"{wr:.6f}"
        wall_s = round(time.perf_counter() - t0, 2)
        msg = f"[w={workers}] iter {i + 1}/{pipe.game_batch_num}  wall_s={wall_s}"
        if loss_s:
            msg += f"  loss={loss_s}"
        if base_s:
            msg += f"  wr_vs_pure={base_s}"
        print(msg)
        rows.append(
            {
                "iteration": i + 1,
                "workers": workers,
                "wall_s": wall_s,
                "loss": loss_s,
                "baseline_vs_pure": base_s,
            }
        )
    return rows, time.perf_counter() - t0


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_walltime(path: Path, wt: list[tuple[int, float]]) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar([str(a) for a, _ in wt], [b for _, b in wt], color="steelblue")
    ax.set_xlabel("workers")
    ax.set_ylabel("wall time (s)")
    ax.set_title(f"Total wall time ({iterations} iterations)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_winrate_iter(path: Path, rows: list[dict]) -> None:
    by_w: dict[int, list[tuple[int, float]]] = {}
    for r in rows:
        if not r.get("baseline_vs_pure"):
            continue
        w = int(r["workers"])
        by_w.setdefault(w, []).append((int(r["iteration"]), float(r["baseline_vs_pure"])))
    if not by_w:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for w in sorted(by_w.keys()):
        pts = sorted(by_w[w], key=lambda x: x[0])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=f"w={w}", ms=3)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("win rate vs Pure MCTS")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_winrate_wall(path: Path, rows: list[dict]) -> None:
    by_w: dict[int, list[tuple[float, float]]] = {}
    for r in rows:
        if not r.get("baseline_vs_pure"):
            continue
        w = int(r["workers"])
        by_w.setdefault(w, []).append((float(r["wall_s"]), float(r["baseline_vs_pure"])))
    if not by_w:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for w in sorted(by_w.keys()):
        pts = sorted(by_w[w], key=lambda x: x[0])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=f"w={w}", ms=3)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("win rate vs Pure MCTS")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    walltimes: list[tuple[int, float]] = []

    for w in workers:
        if w < 1:
            continue
        _set_seed(seed)
        pipe = Exp3Train(self_play_workers=w, use_gpu=use_gpu)
        pipe.check_freq = 10**9
        pipe.game_batch_num = iterations
        pipe.start_batch = 0
        pipe.global_step = 0
        pipe.lr_multiplier = 1.0
        pipe.data_buffer.clear()

        rows, total_s = run_branch(pipe, w, eval_every, eval_games)
        all_rows.extend(rows)
        walltimes.append((w, total_s))
        print(f"workers={w}  wall_s={total_s:.1f}  rows={len(rows)}")

    _csv(out_dir / "exp3_runs.csv", all_rows)
    _plot_walltime(out_dir / "exp3_walltime.png", walltimes)
    _plot_winrate_iter(out_dir / "exp3_winrate_vs_iteration.png", all_rows)
    _plot_winrate_wall(out_dir / "exp3_winrate_vs_walltime.png", all_rows)
    print(out_dir / "exp3_runs.csv")


if __name__ == "__main__":
    main()
