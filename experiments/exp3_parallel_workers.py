"""
Exp 3 (course spec):
- Engineering: wall time to finish the same number of training iterations for different worker counts.
- Algorithm: plot loss vs iteration and gradient-norm dispersion vs iteration (more workers -> richer buffer -> typically smoother updates).

Uses TrainPipeline from train.py; only this file + plots/CSV under experiments/outputs/ are added.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import TrainPipeline


class _NoMetrics:
    def write(self, row):
        pass

    def close(self):
        pass


class Exp3Train(TrainPipeline):
    """Same as TrainPipeline but no checkpoint resume, no metrics file writes."""

    def _try_resume_from_checkpoint(self):
        return

    def __init__(self, self_play_workers: int, use_gpu=None):
        super().__init__(use_gpu=use_gpu, self_play_workers=self_play_workers)
        try:
            self._metrics.close()
        except Exception:
            pass
        self._metrics = _NoMetrics()

    def policy_update(self):
        """Same logic as TrainPipeline.policy_update, plus grad L2 norms before each optimizer step."""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        self.policy_value_net.set_lr_multiplier(self.lr_multiplier)

        state_batch_np = np.ascontiguousarray(np.asarray(state_batch, dtype=np.float32))
        mcts_probs_np = np.ascontiguousarray(np.asarray(mcts_probs_batch, dtype=np.float32))
        winner_batch_np = np.ascontiguousarray(np.asarray(winner_batch, dtype=np.float32))
        state_batch_t = torch.from_numpy(state_batch_np).to(self.policy_value_net.device)
        mcts_probs_t = torch.from_numpy(mcts_probs_np).to(self.policy_value_net.device)
        winner_batch_t = torch.from_numpy(winner_batch_np).to(self.policy_value_net.device)

        grad_norms: list[float] = []
        policy_loss = value_loss = None
        kl = 0.0
        loss_t = None

        for _ in range(self.epochs):
            self.policy_value_net.optimizer.zero_grad()
            self.policy_value_net.policy_value_net.train()
            log_act_probs, value = self.policy_value_net.policy_value_net(state_batch_t)
            value_loss = F.mse_loss(value.view(-1), winner_batch_t)
            policy_loss = -torch.mean(torch.sum(mcts_probs_t * log_act_probs, 1))
            loss_t = value_loss + policy_loss
            loss_t.backward()
            sq = 0.0
            for p in self.policy_value_net.policy_value_net.parameters():
                if p.grad is not None:
                    sq += float(p.grad.detach().double().pow(2).sum().cpu())
            grad_norms.append(sq**0.5)
            self.policy_value_net.optimizer.step()
            self.global_step += 1
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = float(
                np.mean(
                    np.sum(
                        old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                        axis=1,
                    )
                )
            )
            if kl > self.kl_targ * 4:
                break

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        self.policy_value_net.policy_value_net.train()
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net.policy_value_net(state_batch_t)
            entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))

        self.last_loss = float(loss_t.item())
        self.last_entropy = float(entropy.item())
        self.last_kl = kl
        self._last_grad_norm_mean = float(np.mean(grad_norms)) if grad_norms else float("nan")
        self._last_grad_norm_std = float(np.std(grad_norms)) if len(grad_norms) > 1 else 0.0

        return (
            self.last_loss,
            self.last_entropy,
            float(policy_loss.item()),
            float(value_loss.item()),
        )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed % (2**32))


def _load_weights(pipe: Exp3Train, model_path: Path) -> None:
    sd = torch.load(str(model_path), map_location=pipe.policy_value_net.device, weights_only=True)
    pipe.policy_value_net.policy_value_net.load_state_dict(sd)


def run_iterations(pipe: Exp3Train) -> tuple[list[dict], float]:
    """Run game_batch_num outer iterations (collect workers games + maybe train). No arena eval."""
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i in range(pipe.start_batch, pipe.game_batch_num):
        pipe.collect_selfplay_data(pipe.play_batch_size)
        if len(pipe.data_buffer) <= pipe.batch_size:
            continue
        t_u = time.perf_counter()
        loss, ent, _, _ = pipe.policy_update()
        dt_u = time.perf_counter() - t_u
        rows.append(
            {
                "iteration": i + 1,
                "loss": loss,
                "entropy": ent,
                "grad_norm_mean": pipe._last_grad_norm_mean,
                "grad_norm_std": pipe._last_grad_norm_std,
                "update_s": dt_u,
                "wall_s": time.perf_counter() - t0,
            }
        )
    total_s = time.perf_counter() - t0
    return rows, total_s


def _write_csv(path: Path, all_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not all_rows:
        return
    keys = list(all_rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)


def _plot_engineering(out: Path, worker_times: list[tuple[int, float]]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    ws = [w for w, _ in worker_times]
    ts = [t for _, t in worker_times]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(w) for w in ws], ts, color="steelblue")
    ax.set_xlabel("self_play_workers")
    ax.set_ylabel("wall time (s)")
    ax.set_title("Exp3 — same training iterations, time vs workers")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_algorithm(out: Path, all_rows: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not all_rows:
        return
    by_w: dict[int, list[dict]] = {}
    for r in all_rows:
        by_w.setdefault(int(r["workers"]), []).append(r)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for w in sorted(by_w.keys()):
        pts = sorted(by_w[w], key=lambda x: x["iteration"])
        it = [p["iteration"] for p in pts]
        loss = [p["loss"] for p in pts]
        gstd = [p["grad_norm_std"] for p in pts]
        axes[0].plot(it, loss, label=f"workers={w}")
        axes[1].plot(it, gstd, label=f"workers={w}")

    axes[0].set_ylabel("loss (after update)")
    axes[0].set_title("Training dynamics vs iteration")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("std(grad L2 per epoch block)")
    axes[1].set_xlabel("iteration (game batch index)")
    axes[1].set_title("Gradient dispersion proxy (lower often means stabler minibatch updates)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="initial weights (.pth state_dict)")
    p.add_argument("--iterations", type=int, default=40, help="outer game_batches (same for every worker)")
    p.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", action="store_true")
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs" / "exp3"),
    )
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    worker_times: list[tuple[int, float]] = []
    all_rows: list[dict] = []

    for w in args.workers:
        if w < 1:
            continue
        _set_seed(args.seed)
        pipe = Exp3Train(self_play_workers=w, use_gpu=args.gpu if args.gpu else None)
        pipe.check_freq = 10**9
        pipe.game_batch_num = args.iterations
        pipe.start_batch = 0
        pipe.global_step = 0
        pipe.lr_multiplier = 1.0
        pipe.data_buffer.clear()

        _load_weights(pipe, model_path)

        rows, total_s = run_iterations(pipe)
        worker_times.append((w, total_s))
        for r in rows:
            r["workers"] = w
            all_rows.append(r)
        print(f"workers={w}: wall_s={total_s:.2f} training_rows={len(rows)}")

    csv_path = out_dir / "exp3_runs.csv"
    _write_csv(csv_path, all_rows)
    _plot_engineering(out_dir / "exp3_walltime.png", worker_times)
    _plot_algorithm(out_dir / "exp3_algorithm.png", all_rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
