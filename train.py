"""
An implementation of the training pipeline of AlphaZero for Gomoku
"""

from __future__ import annotations

import logging
import os
import random
import time
import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import numpy as np
from collections import defaultdict, deque
import torch
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet 
from az_logging import setup_logging
from az_metrics import MetricsWriter, DEFAULT_METRICS_FIELDS, wilson_ci


logger = logging.getLogger(__name__)


def _selfplay_worker(args: tuple) -> tuple:
    """
    Run one self-play game in a separate process. Loads policy weights from disk (CPU inference).
    """
    (
        model_path,
        board_width,
        board_height,
        n_in_row,
        temp,
        n_playout,
        c_puct,
        seed,
    ) = args
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed % (2**32))

    board = Board(width=board_width, height=board_height, n_in_row=n_in_row)
    game = Game(board)
    net = PolicyValueNet(
        board_width,
        board_height,
        model_file=model_path,
        use_gpu=False,
    )
    player = MCTSPlayer(
        net.policy_value_fn,
        c_puct=c_puct,
        n_playout=n_playout,
        is_selfplay=1,
    )
    _winner, play_data = game.start_self_play(player, temp=temp)
    play_data = list(play_data)
    serializable = [
        (np.asarray(s, dtype=np.float32), np.asarray(p, dtype=np.float32), float(z))
        for s, p, z in play_data
    ]
    return serializable


def _archive_and_clear_outputs(*, archive_root: str = "runs") -> str:
    """
    Move existing training outputs into a fresh archive folder.
    Returns the archive directory path.
    """
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    archive_dir = Path(archive_root) / f"archive_{ts}"
    archive_dir.mkdir(parents=True, exist_ok=False)

    def move_path(p: Path, rel_base: Path) -> None:
        if not p.exists():
            return
        rel = p.relative_to(rel_base)
        dest = archive_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dest))

    root = Path(".").resolve()

    # logs (train.log, metrics.csv, metrics.png, etc.)
    logs_dir = root / "logs"
    if logs_dir.exists():
        for p in logs_dir.glob("*"):
            if p.is_file():
                move_path(p, root)

    # top-level model snapshots
    for p in root.glob("*.pth"):
        if p.is_file():
            move_path(p, root)

    # checkpoints (recursive)
    ckpt_dir = root / "checkpoints"
    if ckpt_dir.exists():
        for p in ckpt_dir.rglob("*"):
            if p.is_file():
                move_path(p, root)

    return str(archive_dir)


class TrainPipeline():
    def __init__(self, use_gpu: bool | None = None, self_play_workers: int = 1):
        # params of the board and the game
        self.board_width = 8
        self.board_height = 8
        self.n_in_row = 5
        self.current_policy_path = "./current_policy_{}_{}_{}.pth".format(
            self.board_width, self.board_height, self.n_in_row
        )
        self.best_policy_path = "./best_policy_{}_{}_{}.pth".format(
            self.board_width, self.board_height, self.n_in_row
        )
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)

        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5 # Exploration Coefficient (PUCT:Predictor + Upper Confidence Bound applied to Trees)
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.self_play_workers = int(self_play_workers)
        if self.self_play_workers < 1:
            raise ValueError("self_play_workers must be >= 1")
        self.play_batch_size = self.self_play_workers
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50 
        self.game_batch_num = 1500
        
        # Evaluation / arena settings
        self.eval_games = 20  # number of games per evaluation
        self.arena_update_threshold = 0.55  # current beats best if winrate > threshold
        self.global_step = 0
        self.start_batch = 0

        # Store checkpoints per board setting to avoid loading a mismatched model (e.g., 6_6_4 vs 8_8_5).
        self.checkpoint_dir = "./checkpoints/{}_{}_{}".format(
            self.board_width, self.board_height, self.n_in_row
        )
        self.latest_checkpoint_path = os.path.join(self.checkpoint_dir, "latest.pth")
        self.best_checkpoint_path = os.path.join(self.checkpoint_dir, "best.pth")

        # num of simulations used for the pure mcts, which is used as the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000

        # Rolling stats for logging/monitoring
        self.episode_lens = deque(maxlen=200)
        self.baseline_winrates = deque(maxlen=50)
        self.last_loss = None
        self.last_entropy = None
        self.last_kl = None

        # Device selection (GPU if available unless explicitly disabled).
        if use_gpu is None:
            self.use_gpu = bool(torch.cuda.is_available())
        else:
            if use_gpu and (not torch.cuda.is_available()):
                raise SystemExit("CUDA is not available, but --use-gpu was specified.")
            self.use_gpu = bool(use_gpu)

        self.policy_value_net = PolicyValueNet(
            self.board_width,
            self.board_height,
            use_gpu=self.use_gpu,
            base_lr=self.learn_rate,
        )
        self._try_resume_from_checkpoint()

        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1,
        )

        self.metrics_path = os.path.join("logs", "metrics_{}_{}_{}.csv".format(
            self.board_width, self.board_height, self.n_in_row
        ))
        self._metrics = MetricsWriter(self.metrics_path, fieldnames=DEFAULT_METRICS_FIELDS)

        device_str = "cuda" if getattr(self.policy_value_net, "device", None) and str(self.policy_value_net.device).startswith("cuda") else "cpu"
        logger.info(
            "Config | board=%sx%s connect-%s | device=%s | n_playout=%s c_puct=%s | lr=%.4g batch=%s epochs=%s buffer=%s | check_freq=%s eval_games=%s pure_mcts_playout=%s",
            self.board_width,
            self.board_height,
            self.n_in_row,
            device_str,
            self.n_playout,
            self.c_puct,
            self.learn_rate,
            self.batch_size,
            self.epochs,
            self.buffer_size,
            self.check_freq,
            self.eval_games,
            self.pure_mcts_playout_num,
        )
        logger.info("Self-play workers: %s", self.self_play_workers)

    def _atomic_save_checkpoint(self, path: str, extra_state: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        self.policy_value_net.save_checkpoint(tmp_path, extra_state=extra_state)
        os.replace(tmp_path, path)

    def _try_resume_from_checkpoint(self) -> None:
        if not os.path.exists(self.latest_checkpoint_path):
            return
        try:
            extra = self.policy_value_net.load_checkpoint(self.latest_checkpoint_path)
            self.start_batch = int(extra.get("batch_i", 0))
            self.global_step = int(extra.get("global_step", 0))
            self.lr_multiplier = float(extra.get("lr_multiplier", self.lr_multiplier))
            self.pure_mcts_playout_num = int(extra.get("pure_mcts_playout_num", self.pure_mcts_playout_num))
            # Optional restored knobs (if present).
            self.eval_games = int(extra.get("eval_games", self.eval_games))
            self.arena_update_threshold = float(extra.get("arena_update_threshold", self.arena_update_threshold))

            # Restore RNG (Random Number Generator) state when available.
            py_rng = extra.get("python_random_state")
            if py_rng is not None:
                random.setstate(py_rng)
            np_rng = extra.get("numpy_random_state")
            if np_rng is not None:
                np.random.set_state(np_rng)
            torch_rng = extra.get("torch_random_state")
            if torch_rng is not None:
                if torch.is_tensor(torch_rng):
                    torch_rng = torch_rng.detach().cpu().to(torch.uint8)
                torch.set_rng_state(torch_rng)
            cuda_rng = extra.get("torch_cuda_random_state_all")
            if cuda_rng is not None and torch.cuda.is_available():
                if isinstance(cuda_rng, (list, tuple)):
                    cuda_rng = [
                        (s.detach().cpu().to(torch.uint8) if torch.is_tensor(s) else s)
                        for s in cuda_rng
                    ]
                torch.cuda.set_rng_state_all(cuda_rng)

            logger.info(
                "Resumed from checkpoint: %s (start_batch=%s, global_step=%s)",
                self.latest_checkpoint_path,
                self.start_batch,
                self.global_step,
            )
        except Exception as e:
            logger.warning("Skip checkpoint resume: %s (%s)", self.latest_checkpoint_path, e)

    def get_equi_data(self, play_data):
        """
        augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        buf_before = len(self.data_buffer)
        t_batch = time.time()

        if self.self_play_workers == 1:
            logger.info(
                "[Self-play] dispatch | games=%s | workers=1",
                n_games,
            )
            for idx in range(n_games):
                _winner, play_data = self.game.start_self_play(
                    self.mcts_player, temp=self.temp
                )
                play_data = list(play_data)[:]
                self.episode_len = len(play_data)
                self.episode_lens.append(self.episode_len)
                logger.info(
                    "[Self-play] collected %s/%s | moves=%s",
                    idx + 1,
                    n_games,
                    self.episode_len,
                )
                play_data = self.get_equi_data(play_data)
                self.data_buffer.extend(play_data)
            buffer_added = len(self.data_buffer) - buf_before
            logger.info(
                "[Self-play] batch done | duration=%.1fs | buffer_added=%s | buffer=%s/%s",
                time.time() - t_batch,
                buffer_added,
                len(self.data_buffer),
                self.buffer_size,
            )
            return

        self.policy_value_net.save_model(self.current_policy_path)
        model_abs = os.path.abspath(self.current_policy_path)
        logger.info(
            "[Self-play] dispatch | games=%s | workers=%s",
            n_games,
            self.self_play_workers,
        )

        base_seed = random.randint(0, 2**31 - 1)
        arg_lists = [
            (
                model_abs,
                self.board_width,
                self.board_height,
                self.n_in_row,
                self.temp,
                self.n_playout,
                self.c_puct,
                base_seed + i,
            )
            for i in range(n_games)
        ]
        # With play_batch_size == self.self_play_workers, n_games equals self.self_play_workers here.
        # max_workers = min(self.self_play_workers, n_games)
        max_workers = self.self_play_workers
        collected = 0
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_selfplay_worker, a) for a in arg_lists]
            for fut in as_completed(futures):
                play_data = fut.result()
                self.episode_len = len(play_data)
                self.episode_lens.append(self.episode_len)
                collected += 1
                logger.info(
                    "[Self-play] collected %s/%s | moves=%s",
                    collected,
                    n_games,
                    self.episode_len,
                )
                play_data = self.get_equi_data(play_data)
                self.data_buffer.extend(play_data)
        buffer_added = len(self.data_buffer) - buf_before
        logger.info(
            "[Self-play] batch done | duration=%.1fs | buffer_added=%s | buffer=%s/%s",
            time.time() - t_batch,
            buffer_added,
            len(self.data_buffer),
            self.buffer_size,
        )

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        # Apply the current KL-adaptive multiplier once per policy update.
        self.policy_value_net.set_lr_multiplier(self.lr_multiplier)
        policy_loss = None
        value_loss = None
        for i in range(self.epochs):
            loss, entropy, policy_loss, value_loss = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch)
            self.global_step += 1
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
            
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        self.last_loss = loss
        self.last_entropy = entropy
        self.last_kl = kl
        return loss, entropy, policy_loss, value_loss

    def _evaluate_current_vs_pure(self, n_games: int):
        """
        Baseline monitoring: current NN+MCTS vs pure MCTS.
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        return win_ratio, win_cnt

    def _evaluate_current_vs_best(self, n_games: int):
        """
        Arena evaluation: current NN+MCTS vs best NN+MCTS.
        Returns (win_ratio_for_current, win_cnt).
        """
        if not os.path.exists(self.best_policy_path):
            # Bootstrap: initialize best policy to current if missing.
            self.policy_value_net.save_model(self.best_policy_path)
            logger.info("Initialized best policy from current: %s", self.best_policy_path)

        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        best_policy = PolicyValueNet(
            self.board_width,
            self.board_height,
            model_file=self.best_policy_path,
            use_gpu=self.use_gpu,
        )
        best_mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          best_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        return win_ratio, win_cnt

    def _trend_label(self, series: deque) -> str:
        if len(series) < 3:
            return "N/A"
        a = float(np.mean(list(series)[-3:]))
        b = float(np.mean(list(series)[-6:-3])) if len(series) >= 6 else float(np.mean(list(series)[:-3]))
        if a > b + 0.02:
            return "Increasing"
        if a < b - 0.02:
            return "Decreasing"
        return "Flat"

    def _log_eval_block(
        self,
        *,
        iteration: int,
        arena_winrate: float,
        arena_update_best: bool,
        baseline_winrate: float,
        arena_ci: tuple[float, float],
        baseline_ci: tuple[float, float],
    ) -> None:
        line = "-" * 50
        avg_len = float(np.mean(self.episode_lens)) if self.episode_lens else float("nan")
        trend = self._trend_label(self.baseline_winrates)
        loss = "N/A" if self.last_loss is None else "{:.4f}".format(self.last_loss)
        entropy = "N/A" if self.last_entropy is None else "{:.4f}".format(self.last_entropy)
        a_lo, a_hi = arena_ci
        b_lo, b_hi = baseline_ci

        logger.info("Iteration [%s]", iteration)
        logger.info(line)
        logger.info(
            "[Arena]   Current vs Best: Winrate %.1f%% (95%% Wilson CI: %.1f%%–%.1f%%) | Result: %s",
            arena_winrate * 100.0,
            a_lo * 100.0,
            a_hi * 100.0,
            "UPDATE BEST!" if arena_update_best else "keep best",
        )
        logger.info(
            "[Monitor] Current vs Pure MCTS: Winrate %.1f%% (95%% Wilson CI: %.1f%%–%.1f%%) | Trend: %s",
            baseline_winrate * 100.0,
            b_lo * 100.0,
            b_hi * 100.0,
            trend,
        )
        logger.info(
            "[Stats]   Avg Game Length: %.1f steps | Loss: %s | Entropy: %s",
            avg_len,
            loss,
            entropy,
        )
        logger.info(line)

    def run(self):
        """run the training pipeline"""
        last_batch_i = self.start_batch
        run_device = str(getattr(self.policy_value_net, "device", "cpu"))
        try:
            for i in range(self.start_batch, self.game_batch_num):
                last_batch_i = i + 1
                logger.info(
                    "game_batch %s/%s | collect_selfplay_data (%s games)",
                    i + 1,
                    self.game_batch_num,
                    self.play_batch_size,
                )
                self.collect_selfplay_data(self.play_batch_size)
                self._metrics.write(
                    {
                        "timestamp": time.time(),
                        "phase": "selfplay",
                        "iteration": i + 1,
                        "global_step": self.global_step,
                        "episode_len": self.episode_len,
                        "buffer_size": len(self.data_buffer),
                        "loss": "",
                        "policy_loss": "",
                        "value_loss": "",
                        "entropy": "",
                        "kl": "",
                        "lr": self.policy_value_net.current_lr(),
                        "lr_multiplier": self.lr_multiplier,
                        "arena_winrate": "",
                        "baseline_winrate": "",
                        "updated_best": "",
                        "eval_games": "",
                        "board_w": self.board_width,
                        "board_h": self.board_height,
                        "n_in_row": self.n_in_row,
                        "n_playout": self.n_playout,
                        "c_puct": self.c_puct,
                        "temp": self.temp,
                        "device": run_device,
                    }
                )
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy, policy_loss, value_loss = self.policy_update()
                    kl_s = (
                        "{:.4f}".format(self.last_kl)
                        if self.last_kl is not None
                        else "N/A"
                    )
                    logger.info(
                        "[Train] update | loss=%.4f | entropy=%.4f | kl=%s | lr=%.2e | lr_mult=%.2f | global_step=%s",
                        loss,
                        entropy,
                        kl_s,
                        self.policy_value_net.current_lr(),
                        self.lr_multiplier,
                        self.global_step,
                    )
                    self._metrics.write(
                        {
                            "timestamp": time.time(),
                            "phase": "update",
                            "iteration": i + 1,
                            "global_step": self.global_step,
                            "episode_len": self.episode_len,
                            "buffer_size": len(self.data_buffer),
                            "loss": loss,
                            "policy_loss": policy_loss,
                            "value_loss": value_loss,
                            "entropy": entropy,
                            "kl": self.last_kl,
                            "lr": self.policy_value_net.current_lr(),
                            "lr_multiplier": self.lr_multiplier,
                            "arena_winrate": "",
                            "baseline_winrate": "",
                            "updated_best": "",
                            "eval_games": "",
                            "board_w": self.board_width,
                            "board_h": self.board_height,
                            "n_in_row": self.n_in_row,
                            "n_playout": self.n_playout,
                            "c_puct": self.c_puct,
                            "temp": self.temp,
                            "device": run_device,
                        }
                    )
                else:
                    logger.info(
                        "[Train] skip | need buffer>%s | current=%s",
                        self.batch_size,
                        len(self.data_buffer),
                    )
                # check the performance of the current model and save the model params
                if (i+1) % self.check_freq == 0:
                    # Dual-axis evaluation:
                    # - Arena (relative): current vs best
                    # - Monitor (absolute baseline): current vs pure MCTS
                    arena_win, _ = self._evaluate_current_vs_best(self.eval_games)
                    baseline_win, _ = self._evaluate_current_vs_pure(self.eval_games)
                    a_lo, a_hi = wilson_ci(arena_win, self.eval_games)
                    b_lo, b_hi = wilson_ci(baseline_win, self.eval_games)
                    self.baseline_winrates.append(baseline_win)

                    # Arena gating: update best if current is consistently better.
                    improved = arena_win > self.arena_update_threshold
                    self.policy_value_net.save_model(self.current_policy_path)
                    if improved:
                        logger.info("Arena passed (%.3f > %.3f). Updating best policy.", arena_win, self.arena_update_threshold)
                        self.policy_value_net.save_model(self.best_policy_path)
                        # Save a dedicated "best" checkpoint capturing the improved model.
                        extra_best = {
                            "timestamp": time.time(),
                            "batch_i": i + 1,
                            "global_step": self.global_step,
                            "lr_multiplier": self.lr_multiplier,
                            "pure_mcts_playout_num": self.pure_mcts_playout_num,
                            "arena_winrate": arena_win,
                            "baseline_winrate": baseline_win,
                            "eval_games": self.eval_games,
                            "arena_update_threshold": self.arena_update_threshold,
                            "improved": True,
                            "python_random_state": random.getstate(),
                            "numpy_random_state": np.random.get_state(),
                            "torch_random_state": torch.get_rng_state(),
                            "torch_cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        }
                        self._atomic_save_checkpoint(self.best_checkpoint_path, extra_best)

                    self._log_eval_block(
                        iteration=i + 1,
                        arena_winrate=arena_win,
                        arena_update_best=improved,
                        baseline_winrate=baseline_win,
                        arena_ci=(a_lo, a_hi),
                        baseline_ci=(b_lo, b_hi),
                    )
                    self._metrics.write(
                        {
                            "timestamp": time.time(),
                            "phase": "eval",
                            "iteration": i + 1,
                            "global_step": self.global_step,
                            "episode_len": self.episode_len,
                            "buffer_size": len(self.data_buffer),
                            "loss": "",
                            "policy_loss": "",
                            "value_loss": "",
                            "entropy": "",
                            "kl": "",
                            "lr": self.policy_value_net.current_lr(),
                            "lr_multiplier": self.lr_multiplier,
                            "arena_winrate": arena_win,
                            "baseline_winrate": baseline_win,
                            "updated_best": 1 if improved else 0,
                            "eval_games": self.eval_games,
                            "board_w": self.board_width,
                            "board_h": self.board_height,
                            "n_in_row": self.n_in_row,
                            "n_playout": self.n_playout,
                            "c_puct": self.c_puct,
                            "temp": self.temp,
                            "device": run_device,
                        }
                    )

                    # Save a resumable training checkpoint after updating state.
                    extra = {
                        "timestamp": time.time(),
                        "batch_i": i + 1,
                        "global_step": self.global_step,
                        "lr_multiplier": self.lr_multiplier,
                        "pure_mcts_playout_num": self.pure_mcts_playout_num,
                        "arena_winrate": arena_win,
                        "baseline_winrate": baseline_win,
                        "eval_games": self.eval_games,
                        "arena_update_threshold": self.arena_update_threshold,
                        "improved": improved,
                        "python_random_state": random.getstate(),
                        "numpy_random_state": np.random.get_state(),
                        "torch_random_state": torch.get_rng_state(),
                        "torch_cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    }
                    self._atomic_save_checkpoint(self.latest_checkpoint_path, extra)
        except KeyboardInterrupt:
            try:
                extra = {
                    "timestamp": time.time(),
                    "batch_i": last_batch_i,
                    "global_step": self.global_step,
                    "lr_multiplier": self.lr_multiplier,
                    "pure_mcts_playout_num": self.pure_mcts_playout_num,
                    "reason": "KeyboardInterrupt",
                    "eval_games": self.eval_games,
                    "arena_update_threshold": self.arena_update_threshold,
                    "python_random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(),
                    "torch_random_state": torch.get_rng_state(),
                    "torch_cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                }
                self._atomic_save_checkpoint(self.latest_checkpoint_path, extra)
                logger.info("Saved checkpoint on interrupt: %s", self.latest_checkpoint_path)
            except Exception:
                logger.exception("Failed to save checkpoint on interrupt.")
            logger.info("quit")
        finally:
            try:
                self._metrics.close()
            except Exception:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Archive current outputs (logs/models/checkpoints) and exit.",
    )
    parser.add_argument("--use-gpu", action="store_true", help="Force CUDA (error if unavailable).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument(
        "--self-play-workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel self-play processes.",
    )
    args = parser.parse_args()

    if args.reset:
        archived_to = _archive_and_clear_outputs(archive_root="runs")
        print(f"Archived previous outputs to: {archived_to}")
        sys.exit(0)

    setup_logging(log_file="logs/train.log")

    use_gpu = None
    if args.cpu:
        use_gpu = False
    elif args.use_gpu:
        use_gpu = True

    training_pipeline = TrainPipeline(
        use_gpu=use_gpu,
        self_play_workers=args.self_play_workers,
    )
    training_pipeline.run()
