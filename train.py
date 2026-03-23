"""
An implementation of the training pipeline of AlphaZero for Gomoku
"""

import logging
import os
import random
import time
import argparse
import numpy as np
from collections import defaultdict, deque
import torch
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet 
from az_logging import setup_logging


logger = logging.getLogger(__name__)


class TrainPipeline():
    def __init__(self, init_model=None, use_gpu: bool | None = None):
        # params of the board and the game
        self.board_width = 6
        self.board_height = 6
        self.n_in_row = 4
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
        self.n_playout = 10  #400 # num of simulations for each move
        self.c_puct = 5 # Exploration Coefficient (PUCT:Predictor + Upper Confidence Bound applied to Trees)
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 5 #50
        self.game_batch_num = 20 #1500
        
        # Evaluation / arena settings
        self.eval_games = 10  # number of games per evaluation
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
        self.pure_mcts_playout_num = 200 # 1000

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
            self.use_gpu = bool(use_gpu) and bool(torch.cuda.is_available())
        if (use_gpu is True) and (not torch.cuda.is_available()):
            logger.warning("use_gpu=True but CUDA not available. Falling back to CPU.")

        # Initialization modes:
        # - If init_model is provided (non-empty path): load weights from that file.
        # - Otherwise: create a fresh net and (if available) resume from latest checkpoint.
        has_init_model = init_model is not None and str(init_model).strip() != ""
        if has_init_model:
            self.policy_value_net = PolicyValueNet(
                self.board_width,
                self.board_height,
                model_file=init_model,
                use_gpu=self.use_gpu,
                base_lr=self.learn_rate,
            )
        else:
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
                torch.set_rng_state(torch_rng)
            cuda_rng = extra.get("torch_cuda_random_state_all")
            if cuda_rng is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_rng)

            logger.info(
                "Resumed from checkpoint: %s (start_batch=%s, global_step=%s)",
                self.latest_checkpoint_path,
                self.start_batch,
                self.global_step,
            )
        except Exception:
            logger.exception("Failed to resume from checkpoint: %s", self.latest_checkpoint_path)

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
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            self.episode_lens.append(self.episode_len)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        # Apply the current KL-adaptive multiplier once per policy update.
        self.policy_value_net.set_lr_multiplier(self.lr_multiplier)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
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

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        logger.info(
            "kl:%.5f, lr_multiplier:%.3f, loss:%s, entropy:%s, explained_var_old:%.3f, explained_var_new:%.3f",
            kl,
            self.lr_multiplier,
            loss,
            entropy,
            explained_var_old,
            explained_var_new,
        )
        self.last_loss = loss
        self.last_entropy = entropy
        self.last_kl = kl
        return loss, entropy

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
    ) -> None:
        line = "-" * 50
        avg_len = float(np.mean(self.episode_lens)) if self.episode_lens else float("nan")
        trend = self._trend_label(self.baseline_winrates)
        loss = "N/A" if self.last_loss is None else "{:.4f}".format(self.last_loss)
        entropy = "N/A" if self.last_entropy is None else "{:.4f}".format(self.last_entropy)

        logger.info("Iteration [%s]", iteration)
        logger.info(line)
        logger.info(
            "[Arena]   Current vs Best: Winrate %.1f%% | Result: %s",
            arena_winrate * 100.0,
            "UPDATE BEST!" if arena_update_best else "keep best",
        )
        logger.info(
            "[Monitor] Current vs Pure MCTS: Winrate %.1f%% | Trend: %s",
            baseline_winrate * 100.0,
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
        try:
            for i in range(self.start_batch, self.game_batch_num):
                last_batch_i = i + 1
                self.collect_selfplay_data(self.play_batch_size)
                logger.info("batch i:%s, episode_len:%s", i + 1, self.episode_len)
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    logger.info("current self-play batch: %s", i + 1)

                    # Dual-axis evaluation:
                    # - Arena (relative): current vs best
                    # - Monitor (absolute baseline): current vs pure MCTS
                    arena_win, _ = self._evaluate_current_vs_best(self.eval_games)
                    baseline_win, baseline_cnt = self._evaluate_current_vs_pure(self.eval_games)
                    self.baseline_winrates.append(baseline_win)

                    # Keep original per-opponent breakdown log for baseline.
                    logger.info(
                        "pure_mcts num_playouts:%s, win:%s, lose:%s, tie:%s",
                        self.pure_mcts_playout_num,
                        baseline_cnt[1],
                        baseline_cnt[2],
                        baseline_cnt[-1],
                    )

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


if __name__ == '__main__':
    setup_logging(log_file="logs/train.log")
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-model", default=None, help="Optional path to a .pth model to initialize from.")
    parser.add_argument("--use-gpu", action="store_true", help="Use CUDA if available.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = parser.parse_args()

    use_gpu = None
    if args.cpu:
        use_gpu = False
    elif args.use_gpu:
        use_gpu = True

    training_pipeline = TrainPipeline(init_model=args.init_model, use_gpu=use_gpu)
    training_pipeline.run()
