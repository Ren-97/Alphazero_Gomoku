import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from mcts_pure import MCTSPlayer as MCTS_Pure
from policy_value_net import PolicyValueNet

BOARD_WIDTH = 8
BOARD_HEIGHT = 8
N_IN_ROW = 5
C_PUCT = 5


def make_game(
    width: int = BOARD_WIDTH,
    height: int = BOARD_HEIGHT,
    n_in_row: int = N_IN_ROW,
) -> tuple[Board, Game]:
    board = Board(width=width, height=height, n_in_row=n_in_row)
    return board, Game(board)


def load_policy(model_path, use_gpu=False):
    """Load a PolicyValueNet from a .pth file path."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"No model at {path}")
    return PolicyValueNet(BOARD_WIDTH, BOARD_HEIGHT, model_file=str(path), use_gpu=use_gpu)


def az_vs_pure_win_rate(net, n_games, n_playout_az, n_playout_pure, c_puct=C_PUCT):
    """
    AZ (player 1) vs Pure MCTS (player 2). Same protocol as train._evaluate_current_vs_pure.
    Win rate = (wins for player 1 + 0.5 * ties) / n_games.
    """
    _, game = make_game()
    az = MCTSPlayer(
        net.policy_value_fn,
        c_puct=c_puct,
        n_playout=n_playout_az,
        is_selfplay=0,
    )
    pure = MCTS_Pure(c_puct=c_puct, n_playout=n_playout_pure)
    win_cnt: dict = defaultdict(int)
    for i in range(n_games):
        winner = game.start_play(
            az,
            pure,
            start_player=i % 2,
            is_shown=0,
        )
        win_cnt[winner] += 1
    win_ratio = (win_cnt[1] + 0.5 * win_cnt[-1]) / max(n_games, 1)
    return float(win_ratio), dict(win_cnt)


class TimedPlayer:
    """Wraps any player with get_action(board); records each get_action duration in seconds."""

    def __init__(self, inner):
        self._inner = inner
        self.latencies: list[float] = []

    def set_player_ind(self, p):
        self._inner.set_player_ind(p)

    def reset_player(self):
        self._inner.reset_player()

    def get_action(self, board):
        t0 = time.perf_counter()
        move = self._inner.get_action(board)
        self.latencies.append(time.perf_counter() - t0)
        return move


def policy_matrix_from_board(
    net: PolicyValueNet,
    board: Board,
) -> np.ndarray:
    """Full board (height, width) neural policy mass; illegal moves may still carry mass."""
    state = board.current_state().reshape(-1, 4, board.width, board.height)
    probs, _ = net.policy_value([state])
    flat = probs[0]
    h, w = board.height, board.width
    return flat.reshape(h, w)
