"""
Exp 4: visualize raw neural policy heatmap on the board.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from helpers import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    Board,
    N_IN_ROW,
    load_policy,
    policy_matrix_from_board,
)


def _apply_moves(board: Board, moves: list[int]) -> None:
    board.init_board(start_player=0)
    for m in moves:
        board.do_move(m)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--gpu", action="store_true")
    p.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs" / "policy_heatmap.png"),
    )
    p.add_argument(
        "--moves",
        type=str,
        default="",
        help="comma-separated move indices (empty = empty board). Example: 27,35,28",
    )
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    net = load_policy(args.model, use_gpu=args.gpu)
    board = Board(width=BOARD_WIDTH, height=BOARD_HEIGHT, n_in_row=N_IN_ROW)
    moves = []
    if args.moves.strip():
        moves = [int(x.strip()) for x in args.moves.split(",") if x.strip()]
    _apply_moves(board, moves)

    mat = policy_matrix_from_board(net, board)
    masked = np.full_like(mat, np.nan, dtype=float)
    for idx in board.availables:
        h, w = idx // board.width, idx % board.width
        masked[h, w] = mat[h, w]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(masked, origin="lower", cmap="magma", interpolation="nearest")
    ax.set_title("Neural policy (legal squares only)")
    ax.set_xlabel("w")
    ax.set_ylabel("h")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
