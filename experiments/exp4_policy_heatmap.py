"""
Exp 4: Visualize the policy network's raw move distribution p(a|s) on the board
(MCTS is not run; CNN policy head, softmax over all cells).

Outputs:
  - experiments/outputs/exp4/policy_heatmap_<name>.png (one per scenario)
  - experiments/outputs/exp4/policy_heatmap_grid_2x3.png

Run:
    python experiments/exp4_policy_heatmap.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.patches import Circle

from helpers import Board, board_height, board_width, load_policy, n_in_row, policy_matrix_from_board

# --- experiment knobs ---
MODEL_PATH = str(Path(__file__).resolve().parent.parent / "results" / "best_policy_8_8_5.pth")
USE_GPU = False
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "exp4"
# After per-scenario PNGs, stitch them into one figure (2×3 when there are 6 scenarios).
SAVE_COMBINED_GRID = True
COMBINED_GRID_NAME = "policy_heatmap_grid_2x3.png"

# Annotate each legal empty cell with raw policy value.
SHOW_CELL_PROB = True
PROB_FONT_SIZE = 6.5

# One PNG per entry. Move index = row * board_width + col (0-based); bottom-left is (0,0).
SCENARIOS: list[dict] = [
    {
        "name": "01_Opening_Start",
        "x_moves": [],
        "o_moves": [],
        "to_play": "x",
        "last_move": -1,
    },
    {
        "name": "02_Center_Opening",
        "x_moves": [27],
        "o_moves": [],
        "to_play": "o",
        "last_move": 27,
    },
    {
        "name": "03_Live_Three_Threat",
        "x_moves": [27, 35, 36],
        "o_moves": [19, 28, 37],
        "to_play": "x",
        "last_move": 19,
    },
    {
        "name": "04_Open_Four",
        "x_moves": [10, 34, 35, 36],
        "o_moves": [19, 28, 37, 46],
        "to_play": "x",
        "last_move": 46,
    },
    {
        "name": "05_Double_Three",
        "x_moves": [19, 35, 36, 45],
        "o_moves": [28, 37, 44],
        "to_play": "o",
        "last_move": 44,
    },
    {
        "name": "06_Double_Three_Prev",
        "x_moves": [19, 35, 36],
        "o_moves": [28, 37],
        "to_play": "o",
        "last_move": 37,
    },
]


def _apply_xo(board: Board, *, x_moves: list[int], o_moves: list[int], last_move: int, to_play: str) -> None:
    board.init_board(start_player=0)
    n = board.width * board.height

    x_set = set(x_moves)
    o_set = set(o_moves)
    if len(x_set) != len(x_moves) or len(o_set) != len(o_moves):
        raise SystemExit("Duplicate move indices in x_moves or o_moves.")
    if x_set & o_set:
        raise SystemExit("Overlapping move indices between x_moves and o_moves.")
    for m in list(x_set) + list(o_set):
        if m < 0 or m >= n:
            raise SystemExit(f"Move index out of range: {m} (valid 0..{n-1})")

    if to_play == "auto":
        if len(x_moves) == len(o_moves):
            board.current_player = board.players[0]  # X to play
        elif len(x_moves) == len(o_moves) + 1:
            board.current_player = board.players[1]  # O to play
        else:
            raise SystemExit(
                f"Cannot infer side to play from counts: |X|={len(x_moves)} |O|={len(o_moves)}. "
                "Set to_play to 'x' or 'o', or fix stone counts."
            )
    elif to_play == "x":
        board.current_player = board.players[0]
    elif to_play == "o":
        board.current_player = board.players[1]
    else:
        raise SystemExit(f"Invalid to_play: {to_play}")

    # Populate occupied squares.
    for m in x_moves:
        board.states[m] = board.players[0]
    for m in o_moves:
        board.states[m] = board.players[1]
    board.availables = [i for i in range(n) if i not in board.states]

    if last_move != -1:
        if last_move not in board.states:
            raise SystemExit("last_move must be an occupied square listed in x_moves or o_moves.")
        board.last_move = last_move


def _render_one(net, board: Board, out_path: Path, scenario_name: str) -> None:
    mat = policy_matrix_from_board(net, board)
    masked = np.full_like(mat, np.nan, dtype=float)
    for idx in board.availables:
        h, w = idx // board.width, idx % board.width
        masked[h, w] = mat[h, w]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(
        masked,
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        extent=(-0.5, board.width - 0.5, -0.5, board.height - 0.5),
        aspect="equal",
    )
    ax.set_xlim(-0.5, board.width - 0.5)
    ax.set_ylim(-0.5, board.height - 0.5)
    ax.set_xticks(range(board.width))
    ax.set_yticks(range(board.height))
    ax.set_xticks(np.arange(-0.5, board.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, board.height, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6, alpha=0.35)
    ax.tick_params(which="minor", length=0)

    stroke = [pe.withStroke(linewidth=2.5, foreground="black")]
    if SHOW_CELL_PROB:
        for idx in board.availables:
            h = idx // board.width
            w = idx % board.width
            p = float(mat[h, w])
            if p >= 0.01:
                label = f"{p:.3f}"
            elif p >= 1e-4:
                label = f"{p:.4f}"
            else:
                label = f"{p:.0e}"
            ax.text(
                w,
                h,
                label,
                ha="center",
                va="center",
                fontsize=PROB_FONT_SIZE,
                color="white",
                path_effects=stroke,
                zorder=4,
            )

    piece_stroke = [pe.withStroke(linewidth=3, foreground="black")]
    for move, pid in board.states.items():
        h = move // board.width
        w = move % board.width
        if pid == board.players[0]:
            face, edge = "#74c0fc", "#1864ab"
        else:
            face, edge = "#ffe066", "#f08c00"
        ax.add_patch(
            Circle(
                (w, h),
                0.38,
                facecolor=face,
                edgecolor=edge,
                linewidth=2,
                zorder=5,
            )
        )
        ch = "X" if pid == board.players[0] else "O"
        ax.text(
            w,
            h,
            ch,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="#212529",
            path_effects=piece_stroke,
            zorder=6,
        )

    nxt = "X" if board.current_player == board.players[0] else "O"
    label = str(scenario_name).replace("_", " ")
    ax.set_title(f"{label} (next: {nxt})")
    ax.set_xlabel("column (w)")
    ax.set_ylabel("row (h)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$p(a|s)$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _combine_heatmap_grid(out_dir: Path, slugs: list[str], out_name: str) -> None:
    paths = [out_dir / f"policy_heatmap_{s}.png" for s in slugs]
    if not paths or any(not p.is_file() for p in paths):
        return
    n = len(paths)
    if n == 6:
        rows, cols = 2, 3
    else:
        cols = min(3, n)
        rows = (n + cols - 1) // cols
    fig_w, fig_h = 5.5 * cols, 5.0 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes_flat = np.atleast_1d(axes).ravel()
    for i, path in enumerate(paths):
        axes_flat[i].imshow(plt.imread(str(path)))
        axes_flat[i].axis("off")
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0.05, hspace=0.05)
    fig.savefig(out_dir / out_name, dpi=150)
    plt.close(fig)


def main() -> None:
    net = load_policy(MODEL_PATH, use_gpu=USE_GPU)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sc in SCENARIOS:
        name = str(sc.get("name", "case")).replace(" ", "_")
        board = Board(width=board_width, height=board_height, n_in_row=n_in_row)
        _apply_xo(
            board,
            x_moves=list(sc["x_moves"]),
            o_moves=list(sc["o_moves"]),
            last_move=int(sc.get("last_move", -1)),
            to_play=str(sc.get("to_play", "auto")),
        )
        path = out_dir / f"policy_heatmap_{name}.png"
        _render_one(net, board, path, scenario_name=name)

    if SAVE_COMBINED_GRID:
        slugs = [str(sc.get("name", "case")).replace(" ", "_") for sc in SCENARIOS]
        _combine_heatmap_grid(out_dir, slugs, COMBINED_GRID_NAME)


if __name__ == "__main__":
    main()
