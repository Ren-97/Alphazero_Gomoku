"""
Minimal UI (Streamlit) for Human vs AI Gomoku.

Only includes:
- choose model (from `results/` folder)
- choose who plays first
- play by clicking the board
- show winner at the end

Run:
  pip install -r requirements.txt
  streamlit run UI_game.py
"""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from az_logging import setup_logging
from game import Board
from mcts_alphaZero import MCTSPlayer
from policy_value_net import PolicyValueNet


def _resolve_model_path(model_file: str) -> Path:
    p = Path(model_file)
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parent / p


@st.cache_resource(show_spinner=False)
def _load_policy(w: int, h: int, model_file: str) -> PolicyValueNet:
    model_path = _resolve_model_path(model_file)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return PolicyValueNet(w, h, model_file=str(model_path), use_gpu=False)


def _make_ai(*, w: int, h: int, model_file: str, playouts: int = 400, c_puct: float = 5.0) -> MCTSPlayer:
    pv = _load_policy(w, h, model_file)
    return MCTSPlayer(
        pv.policy_value_fn,
        c_puct=c_puct,
        n_playout=playouts,
        is_selfplay=0,
    )


def _infer_board_params_from_model_name(model_path: Path) -> tuple[int, int, int] | None:
    """
    Try to infer (w, h, n_in_row) from filenames like:
      best_policy_6_6_4.pth
      best_policy_8_8_5.pth
    """
    m = re.search(r"(\d+)[^\d]+(\d+)[^\d]+(\d+)\.pth$", model_path.name)
    if not m:
        return None
    w, h, n = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if w <= 0 or h <= 0 or n <= 0:
        return None
    return w, h, n


def _discover_models(base_dir: Path) -> list[Path]:
    """
    Discover play models from `results/` folder only.
    """
    results_dir = base_dir / "results"
    if not results_dir.exists():
        return []
    paths = list(results_dir.glob("*.pth")) + list(results_dir.glob("**/*.pth"))
    # Deduplicate while keeping most recently modified first.
    uniq: dict[Path, Path] = {}
    for p in paths:
        uniq[p.resolve()] = p
    out = list(uniq.values())
    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out


def _board_image(board: Board, last_move: int | None) -> tuple[Image.Image, dict]:
    w, h = board.width, board.height
    cell = 56
    pad = 20
    img_w = pad * 2 + cell * (w - 1)
    img_h = pad * 2 + cell * (h - 1)

    img = Image.new("RGB", (img_w, img_h), (222, 184, 135))
    draw = ImageDraw.Draw(img)

    def xy(r: int, c: int) -> tuple[int, int]:
        # r=0 is bottom row; image y grows downward
        x = pad + c * cell
        y = pad + (h - 1 - r) * cell
        return x, y

    # grid
    for c in range(w):
        x1, y1 = xy(0, c)
        x2, y2 = xy(h - 1, c)
        draw.line((x1, y1, x2, y2), fill=(34, 34, 34), width=2)
    for r in range(h):
        x1, y1 = xy(r, 0)
        x2, y2 = xy(r, w - 1)
        draw.line((x1, y1, x2, y2), fill=(34, 34, 34), width=2)

    # stones
    r_stone = 18
    for move, player in board.states.items():
        rr = move // w
        cc = move % w
        cx, cy = xy(rr, cc)
        if player == 1:
            fill = (20, 20, 20)
            outline = (5, 5, 5)
        else:
            fill = (247, 247, 247)
            outline = (30, 30, 30)
        draw.ellipse((cx - r_stone, cy - r_stone, cx + r_stone, cy + r_stone), fill=fill, outline=outline, width=2)

    # last move marker
    if last_move is not None and last_move in board.states:
        rr = last_move // w
        cc = last_move % w
        cx, cy = xy(rr, cc)
        draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=(220, 20, 60))

    meta = {"cell": cell, "pad": pad, "w": w, "h": h}
    return img, meta


def _click_to_move(click: dict | None, meta: dict) -> int | None:
    if not click:
        return None
    x = click.get("x")
    y = click.get("y")
    if x is None or y is None:
        return None
    cell = meta["cell"]
    pad = meta["pad"]
    w = meta["w"]
    h = meta["h"]

    c = int(round((x - pad) / cell))
    r_from_top = int(round((y - pad) / cell))
    r = (h - 1) - r_from_top
    if r < 0 or r >= h or c < 0 or c >= w:
        return None
    return r * w + c


def _init_state() -> None:
    st.session_state.setdefault("board", None)
    st.session_state.setdefault("ai", None)
    st.session_state.setdefault("human_player", 1)
    st.session_state.setdefault("last_move", None)
    st.session_state.setdefault("game_over", False)
    st.session_state.setdefault("winner", None)  # int | None (None=tie)
    st.session_state.setdefault("model_file", None)
    st.session_state.setdefault("board_params", None)  # tuple[int,int,int] | None


def _new_game(*, w: int, h: int, n_in_row: int, model_file: str, first: str) -> None:
    board = Board(width=w, height=h, n_in_row=n_in_row)
    # Gomoku rule: Black (player 1) always moves first. Do not use start_player=1,
    # or White would open — then "AI first" looked like you play White but still opened.
    board.init_board(start_player=0)

    # Human first: human is Black (1), AI is White (2).
    # AI first: AI is Black (1), human is White (2).
    human_player = 1 if first == "Human first" else 2

    ai = _make_ai(w=w, h=h, model_file=model_file, playouts=400, c_puct=5.0)

    st.session_state.board = board
    st.session_state.ai = ai
    st.session_state.human_player = human_player
    st.session_state.last_move = None
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.model_file = model_file
    st.session_state.board_params = (w, h, n_in_row)

    if first == "AI first":
        _ai_move(temp=1e-3)


def _do_move(move: int) -> None:
    board: Board = st.session_state.board
    if st.session_state.game_over or move not in board.availables:
        return
    board.do_move(move)
    st.session_state.last_move = move
    end, wnr = board.game_end()
    if end:
        st.session_state.game_over = True
        st.session_state.winner = None if wnr == -1 else wnr


def _ai_move(*, temp: float = 1e-3) -> None:
    board: Board = st.session_state.board
    ai = st.session_state.ai
    move = int(ai.get_action(board, temp=temp))
    _do_move(move)


def main() -> None:
    setup_logging(log_file=None)
    st.set_page_config(page_title="Gomoku UI", layout="centered")
    _init_state()

    st.title("Gomoku – Human vs AI")

    base_dir = Path(__file__).resolve().parent
    model_paths = _discover_models(base_dir)
    model_labels: list[str] = []
    for p in model_paths:
        try:
            model_labels.append(str(p.resolve().relative_to(base_dir.resolve())))
        except Exception:
            model_labels.append(str(p))

    with st.sidebar:
        st.markdown("### Settings")
        if not model_labels:
            st.warning("No models found in `results/`. Put your `.pth` files into `results/`.")
            model_file = ""
        else:
            model_file = st.selectbox("Model", model_labels, index=0)
        first = st.radio("First", ["Human first", "AI first"], horizontal=True)

        if model_file:
            inferred = _infer_board_params_from_model_name(_resolve_model_path(model_file))
        else:
            inferred = None
        if inferred is None:
            with st.expander("Board params (only needed if not in filename)", expanded=True):
                w = st.number_input("width", min_value=3, max_value=25, value=8, step=1)
                h = st.number_input("height", min_value=3, max_value=25, value=8, step=1)
                n_in_row = st.number_input("n_in_row", min_value=3, max_value=min(int(w), int(h)), value=5, step=1)
            w, h, n_in_row = int(w), int(h), int(n_in_row)
        else:
            w, h, n_in_row = inferred
            st.caption(f"Board: {w}x{h}, connect-{n_in_row}")

        can_start = bool(model_file)
        if st.button("New game", type="primary", disabled=not can_start):
            try:
                _new_game(w=w, h=h, n_in_row=n_in_row, model_file=model_file, first=first)
            except Exception as e:
                st.error(str(e))

    if st.session_state.board is None:
        st.info("Choose a model and who plays first, then click **New game**.")
        return

    board: Board = st.session_state.board
    human_player: int = st.session_state.human_player

    if st.session_state.game_over:
        if st.session_state.winner is None:
            st.success("Game over: Tie")
        else:
            st.success(f"Game over: {'Black(●)' if st.session_state.winner == 1 else 'White(○)'} wins")
    else:
        cur = board.get_current_player()
        turn = "Black(●)" if cur == 1 else "White(○)"
        st.write(f"Turn: {turn}")

    img, meta = _board_image(board, st.session_state.last_move)
    click = streamlit_image_coordinates(img, key=f"board-{len(board.states)}")
    st.caption("Click near an intersection to play.")

    if not st.session_state.game_over:
        move = _click_to_move(click, meta)
        is_human_turn = board.get_current_player() == human_player
        if move is not None and is_human_turn and move in board.availables:
            _do_move(move)
            if not st.session_state.game_over:
                _ai_move(temp=1e-3)
            st.rerun()


if __name__ == "__main__":
    main()

