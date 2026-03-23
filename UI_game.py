"""
Minimal UI (Streamlit) for AlphaZero_Gomoku.

Run:
  pip install -r requirements.txt
  streamlit run UI_game.py
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import time

import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from az_logging import setup_logging
from game import Board
from mcts_alphaZero import MCTSPlayer
from mcts_pure import MCTSPlayer as MCTS_Pure
from policy_value_net import PolicyValueNet


PRESETS = {
    "6x6 connect-4": {"w": 6, "h": 6, "n": 4, "model": "best_policy_6_6_4.pth"},
    "8x8 connect-5": {"w": 8, "h": 8, "n": 5, "model": "best_policy_8_8_5.pth"},
}


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


def _make_ai(*, preset: dict, pure_mcts: bool, playouts: int, c_puct: float, self_play: bool):
    if pure_mcts:
        return MCTS_Pure(c_puct=c_puct, n_playout=max(1000, playouts))
    pv = _load_policy(preset["w"], preset["h"], preset["model"])
    return MCTSPlayer(
        pv.policy_value_fn,
        c_puct=c_puct,
        n_playout=playouts,
        is_selfplay=1 if self_play else 0,
    )


def _stone(p: int) -> str:
    return "●" if p == 1 else ("○" if p == 2 else "·")


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
    st.session_state.setdefault("human_player", 1)  # int | None
    st.session_state.setdefault("last_move", None)
    st.session_state.setdefault("game_over", False)
    st.session_state.setdefault("winner", None)  # int | None (None=tie)
    st.session_state.setdefault("mode", "Human vs AI")


def _new_game(*, preset: dict, mode: str, pure_mcts: bool, playouts: int, c_puct: float, first: str):
    board = Board(width=preset["w"], height=preset["h"], n_in_row=preset["n"])
    start_player_idx = 0
    if mode == "Human vs AI" and first == "AI first":
        start_player_idx = 1
    board.init_board(start_player=start_player_idx)

    ai = _make_ai(
        preset=preset,
        pure_mcts=pure_mcts,
        playouts=playouts,
        c_puct=c_puct,
        self_play=(mode == "Self-play"),
    )

    if mode == "Self-play":
        human_player = None
    else:
        p1, p2 = board.players
        human_player = p1 if start_player_idx == 0 else p2

    st.session_state.board = board
    st.session_state.ai = ai
    st.session_state.human_player = human_player
    st.session_state.last_move = None
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.mode = mode


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


def _ai_move(*, temp: float) -> None:
    board: Board = st.session_state.board
    ai = st.session_state.ai
    if isinstance(ai, MCTSPlayer):
        move = int(ai.get_action(board, temp=temp))
    else:
        move = int(ai.get_action(board))
    _do_move(move)


def _is_ai_turn(board: Board) -> bool:
    hp: int | None = st.session_state.human_player
    if hp is None:
        return True
    return board.get_current_player() != hp


def main() -> None:
    setup_logging(log_file=None)
    st.set_page_config(page_title="Gomoku UI", layout="wide")
    _init_state()

    st.title("Gomoku (AlphaZero) – UI")

    with st.sidebar:
        preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
        preset = PRESETS[preset_name]
        mode = st.radio("Mode", ["Human vs AI", "Self-play"], horizontal=True)
        first = "Human first"
        if mode == "Human vs AI":
            first = st.radio("First", ["Human first", "AI first"], horizontal=True)

        pure_mcts = st.toggle("Pure MCTS (no net)", value=False)
        playouts = st.slider("Playouts", 50, 2000, 400, 50)
        c_puct = st.slider("c_puct", 0.5, 10.0, 5.0, 0.5)
        temp = st.slider("temp", 0.01, 2.0, 0.10 if mode == "Self-play" else 0.01, 0.01)

        if st.button("New game", type="primary"):
            try:
                _new_game(
                    preset=preset,
                    mode=mode,
                    pure_mcts=pure_mcts,
                    playouts=playouts,
                    c_puct=c_puct,
                    first=first,
                )
            except Exception as e:
                st.error(str(e))

        st.caption("Tip: if model file not found, put the `.pth` next to this script.")

    if st.session_state.board is None:
        st.info("Click **New game** in the sidebar.")
        return

    board: Board = st.session_state.board
    human_player: int | None = st.session_state.human_player
    cur = board.get_current_player()
    turn = "Black(●)" if cur == 1 else "White(○)"

    if st.session_state.game_over:
        if st.session_state.winner is None:
            st.success("Game over: Tie")
        else:
            st.success(f"Game over: {'Black(●)' if st.session_state.winner == 1 else 'White(○)'} wins")
    else:
        st.write(f"Turn: {turn}")

    left, right = st.columns([3, 1], gap="large")
    with left:
        img, meta = _board_image(board, st.session_state.last_move)
        click = streamlit_image_coordinates(img, key=f"board-{len(board.states)}")
        st.caption("Tip: click near an intersection to place a stone.")

        if st.session_state.mode == "Human vs AI" and not st.session_state.game_over:
            move = _click_to_move(click, meta)
            is_human_turn = human_player is not None and board.get_current_player() == human_player
            if move is not None and is_human_turn and move in board.availables:
                _do_move(move)
                if not st.session_state.game_over:
                    _ai_move(temp=temp)
                st.rerun()

            with st.expander("Or enter row/col", expanded=False):
                rr, cc = st.columns(2)
                row = rr.number_input("row", min_value=0, max_value=board.height - 1, value=0, step=1)
                col = cc.number_input("col", min_value=0, max_value=board.width - 1, value=0, step=1)
                move2 = int(row) * board.width + int(col)
                can_play = is_human_turn and move2 in board.availables
                if st.button("Play", disabled=not can_play):
                    _do_move(move2)
                    if not st.session_state.game_over:
                        _ai_move(temp=temp)
                    st.rerun()

    with right:
        st.markdown("### Controls")
        if st.session_state.mode == "Self-play":
            if st.button("Step", disabled=st.session_state.game_over):
                _ai_move(temp=temp)
                st.rerun()
            n = st.number_input("Auto N", 1, 200, 30, 1)
            if st.button("Auto", disabled=st.session_state.game_over):
                for _ in range(int(n)):
                    if st.session_state.game_over:
                        break
                    _ai_move(temp=temp)
                    time.sleep(0.02)
                st.rerun()
        else:
            if st.button("AI move", disabled=st.session_state.game_over or (human_player is not None and board.get_current_player() == human_player)):
                _ai_move(temp=temp)
                st.rerun()

        st.markdown("### Info")
        st.write(
            {
                "board": f'{board.width}x{board.height}, connect-{board.n_in_row}',
                "pure_mcts": pure_mcts,
                "model": None if pure_mcts else str(_resolve_model_path(preset["model"])),
            }
        )


if __name__ == "__main__":
    main()

