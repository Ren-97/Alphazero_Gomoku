"""
Exp 2: wall-clock per move for AZ vs Pure at chosen playout counts.
"""

import argparse
import statistics

from helpers import (
    MCTSPlayer,
    MCTS_Pure,
    TimedPlayer,
    board_height,
    board_width,
    c_puct,
    load_policy,
    make_game,
    n_in_row,
    root,
)

model = root / "current_policy_8_8_5.pth"
n_games = 10
n_az = 20
n_pure = 800
use_gpu = False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=str(model))
    p.add_argument("--games", type=int, default=n_games, help="short matches; more = stable mean")
    p.add_argument("--n-az", type=int, default=n_az)
    p.add_argument("--n-pure", type=int, default=n_pure)
    p.add_argument("--gpu", action="store_true", default=use_gpu)
    args = p.parse_args()

    net = load_policy(args.model, use_gpu=args.gpu)
    _, game = make_game()

    az_inner = MCTSPlayer(
        net.policy_value_fn,
        c_puct=c_puct,
        n_playout=args.n_az,
        is_selfplay=0,
    )
    pure_inner = MCTS_Pure(c_puct=c_puct, n_playout=args.n_pure)

    az_timed = TimedPlayer(az_inner)
    pure_timed = TimedPlayer(pure_inner)

    for g in range(args.games):
        az_timed.reset_player()
        pure_timed.reset_player()
        game.start_play(
            az_timed,
            pure_timed,
            start_player=g % 2,
            is_shown=0,
        )

    def summarize(name: str, lat: list[float]) -> None:
        if not lat:
            print(f"{name}: no samples")
            return
        ms = [x * 1000.0 for x in lat]
        print(
            f"{name}: n_moves={len(ms)}  mean_ms={statistics.mean(ms):.2f}  "
            f"median_ms={statistics.median(ms):.2f}"
        )

    print(f"board={board_width}x{board_height} connect-{n_in_row}")
    print(f"AZ n_playout={args.n_az} | Pure n_playout={args.n_pure} | games={args.games}")
    print(f"ratio (Pure/AZ) = {args.n_pure / args.n_az:.2f}")
    summarize("AZ (all moves by the neural MCTS player)", az_timed.latencies)
    summarize("Pure MCTS (all moves by the rollout player)", pure_timed.latencies)


if __name__ == "__main__":
    main()
