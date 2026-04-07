"""
Exp 2: wall-clock per move for AZ vs Pure at chosen playout counts.
"""

import argparse
import statistics

from helpers import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    C_PUCT,
    MCTSPlayer,
    MCTS_Pure,
    N_IN_ROW,
    TimedPlayer,
    load_policy,
    make_game,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--games", type=int, default=2, help="short matches; more = stable mean")
    p.add_argument("--n-az", type=int, default=20)
    p.add_argument("--n-pure", type=int, default=800)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()

    net = load_policy(args.model, use_gpu=args.gpu)
    _, game = make_game()

    az_inner = MCTSPlayer(
        net.policy_value_fn,
        c_puct=C_PUCT,
        n_playout=args.n_az,
        is_selfplay=0,
    )
    pure_inner = MCTS_Pure(c_puct=C_PUCT, n_playout=args.n_pure)

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

    print(f"board={BOARD_WIDTH}x{BOARD_HEIGHT} connect-{N_IN_ROW}")
    print(f"AZ n_playout={args.n_az} | Pure n_playout={args.n_pure} | games={args.games}")
    summarize("AZ (all moves by the neural MCTS player)", az_timed.latencies)
    summarize("Pure MCTS (all moves by the rollout player)", pure_timed.latencies)


if __name__ == "__main__":
    main()
