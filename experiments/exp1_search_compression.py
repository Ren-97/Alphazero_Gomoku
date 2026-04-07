"""
Exp 1: AZ vs Pure MCTS — edit the constants below, then run:
    python experiments/exp1_search_compression.py
Writes table to console + CSV + PNG under experiments/outputs/exp1/.
"""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

from helpers import BOARD_HEIGHT, BOARD_WIDTH, C_PUCT, N_IN_ROW, ROOT, az_vs_pure_win_rate, load_policy

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "exp1"

MODEL = ROOT / "current_policy_8_8_5.pth"
N_AZ = 20
N_PURE_LIST = [400, 800, 1600, 2000]
N_GAMES = 24
USE_GPU = False


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = max(0.0, min(1.0, float(p)))
    n = int(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def main() -> None:
    net = load_policy(MODEL, use_gpu=USE_GPU)
    rows: list[dict] = []

    for n_pure in sorted(N_PURE_LIST):
        wr, counts = az_vs_pure_win_rate(
            net,
            n_games=N_GAMES,
            n_playout_az=N_AZ,
            n_playout_pure=n_pure,
            c_puct=C_PUCT,
        )
        lo, hi = wilson_ci(wr, N_GAMES)
        rows.append(
            {
                "N_pure": n_pure,
                "WinRate_AZ": round(wr, 4),
                "Wilson95_lo": round(lo, 4),
                "Wilson95_hi": round(hi, 4),
                "AZ_wins": int(counts.get(1, 0)),
                "Pure_wins": int(counts.get(2, 0)),
                "Ties": int(counts.get(-1, 0)),
                "N_pure_div_N_az": round(n_pure / N_AZ, 2),
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "az_vs_pure_mcts.csv"
    png_path = OUT_DIR / "az_vs_pure_mcts.png"

    keys = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    xs = [r["N_pure"] for r in rows]
    ys = [r["WinRate_AZ"] for r in rows]
    lo = [r["Wilson95_lo"] for r in rows]
    hi = [r["Wilson95_hi"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(xs, ys, yerr=([ys[i] - lo[i] for i in range(len(rows))], [hi[i] - ys[i] for i in range(len(rows))]), fmt="o-", capsize=4)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Pure MCTS N (simulations / move)")
    ax.set_ylabel("AZ win rate (P1)")
    ax.set_title("AZ vs Pure MCTS")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print("AZ vs Pure MCTS | P1=AZ | alternating start")
    print(f"  model={MODEL}  board={BOARD_WIDTH}x{BOARD_HEIGHT}-{N_IN_ROW}  N_az={N_AZ}  games/row={N_GAMES}")
    print()
    hdr = f"{'N_pure':>7} {'WinRate':>9} {'Wilson95':^17} {'AZ':>4} {'Pu':>4} {'Dr':>4} {'ratio':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        w = f"[{r['Wilson95_lo']:.3f},{r['Wilson95_hi']:.3f}]"
        print(
            f"{r['N_pure']:>7} {r['WinRate_AZ']:>9.4f} {w:^17} "
            f"{r['AZ_wins']:>4} {r['Pure_wins']:>4} {r['Ties']:>4} {r['N_pure_div_N_az']:>7.2f}"
        )
    print()
    print(csv_path)
    print(png_path)


if __name__ == "__main__":
    main()
