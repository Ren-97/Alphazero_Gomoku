from __future__ import annotations

import argparse
import csv
from pathlib import Path

from az_metrics import wilson_ci


def _to_float(x: str):
    if x is None:
        return None
    x = str(x).strip()
    if x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _to_int(x: str):
    f = _to_float(x)
    if f is None:
        return None
    return int(f)


def read_metrics_csv(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="Path to metrics CSV (default: newest logs/metrics_*.csv).")
    ap.add_argument("--output", default=None, help="Output PNG path (default: same name as input, .png).")
    ap.add_argument("--z", type=float, default=1.96, help="Z value for CI (default 1.96 ~ 95%%).")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with:\n"
            "  pip install matplotlib\n\n"
            f"Import error: {e}"
        )

    if args.input is None:
        logs_dir = Path("logs")
        candidates = sorted(logs_dir.glob("metrics_*.csv"), key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
        if not candidates:
            raise SystemExit("No metrics CSV found. Expected something like logs/metrics_6_6_4.csv")
        inp = candidates[-1]
    else:
        inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"metrics file not found: {inp}")

    rows = read_metrics_csv(inp)
    if not rows:
        raise SystemExit(f"metrics file is empty: {inp}")

    eval_it: list[int] = []
    arena_w: list[float] = []
    arena_lo: list[float] = []
    arena_hi: list[float] = []
    base_w: list[float] = []
    base_lo: list[float] = []
    base_hi: list[float] = []

    upd_step: list[int] = []
    loss: list[float] = []
    policy_loss: list[float] = []
    value_loss: list[float] = []

    for row in rows:
        phase = (row.get("phase") or "").strip()
        it = _to_int(row.get("iteration"))
        gs = _to_int(row.get("global_step"))

        if phase == "eval":
            aw = _to_float(row.get("arena_winrate"))
            bw = _to_float(row.get("baseline_winrate"))
            n = _to_int(row.get("eval_games"))
            if it is not None and aw is not None and n is not None:
                eval_it.append(it)
                arena_w.append(aw)
                alo, ahi = wilson_ci(aw, n, z=float(args.z))
                arena_lo.append(alo)
                arena_hi.append(ahi)

                bwv = bw if bw is not None else float("nan")
                base_w.append(bwv)
                if bw is not None:
                    blo, bhi = wilson_ci(bw, n, z=float(args.z))
                else:
                    blo, bhi = (float("nan"), float("nan"))
                base_lo.append(blo)
                base_hi.append(bhi)
        elif phase == "update":
            lo = _to_float(row.get("loss"))
            plo = _to_float(row.get("policy_loss"))
            vlo = _to_float(row.get("value_loss"))
            if gs is not None and lo is not None:
                upd_step.append(gs)
                loss.append(lo)
                policy_loss.append(plo if plo is not None else float("nan"))
                value_loss.append(vlo if vlo is not None else float("nan"))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    ax = axes[0]
    if eval_it:
        l1 = ax.plot(eval_it, arena_w, label="arena_winrate (current vs best)")[0]
        l2 = ax.plot(eval_it, base_w, label="baseline_winrate (vs pure MCTS)")[0]
        ax.fill_between(eval_it, arena_lo, arena_hi, color=l1.get_color(), alpha=0.18, linewidth=0)
        ax.fill_between(eval_it, base_lo, base_hi, color=l2.get_color(), alpha=0.12, linewidth=0)
    ax.set_title("Winrates (with confidence interval)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("winrate")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    if upd_step:
        ax.plot(upd_step, loss, label="loss")
        ax.plot(upd_step, policy_loss, label="policy_loss")
        ax.plot(upd_step, value_loss, label="value_loss")
    ax.set_title("Loss components")
    ax.set_xlabel("global_step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(str(inp).replace("\\", "/"), fontsize=10)
    fig.tight_layout()

    if args.output is None:
        out = inp.with_suffix(".png")
    else:
        out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

