import csv
import math
import os
from typing import Iterable


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion (approximate when p comes from
    tie-adjusted scores, same convention as plot_metrics / training eval).
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = max(0.0, min(1.0, float(p)))
    n = int(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


DEFAULT_METRICS_FIELDS = [
    "timestamp",
    "phase",
    "iteration",
    "global_step",
    "episode_len",
    "buffer_size",
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "kl",
    "lr",
    "lr_multiplier",
    "arena_winrate",
    "baseline_winrate",
    "updated_best",
    "eval_games",
    "board_w",
    "board_h",
    "n_in_row",
    "n_playout",
    "c_puct",
    "temp",
    "device",
]


class MetricsWriter:
    def __init__(self, path: str, fieldnames: Iterable[str] = DEFAULT_METRICS_FIELDS):
        fieldnames = list(fieldnames)
        self.path = path
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        self._fieldnames = fieldnames
        if os.path.exists(self.path):
            self._ensure_header()
        exists = os.path.exists(self.path)
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
        if not exists:
            self._writer.writeheader()
            self._fh.flush()

    def _ensure_header(self) -> None:
        expected = ",".join(self._fieldnames)
        with open(self.path, "r", encoding="utf-8", newline="") as rf:
            first = (rf.readline() or "").strip()
        if first == "" or first == expected:
            return

        tmp = self.path + ".tmp"
        with open(self.path, "r", encoding="utf-8", newline="") as rf:
            reader = csv.DictReader(rf)
            old_rows = list(reader)
        with open(tmp, "w", encoding="utf-8", newline="") as wf:
            writer = csv.DictWriter(wf, fieldnames=self._fieldnames)
            writer.writeheader()
            for row in old_rows:
                writer.writerow({k: row.get(k, "") for k in self._fieldnames})
        os.replace(tmp, self.path)

    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
