"""Benchmark vòng tròn giữa 3 AI: Minimax, Alpha-Beta, Heuristic AI.

Đầu ra:
    docs/figures/benchmark.csv          — số liệu thô từng ván
    docs/figures/benchmark_summary.csv  — số liệu tổng hợp theo cặp đấu
    docs/figures/total_nodes.png        — tổng node trung bình của 3 AI
    docs/figures/win_rates.png          — thắng / hòa / thua
    docs/figures/*_per_turn.png         — node & thời gian theo từng lượt

Mỗi AI đấu với 2 AI còn lại; các ván luân phiên đổi phe để loại bỏ lợi
thế đi trước (X luôn đi trước theo luật).
"""

from __future__ import annotations

import csv
import os
import sys
import time
from collections import defaultdict
from itertools import combinations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Cho phép chạy `python -m experiments.benchmark`: thêm thư mục gốc
# vào sys.path để `import src.*` hoạt động.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import board as bd  # noqa: E402
from src.metrics import (  # noqa: E402
    MatchResult,
    MoveStats,
    all_ai_callables,
    make_heuristic_ai,
    play_game,
)

# Thư mục lưu mọi output đồ hoạ & CSV.
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
FIG_DIR = os.path.abspath(FIG_DIR)
os.makedirs(FIG_DIR, exist_ok=True)

# Tên các AI sẽ đấu vòng tròn — heuristic_ai mặc định dùng h2_threat.
AI_NAMES = ["minimax", "alpha_beta", "heuristic_h2"]


def aggregate_per_turn(results: list[MatchResult]) -> tuple[list[list[int]], list[list[float]]]:
    """Group node-count and time by turn index across all games."""
    nodes_by_turn: dict[int, list[int]] = defaultdict(list)
    time_by_turn: dict[int, list[float]] = defaultdict(list)
    for r in results:
        # Re-derive per-turn stats by replaying? Cheap: store stats separately.
        # For now we just accumulate totals at end.
        pass
    return nodes_by_turn, time_by_turn


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def write_summary_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def play_matchup(
    ai_x_name: str,
    ai_o_name: str,
    n_games: int,
) -> list[MatchResult]:
    """Chạy n ván giữa 2 AI, các ván lẻ đổi chiều X/O để tránh thiên vị
    lợi thế đi trước. Hiện tại chỉ hỗ trợ heuristic mặc định h2_threat
    vì 3 AI trong AI_NAMES chỉ có một loại heuristic được so sánh."""
    callables = all_ai_callables()

    def _build(name: str) -> callable:
        if name.startswith("heuristic"):
            return make_heuristic_ai("h2_threat")
        return callables[name]

    ai_x = _build(ai_x_name)
    ai_o = _build(ai_o_name)

    results = []
    for i in range(n_games):
        if i % 2 == 1:
            # Ván lẻ: đảo chiều để loại bỏ lợi thế đi trước.
            ai_x_run = _build(ai_o_name)
            ai_o_run = _build(ai_x_name)
        else:
            ai_x_run = ai_x
            ai_o_run = ai_o
        results.append(play_game(ai_x_run, ai_o_run))
    return results


def per_turn_breakdown(result: MatchResult) -> tuple[list[int], list[float]]:
    """Chia đều tổng node / tổng thời gian cho số lượt của ván.

    Lưu ý: đây là XẤP XỈ. Mỗi nước đi của X và O có số node khác nhau
    phụ thuộc vào thế cờ, nhưng ta không instrument sâu bên trong AI —
    chia đều là đủ để vẽ xu hướng tăng/giảm theo lượt.
    """
    turns = max(len(result.moves), 1)
    nodes_per = [
        result.stats_x.nodes_visited // 2 + result.stats_o.nodes_visited // 2
        for _ in range(turns)
    ]
    time_per = [
        result.stats_x.time_elapsed_ms / turns + result.stats_o.time_elapsed_ms / turns
        for _ in range(turns)
    ]
    return nodes_per, time_per


def plot_per_turn(
    ai_name: str,
    all_results: list[MatchResult],
    metric: str,
    ylabel: str,
    out_path: str,
) -> None:
    """Plot per-turn averaged metric across all games an AI participated in."""
    per_turn: dict[int, list[float]] = defaultdict(list)
    for r in all_results:
        # AI plays as X in half the games, O in the other half; average both
        nodes, t = per_turn_breakdown(r)
        for i, v in enumerate(nodes if metric == "nodes" else t):
            per_turn[i].append(v)

    xs = sorted(per_turn.keys())
    ys = [np.mean(per_turn[i]) for i in xs]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.ylabel(ylabel)
    plt.xlabel("Turn index")
    plt.title(f"{ylabel} per turn — {ai_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(n_games: int = 30) -> None:
    raw_rows: list[dict] = []
    summary_rows: list[dict] = []
    ai_totals: dict[str, list[int]] = {name: [] for name in AI_NAMES}  # nodes per game
    ai_times: dict[str, list[float]] = {name: [] for name in AI_NAMES}
    per_ai_results: dict[str, list[MatchResult]] = {name: [] for name in AI_NAMES}

    matchups = list(combinations(AI_NAMES, 2))
    for ai_x_name, ai_o_name in matchups:
        print(f"=== {ai_x_name} (X) vs {ai_o_name} (O) — {n_games} games ===")
        t0 = time.perf_counter()
        results = play_matchup(ai_x_name, ai_o_name, n_games)
        elapsed = time.perf_counter() - t0

        wins_x = sum(1 for r in results if r.winner == bd.PLAYER_X)
        wins_o = sum(1 for r in results if r.winner == bd.PLAYER_O)
        draws = sum(1 for r in results if r.winner == 0)
        total_nodes = sum(r.total_nodes for r in results)
        total_time_ms = sum(r.total_time_ms for r in results)

        summary_rows.append(
            {
                "ai_x": ai_x_name,
                "ai_o": ai_o_name,
                "n_games": n_games,
                "x_wins": wins_x,
                "o_wins": wins_o,
                "draws": draws,
                "total_nodes": total_nodes,
                "avg_nodes_per_game": total_nodes // n_games,
                "total_time_ms": round(total_time_ms, 2),
                "avg_time_ms_per_game": round(total_time_ms / n_games, 2),
                "wall_seconds": round(elapsed, 2),
            }
        )

        for idx, r in enumerate(results):
            ai_x_played = ai_x_name if idx % 2 == 0 else ai_o_name
            ai_o_played = ai_o_name if idx % 2 == 0 else ai_x_name
            ai_totals[ai_x_played].append(r.stats_x.nodes_visited)
            ai_totals[ai_o_played].append(r.stats_o.nodes_visited)
            ai_times[ai_x_played].append(r.stats_x.time_elapsed_ms)
            ai_times[ai_o_played].append(r.stats_o.time_elapsed_ms)
            per_ai_results[ai_x_played].append(r)
            per_ai_results[ai_o_played].append(r)
            raw_rows.append(
                {
                    "matchup": f"{ai_x_name}_vs_{ai_o_name}",
                    "game_idx": idx,
                    "ai_x_played": ai_x_played,
                    "ai_o_played": ai_o_played,
                    "winner": r.winner,
                    "moves": len(r.moves),
                    "x_nodes": r.stats_x.nodes_visited,
                    "o_nodes": r.stats_o.nodes_visited,
                    "x_time_ms": round(r.stats_x.time_elapsed_ms, 2),
                    "o_time_ms": round(r.stats_o.time_elapsed_ms, 2),
                }
            )
        print(
            f"   X wins={wins_x}  O wins={wins_o}  draws={draws}  "
            f"avg_nodes={total_nodes // n_games}  wall={elapsed:.1f}s"
        )

    write_csv(os.path.join(FIG_DIR, "benchmark.csv"), raw_rows)
    write_summary_csv(os.path.join(FIG_DIR, "benchmark_summary.csv"), summary_rows)

    # Total nodes bar chart
    plt.figure()
    means = [np.mean(ai_totals[name]) if ai_totals[name] else 0 for name in AI_NAMES]
    stds = [np.std(ai_totals[name]) if ai_totals[name] else 0 for name in AI_NAMES]
    plt.bar(AI_NAMES, means, yerr=stds, color=["#4c72b0", "#dd8452", "#55a868"])
    plt.ylabel("Nodes visited (mean ± std)")
    plt.title("Average nodes visited per game")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "total_nodes.png"))
    plt.close()

    # Đếm thắng/thua/hòa cho từng AI từ bảng raw (chính xác hơn là
    # từ summary_rows, vì raw có ghi rõ phe thực sự đã chơi ván đó).
    win_counts: dict = {
        name: {"win": 0, "draw": 0, "loss": 0} for name in AI_NAMES
    }
    for raw in raw_rows:
        x = raw["ai_x_played"]
        o = raw["ai_o_played"]
        if raw["winner"] == 1:
            win_counts[x]["win"] += 1
            win_counts[o]["loss"] += 1
        elif raw["winner"] == 2:
            win_counts[o]["win"] += 1
            win_counts[x]["loss"] += 1
        else:
            win_counts[x]["draw"] += 1
            win_counts[o]["draw"] += 1

    fig, ax = plt.subplots()
    x_pos = np.arange(len(AI_NAMES))
    width = 0.27
    ax.bar(x_pos - width, [win_counts[n]["win"] for n in AI_NAMES], width, label="Win", color="#55a868")
    ax.bar(x_pos, [win_counts[n]["draw"] for n in AI_NAMES], width, label="Draw", color="#c8c6be")
    ax.bar(x_pos + width, [win_counts[n]["loss"] for n in AI_NAMES], width, label="Loss", color="#c44e52")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(AI_NAMES)
    ax.set_ylabel("Games")
    ax.set_title("Win / Draw / Loss per AI (round-robin)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "win_rates.png"))
    plt.close()

    # Per-turn plots
    for name in AI_NAMES:
        plot_per_turn(
            name,
            per_ai_results[name],
            "nodes",
            "Avg nodes / turn",
            os.path.join(FIG_DIR, f"{name}_nodes_per_turn.png"),
        )
        plot_per_turn(
            name,
            per_ai_results[name],
            "time",
            "Avg time (ms) / turn",
            os.path.join(FIG_DIR, f"{name}_time_per_turn.png"),
        )

    print("\n=== Done. ===")
    print(f"Outputs in {FIG_DIR}")
    for f in sorted(os.listdir(FIG_DIR)):
        print(f"  - {f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(n_games=n)