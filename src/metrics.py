"""Bộ đếm số liệu và bộ chạy ván AI-vs-AI.

Mỗi AI là một callable có dạng `ai(state, player, metrics) -> move`.
Minimax và Alpha-Beta thoả ngay; Heuristic AI cần truyền thêm tên
heuristic nên ta bọc (wrap) bằng `make_heuristic_ai` để đồng nhất giao
diện — GUI/benchmark chỉ cần gọi 1 hàm duy nhất cho cả 3 thuật toán.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src import board as bd
from src.algorithms import alpha_beta, heuristic_ai, minimax


# Registry các AI "có sẵn" — heuristic AI tách riêng vì cần tham số phụ.
AI_REGISTRY: dict[str, Callable] = {
    "minimax": minimax.choose_move,
    "alpha_beta": alpha_beta.choose_move,
}


def make_heuristic_ai(heuristic_name: str = "h2_threat") -> Callable:
    """Đóng gói `heuristic_ai.choose_move` thành callable cùng chữ ký
    `ai(state, player, metrics) -> move`. Đặt `__name__` để dễ debug
    khi in log benchmark."""
    fn = heuristic_ai.choose_move

    def ai(state, player, metrics):
        return fn(state, player, metrics, heuristic_name=heuristic_name)

    ai.__name__ = f"heuristic_ai[{heuristic_name}]"
    return ai


def all_ai_callables() -> dict[str, Callable]:
    """Liệt kê tất cả agent — GUI và benchmark dùng chung từ điển này
    để tra cứu theo tên."""
    return {
        **AI_REGISTRY,
        "heuristic_h1": make_heuristic_ai("h1_position"),
        "heuristic_h2": make_heuristic_ai("h2_threat"),
    }


@dataclass
class MoveStats:
    """Thống kê cộng dồn cho 1 phe trong 1 ván: tổng số nút duyệt và
    tổng thời gian AI suy nghĩ (tính bằng mili-giây)."""
    nodes_visited: int = 0
    time_elapsed_ms: float = 0.0


@dataclass
class MatchResult:
    """Kết quả 1 ván đấu: ai thắng, danh sách nước đã đi, và thống kê
    tách riêng cho X và O để có thể phân tích theo phe."""
    winner: int = 0  # 0 = hòa, 1 = X thắng, 2 = O thắng
    moves: list[tuple[int, int]] = field(default_factory=list)
    stats_x: MoveStats = field(default_factory=MoveStats)
    stats_o: MoveStats = field(default_factory=MoveStats)

    @property
    def total_nodes(self) -> int:
        """Tổng node cả 2 phe — dùng để so sánh sức mạnh thuật toán."""
        return self.stats_x.nodes_visited + self.stats_o.nodes_visited

    @property
    def total_time_ms(self) -> float:
        return self.stats_x.time_elapsed_ms + self.stats_o.time_elapsed_ms


def _play_turn(
    state: np.ndarray,
    player: int,
    ai: Callable,
    stats: MoveStats,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Cho 1 phe đi 1 nước, đo thời gian, cộng dồn vào `stats`.

    Trả về (bàn cờ mới, nước vừa đi). Bàn cờ trả về qua `apply_move`
    nên input `state` không bị mutate — quan trọng vì caller có thể
    cần state cũ để hiển thị.
    """
    metrics = {"nodes_visited": 0}
    t0 = time.perf_counter()
    move = ai(state, player, metrics)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    stats.nodes_visited += int(metrics["nodes_visited"])
    stats.time_elapsed_ms += elapsed_ms
    return bd.apply_move(state, move, player), move


def play_game(
    ai_x: Callable,
    ai_o: Callable,
    starting_player: int = bd.PLAYER_X,
) -> MatchResult:
    """Chạy 1 ván Tic-Tac-Toe đầy đủ.

    Mặc định X đi trước; nếu cần đổi phe (ví dụ benchmark) thì truyền
    `starting_player`. Cờ `X đi trước` được suy ra bằng `np.sum` quân
    mỗi phe — không cần lưu thêm biến turn.
    """
    state = bd.empty_board()
    result = MatchResult()
    current = starting_player
    ais = {bd.PLAYER_X: ai_x, bd.PLAYER_O: ai_o}
    stats_map = {bd.PLAYER_X: result.stats_x, bd.PLAYER_O: result.stats_o}

    while True:
        state, move = _play_turn(state, current, ais[current], stats_map[current])
        result.moves.append(move)
        if bd.winner(state) != 0:
            result.winner = bd.winner(state)
            return result
        if not np.any(state == bd.EMPTY):
            result.winner = 0
            return result
        current = bd.other(current)


def run_match(
    ai_x_name: str,
    ai_o_name: str,
    n_games: int,
    heuristic_x: str = "h2_threat",
    heuristic_o: str = "h2_threat",
    swap_sides: bool = True,
) -> list[MatchResult]:
    """Chạy n ván giữa 2 AI, có thể đổi phe ở nửa sau để loại bỏ
    lợi thế đi trước (X luôn đi trước theo luật Tic-Tac-Toe).

    Trả về danh sách `MatchResult` theo thứ tự thời gian.
    """
    callables = all_ai_callables()

    def _build(name: str, h: str) -> Callable:
        if name.startswith("heuristic"):
            return make_heuristic_ai(h)
        return callables[name]

    results: list[MatchResult] = []
    for i in range(n_games):
        # Ván lẻ (chỉ số lẻ) đổi chiều X/O để cân bằng lợi thế đi trước.
        if swap_sides and i % 2 == 1:
            ai_x = _build(ai_o_name, heuristic_o)
            ai_o = _build(ai_x_name, heuristic_x)
        else:
            ai_x = _build(ai_x_name, heuristic_x)
            ai_o = _build(ai_o_name, heuristic_o)
        results.append(play_game(ai_x, ai_o))
    return results