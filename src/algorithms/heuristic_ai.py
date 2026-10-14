"""Heuristic AI = Alpha-Beta + sắp xếp nước đi theo heuristic.

Cùng khung đệ quy với `alpha_beta.py`, nhưng trước mỗi lần duyệt các
nước đi hợp lệ, ta SẮP XẾP chúng theo điểm heuristic giảm dần. Ý tưởng:
nếu nước đi tốt nhất được thử trước thì α và β sẽ co lại nhanh hơn,
từ đó cắt được nhiều nhánh hơn ở các nước đi sau.

Cảm hứng từ paper Don Yitong Ma (2023) "Optimization of Alpha-Beta
Pruning based on Heuristic Algorithm".

Điểm quan trọng: heuristic ở đây chỉ ảnh hưởng THỨ TỰ duyệt, KHÔNG ảnh
hưởng giá trị cận α / β. Vì vậy thuật toán vẫn trả về nước đi tối ưu
giống Alpha-Beta thuần — chỉ nhanh hơn.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src import board as bd
from src.heuristics import h1_position_score, h2_threat_score


# Tra cứu heuristic theo tên — cho phép GUI / benchmark chọn.
DEFAULT_HEURISTICS: dict[str, Callable] = {
    "h1_position": h1_position_score,
    "h2_threat": h2_threat_score,
}


def _score_terminal(state: np.ndarray, player: int) -> int | None:
    """Chấm điểm trạng thái kết thúc: +1 thắng, -1 thua, 0 hòa, None nếu chưa kết thúc."""
    w = bd.winner(state)
    if w == player:
        return 1
    if w == bd.other(player):
        return -1
    if bd.is_terminal(state):
        return 0
    return None


def _ordered_moves(
    state: np.ndarray,
    player: int,
    heuristic: Callable[[np.ndarray, int], int],
) -> list[tuple[int, int]]:
    """Trả về danh sách nước đi hợp lệ, sắp xếp theo điểm heuristic.

    Ở lớp MAX (player) → sort giảm dần: nước tốt nhất thử trước
    → α co nhanh → beta cutoff nhiều hơn.
    Ở lớp MIN (đối thủ) → sort TĂNG DẦN: nước có điểm heuristic THẤP
    (từ góc MAX) thử trước → β co nhanh → alpha cutoff nhiều hơn.
    """
    scored = []
    for m in bd.legal_moves(state):
        child = bd.apply_move(state, m, player)
        scored.append((heuristic(child, player), m))
    # MAX: sort giảm dần (best first) | MIN: sort tăng dần (best for MAX first)
    scored.sort(key=lambda x: x[0], reverse=(player == bd.PLAYER_X))
    return [m for _, m in scored]


def _alpha_beta_ordered(
    state: np.ndarray,
    player: int,
    alpha: int,
    beta: int,
    heuristic: Callable[[np.ndarray, int], int],
    metrics: dict,
) -> int:
    """Alpha-Beta có move ordering. Logic giống hệt `_alpha_beta` của
    `alpha_beta.py`, chỉ thay vòng lặp duyệt moves bằng `_ordered_moves`."""
    metrics["nodes_visited"] += 1
    s = _score_terminal(state, player)
    if s is not None:
        return s

    opp = bd.other(player)
    best = -2
    for m in _ordered_moves(state, player, heuristic):
        child = bd.apply_move(state, m, player)
        score = -_alpha_beta_ordered(child, opp, -beta, -alpha, heuristic, metrics)
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def choose_move(
    state: np.ndarray,
    player: int,
    metrics: dict,
    heuristic_name: str = "h2_threat",
) -> tuple[int, int]:
    """Chọn nước đi bằng Alpha-Beta có move ordering theo heuristic `heuristic_name`.

    Mặc định dùng h2_threat vì nó thể hiện rõ hơn các mối đe dọa 2-quân
    (so với h1 chỉ dựa trên vị trí tĩnh).
    """
    metrics.setdefault("nodes_visited", 0)
    heuristic = DEFAULT_HEURISTICS[heuristic_name]
    best_move = None
    best_score = -2
    alpha = -2
    beta = 2
    opp = bd.other(player)
    for m in _ordered_moves(state, player, heuristic):
        child = bd.apply_move(state, m, player)
        score = -_alpha_beta_ordered(child, opp, -beta, -alpha, heuristic, metrics)
        if score > best_score:
            best_score = score
            best_move = m
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best_move