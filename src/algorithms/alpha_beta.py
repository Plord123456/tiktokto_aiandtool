"""Alpha-Beta pruning trên nền Minimax.

Cùng kết quả tối ưu như Minimax nhưng cắt bỏ những nhánh chắc chắn
không ảnh hưởng đến quyết định cuối cùng, nhờ hai cận α / β:
    - α: điểm tốt nhất mà MAX (player hiện tại) có thể đảm bảo
    - β : điểm tốt nhất mà MIN (đối thủ) có thể đảm bảo
Khi α ≥ β thì nhánh còn lại không thể cải thiện kết quả → dừng duyệt.

Đếm số nút truy cập (gốc + mọi lệnh gọi đệ quy) vào `metrics` để so sánh
với Minimax thuần.
"""

from __future__ import annotations

import numpy as np

from src import board as bd


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


def _alpha_beta(
    state: np.ndarray,
    player: int,
    alpha: int,
    beta: int,
    metrics: dict,
) -> int:
    """Alpha-Beta đệ quy dạng negamax.

    Vì đổi vai trò người chơi mỗi lần đệ quy, ta đảo cả dấu điểm lẫn
    cận (alpha, beta) -> (-beta, -alpha) — đây là cách viết gọn và
    tương đương với Alpha-Beta cổ điển.
    """
    metrics["nodes_visited"] += 1
    s = _score_terminal(state, player)
    if s is not None:
        return s

    moves = bd.legal_moves(state)
    opp = bd.other(player)
    best = -2
    for m in moves:
        child = bd.apply_move(state, m, player)
        score = -_alpha_beta(child, opp, -beta, -alpha, metrics)
        if score > best:
            best = score
        # Cập nhật α. Nếu α vượt β thì nhánh còn lại của đối thủ chắc chắn
        # không cho MAX điểm tốt hơn -> cắt tỉa (beta cutoff).
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def choose_move(state: np.ndarray, player: int, metrics: dict) -> tuple[int, int]:
    """Chọn nước đi tối ưu bằng Alpha-Beta ở lớp gốc.

    Khởi tạo alpha=-2, beta=2 vì điểm nằm trong khoảng {-1, 0, 1}.
    """
    metrics.setdefault("nodes_visited", 0)
    best_move = None
    best_score = -2
    alpha = -2
    beta = 2
    opp = bd.other(player)
    for m in bd.legal_moves(state):
        child = bd.apply_move(state, m, player)
        score = -_alpha_beta(child, opp, -beta, -alpha, metrics)
        if score > best_score:
            best_score = score
            best_move = m
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best_move