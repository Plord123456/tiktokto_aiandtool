"""Minimax thuần cho Tic-Tac-Toe.

Là thuật toán baseline: duyệt TOÀN BỘ cây trò chơi đến trạng thái kết thúc
(không cắt tỉa). Dùng làm mốc so sánh với Alpha-Beta và Heuristic AI.

Trả về nước đi tốt nhất qua `choose_move()` và cập nhật dict `metrics`
để đếm số nút đã duyệt.
"""

from __future__ import annotations

import numpy as np

from src import board as bd


def _score_terminal(state: np.ndarray, player: int) -> int | None:
    """Chấm điểm trạng thái kết thúc từ góc nhìn của `player`.

    Quy ước điểm: +1 thắng, -1 thua, 0 hòa. Trả về None nếu trạng thái
    chưa kết thúc — caller sẽ tiếp tục đệ quy.
    """
    w = bd.winner(state)
    if w == player:
        return 1
    if w == bd.other(player):
        return -1
    if bd.is_terminal(state):
        return 0
    return None


def _minimax(
    state: np.ndarray,
    player: int,
    metrics: dict,
) -> int:
    """Hàm đệ quy Minimax. Trả về điểm tốt nhất mà `player` đạt được.

    Mẹo "negamax" (-rec(...) ở dòng dưới): thay vì viết 2 hàm MAX/MIN
    riêng, ta đổi vai trò người chơi mỗi lần đệ quy và đảo dấu điểm.
    Cùng kết quả, code gọn hơn.
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
        score = -_minimax(child, opp, metrics)
        if score > best:
            best = score
    return best


def choose_move(state: np.ndarray, player: int, metrics: dict) -> tuple[int, int]:
    """Chọn nước đi tối ưu cho `player` bằng Minimax thuần.

    Thử từng nước đi hợp lệ ở lớp hiện tại, mỗi nước gọi đệ quy xuống
    lớp đối thủ và lấy điểm (đã đảo dấu). Nước nào có điểm cao nhất là
    nước đi tốt nhất.
    """
    metrics.setdefault("nodes_visited", 0)
    best_move = None
    best_score = -2
    opp = bd.other(player)
    for m in bd.legal_moves(state):
        child = bd.apply_move(state, m, player)
        score = -_minimax(child, opp, metrics)
        if score > best_score:
            best_score = score
            best_move = m
    return best_move