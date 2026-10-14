"""Hai hàm heuristic cho Tic-Tac-Toe.

Hai hàm heuristic này chỉ dùng để SẮP XẾP THỨ TỰ nước đi (move ordering)
trước khi Alpha-Beta duyệt — KHÔNG cắt tỉa kết quả. Vì vậy chúng không
cần thoả mãn tính chất admissible/consistent như trong A*.

h1 — Position score (lấy từ paper AsadJatoi 2023):
    Trọng số vị trí: ô giữa = 3, góc = 2, cạnh = 1 cho quân của player;
    trừ đi điểm tương ứng nếu ô đó thuộc về đối thủ.

h2 — Threat / line-formation score (cảm hứng từ paper Don Yitong Ma 2023):
    Quét 8 đường (3 hàng, 3 cột, 2 chéo). Mỗi đường đóng góp điểm dựa
    trên số quân thân thiện / đối thủ còn lại. Đường bị chặn (có cả 2 phe)
    thì đóng góp 0.
"""

from __future__ import annotations

import numpy as np

from src import board as bd

# Bảng trọng số vị trí — định nghĩa 1 lần để dùng cho h1.
# Center = 3, Corner = 2, Edge = 1.
_POSITION_WEIGHT = np.array(
    [
        [2, 1, 2],
        [1, 3, 1],
        [2, 1, 2],
    ],
    dtype=np.int32,
)


def h1_position_score(state: np.ndarray, player: int) -> int:
    """Tổng trọng số vị trí: cộng điểm các ô của player, trừ điểm các ô của đối thủ.

    Hàm đối xứng: nếu đổi `player` ↔ `other(player)` thì điểm đổi dấu.
    """
    opp = bd.other(player)
    score = 0
    for r in range(bd.BOARD_SIZE):
        for c in range(bd.BOARD_SIZE):
            cell = int(state[r, c])
            if cell == player:
                score += int(_POSITION_WEIGHT[r, c])
            elif cell == opp:
                score -= int(_POSITION_WEIGHT[r, c])
    return score


def _lines(state: np.ndarray) -> list[np.ndarray]:
    """Sinh ra 8 đường có thể thắng: 3 hàng, 3 cột, 2 chéo."""
    lines = [state[r, :] for r in range(bd.BOARD_SIZE)]
    lines += [state[:, c] for c in range(bd.BOARD_SIZE)]
    lines.append(state.diagonal())
    lines.append(np.flip(state).diagonal())
    return lines


def _line_score(line: np.ndarray, player: int) -> int:
    """Chấm điểm 1 đường 3 ô từ góc nhìn của `player`.

    Quy ước điểm (đã chuẩn hoá so với bản gốc của code Caro):
        +1000 nếu player có đủ 3 (thắng)
         -1000 nếu đối thủ có đủ 3 (thua)
        ±50  nếu phe đó có 2 quân + 1 ô trống (đe dọa thắng nước kế)
        ±5   nếu phe đó có 1 quân + 2 ô trống (tiềm năng)
    Đường có cả 2 phe thì đóng góp 0 — bị chặn không thể thắng.
    """
    opp = bd.other(player)
    p_count = int(np.sum(line == player))
    o_count = int(np.sum(line == opp))
    empty = int(np.sum(line == bd.EMPTY))
    if p_count > 0 and o_count > 0:
        return 0  # đường bị chặn — cả 2 phe đều không thể thắng
    if p_count == 0 and o_count == 0:
        return 0  # đường rỗng — không mang thông tin
    if p_count == 3:
        return 1000
    if o_count == 3:
        return -1000
    if p_count == 2 and empty == 1:
        return 50   # đe dọa mạnh — đối thủ phải chặn
    if o_count == 2 and empty == 1:
        return -50  # bị đe doạ — phải phòng thủ
    if p_count == 1 and empty == 2:
        return 5
    if o_count == 1 and empty == 2:
        return -5
    return 0


def h2_threat_score(state: np.ndarray, player: int) -> int:
    """Tổng điểm đe dọa: cộng điểm của cả 8 đường, đã bao gồm cả dương
    (lợi thế cho player) lẫn âm (bất lợi / bị đe doạ)."""
    return sum(_line_score(line, player) for line in _lines(state))


def all_heuristics() -> dict:
    """Tra cứu theo tên — dùng cho GUI/benchmark chọn heuristic."""
    return {"h1_position": h1_position_score, "h2_threat": h2_threat_score}