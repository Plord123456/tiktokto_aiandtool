"""Tic-Tac-Toe board state, transition, and win detection.

Grid is a 3x3 numpy int array:
    0 = empty
    1 = X
    2 = O

Players alternate turns. Win = 3 in a row (horizontal, vertical, diagonal).
"""

from __future__ import annotations

import numpy as np

BOARD_SIZE = 3
WIN_LENGTH = 3
EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2

SYMBOLS = {PLAYER_X: "X", PLAYER_O: "O", EMPTY: "."}


def empty_board() -> np.ndarray:
    return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)


def legal_moves(board: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(c) for c in np.argwhere(board == EMPTY)]


def apply_move(board: np.ndarray, move: tuple[int, int], player: int) -> np.ndarray:
    """Return a new board after applying `move` for `player`. Does not mutate input."""
    r, c = move
    if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
        raise ValueError(f"Move out of bounds: {move}")
    if board[r, c] != EMPTY:
        raise ValueError(f"Cell {move} already taken")
    new = board.copy()
    new[r, c] = player
    return new


def winner(board: np.ndarray) -> int:
    """Return 1, 2, or 0 (no winner yet / draw)."""
    for p in (PLAYER_X, PLAYER_O):
        for r in range(BOARD_SIZE):
            if all(board[r, :] == p):
                return p
        for c in range(BOARD_SIZE):
            if all(board[:, c] == p):
                return p
        if all(board.diagonal() == p):
            return p
        if all(np.flip(board).diagonal() == p):
            return p
    return 0


def is_terminal(board: np.ndarray) -> bool:
    return winner(board) != 0 or not np.any(board == EMPTY)


def other(player: int) -> int:
    return PLAYER_O if player == PLAYER_X else PLAYER_X


def render(board: np.ndarray) -> str:
    rows = []
    for r in range(BOARD_SIZE):
        rows.append(" ".join(SYMBOLS[int(board[r, c])] for c in range(BOARD_SIZE)))
    return "\n".join(rows)