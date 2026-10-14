"""GUI Tkinter cho Tic-Tac-Toe với 3 AI.

Bố cục:
    [ Bàn cờ 3×3 ]   [ Chọn AI cho X ]
                     [ Chọn AI cho O ]
                     [ Chọn heuristic  ]
                     [ Chọn chế độ     ]
                     [ Nút Reset       ]
                     [ Thanh trạng thái ]
                     [ Bảng metrics (node/time) ]

Chế độ chơi:
    * Human vs AI  — chọn AI cho phe còn lại, bạn điều khiển phe còn lại
    * AI vs AI     — chọn AI cho cả 2 phe, xem chúng tự đấu
    * Human vs Human
"""

from __future__ import annotations

import os
import sys
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

# Cho phép chạy file này trực tiếp (`python src/gui/tk_ui.py`) bằng cách
# thêm thư mục gốc của project vào sys.path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src import board as bd
from src.metrics import all_ai_callables, make_heuristic_ai


# Tên hiển thị trong combobox phải khớp với key trong all_ai_callables().
AI_OPTIONS = ["minimax", "alpha_beta", "heuristic_ai"]
HEURISTIC_OPTIONS = ["h1_position", "h2_threat"]


def _ai_callable(name: str, heuristic: str):
    """Lookup AI theo tên; heuristic chỉ cần khi AI là `heuristic_ai`."""
    if name == "heuristic_ai":
        return make_heuristic_ai(heuristic)
    return all_ai_callables()[name]


class TicTacToeApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Tic-Tac-Toe — Alpha-Beta + Heuristic AI")
        self.state = bd.empty_board()
        self.buttons: list[tk.Button] = []
        self.busy = False

        self._build_layout()

    def _build_layout(self) -> None:
        # 2 cột: trái là bàn cờ, phải là bảng điều khiển.
        left = tk.Frame(self.root, padx=10, pady=10)
        left.grid(row=0, column=0)
        right = tk.Frame(self.root, padx=10, pady=10)
        right.grid(row=0, column=1, sticky="n")

        # 3×3 nút bấm — mỗi nút gắn với (r, c) qua default-arg trick
        # (lambda r=r, c=c) để tránh bị Python bắt biến vòng lặp.
        for r in range(bd.BOARD_SIZE):
            for c in range(bd.BOARD_SIZE):
                b = tk.Button(
                    left,
                    text="",
                    font=("Helvetica", 32, "bold"),
                    width=4,
                    height=2,
                    command=lambda r=r, c=c: self._on_cell_click(r, c),
                )
                b.grid(row=r, column=c, padx=2, pady=2)
                self.buttons.append(b)

        # Bảng điều khiển bên phải — thứ tự widget xếp theo dòng.
        tk.Label(right, text="AI for X", font=("Helvetica", 11)).grid(row=0, column=0, sticky="w")
        self.ai_x_var = tk.StringVar(value="alpha_beta")
        ttk.Combobox(right, textvariable=self.ai_x_var, values=AI_OPTIONS, state="readonly").grid(
            row=0, column=1, sticky="ew"
        )

        tk.Label(right, text="AI for O", font=("Helvetica", 11)).grid(row=1, column=0, sticky="w")
        self.ai_o_var = tk.StringVar(value="heuristic_ai")
        ttk.Combobox(right, textvariable=self.ai_o_var, values=AI_OPTIONS, state="readonly").grid(
            row=1, column=1, sticky="ew"
        )

        tk.Label(right, text="Heuristic", font=("Helvetica", 11)).grid(row=2, column=0, sticky="w")
        self.heuristic_var = tk.StringVar(value="h2_threat")
        ttk.Combobox(
            right, textvariable=self.heuristic_var, values=HEURISTIC_OPTIONS, state="readonly"
        ).grid(row=2, column=1, sticky="ew")

        self.mode_var = tk.StringVar(value="Human vs AI (you = X)")
        ttk.Combobox(
            right,
            textvariable=self.mode_var,
            values=["Human vs AI (you = X)", "Human vs AI (you = O)", "AI vs AI", "Human vs Human"],
            state="readonly",
        ).grid(row=3, column=0, columnspan=2, sticky="ew")

        tk.Button(right, text="Reset", command=self._reset).grid(row=4, column=0, columnspan=2, sticky="ew", pady=4)

        self.status_var = tk.StringVar(value="Your turn (X).")
        tk.Label(right, textvariable=self.status_var, font=("Helvetica", 11, "bold")).grid(
            row=5, column=0, columnspan=2, sticky="w"
        )

        # Bảng metrics — text widget để in nhiều dòng dễ đọc.
        self.metrics_text = tk.Text(right, width=46, height=12, font=("Courier", 10))
        self.metrics_text.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=4)
        self._update_metrics_panel(0, 0.0, 0, None)

    def _reset(self) -> None:
        self.state = bd.empty_board()
        self._render()
        self._update_metrics_panel(0, 0.0, 0, None)
        mode = self.mode_var.get()
        if mode == "AI vs AI":
            self.status_var.set("AI vs AI starting…")
            self.root.after(200, self._ai_vs_ai_step)
        elif "you = X" in mode:
            self.status_var.set("Your turn (X).")
        else:
            self.status_var.set("AI starts.")

    def _render(self) -> None:
        for r in range(bd.BOARD_SIZE):
            for c in range(bd.BOARD_SIZE):
                v = int(self.state[r, c])
                txt = "" if v == bd.EMPTY else bd.SYMBOLS[v]
                color = {"": "#f0f0f0", "X": "#1f77b4", "O": "#d62728"}.get(txt, "black")
                idx = r * bd.BOARD_SIZE + c
                self.buttons[idx].config(text=txt, fg=color, state=tk.NORMAL if v == bd.EMPTY else tk.DISABLED)

    def _on_cell_click(self, r: int, c: int) -> None:
        """Bắt sự kiện click ô cờ. Bỏ qua khi AI đang suy nghĩ (`self.busy`)
        hoặc khi đang ở chế độ AI-vs-AI (click vô hiệu)."""
        if self.busy:
            return
        mode = self.mode_var.get()
        if mode == "AI vs AI":
            return  # AI đấu AI — click chuột bị vô hiệu hoá
        if mode == "Human vs Human":
            # Ở chế độ 2 người, người chơi hiện tại là phe có số quân ít hơn.
            current = bd.PLAYER_X if np.sum(self.state == bd.EMPTY) == 9 - 0 else (
                bd.PLAYER_X if (np.sum(self.state == bd.PLAYER_X) <= np.sum(self.state == bd.PLAYER_O))
                else bd.PLAYER_O
            )
            self._apply_human_move(r, c, current)
            return

        # Human-vs-AI: chỉ cho người đi khi đúng lượt.
        human_player = bd.PLAYER_X if "you = X" in mode else bd.PLAYER_O
        if self._current_player() != human_player:
            return
        self._apply_human_move(r, c, human_player)

    def _apply_human_move(self, r: int, c: int, player: int) -> None:
        if self.state[r, c] != bd.EMPTY:
            return
        self.state[r, c] = player
        self._after_move(player)

    def _after_move(self, last_player: int) -> None:
        """Xử lý sau mỗi nước đi (của người lẫn AI): vẽ lại bàn cờ, kiểm
        tra thắng/hòa, và quyết định phe tiếp theo do AI hay người điều khiển.

        Việc gọi AI qua `root.after(...)` (không gọi trực tiếp) là để tránh
        đứng GUI: AI có thể mất vài giây, nếu chạy trong luồng GUI chính sẽ
        treo cửa sổ. `self.busy` chặn người dùng click khi AI đang suy nghĩ.
        """
        self._render()
        winner = bd.winner(self.state)
        if winner != 0:
            self._end_game(f"{bd.SYMBOLS[winner]} wins!")
            return
        if not np.any(self.state == bd.EMPTY):
            self._end_game("Draw.")
            return

        mode = self.mode_var.get()
        next_player = bd.other(last_player)
        if mode == "Human vs Human":
            self.status_var.set(f"{bd.SYMBOLS[next_player]}'s turn.")
            return

        if mode == "AI vs AI":
            self.root.after(150, self._ai_vs_ai_step)
            return

        human_player = bd.PLAYER_X if "you = X" in mode else bd.PLAYER_O
        if next_player == human_player:
            self.status_var.set(f"Your turn ({bd.SYMBOLS[next_player]}).")
        else:
            self.status_var.set(f"AI ({bd.SYMBOLS[next_player]}) thinking…")
            self.busy = True
            self.root.after(50, lambda: self._ai_step(next_player))

    def _ai_step(self, player: int) -> None:
        """Một lượt đi của AI: tra AI theo phe, đo thời gian, cập nhật
        bảng metrics rồi áp dụng nước đi. Đây là nơi GUI "giao tiếp" với
        engine AI ở bên dưới."""
        name = self.ai_x_var.get() if player == bd.PLAYER_X else self.ai_o_var.get()
        heuristic = self.heuristic_var.get()
        ai = _ai_callable(name, heuristic)

        metrics = {"nodes_visited": 0}
        t0 = time.perf_counter()
        move = ai(self.state, player, metrics)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self._update_metrics_panel(
            metrics["nodes_visited"],
            elapsed_ms,
            player,
            move,
        )
        self.state = bd.apply_move(self.state, move, player)
        self.busy = False
        self._after_move(player)

    def _ai_vs_ai_step(self) -> None:
        if self.busy:
            return
        current = bd.PLAYER_X if np.sum(self.state == bd.PLAYER_X) <= np.sum(self.state == bd.PLAYER_O) else bd.PLAYER_O
        if bd.winner(self.state) != 0 or not np.any(self.state == bd.EMPTY):
            return
        self.busy = True
        self.root.after(50, lambda: self._ai_step(current))

    def _current_player(self) -> int:
        n_x = int(np.sum(self.state == bd.PLAYER_X))
        n_o = int(np.sum(self.state == bd.PLAYER_O))
        return bd.PLAYER_X if n_x <= n_o else bd.PLAYER_O

    def _end_game(self, msg: str) -> None:
        self.status_var.set(msg)
        for b in self.buttons:
            b.config(state=tk.DISABLED)

    def _update_metrics_panel(
        self,
        nodes: int,
        elapsed_ms: float,
        player: int,
        move: tuple[int, int] | None,
    ) -> None:
        self.metrics_text.delete("1.0", tk.END)
        move_str = f"({move[0]+1}, {move[1]+1})" if move is not None else "n/a"
        self.metrics_text.insert(
            tk.END,
            f"Last AI move: {bd.SYMBOLS.get(player, '?')} -> {move_str}\n"
            f"Nodes visited: {nodes:,}\n"
            f"Time elapsed : {elapsed_ms:8.2f} ms\n"
            f"\n"
            f"Current heuristic: {self.heuristic_var.get()}\n"
            f"AI for X       = {self.ai_x_var.get()}\n"
            f"AI for O       = {self.ai_o_var.get()}\n",
        )


def main() -> None:
    root = tk.Tk()
    TicTacToeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()