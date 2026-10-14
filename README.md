# Tic-Tac-Toe — Alpha-Beta Pruning & Heuristic AI

**Đề tài 6** — mở rộng từ đề 5 (Minimax).

3 agent tự động chơi Tic-Tac-Toe:
- **Minimax** — tìm kiếm tuyệt đối, không cắt tỉa
- **Alpha-Beta** — cắt tỉa bằng alpha-beta pruning
- **Heuristic AI** — alpha-beta + move ordering bằng hàm heuristic (h1/h2)

---

## Cài đặt (1 lần)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Cách demo

### Cách 1 — Nhanh nhất: chạy notebook trên VS Code (Recommended)

```bash
code demo.ipynb
```

Mở file `demo.ipynb` trong VS Code → cài extension **Jupyter** nếu chưa có →
chạy từng cell bằng `Shift+Enter`. Không cần cài thêm gì.

Notebook gồm **17 phần**:

| Cell | Nội dung |
|---|---|
| 1–3 | Giới thiệu + PEAS |
| 4–6 | **Tương tác người vs AI** — click ô trên bàn cờ, chọn AI nào cho X/O, log in số node mỗi nước đi |
| 7–10 | Hàm heuristic + **1 ván AI vs AI** (in từng nước + node + ms) |
| 11–12 | **Vẽ cây tìm kiếm 25 nodes** (yêu cầu đề bài) |
| 13–16 | **Benchmark vòng tròn** 3 AI — so sánh số node, thời gian, win rate |
| 17 | Kết luận |

---

### Cách 2 — GUI tương tác

```bash
.venv/bin/python -m src.gui.tk_ui
```

Mở cửa sổ Tkinter: chọn AI cho X và O → chơi hoặc để AI đấu nhau → xem số node/ms mỗi nước.

---

### Cách 3 — Benchmark đầy đủ (tạo biểu đồ)

```bash
.venv/bin/python -m experiments.benchmark
```

Chạy 3 AI đấu vòng tròn → xuất biểu đồ vào `docs/figures/`:
`total_nodes.png`, `win_rates.png`, `*_time_per_turn.png`, `search_tree_*.png`.

---

## Cấu trúc

```
src/
  board.py             # Bàn cờ, nước đi, kiểm tra thắng
  heuristics.py       # h1 (điểm vị trí) + h2 (mối đe dọa)
  metrics.py          # đếm node, thời gian, chạy ván
  algorithms/
    minimax.py        # Minimax thuần
    alpha_beta.py     # Alpha-Beta Pruning
    heuristic_ai.py   # Alpha-Beta + move ordering
  gui/
    tk_ui.py          # Giao diện Tkinter
experiments/
  benchmark.py         # Chạy vòng tròn + vẽ biểu đồ
  draw_trees.py       # Vẽ cây tìm kiếm
  build_notebook.py   # Sinh demo.ipynb
docs/
  report.docx         # Báo cáo nộp thầy
  figures/             # Biểu đồ benchmark + cây tìm kiếm
demo.ipynb            # Notebook demo thuyết trình
```
# tiktokto_aiandtool
