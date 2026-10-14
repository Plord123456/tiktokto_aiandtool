# Tic-Tac-Toe với Alpha-Beta Pruning & Heuristic AI — Slide thuyết trình

> Đề tài 6 — mở rộng đề tài 5.
> Trình bày ~10–15 phút.

## Slide 1 — Bìa

- Tên đề tài: **Tic-Tac-Toe với Alpha-Beta Pruning và Heuristic AI**
- Tên nhóm / MSSV
- Môn: Trí tuệ nhân tạo — GVHD:

## Slide 2 — Tổng quan đề tài

- Đề tài 6: mở rộng đề 5 (Minimax) bằng Alpha-Beta Pruning
- Mục tiêu: so sánh số nút duyệt, thời gian, tỉ lệ thắng của 3 agent
  - Minimax thuần
  - Alpha-Beta Pruning
  - Alpha-Beta + Heuristic ordering

## Slide 3 — Đặc tả PEAS

- **P**erformance: thắng / hòa / thua + số node + thời gian
- **E**nvironment: bàn cờ 3×3
- **A**ctuators: đặt X / O vào ô trống
- **S**ensors: quan sát toàn bộ 9 ô
- Phân loại: quan sát được đầy đủ, xác định, tĩnh, rời rạc, đa tác nhân cạnh tranh

## Slide 4 — Định nghĩa bài toán tìm kiếm

- State, initial state, Actions, Transition, Goal test, Path cost (bảng)
- Nhấn mạnh: Tic-Tac-Toe có không gian trạng thái rất nhỏ (3^9 = 19683) nên cả 3 thuật toán đều chạy được

## Slide 5 — Transition Function

- Công thức RESULT(s, a): thay ô (r, c) của trạng thái s bằng giá trị player hiện tại
- Minh họa 1 transition: bàn cờ rỗng → X đặt vào giữa
- (Có bảng 12 dòng trong report)

## Slide 6 — Cây tìm kiếm Minimax

- Hình: search_tree_minimax.png
- Nhận xét: duyệt tất cả nhánh → ~550k nút / ván

## Slide 7 — Cây tìm kiếm Alpha-Beta

- Hình: search_tree_alpha_beta.png
- Nhận xét: cắt nhờ α ≥ β → ~9.2k nút / ván (-98%)

## Slide 8 — Cây tìm kiếm Alpha-Beta + Heuristic

- Hình: search_tree_heuristic.png
- Nhận xét: move ordering theo h2_threat → ~7.5k nút / ván (cắt thêm nhờ α-β co nhanh hơn)

## Slide 9 — Heuristic

- **h1 — Position score**: center=3, corner=2, edge=1 (theo AsadJatoi)
- **h2 — Threat / line-formation score**: quét 8 đường; +1000 thắng, +50 đe dọa thắng, +5 tạo tiềm năng (theo Don Yitong Ma)
- Heuristic không đổi kết quả cuối — chỉ tăng tốc nhờ move ordering

## Slide 10 — Kết quả benchmark

- Bảng tổng hợp:
  | Matchup | avg nodes | avg time |
  |---|---|---|
  | minimax vs alpha_beta | 350 868 | 11 887 ms |
  | minimax vs heuristic | 344 483 | 11 966 ms |
  | alpha_beta vs heuristic | 9 190 | 565 ms |
- → Alpha-Beta giảm ~98% node so với Minimax; Heuristic cắt thêm ~18% so với Alpha-Beta thuần

## Slide 11 — Tỉ lệ thắng / hòa / thua

- 100% các ván đều hòa vì cả 3 AI đều optimal
- Trên Caro (bàn cờ lớn hơn) kết quả sẽ phân hóa rõ rệt — xem paper Don Yitong Ma

## Slide 12 — Minh họa GUI

- Chụp màn hình GUI Tkinter
- Chỉ ra: dropdown chọn AI, dropdown heuristic, bảng metrics realtime (nút duyệt, ms)

## Slide 13 — Kết luận & hướng phát triển

- Alpha-Beta pruning cắt giảm số nút từ O(b^d) xuống O(b^(d/2)) trong trường hợp tốt nhất
- Move ordering theo heuristic giảm thêm ~15–20% số nút trong thực tế
- Hướng phát triển: mở rộng sang Caro (NxN), áp dụng transposition table, iterative deepening

## Slide 14 — Q & A