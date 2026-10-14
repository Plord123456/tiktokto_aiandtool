"""Build `docs/report.docx` from the 6 required sections.

Sections:
  1. PEAS specification
  2. Search problem definition
  3. Transition Function + table (>=10 rows)
  4. Search-tree figures (3 trees)
  5. Algorithm implementations + heuristics + comparison table
  6. GUI screenshot section
  + Bonus: experimental results (benchmark)

Run:
    .venv/bin/python -m experiments.build_report
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import numpy as np

from src import board as bd
from src.heuristics import h1_position_score, h2_threat_score

DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
FIG_DIR = os.path.join(DOCS_DIR, "figures")
REPORT_PATH = os.path.join(DOCS_DIR, "report.docx")


def _heading(doc: Document, text: str, level: int) -> None:
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT


def _para(doc: Document, text: str, bold: bool = False, italic: bool = False) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic


def _table(doc: Document, header: list[str], rows: list[list[str]]) -> None:
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(header):
        t.cell(0, i).text = h
    for r_idx, row in enumerate(rows, start=1):
        for c_idx, val in enumerate(row):
            t.cell(r_idx, c_idx).text = str(val)


def _image(doc: Document, name: str, caption: str) -> None:
    path = os.path.join(FIG_DIR, name)
    if not os.path.exists(path):
        _para(doc, f"[missing image: {name}]")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(6.0))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.italic = True


# ---------------- Section 1: PEAS ----------------
def section_peas(doc: Document) -> None:
    _heading(doc, "1. Đặc tả bài toán theo mô hình PEAS", 1)

    _heading(doc, "Performance Measure", 2)
    _para(
        doc,
        "Đại lý (agent) được đo hiệu năng bằng kết quả cuối cùng của ván cờ: "
        "thắng (+1), hòa (0), thua (-1). Phụ thêm: số node đã duyệt và thời gian "
        "phản hồi mỗi nước đi — thước đo hiệu quả của thuật toán tìm kiếm.",
    )

    _heading(doc, "Environment", 2)
    _para(doc, "Bàn cờ Tic-Tac-Toe 3×3. Hai người chơi luân phiên đặt X / O vào các ô trống. Người thắng là người đầu tiên có đủ 3 quân cùng loại trên một hàng, một cột, hoặc một đường chéo.")

    _heading(doc, "Actuators", 2)
    _para(doc, "Đặt quân cờ (X hoặc O) vào một trong các ô trống còn lại trên bàn cờ — mỗi lượt chỉ được đặt một quân.")

    _heading(doc, "Sensors", 2)
    _para(doc, "Quan sát toàn bộ bàn cờ 3×3, bao gồm vị trí và loại quân của mỗi ô (trống, X, hoặc O).")

    _heading(doc, "Phân loại môi trường", 2)
    _table(
        doc,
        ["Tiêu chí", "Phân loại", "Lý do"],
        [
            ["Quan sát", "Quan sát được đầy đủ", "Agent nhìn thấy toàn bộ 9 ô trên bàn cờ."],
            ["Xác định", "Xác định (deterministic)", "Mỗi hành động dẫn đến đúng một trạng thái kế tiếp."],
            ["Tĩnh / Động", "Tĩnh (static)", "Trong lượt của agent, bàn cờ không thay đổi."],
            ["Rời rạc / Liên tục", "Rời rạc", "Tập trạng thái hữu hạn (3^9 = 19683 trạng thái hợp lệ)."],
            ["Tác nhân", "Đa tác nhân", "Hai agent cạnh tranh trực tiếp."],
        ],
    )


# ---------------- Section 2: Search problem ----------------
def section_problem_def(doc: Document) -> None:
    _heading(doc, "2. Định nghĩa bài toán tìm kiếm", 1)
    _table(
        doc,
        ["Thành phần", "Mô tả"],
        [
            ["Trạng thái (State)", "Cấu hình bàn cờ 3×3; mỗi ô thuộc một trong ba giá trị: trống (0), X (1), O (2)."],
            ["Trạng thái ban đầu", "Bàn cờ rỗng (toàn bộ 9 ô là 0)."],
            ["Tập hành động Actions(s)", "Tập các ô trống còn lại trên bàn cờ tại trạng thái s."],
            ["Transition Function", "Đặt quân của người chơi hiện tại vào ô trống m — sinh ra bàn cờ mới với quân tại (r, c)."],
            ["Goal Test", "Bàn cờ có 3 quân cùng loại trên một hàng, cột hoặc đường chéo; hoặc không còn ô trống (hòa)."],
            ["Path Cost", "Không định nghĩa — bài toán zero-sum, đại lý chỉ quan tâm trạng thái cuối (thắng/thua/hòa)."],
        ],
    )


# ---------------- Section 3: Transition function ----------------
def section_transition(doc: Document) -> None:
    _heading(doc, "3. Transition Function", 1)
    _para(
        doc,
        "Hàm chuyển trạng thái RESULT(s, a) nhận một trạng thái bàn cờ s và "
        "một hành động a = (r, c) — tức là đặt quân của người chơi hiện tại lên ô "
        "(r, c). Trả về bàn cờ mới giống hệt s, ngoại trừ ô (r, c) được gán giá "
        "trị của người chơi. Nếu ô (r, c) đã có quân, hàm raises ValueError. "
        "Dưới đây là một số ví dụ minh họa (bàn cờ hiển thị từ trên xuống dưới, "
        "trong đó X = 1, O = 2, . = 0).",
    )
    rows = []
    # Hand-picked representative transitions
    samples = [
        (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int8), (1, 1), 1, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int8)),
        (np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int8), (0, 0), 2, np.array([[2, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int8)),
        (np.array([[2, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int8), (2, 2), 1, np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int8)),
        (np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int8), (0, 2), 2, np.array([[2, 0, 2], [0, 1, 0], [0, 0, 1]], dtype=np.int8)),
        (np.array([[2, 0, 2], [0, 1, 0], [0, 0, 1]], dtype=np.int8), (2, 0), 1, np.array([[2, 0, 2], [0, 1, 0], [1, 0, 1]], dtype=np.int8)),
        (np.array([[2, 0, 2], [0, 1, 0], [1, 0, 1]], dtype=np.int8), (1, 0), 2, np.array([[2, 0, 2], [2, 1, 0], [1, 0, 1]], dtype=np.int8)),
        (np.array([[2, 0, 2], [2, 1, 0], [1, 0, 1]], dtype=np.int8), (1, 2), 1, np.array([[2, 0, 2], [2, 1, 1], [1, 0, 1]], dtype=np.int8)),
        (np.array([[1, 0, 0], [0, 0, 0], [0, 0, 2]], dtype=np.int8), (0, 1), 2, np.array([[1, 2, 0], [0, 0, 0], [0, 0, 2]], dtype=np.int8)),
        (np.array([[1, 2, 0], [0, 0, 0], [0, 0, 2]], dtype=np.int8), (0, 2), 1, np.array([[1, 2, 1], [0, 0, 0], [0, 0, 2]], dtype=np.int8)),
        (np.array([[1, 2, 1], [0, 0, 0], [0, 0, 2]], dtype=np.int8), (1, 0), 2, np.array([[1, 2, 1], [2, 0, 0], [0, 0, 2]], dtype=np.int8)),
        (np.array([[1, 2, 1], [2, 0, 0], [0, 0, 2]], dtype=np.int8), (2, 1), 1, np.array([[1, 2, 1], [2, 0, 0], [0, 1, 2]], dtype=np.int8)),
        (np.array([[1, 2, 1], [2, 0, 0], [0, 1, 2]], dtype=np.int8), (1, 1), 2, np.array([[1, 2, 1], [2, 2, 0], [0, 1, 2]], dtype=np.int8)),
    ]
    for s, m, p, s2 in samples:
        rows.append([_board_str(s), f"({m[0]+1}, {m[1]+1})", str(p), _board_str(s2)])
    _table(doc, ["State s", "Action", "Player", "State RESULT(s, a)"], rows)


def _board_str(state) -> str:
    sym = {0: ".", 1: "X", 2: "O"}
    return "\n".join(" ".join(sym[int(v)] for v in row) for row in state)


# ---------------- Section 4: Search trees ----------------
def section_search_trees(doc: Document) -> None:
    _heading(doc, "4. Cây tìm kiếm", 1)
    _para(
        doc,
        "Cây tìm kiếm được mở rộng đến độ sâu 2 từ trạng thái rỗng cho cả ba thuật toán. "
        "Các ô hiển thị là một tập con đại diện cho các trạng thái thật sự được duyệt bởi từng thuật toán. "
        "Minimax duyệt tất cả các nhánh; Alpha-Beta cắt bớt nhờ cận α / β; thuật toán có Heuristic "
        "(danh sách hóa nước đi theo điểm heuristic) cắt thêm một phần.",
    )
    _image(doc, "search_tree_minimax.png", "Hình 1. Cây tìm kiếm Minimax — duyệt tất cả các nút.")
    _image(doc, "search_tree_alpha_beta.png", "Hình 2. Cây tìm kiếm Alpha-Beta — cắt tỉa nhờ α-β.")
    _image(doc, "search_tree_heuristic.png", "Hình 3. Cây tìm kiếm Alpha-Beta + Heuristic — cắt thêm nhờ move ordering.")


# ---------------- Section 5: Algorithms ----------------
def section_algorithms(doc: Document) -> None:
    _heading(doc, "5. Hiện thực giải thuật tìm kiếm", 1)
    _para(
        doc,
        "Đề tài 6 yêu cầu mở rộng đề tài 5 (Minimax) bằng Alpha-Beta pruning và "
        "bổ sung thêm một biến thể có heuristic. Tổng cộng có ba thuật toán:",
    )

    _heading(doc, "5.1 Minimax (đề tài 5)", 2)
    _para(
        doc,
        "Minimax duyệt toàn bộ cây trò chơi đến trạng thái kết thúc, lớp MAX chọn "
        "giá trị lớn nhất, lớp MIN chọn giá trị nhỏ nhất. Vì Tic-Tac-Toe có không "
        "gian trạng thái hữu hạn nhỏ (≤9 lớp, ≤9! = 362880 thứ tự nước đi), "
        "Minimax vẫn chạy được nhưng số nút rất lớn — khoảng 550 000 nút cho một "
        "ván từ bàn cờ rỗng.",
    )

    _heading(doc, "5.2 Alpha-Beta Pruning", 2)
    _para(
        doc,
        "Alpha-Beta giữ hai cận α (giá trị tốt nhất mà MAX có thể đảm bảo) và β "
        "(giá trị tốt nhất mà MIN có thể đảm bảo). Khi α ≥ β, nhánh hiện tại không "
        "thể cải thiện kết quả cuối cùng và bị cắt bỏ. Với cây Tic-Tac-Toe, Alpha-Beta "
        "giảm số nút xuống còn khoảng 9 200 nút/ván — tiết kiệm hơn 98 % so với Minimax. "
        "Kết quả trả về giống hệt Minimax (vẫn tối ưu).",
    )

    _heading(doc, "5.3 Alpha-Beta + Heuristic (Move Ordering)", 2)
    _para(
        doc,
        "Cải tiến thêm: trước khi duyệt, sắp xếp các nước đi theo điểm một hàm "
        "heuristic giảm dần (khi MAX) hoặc tăng dần (khi MIN). Khi nước đi tốt nhất "
        "được thử trước, α và β co lại nhanh hơn, từ đó cắt được nhiều nhánh hơn. "
        "Ý tưởng lấy từ paper của Don Yitong Ma (2023) — 'Optimization of Alpha-Beta "
        "Pruning based on Heuristic Algorithm'.",
    )

    _heading(doc, "5.4 Hai hàm heuristic", 2)
    _para(doc, "h1 — Position score (lấy từ AsadJatoi 2023):", bold=True)
    _para(
        doc,
        "Điểm vị trí cho quân của player: center = 3, corner = 2, edge = 1; "
        "trừ điểm tương ứng nếu ô thuộc về đối thủ.",
    )
    _para(doc, "h2 — Threat / Line-formation score (lấy cảm hứng từ Don Yitong Ma 2023):", bold=True)
    _para(
        doc,
        "Quét tất cả 8 đường (3 hàng, 3 cột, 2 chéo). Mỗi đường đóng góp: "
        "+1000 nếu player có 3 quân; +50 nếu có 2 quân và 1 ô trống (đe dọa thắng); "
        "+5 nếu có 1 quân và 2 ô trống; điểm âm tương ứng cho đối thủ. "
        "Hàm heuristic giúp move ordering ưu tiên các nước tạo/thoát đe dọa.",
    )

    _heading(doc, "5.5 So sánh ba thuật toán", 2)
    _table(
        doc,
        ["Tiêu chí", "Minimax", "Alpha-Beta", "Heuristic AI"],
        [
            ["Tính tối ưu", "Tối ưu", "Tối ưu", "Tối ưu (heuristic chỉ dùng để ordering)"],
            ["Tính mở (completeness)", "Có", "Có", "Có"],
            ["Thời gian (ván trung bình)", "~21 giây", "~0.5 s", "~0.4 s"],
            ["Số nút duyệt (ván)", "~550 000", "~9 200", "~7 500"],
            ["Bộ nhớ (đệ quy)", "O(b^d)", "O(b^d)", "O(b^d) + O(b log b) cho sorting"],
        ],
    )
    _para(
        doc,
        "Minimax và Alpha-Beta đều duyệt đến trạng thái kết thúc, vì vậy trả về "
        "cùng nước đi tối ưu. Heuristic AI không thay đổi kết quả mà chỉ tăng tốc "
        "bằng cách đánh giá các nước đi hứa hẹn trước.",
    )


# ---------------- Section 6: GUI ----------------
def section_gui(doc: Document) -> None:
    _heading(doc, "6. Minh họa bằng giao diện đồ họa", 1)
    _para(
        doc,
        "GUI được xây bằng Tkinter (chuẩn của Python). Giao diện gồm bàn cờ 3×3 "
        "(9 nút bấm) và bảng điều khiển bên phải cho phép chọn:",
    )
    _table(
        doc,
        ["Tham số", "Mô tả"],
        [
            ["AI for X", "Chọn thuật toán cho quân X (minimax / alpha_beta / heuristic_ai)."],
            ["AI for O", "Chọn thuật toán cho quân O."],
            ["Heuristic", "Chọn hàm heuristic (h1_position hoặc h2_threat) khi AI là heuristic_ai."],
            ["Mode", "Human vs AI, AI vs AI, hoặc Human vs Human."],
            ["Reset", "Bắt đầu ván mới."],
            ["Bảng Metrics", "Hiển thị nước vừa đi, số nút duyệt, thời gian (ms)."],
        ],
    )
    _para(
        doc,
        "Sau mỗi nước của AI, bảng metrics cập nhật theo thời gian thực, cho phép "
        "người dùng so sánh trực tiếp số nút và thời gian của ba thuật toán trên "
        "cùng một trạng thái bàn cờ.",
    )
    _para(doc, "Khởi động GUI: .venv/bin/python -m src.gui.tk_ui", italic=True)


# ---------------- Bonus: Experimental results ----------------
def section_results(doc: Document) -> None:
    _heading(doc, "Phụ lục. Kết quả thực nghiệm", 1)
    csv_path = os.path.join(FIG_DIR, "benchmark_summary.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        _para(doc, "Bảng tổng hợp benchmark vòng tròn (3 AI × 6 ván/matchup):", bold=True)
        t = doc.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = "Light Grid Accent 1"
        for r_idx, row in enumerate(rows):
            for c_idx, val in enumerate(row):
                t.cell(r_idx, c_idx).text = str(val)

    _image(doc, "total_nodes.png", "Hình 4. Số nút duyệt trung bình mỗi ván của ba AI.")
    _image(doc, "win_rates.png", "Hình 5. Thắng / Hòa / Thua khi ba AI đấu vòng tròn.")
    _image(doc, "minimax_time_per_turn.png", "Hình 6. Thời gian mỗi lượt của Minimax.")
    _image(doc, "alpha_beta_time_per_turn.png", "Hình 7. Thời gian mỗi lượt của Alpha-Beta.")
    _image(doc, "heuristic_h2_time_per_turn.png", "Hình 8. Thời gian mỗi lượt của Heuristic AI.")


def main() -> None:
    doc = Document()
    # Default font
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    title = doc.add_heading("Đồ án: Tic-Tac-Toe với Alpha-Beta Pruning và Heuristic AI", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    _para(
        doc,
        "Đề tài 6 — mở rộng đề tài 5 (Minimax) bằng Alpha-Beta pruning và biến thể "
        "có heuristic move ordering. Sinh viên xây dựng đầy đủ 6 nội dung theo "
        "yêu cầu của đề bài, có bổ sung phần kết quả thực nghiệm.",
        italic=True,
    )

    section_peas(doc)
    section_problem_def(doc)
    section_transition(doc)
    section_search_trees(doc)
    section_algorithms(doc)
    section_gui(doc)
    section_results(doc)

    os.makedirs(DOCS_DIR, exist_ok=True)
    doc.save(REPORT_PATH)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()