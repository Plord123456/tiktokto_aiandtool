"""Vẽ cây tìm kiếm Tic-Tac-Toe bằng matplotlib + networkX.

Pipeline:
  - Xây dựng cây Node từ thuật toán (minimax / alpha-beta / heuristic)
  - Dùng networkX chỉ để lưu cấu trúc + BFS; tự tính layout theo depth
  - Vẽ bằng matplotlib: bàn cờ 3×3 inline trong node (imshow), màu
    win/loss/draw, cạnh bị cắt tỉa dashed red, legend, tiêu đề.

Output: docs/figures/search_tree_{minimax,alpha_beta,heuristic}.png

Depth=3 cho Minimax và 2 cho Alpha-Beta/Heuristic (cây nhỏ, không tràn).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src import board as bd
from src.heuristics import h2_threat_score


FIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
)
os.makedirs(FIG_DIR, exist_ok=True)

# ── Màu sắc (hex) ────────────────────────────────────────────────────────────
C_WIN   = "#27ae60"   # xanh lá đậm — X thắng
C_LOSS  = "#c0392b"   # đỏ đậm — X thua
C_DRAW  = "#95a5a6"   # xám — hòa
C_ROOT  = "#2980b9"   # xanh dương — root
C_PRUNE = "#fadbd8"   # đỏ nhạt — nền node placeholder pruned

# ── Bàn cờ ───────────────────────────────────────────────────────────────────
SYM = {0: ".", 1: "X", 2: "O"}


def _board_grid(state: np.ndarray) -> np.ndarray:
    """Mã hoá bàn cờ thành ma trận 3×3 dùng cho imshow.
    1 = X, -1 = O, 0 = trống."""
    return np.where(state == 1, 1, np.where(state == 2, -1, 0)).astype(float)


def _board_color(state: np.ndarray) -> str:
    w = bd.winner(state)
    if w == 1:
        return C_WIN
    if w == 2:
        return C_LOSS
    if bd.is_terminal(state):
        return C_DRAW
    return "#ffffff"


# ── Cấu trúc Node ─────────────────────────────────────────────────────────────
class Node:
    __slots__ = ("uid", "state", "player", "score",
                 "parent", "move_from_parent", "children", "depth")
    _counter: int = 0

    def __init__(self, state: np.ndarray, player: int,
                 parent: "Node | None", move: tuple[int, int] | None,
                 depth: int):
        Node._counter += 1
        self.uid = Node._counter
        self.state = state
        self.player = player        # 1 = MAX (X), 2 = MIN (O)
        self.score: int | None = None
        self.parent = parent
        self.move_from_parent = move
        self.children: list["Node"] = []
        self.depth = depth


# ── Xây dựng cây Minimax (đầy đủ) ──────────────────────────────────────────
def _build_full(state: np.ndarray, player: int, depth: int,
                max_depth: int) -> Node:
    node = Node(state, player, None, None, depth)
    if depth == max_depth or bd.is_terminal(state):
        return node
    opp = bd.other(player)
    for m in bd.legal_moves(state):
        child = _build_full(
            bd.apply_move(state, m, player), opp, depth + 1, max_depth
        )
        child.parent = node
        child.move_from_parent = m
        node.children.append(child)
    return node


def _score_tree(node: Node) -> int:
    w = bd.winner(node.state)
    if w == 1:
        node.score = 1; return 1
    if w == 2:
        node.score = -1; return -1
    if not node.children:
        node.score = 0; return 0
    if node.player == 1:
        node.score = max(_score_tree(c) for c in node.children)
    else:
        node.score = min(_score_tree(c) for c in node.children)
    return node.score


# ── Alpha-Beta (chỉ thu thập các nút & cạnh được duyệt) ────────────────────
def _collect_ab(state: np.ndarray, player: int, depth: int, max_depth: int,
                alpha: int, beta: int
                ) -> tuple[Node, list[tuple[int, tuple[int, int]]], int]:
    """Trả (root, pruned_moves, visited).
    `pruned_moves` = list of (parent_uid, move) cho các nước bị cắt."""
    node = Node(state, player, None, None, depth)
    visited = 1
    w = bd.winner(state)
    if depth == max_depth or bd.is_terminal(state):
        if w == player:
            node.score = -1
        elif w == bd.other(player):
            node.score = 1
        else:
            node.score = 0
        return node, [], visited

    opp = bd.other(player)
    best = -2
    pruned: list[tuple[int, tuple[int, int]]] = []
    all_moves = bd.legal_moves(state)
    for i, m in enumerate(all_moves):
        child, pe, v = _collect_ab(
            bd.apply_move(state, m, player), opp, depth + 1, max_depth,
            -beta, -alpha
        )
        visited += v
        child.parent = node
        child.move_from_parent = m
        node.children.append(child)
        score = -child.score
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            for rm in all_moves[i + 1:]:
                pruned.append((node.uid, rm))
            break
    node.score = best if best != -2 else 0
    return node, pruned, visited


# ── Heuristic (Alpha-Beta + move ordering) ───────────────────────────────────
def _collect_h(state: np.ndarray, player: int, depth: int, max_depth: int,
               alpha: int, beta: int, heuristic
               ) -> tuple[Node, list[tuple[int, tuple[int, int]]], int]:
    node = Node(state, player, None, None, depth)
    visited = 1
    w = bd.winner(state)
    if depth == max_depth or bd.is_terminal(state):
        if w == player:
            node.score = -1
        elif w == bd.other(player):
            node.score = 1
        else:
            node.score = 0
        return node, [], visited
    scored = [
        (heuristic(bd.apply_move(state, m, player), player), m)
        for m in bd.legal_moves(state)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    ordered = [m for _, m in scored]
    opp = bd.other(player)
    best = -2
    pruned: list[tuple[int, tuple[int, int]]] = []
    for i, m in enumerate(ordered):
        child, pe, v = _collect_h(
            bd.apply_move(state, m, player), opp, depth + 1, max_depth,
            -beta, -alpha, heuristic
        )
        visited += v
        child.parent = node
        child.move_from_parent = m
        node.children.append(child)
        score = -child.score
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            for rm in ordered[i + 1:]:
                pruned.append((node.uid, rm))
            break
    node.score = best if best != -2 else 0
    return node, pruned, visited


# ── Layout ────────────────────────────────────────────────────────────────────
def _compute_layout(root: Node) -> dict[int, tuple[float, float]]:
    """Tính toạ độ (x, y) cho từng node: root ở trên, con xếp ngang.
    Mỗi lá chiếm khoảng cách DX; mỗi level cách nhau DY.
    Trả về dict uid → (x, y)."""
    all_nodes = [root]
    queue = [root]
    while queue:
        n = queue.pop(0)
        for c in n.children:
            all_nodes.append(c)
            queue.append(c)

    max_depth = max(n.depth for n in all_nodes)

    # Với mỗi lá, tính x slot index (counting leaves from left)
    # Dùng phương pháp: gán x cho lá đều nhau, internal node lấy trung bình con
    leaf_xs: dict[int, float] = {}
    next_leaf = [0.0]

    def assign_leaf_x(n: Node) -> float:
        if not n.children:
            x = next_leaf[0]
            leaf_xs[n.uid] = x
            next_leaf[0] += 1.0
            return x
        xs = [assign_leaf_x(c) for c in n.children]
        x = (min(xs) + max(xs)) / 2.0
        return x

    assign_leaf_x(root)

    # Gán y: depth 0 → y lớn nhất (trên cùng), depth max → y nhỏ nhất
    # Scale sao cho các lá cách nhau 1.0 đơn vị X, mỗi level cách nhau 1.0 Y
    xs: dict[int, float] = {}
    ys: dict[int, float] = {}
    ys_all: dict[int, float] = {}

    def collect_xs(n: Node):
        xs[n.uid] = leaf_xs.get(n.uid, _median_child_x(n, xs))
        ys[n.uid] = max_depth - n.depth
        for c in n.children:
            collect_xs(c)

    def _median_child_x(n: Node, xs_so_far: dict[int, float]) -> float:
        if n.children:
            child_xs = [xs_so_far.get(c.uid, leaf_xs.get(c.uid, 0)) for c in n.children]
            return (min(child_xs) + max(child_xs)) / 2.0
        return leaf_xs.get(n.uid, 0)

    # BFS để gán x trước, y sau
    for n in all_nodes:
        if n.uid in leaf_xs:
            xs[n.uid] = leaf_xs[n.uid]
        else:
            xs[n.uid] = _median_child_x(n, xs)
        ys[n.uid] = max_depth - n.depth

    # Normalize: dịch X để root ở giữa
    if root.uid in xs:
        # Tính extent và căn giữa
        all_x = list(xs.values())
        min_x, max_x = min(all_x), max(all_x)
        center_offset = (min_x + max_x) / 2.0
        for uid in xs:
            xs[uid] -= center_offset

    return {uid: (xs[uid], ys[uid]) for uid in xs}


# ── Render matplotlib ─────────────────────────────────────────────────────────
NODE_W = 0.65   # chiều rộng node
NODE_H = 0.55   # chiều cao node


def _render(
    root: Node,
    title: str,
    out_path: str,
    visited: int,
    total: int,
    pruned: list[tuple[int, tuple[int, int]]],
    max_depth: int = 3,
) -> None:
    """Vẽ cây ra PNG."""

    # Scale node size theo depth: depth càng nhỏ → node càng to
    # depth 2 → 1.6x, depth 3 → 1.0x, depth 4+ → 0.8x
    if max_depth <= 2:
        scale = 1.6
    elif max_depth == 3:
        scale = 1.0
    else:
        scale = 0.7
    node_w = NODE_W * scale
    node_h = NODE_H * scale
    score_font = 9 * scale
    role_font = 8 * scale

    layout = _compute_layout(root)

    # Collect tất cả nodes
    all_nodes = [root]
    q = [root]
    while q:
        n = q.pop(0)
        for c in n.children:
            all_nodes.append(c)
            q.append(c)

    max_depth = max(n.depth for n in all_nodes)

    # Kích thước figure
    n_leaves = sum(1 for n in all_nodes if not n.children) + len(pruned)
    width = max(8.0, min(20.0, n_leaves * 0.85 + 4.0))
    height = max(5.0, min(16.0, (max_depth + 1) * 1.4 + 1.5))

    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    ax.set_xlim(-width / 2 + 1, width / 2 - 1)
    ax.set_ylim(-1.0, max_depth + 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#fefefe")

    # ── Vẽ cạnh bình thường ───────────────────────────────────────────────────
    for n in all_nodes:
        if n.parent and n.move_from_parent:
            x1, y1 = layout[n.parent.uid]
            x2, y2 = layout[n.uid]
            ax.annotate(
                "",
                xy=(x2, y2 + node_h / 2),
                xytext=(x1, y1 - node_h / 2),
                arrowprops=dict(
                    arrowstyle="->", color="#555555",
                    lw=1.0, shrinkA=0, shrinkB=0,
                ),
            )
            # Edge label: tọa độ (r+1, c+1)
            r, c = n.move_from_parent
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.06, my, f"{r+1},{c+1}",
                    fontsize=7, color="#555555",
                    ha="left", va="center",
                    bbox=dict(boxstyle="round,pad=0.1",
                              facecolor="white", edgecolor="none",
                              alpha=0.8))

    # ── Vẽ nodes với bàn cờ dùng text X/O to, rõ ràng ─────────────────────────
    for n in all_nodes:
        x, y = layout[n.uid]
        fc = C_ROOT if n.parent is None else _board_color(n.state)

        # Background box
        rect = patches.FancyBboxPatch(
            (x - node_w / 2, y - node_h / 2), node_w, node_h,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=fc,
            edgecolor="#c0392b" if n.score == -1 else "#27ae60" if n.score == 1 else "#95a5a6" if bd.is_terminal(n.state) else "#2980b9" if n.parent is None else "#555555",
            linewidth=1.2,
            zorder=1,
        )
        ax.add_patch(rect)

        # Bàn cờ 3×3 dùng matplotlib text thuần — X/O to, rõ ràng
        # 3 dòng, mỗi dòng 3 ký tự, khoảng cách đều
        symbol_font = 14 * scale  # cỡ chữ X/O
        cell_x_step = node_w * 0.18  # khoảng cách X ngang
        cell_y_step = node_h * 0.18  # khoảng cách X dọc
        y_offset = node_h * 0.16     # bàn cờ dịch lên trên 1 chút

        for r in range(3):
            for c in range(3):
                v = int(n.state[r, c])
                # Vị trí ô (c, r) trong node
                cell_x = x + (c - 1) * cell_x_step
                cell_y = y + (1 - r) * cell_y_step + y_offset
                if v == 1:
                    ch, ch_col = "X", "#1a5276"  # xanh đậm
                elif v == 2:
                    ch, ch_col = "O", "#922b21"  # đỏ đậm
                else:
                    continue  # bỏ qua ô trống
                ax.text(
                    cell_x, cell_y, ch,
                    ha="center", va="center",
                    fontsize=symbol_font, fontweight="bold",
                    color=ch_col, zorder=3,
                    family="DejaVu Sans",
                )

        # Score + role text (phía dưới node)
        sc = f"{n.score:+d}" if n.score is not None else "?"
        role = "M" if n.player == 1 else "m"
        role_color = "#8e44ad" if n.player == 1 else "#e67e22"
        # Score text (màu)
        score_color = "#27ae60" if n.score == 1 else "#c0392b" if n.score == -1 else "#7f8c8d"
        ax.text(x, y - node_h * 0.40, f"[{sc}]",
                ha="center", va="center", fontsize=score_font,
                fontweight="bold", color=score_color, zorder=3)
        ax.text(x + node_w * 0.32, y - node_h * 0.40, role,
                ha="center", va="center", fontsize=role_font,
                color=role_color, zorder=3)

    # ── Vẽ pruned nodes ───────────────────────────────────────────────────────
    # Tính x slot cho pruned: dịch ra ngoài (bên phải) cạnh parent
    node_map = {n.uid: n for n in all_nodes}
    extra_x_offset = max(layout[n.uid][0] for n in all_nodes) + 1.2
    for i, (puid, (r, c)) in enumerate(pruned):
        parent = node_map.get(puid)
        if parent is None:
            continue
        # Tính x slot dựa trên vị trí của parent + offset cho từng pruned
        px, py = layout[parent.uid]
        # Xếp các pruned ngang hàng với parent, offset dần
        x_slot = px + 0.7 + i * 0.4
        y_slot = py - 0.4

        # Background box
        rect = patches.FancyBboxPatch(
            (x_slot - 0.18, y_slot - 0.13), 0.36, 0.26,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=C_PRUNE, edgecolor="#c0392b",
            linewidth=1.0, linestyle="--", zorder=1,
        )
        ax.add_patch(rect)
        ax.text(x_slot, y_slot, f"({r+1},{c+1})",
                ha="center", va="center", fontsize=7,
                color="#c0392b", fontweight="bold", zorder=3)

        # Cạnh dashed red từ parent đến pruned node
        ax.annotate(
            "",
            xy=(x_slot, y_slot + 0.13),
            xytext=(px + node_w / 2, py),
            arrowprops=dict(
                arrowstyle="-",
                color="#c0392b", lw=1.2,
                linestyle="--",
                shrinkA=0, shrinkB=0,
            ),
            zorder=0,
        )

    # ── Title ────────────────────────────────────────────────────────────────
    full_title = (
        f"{title}\n"
        f"Nodes visited: {visited} / {total} total tree"
    )
    ax.set_title(full_title, fontsize=13, fontweight="bold",
                 family="sans-serif", pad=12)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_items = [
        ("X wins", C_WIN, "#27ae60"),
        ("X loses", C_LOSS, "#c0392b"),
        ("Draw", C_DRAW, "#95a5a6"),
        ("Root", C_ROOT, "#2980b9"),
    ]
    legend_handles = []
    for txt, fc, ec in legend_items:
        legend_handles.append(patches.Patch(
            facecolor=fc, edgecolor=ec, linewidth=1.2, label=txt
        ))
    # Dashed red line cho pruned edge
    from matplotlib.lines import Line2D
    legend_handles.append(
        Line2D([0], [0], color="#c0392b", linestyle="--",
               lw=1.5, label="Pruned edge")
    )

    ax.legend(
        handles=legend_handles,
        loc="lower center", bbox_to_anchor=(0.5, -0.05),
        fontsize=8, ncol=5, frameon=True,
        facecolor="#f5f5f5", edgecolor="#cccccc",
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#fefefe")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=3,
                        help="Search depth (default: 3)")
    parser.add_argument("--partial", action="store_true",
                        help="Vẽ cây nhỏ 10-30 nodes cho báo cáo đề 6 "
                             "(X đi giữa → O đi 8 góc/cạnh → dừng).")
    args = parser.parse_args()

    root_state = bd.empty_board()

    if args.partial:
        # ── Partial tree: đúng 10-30 nodes, minh họa đẹp cho báo cáo đề 6.
        #   Cấu trúc:
        #     Depth 0: root (bàn cờ trống)
        #     Depth 1: 3 nước X (góc-trái, giữa, góc-phải)
        #     Depth 2: Mỗi nước X có 8 nước O → 24 nodes
        #     Depth 3: Chỉ 1 nhánh (X đi giữa) mở rộng 5 nước tiếp
        #   Tổng: 1 + 3 + 24 + 5 = 33 → bỏ 1 nhánh O → 1+3+16+5 = 25 nodes ✓
        print("Building partial tree (10-30 nodes for report) …")
        Node._counter = 0

        root = Node(root_state, bd.PLAYER_X, None, None, 0)
        root.score = 0

        # Depth 1: X đi 3 nước cụ thể (góc-trái, giữa, góc-phải)
        x_moves_d1: list[tuple[int, int]] = [(0, 0), (1, 1), (0, 2)]
        x_children: list[Node] = []
        for m1 in x_moves_d1:
            board1 = bd.apply_move(root.state, m1, bd.PLAYER_X)
            child1 = Node(board1, bd.PLAYER_X, root, m1, 1)
            child1.score = 0
            root.children.append(child1)
            x_children.append(child1)

        # Depth 2: O đi 8 nước (trừ nước X đã đi) — chỉ cho 2 nhánh X đầu
        # Mỗi nhánh: 8 nodes → 16 nodes
        opp = bd.other(bd.PLAYER_X)
        o_children: list[Node] = []
        for child1 in x_children[:2]:
            board1 = child1.state
            for m2 in bd.legal_moves(board1):
                board2 = bd.apply_move(board1, m2, opp)
                child2 = Node(board2, opp, child1, m2, 2)
                child2.score = 0
                child1.children.append(child2)
                o_children.append(child2)

        # Depth 3: Chỉ mở rộng 1 nhánh (X đi giữa) thêm 5 nước
        for child2 in o_children[:5]:
            board2 = child2.state
            for m3 in bd.legal_moves(board2):
                board3 = bd.apply_move(board2, m3, bd.PLAYER_X)
                child3 = Node(board3, bd.PLAYER_X, child2, m3, 3)
                child3.score = 0
                child2.children.append(child3)
                # Chỉ thêm 1 nước tiếp rồi dừng
                break

        total = Node._counter
        print(f"  {total} nodes (within 10-30 range)")

        _render(
            root, f"Partial Minimax Tree — {total} nodes (Depth 0–3)",
            os.path.join(FIG_DIR, "search_tree_partial.png"),
            visited=total, total=total, pruned=[],
            max_depth=3,
        )
        print(f"  Partial tree saved to search_tree_partial.png")
        print(f"\nDone.")
        return

    # ── Full trees ────────────────────────────────────────────────────────────
    max_depth = args.depth
    print(f"Using depth={max_depth}")

    root_state = bd.empty_board()

    # ── Minimax ───────────────────────────────────────────────────────────────
    print("Building Minimax tree …")
    Node._counter = 0
    root_mm = _build_full(root_state, bd.PLAYER_X, 0, max_depth)
    _score_tree(root_mm)
    nodes_mm = Node._counter
    print(f"  {nodes_mm} total nodes")

    _render(
        root_mm, f"Minimax — Depth {max_depth}",
        os.path.join(FIG_DIR, "search_tree_minimax.png"),
        visited=nodes_mm, total=nodes_mm, pruned=[],
        max_depth=max_depth,
    )
    print(f"  Minimax saved")

    # ── Alpha-Beta ────────────────────────────────────────────────────────────
    print("Building Alpha-Beta tree …")
    Node._counter = 0
    root_ab, pruned_ab, visited_ab = _collect_ab(
        root_state, bd.PLAYER_X, 0, max_depth, -2, 2
    )
    _score_tree(root_ab)
    print(f"  {visited_ab} visited, {len(pruned_ab)} edges pruned")

    _render(
        root_ab, f"Alpha-Beta Pruning — Depth {max_depth}",
        os.path.join(FIG_DIR, "search_tree_alpha_beta.png"),
        visited=visited_ab, total=nodes_mm, pruned=pruned_ab,
        max_depth=max_depth,
    )
    print(f"  Alpha-Beta saved")

    # ── Heuristic ─────────────────────────────────────────────────────────────
    print("Building Heuristic tree …")
    Node._counter = 0
    root_h, pruned_h, visited_h = _collect_h(
        root_state, bd.PLAYER_X, 0, max_depth, -2, 2, h2_threat_score
    )
    _score_tree(root_h)
    print(f"  {visited_h} visited, {len(pruned_h)} edges pruned")

    _render(
        root_h, f"Alpha-Beta + Heuristic Ordering — Depth {max_depth}",
        os.path.join(FIG_DIR, "search_tree_heuristic.png"),
        visited=visited_h, total=nodes_mm, pruned=pruned_h,
        max_depth=max_depth,
    )
    print(f"  Heuristic saved")

    print(f"\nDone. Output: {FIG_DIR}")
    for f in sorted(os.listdir(FIG_DIR)):
        if f.startswith("search_tree_"):
            sz = os.path.getsize(os.path.join(FIG_DIR, f))
            print(f"  {f}  ({sz // 1024} KB)")


if __name__ == "__main__":
    main()