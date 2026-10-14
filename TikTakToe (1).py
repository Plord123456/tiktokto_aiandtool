import math
import random
import numpy as np
import time
MAX_DEPTH = 1

# thông tin bàn cờ
N = 100 #1000
NUM_WIN = 5
# Khởi tạo bàn cờ
caroState = np.zeros( (N,N))
caroState

# Hàm hiển thị trạng thái bàn cờ với bán kính nhất định
def showPartialCaroState(state, row, col, radius=None):
    N = state.shape[0]
    if radius is None:
        if N <= 50:
            radius = 5
        elif N <= 200:
            radius = 10
        elif N <= 500:
            radius = 15
        else:
            radius = 20  # đảm bảo luôn hiển thị ít nhất 10 ô (5*2)

    start_row = max(0, row - radius)
    end_row = min(N, row + radius + 1)
    start_col = max(0, col - radius)
    end_col = min(N, col + radius + 1)

    print(f"Vị trí nước đi: ({row + 1}, {col + 1}) | Kích thước hiển thị: {end_row - start_row}x{end_col - start_col}")

    # In số cột
    print("     ", end="")
    for c in range(start_col, end_col):
        print(f"{c+1:>3}", end="")
    print()

    # In từng dòng
    for r in range(start_row, end_row):
        print(f"{r+1:>4} ", end="")
        for c in range(start_col, end_col):
            val = state[r][c]
            if val == 1:
                x = " X "
            elif val == 2:
                x = " O "
            else:
                x = " - "
            print(x, end="")
        print()

#Kiểm tra giá trị dòng và cột có đúng
def checkRowColValid(row, col):
  return (row >= 0) and (row < N) and (col >= 0) and (col < N)

def checkAvailable(state,row,col):
  if checkRowColValid(row,col):
    return state[row][col] == 0
  return False

def setStatePosition(state,xo,row,col):
  if checkAvailable(state,row,col):
    state[row][col] = xo

#Kiểm tra người/máy thắng
def isWinner(state, player, N, NUM_WIN):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # dọc, ngang, chéo chính, chéo phụ

    for r in range(N):
        for c in range(N):
            if state[r][c] != player:
                continue
            for dx, dy in directions:
                count = 0
                for i in range(NUM_WIN):
                    nr, nc = r + i * dx, c + i * dy
                    if 0 <= nr < N and 0 <= nc < N and state[nr][nc] == player:
                        count += 1
                    else:
                        break
                if count == NUM_WIN:
                    return True
    return False

def isWinnerFast(state, player, row, col):
    for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        count = 1
        for dir in [-1, 1]:
            for i in range(1, NUM_WIN):
                nr, nc = row + dir*i*dx, col + dir*i*dy
                if 0 <= nr < N and 0 <= nc < N and state[nr][nc] == player:
                    count += 1
                else:
                    break
        if count >= NUM_WIN:
            return True
    return False

#Đánh giá một dãy 1 chiều (hàng, cột, đường chéo) để tính tổng điểm theo mức độ nguy hiểm hoặc cơ hội
def check_line_sequence(line, player):
    score = 0
    opponent = 2 if player == 1 else 1

    for i in range(len(line) - NUM_WIN + 1):
        segment = line[i:i + NUM_WIN]

        # Đếm tay thay vì dùng np.count_nonzero
        count_player = sum(1 for x in segment if x == player)
        count_opponent = sum(1 for x in segment if x == opponent)

        if count_opponent > 0 or count_player == 0:
            continue

        left_open = (i == 0 or line[i - 1] == 0)
        right_open = (i + NUM_WIN == len(line) or line[i + NUM_WIN] == 0)
        open_ends = left_open + right_open

        if count_player == 5:
            score += 100000
        elif count_player == 4:
            if open_ends == 2:
                score += 99999
            elif open_ends == 1:
                score += 35000
        elif count_player == 3:
            if open_ends == 2:
                score += 5000
            elif open_ends == 1:
                score += 800
        elif count_player == 2:
            if open_ends == 2:
                score += 150
            elif open_ends == 1:
                score += 50
        elif count_player == 1:
            if open_ends == 2:
                score += 10
            elif open_ends == 1:
                score += 3
    return score

#Tính tổng điểm cho toàn bộ bàn cờ dựa trên khả năng thắng
def calculate_score_general(board, player):
    total_score = 0
    N = board.shape[0]

    # Chỉ lấy những hàng/cột/chéo có quân của player
    positions = np.argwhere(board == player)
    active_rows = set()
    active_cols = set()
    main_diagonals = set()
    anti_diagonals = set()

    for r, c in positions:
        active_rows.add(r)
        active_cols.add(c)
        main_diagonals.add(r - c)
        anti_diagonals.add(r + c)

    # Duyệt hàng
    for r in active_rows:
        total_score += check_line_sequence(board[r, :], player)

    # Duyệt cột
    for c in active_cols:
        total_score += check_line_sequence(board[:, c], player)

    # Duyệt chéo chính
    for d in main_diagonals:
        main_diag = np.diagonal(board, offset=d)
        total_score += check_line_sequence(main_diag, player)

    # Duyệt chéo phụ
    flipped = np.fliplr(board)
    for d in anti_diagonals:
        anti_diag = np.diagonal(flipped, offset=d - (N - 1))
        total_score += check_line_sequence(anti_diag, player)

    return total_score

# Hàm lượng giá trạng thái hiện tại
def evaluate(state):
    if isWinner(state, 1, N, NUM_WIN):
        return float('inf')
    if isWinner(state, 2, N, NUM_WIN):
        return -float('inf')

    attack_score = calculate_score_general(state, 1)
    defense_score = calculate_score_general(state, 2)

    # Ưu tiên thắng ngay
    if attack_score >= 99999:
        return float('inf')
    if defense_score >= 99999:
        return -float('inf')

    # Phản ứng với mối đe dọa cao trước
    if defense_score >= 20000:
        return -1e9
    elif defense_score >= 10000:
        return -7e8
    elif defense_score >= 5000:
        return -4e6
    elif defense_score >= 3000:
        if attack_score >= 5000:
            return attack_score * 12 - defense_score * 4
        else:
            return -2e6

    # Ưu tiên thế tấn công cực mạnh
    if attack_score >= 20000:
        return float('inf')
    elif attack_score >= 10000 and defense_score < 3000:
        return attack_score * 15
    elif attack_score >= 5000:
        return attack_score * 12 - defense_score * 5
    elif attack_score >= 3000:
        return attack_score * 10 - defense_score * 6

    # Trạng thái cân bằng hoặc sớm
    filled = np.count_nonzero(state)
    if filled < 30:
        return attack_score * 5 + defense_score * 12  # ưu tiên phòng thủ sớm
    elif filled > 300:
        return attack_score * 13 + defense_score * 5  # ưu tiên tấn công
    else:
        return attack_score * 10 + defense_score * 8  # cân bằng

# Tính nước đi tiềm năng: chỉ các ô trống gần ô đã đánh
def get_candidate_moves(state, radius=None):
    N = state.shape[0]
    filled = np.count_nonzero(state)

    occupied_positions = np.argwhere(state != 0)
    if len(occupied_positions) == 0:
        return [(N // 2, N // 2)]

    min_r = np.min(occupied_positions[:, 0])
    max_r = np.max(occupied_positions[:, 0])
    min_c = np.min(occupied_positions[:, 1])
    max_c = np.max(occupied_positions[:, 1])
    margin = 8
    min_r = max(0, min_r - margin)
    max_r = min(N - 1, max_r + margin)
    min_c = max(0, min_c - margin)
    max_c = min(N - 1, max_c + margin)

    if radius is None:
        if N <= 20:
            radius = 3
        elif N <= 100:
            radius = 2 if filled > 20 else 3
        else:
            radius = 1 if filled > 50 else 2

    moves = set()
    my_cells = []
    enemy_cells = []

    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if state[r][c] == 1:
                my_cells.append((r, c))
            elif state[r][c] == 2:
                enemy_cells.append((r, c))

            if state[r][c] != 0:
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < N and 0 <= nc < N and state[nr][nc] == 0:
                            moves.add((nr, nc))

    if not moves:
        return [(N // 2, N // 2)]

    #Ưu tiên gần đối thủ đầu game
    def move_priority(move):
        r, c = move
        dist_to_my = min(abs(r - r0) + abs(c - c0) for r0, c0 in my_cells) if my_cells else N
        dist_to_enemy = min(abs(r - r0) + abs(c - c0) for r0, c0 in enemy_cells) if enemy_cells else N
        if filled < 10:
            return dist_to_enemy
        return dist_to_my

    limit = 20 if N <= 20 else 10 if filled > 40 else 20
    return sorted(moves, key=move_priority)[:limit]

# Alpha-beta pruning tối ưu nước đi
def alpha_beta(state, depth, alpha, beta, maximizing_player, start_time):
    if time.time() - start_time > MAX_TIME or depth == 0:
        return evaluate(state), None

    best_move = None
    candidate_moves = get_candidate_moves(state)

    if maximizing_player:
        max_eval = -float('inf')
        for r, c in candidate_moves:
            if time.time() - start_time > MAX_TIME:
                break
            state[r][c] = 1

            if isWinnerFast(state, 1, r, c):  # dùng hàm tối ưu
                state[r][c] = 0
                return float('inf'), (r, c)

            eval_score, _ = alpha_beta(state, depth - 1, alpha, beta, False, start_time)

            attack_score = calculate_score_general(state, 1)
            defense_score = calculate_score_general(state, 2)
            if attack_score >= 5000 and defense_score >= 3000:
                eval_score += 10000  # fork mạnh
            elif attack_score >= 5000:
                eval_score += 5000

            state[r][c] = 0

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = (r, c)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move

    else:
        min_eval = float('inf')
        for r, c in candidate_moves:
            if time.time() - start_time > MAX_TIME:
                break
            state[r][c] = 2

            if isWinnerFast(state, 2, r, c):
                state[r][c] = 0
                return -float('inf'), (r, c)

            eval_score, _ = alpha_beta(state, depth - 1, alpha, beta, True, start_time)

            state[r][c] = 0

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = (r, c)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move

# Hàm chọn nước đi chính
def select_best_move(currentState):
    global MAX_DEPTH
    start_time = time.time()
    N = currentState.shape[0]
    filled_cells = np.count_nonzero(currentState != 0)

    if N <= 20:
        MAX_DEPTH = 4 if filled_cells < 10 else 3
    elif N <= 100:
        MAX_DEPTH = 2 if filled_cells < 200 else 1
    else:
        MAX_DEPTH = 1 #bàn lớn
    if filled_cells < 2:
        MAX_DEPTH = 0 #Mới bắt đầu trận đấu

    candidate_moves = get_candidate_moves(currentState)

    # 1. Thắng ngay
    for r, c in candidate_moves:
        currentState[r][c] = 1
        if isWinnerFast(currentState, 1, r, c):
            currentState[r][c] = 0
            duration = round(time.time() - start_time, 3)
            print(f"🏁 Thắng ngay tại ({r+1},{c+1}) | ⏱ {duration}s")
            return r, c
        currentState[r][c] = 0

    # 2. Chặn thắng ngay
    for r, c in candidate_moves:
        currentState[r][c] = 2
        if isWinnerFast(currentState, 2, r, c):
            currentState[r][c] = 1
            duration = round(time.time() - start_time, 3)
            print(f"🛡️ Chặn thắng ngay tại ({r+1},{c+1}) | ⏱ {duration}s")
            return r, c
        currentState[r][c] = 0

    # 3. Tấn công tốt nhất
    best_attack_score = -float('inf')
    best_attack_move = None
    for r, c in candidate_moves:
        currentState[r][c] = 1
        attack_score = calculate_score_general(currentState, 1)
        currentState[r][c] = 0
        if attack_score > best_attack_score:
            best_attack_score = attack_score
            best_attack_move = (r, c)

    # 4. Phòng thủ tốt nhất
    best_threat_score = 0
    best_threat_move = None
    enemy_moves = np.argwhere(currentState == 2)
    checked = set()
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for r0, c0 in enemy_moves:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < N and 0 <= c < N and currentState[r][c] == 0 and (r, c) not in checked:
                    checked.add((r, c))
                    for dx, dy in directions:
                        count = 0
                        open_ends = 0
                        for i in range(1, NUM_WIN):
                            nr, nc = r + i * dx, c + i * dy
                            if 0 <= nr < N and 0 <= nc < N:
                                if currentState[nr][nc] == 2:
                                    count += 1
                                elif currentState[nr][nc] == 0:
                                    open_ends += 1
                                    break
                                else:
                                    break
                        for i in range(1, NUM_WIN):
                            nr, nc = r - i * dx, c - i * dy
                            if 0 <= nr < N and 0 <= nc < N:
                                if currentState[nr][nc] == 2:
                                    count += 1
                                elif currentState[nr][nc] == 0:
                                    open_ends += 1
                                    break
                                else:
                                    break
                        if count >= 3 and open_ends >= 1:
                            threat_score = count * 1000 + open_ends * 100
                            if threat_score > best_threat_score:
                                best_threat_score = threat_score
                                best_threat_move = (r, c)

    # 5. So sánh tấn công và phòng thủ
    if best_threat_score >= 3000 and best_attack_score >= 3000:
        duration = round(time.time() - start_time, 3)
        if best_attack_score >= best_threat_score:
            r, c = best_attack_move
            print(f"⚔️ Ưu tiên tấn công tại ({r+1},{c+1}) | ⏱ {duration}s")
            return r, c
        else:
            r, c = best_threat_move
            print(f"🛡️ Ưu tiên phòng thủ tại ({r+1},{c+1}) | ⏱ {duration}s")
            return r, c

    if best_threat_move:
        r, c = best_threat_move
        duration = round(time.time() - start_time, 3)
        print(f"🛡️ Chọn nước phòng thủ tốt tại ({r+1},{c+1}) | ⏱ {duration}s")
        return r, c

    if best_attack_score >= 3000:
        r, c = best_attack_move
        duration = round(time.time() - start_time, 3)
        print(f"⚔️ Chọn nước tấn công mạnh tại ({r+1},{c+1}) | ⏱ {duration}s")
        return r, c

    # 6. Fallback Alpha-Beta
    if best_attack_score < 1500 and best_threat_score < 2000:
        score, move = alpha_beta(currentState, MAX_DEPTH, -float('inf'), float('inf'), True, start_time)
        duration = round(time.time() - start_time, 3)
        if move:
            r, c = move
            print(f"🤖 Máy chọn nước đi Alpha-Beta: ({r+1},{c+1}) | ⏱ {duration}s | Depth: {MAX_DEPTH}")
            return r, c

    # 7. Chọn lại nước tốt nhất
    if best_attack_move:
        r, c = best_attack_move
        duration = round(time.time() - start_time, 3)
        print(f"⚡ Chọn lại nước tốt nhất đã tính: ({r+1},{c+1}) | ⏱ {duration}s")
        return r, c

    # 8. Random fallback
    r, c = random.choice(np.argwhere(currentState == 0))
    duration = round(time.time() - start_time, 3)
    print(f"⚠️ Chọn ngẫu nhiên: ({r+1},{c+1}) | ⏱ {duration}s")
    return r, c

# Giới hạn thời gian tìm kiếm
MAX_TIME = 1.8  # Giây

def updateMoveStep(playerName, row, col):
    global caroState
    if playerName == "A":
        setStatePosition(caroState, 1, row, col)
    else:
        setStatePosition(caroState, 2, row, col)

def selectMoveStep():
    global caroState
    return select_best_move(caroState)

# ================== Hỗ trợ ==================
def updateMoveStep(playerName, row, col):
    global caroState
    if playerName == "A":
        setStatePosition(caroState, 1, row, col)
    else:
        setStatePosition(caroState, 2, row, col)

def selectMoveStep():
    global caroState
    return select_best_move(caroState)

# ================== Chế độ Người vs Máy ==================
def play_human_vs_ai():
    global caroState, SoNuocCoConLai
    caroState = np.zeros((N, N))
    SoNuocCoConLai = N * N
    print("\n=== Chế độ: Người chơi vs Máy ===")

    while SoNuocCoConLai > 0 and not isWinner(caroState, 1, N, NUM_WIN) and not isWinner(caroState, 2, N, NUM_WIN):
        # Người chơi nhập nước đi
        while True:
            try:
                row = int(input("Nhập dòng (1...N): ")) - 1
                col = int(input("Nhập cột (1...N): ")) - 1
                if checkAvailable(caroState, row, col):
                    break
                print("❌ Ô đã đánh hoặc nằm ngoài phạm vi. Nhập lại.")
            except:
                print("⚠️ Vui lòng nhập số nguyên.")

        updateMoveStep("B", row, col)
        SoNuocCoConLai -= 1
        if isWinner(caroState, 2, N, NUM_WIN):
            print("🎉 Bạn thắng!")
            showPartialCaroState(caroState, row, col)
            return

        # Máy đánh
        row_ai, col_ai = selectMoveStep()
        updateMoveStep("A", row_ai, col_ai)
        SoNuocCoConLai -= 1
        showPartialCaroState(caroState, row_ai, col_ai)

        if isWinner(caroState, 1, N, NUM_WIN):
            print("🤖 Máy thắng!")
            return

    print("🤝 Hòa!")
    showPartialCaroState(caroState, row_ai, col_ai)

# ================== Chế độ AI vs AI ==================
def play_ai_vs_ai():
    global caroState, SoNuocCoConLai
    caroState = np.zeros((N, N))
    SoNuocCoConLai = N * N
    print(f"\n=== Chế độ: AI vs AI trên bàn {N}x{N} ===")
    showPartialCaroState(caroState, N // 2, N // 2)

    # ✅ CHỈ RANDOM NGƯỜI ĐI TRƯỚC MỘT LẦN DUY NHẤT TRƯỚC VÒNG while
    first_player = random.choice(["A", "B"])
    print(f"🎲 AI {first_player} đi trước!")

    turn = 0
    solan = 0

    while SoNuocCoConLai > 0:
        solan += 1
        print(f"\n🔁 Lượt {solan} | Còn lại: {SoNuocCoConLai} ô trống")

        # ✅ Tính người chơi dựa vào lượt hiện tại và người đi đầu
        playerName = first_player if turn % 2 == 0 else ("B" if first_player == "A" else "A")

        row_ai, col_ai = selectMoveStep()
        updateMoveStep(playerName, row_ai, col_ai)
        SoNuocCoConLai -= 1
        showPartialCaroState(caroState, row_ai, col_ai)

        if isWinner(caroState, 1 if playerName == "A" else 2, N, NUM_WIN):
            print(f"\n🏁 AI {playerName} thắng sau {solan} lượt!")
            showPartialCaroState(caroState, row_ai, col_ai)
            return

        turn += 1

    print("\n🤝 Hòa!")
    showPartialCaroState(caroState, row_ai, col_ai)


# ================== Main menu ==================
def main():
    print("🎮 Chào mừng đến với Caro AI")
    print("1. Người chơi vs Máy")
    print("2. Máy vs Máy (AI vs AI)")

    while True:
        choice = input("Chọn chế độ (1 hoặc 2): ").strip()
        if choice == "1":
            play_human_vs_ai()
            break
        elif choice == "2":
            play_ai_vs_ai()
            break
        else:
            print("⚠️ Vui lòng chọn 1 hoặc 2.")

# ================== Chạy chương trình ==================
if __name__ == "__main__":
    main()
