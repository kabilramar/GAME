"""
AI Tic-Tac-Toe — Minimax & Alpha-Beta Pruning
Streamlit version

DSA concepts demonstrated:
- Recursion        -> minimax() calls itself on every possible next board
- Backtracking     -> a move is placed, explored, then undone before the next try
- Game Tree        -> every board state is a node; every legal move is an edge
- Minimax          -> AI maximizes its score assuming the human minimizes it
- Alpha-Beta Pruning -> branches that can't affect the final decision are skipped

Run with:
    streamlit run app.py
"""

import random
import streamlit as st

HUMAN = "X"
AI = "O"
EMPTY = None

WIN_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],   # rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],   # columns
    [0, 4, 8], [2, 4, 6],              # diagonals
]

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="AI Tic-Tac-Toe — Minimax & Alpha-Beta Pruning",
                    page_icon="🎮", layout="wide")

# ----------------------------------------------------------------------
# Session state — Streamlit reruns the whole script on every interaction,
# so game state must live in st.session_state to persist between reruns.
# ----------------------------------------------------------------------
def init_state():
    st.session_state.board = [EMPTY] * 9
    st.session_state.game_over = False
    st.session_state.human_turn = True
    st.session_state.winner = None
    st.session_state.win_line = None
    st.session_state.stats = {"nodes": 0, "pruned": 0, "max_depth": 0}
    st.session_state.chosen_score = None

if "board" not in st.session_state:
    init_state()
    st.session_state.scores = {"win": 0, "loss": 0, "draw": 0}


# ----------------------------------------------------------------------
# Core game logic
# ----------------------------------------------------------------------
def check_result(board):
    """Returns (winner, winning_line) where winner is 'X', 'O', 'draw', or None."""
    for line in WIN_LINES:
        a, b, c = line
        if board[a] and board[a] == board[b] == board[c]:
            return board[a], line
    if all(cell is not None for cell in board):
        return "draw", None
    return None, None


def alpha_beta_minimax(board, depth, is_maximizing, alpha, beta, stats):
    """
    Recursive minimax search with alpha-beta pruning.

    is_maximizing=True  -> AI's turn  (tries to maximize the score)
    is_maximizing=False -> Human's turn (assumed to minimize the score)

    alpha = best score MAX can guarantee so far
    beta  = best score MIN can guarantee so far
    Once beta <= alpha, remaining sibling branches cannot change the
    outcome, so they are pruned (skipped).
    """
    stats["nodes"] += 1
    stats["max_depth"] = max(stats["max_depth"], depth)

    winner, _ = check_result(board)
    if winner is not None:
        if winner == AI:
            return 10 - depth        # winning sooner is better
        if winner == HUMAN:
            return depth - 10        # losing later is better (if forced)
        return 0                     # draw

    if is_maximizing:
        best = float("-inf")
        for i in range(9):
            if board[i] is not None:
                continue
            board[i] = AI                      # try move
            score = alpha_beta_minimax(board, depth + 1, False, alpha, beta, stats)
            board[i] = None                    # BACKTRACK: undo the move
            best = max(best, score)
            alpha = max(alpha, score)
            if beta <= alpha:
                stats["pruned"] += 1
                break                           # prune remaining siblings
        return best
    else:
        best = float("inf")
        for i in range(9):
            if board[i] is not None:
                continue
            board[i] = HUMAN                   # try move
            score = alpha_beta_minimax(board, depth + 1, True, alpha, beta, stats)
            board[i] = None                    # BACKTRACK: undo the move
            best = min(best, score)
            beta = min(beta, score)
            if beta <= alpha:
                stats["pruned"] += 1
                break                           # prune remaining siblings
        return best


def find_best_move(board):
    """Tries every empty cell at the top level, returns the best one plus stats."""
    stats = {"nodes": 0, "pruned": 0, "max_depth": 0}
    best_score = float("-inf")
    best_move = None
    candidates = []

    for i in range(9):
        if board[i] is not None:
            continue
        board[i] = AI
        score = alpha_beta_minimax(board, 0, False, float("-inf"), float("inf"), stats)
        board[i] = None
        candidates.append((i, score))
        if score > best_score:
            best_score = score
            best_move = i

    return best_move, best_score, candidates, stats


def pick_move_for_difficulty(level, candidates, best_move, best_score):
    if level == "Unbeatable (full minimax)":
        return best_move, best_score
    if level == "Easy (random-ish)":
        i, s = random.choice(candidates)
        return i, s
    # Medium: 70% best move, 30% random
    if random.random() < 0.7:
        return best_move, best_score
    i, s = random.choice(candidates)
    return i, s


def human_move(i):
    if st.session_state.game_over or not st.session_state.human_turn:
        return
    if st.session_state.board[i] is not None:
        return

    st.session_state.board[i] = HUMAN
    winner, line = check_result(st.session_state.board)
    if winner:
        finish_game(winner, line)
        return

    st.session_state.human_turn = False
    ai_move()


def ai_move():
    board = st.session_state.board
    best_move, best_score, candidates, stats = find_best_move(board)
    level = st.session_state.get("difficulty", "Unbeatable (full minimax)")
    move, score = pick_move_for_difficulty(level, candidates, best_move, best_score)

    board[move] = AI
    st.session_state.stats = stats
    st.session_state.chosen_score = score

    winner, line = check_result(board)
    if winner:
        finish_game(winner, line)
    else:
        st.session_state.human_turn = True


def finish_game(winner, line):
    st.session_state.game_over = True
    st.session_state.winner = winner
    st.session_state.win_line = line
    if winner == "draw":
        st.session_state.scores["draw"] += 1
    elif winner == HUMAN:
        st.session_state.scores["win"] += 1
    else:
        st.session_state.scores["loss"] += 1


def new_game(first_player):
    scores = st.session_state.scores
    init_state()
    st.session_state.scores = scores
    if first_player == "AI goes first":
        st.session_state.human_turn = False
        ai_move()


# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("Game settings")
difficulty = st.sidebar.selectbox(
    "Difficulty",
    ["Unbeatable (full minimax)", "Medium (70% best move)", "Easy (random-ish)"],
    key="difficulty",
)
first_player = st.sidebar.selectbox("Who goes first?", ["You go first", "AI goes first"])

if st.sidebar.button("🔄 New Game", use_container_width=True):
    new_game(first_player)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Scoreboard")
c1, c2, c3 = st.sidebar.columns(3)
c1.metric("Wins", st.session_state.scores["win"])
c2.metric("Losses", st.session_state.scores["loss"])
c3.metric("Draws", st.session_state.scores["draw"])

st.sidebar.markdown("---")
st.sidebar.subheader("DSA concepts in play")
st.sidebar.markdown(
    "- **Recursion** — self-calls on every future board\n"
    "- **Backtracking** — a move is undone after exploring it\n"
    "- **Game tree** — boards are nodes, moves are edges\n"
    "- **Minimax** — AI maximizes, assumes human minimizes\n"
    "- **Alpha-beta pruning** — skips branches that can't matter"
)

# ----------------------------------------------------------------------
# Main layout
# ----------------------------------------------------------------------
st.title("🎮 AI Tic-Tac-Toe")
st.caption("Minimax & Alpha-Beta Pruning — you are **X**, the computer is **O**.")

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    # Status line
    if st.session_state.game_over:
        if st.session_state.winner == "draw":
            st.info("It's a draw — nobody wins the perfect game.")
        elif st.session_state.winner == HUMAN:
            st.success("You win! X takes it. 🎉")
        else:
            st.error("Computer wins with O.")
    else:
        st.write("**Your move**" if st.session_state.human_turn else "**Computer is thinking…**")

    # 3x3 board rendered as buttons
    board = st.session_state.board
    win_line = st.session_state.win_line or []

    for row in range(3):
        cols = st.columns(3, gap="small")
        for col in range(3):
            i = row * 3 + col
            val = board[i]
            label = val if val else " "
            disabled = (
                val is not None
                or st.session_state.game_over
                or not st.session_state.human_turn
            )
            btn_type = "primary" if i in win_line else "secondary"
            if cols[col].button(
                label, key=f"cell_{i}", disabled=disabled,
                use_container_width=True, type=btn_type
            ):
                human_move(i)
                st.rerun()

with right:
    st.subheader("What the AI just computed")
    st.caption("These numbers update after each AI move.")

    stats = st.session_state.stats
    m1, m2 = st.columns(2)
    m1.metric("Nodes explored", stats["nodes"] if stats["nodes"] else "—")
    m2.metric("Branches pruned", stats["pruned"] if stats["pruned"] else "—")
    m3, m4 = st.columns(2)
    m3.metric("Max recursion depth", stats["max_depth"] if stats["max_depth"] else "—")
    m4.metric("Chosen move score",
               st.session_state.chosen_score if st.session_state.chosen_score is not None else "—")

    st.markdown("---")
    st.subheader("How it works")
    st.markdown(
        "1. When it's the AI's turn, `find_best_move()` tries every empty cell.\n"
        "2. Each try recursively simulates the rest of the game with `alpha_beta_minimax()`.\n"
        "3. The AI assumes you always play your best possible move (minimizing).\n"
        "4. Alpha-beta pruning skips branches that are already proven irrelevant.\n"
        "5. The move with the highest guaranteed score is played."
    )

st.markdown("---")
st.caption("Open app.py to read the fully commented alpha_beta_minimax() function.")