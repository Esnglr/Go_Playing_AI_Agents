import numpy as np

class GoEnv:
    EMPTY = 0
    BLACK = 1
    WHITE = -1

    #in broader tables komi could be around 6.5 but on small board, this causes miss calculations, so i will use 2.5
    def __init__(self, size=5, komi=2.5):
        self.size = size
        self.komi = komi
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = self.BLACK
        self.history = []
        self.captures = {self.BLACK: 0, self.WHITE: 0}
        self.pass_count = 0
        self.directions = [(-1,0),(1,0),(0,-1),(0,1)]


    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = self.BLACK
        self.history = []
        self.captures = {self.BLACK: 0, self.WHITE: 0}
        self.pass_count = 0
        return self.board.flatten()


    def play_move(self, x, y):
        if x is None or y is None:
            self.pass_turn()
            self.history.append((self.current_player,None,None))
            return True        
        if not self.is_on_board(x, y):
            return False
        if self.board[y, x] != self.EMPTY:
            return False
        if self.is_self_capture(x, y, self.current_player):
            return False

        self.board[y, x] = self.current_player
        self.remove_captured_stones(x, y)
        self.history.append((self.current_player, x, y))
        self.current_player *= -1
        self.pass_count = 0
        return True


    def is_on_board(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size


    def get_legal_moves(self):
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.board[y, x] == self.EMPTY and not self.is_self_capture(x, y, self.current_player):
                    moves.append((x, y))
        return moves

    def get_group(self, x, y, board=None):
        if board is None:
            board = self.board
        color = board[y, x]
        group = set()
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in group:
                continue
            group.add((cx, cy))
            for dx, dy in self.directions:
                nx, ny = cx + dx, cy + dy
                if self.is_on_board(nx, ny) and board[ny, nx] == color:
                    stack.append((nx, ny))
        return group

    def has_liberty(self, group, board=None):
        if board is None:
            board = self.board
        for gx, gy in group:
            for dx, dy in self.directions:
                nx, ny = gx + dx, gy + dy
                if self.is_on_board(nx, ny) and board[ny, nx] == self.EMPTY:
                    return True
        return False

    def pass_turn(self):
        self.pass_count += 1
        self.current_player *= -1


    def is_over(self):
        return self.pass_count >= 2 or len(self.get_legal_moves()) == 0


    def remove_captured_stones(self, x, y):
        opponent = -self.current_player
        checked = set()

        # For each adjacent opponent stone, check if that group is captured
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if not self.is_on_board(nx, ny):
                continue
            if self.board[ny, nx] == opponent and (nx, ny) not in checked:
                group = self.get_group(nx, ny)
                checked |= group
                if not self.has_liberty(group):
                    for (gx, gy) in group:
                        self.board[gy, gx] = self.EMPTY
                    self.captures[self.current_player] += len(group)

    def is_self_capture(self, x, y, color):
        if not self.is_on_board(x, y) or self.board[y, x] != self.EMPTY:
            return False

        temp_board = self.board.copy()
        temp_board[y, x] = color

        #removing captured opponent stones
        opponent = -color
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if self.is_on_board(nx, ny) and temp_board[ny, nx] == opponent:
                group = self.get_group(nx, ny, temp_board)
                if not self.has_liberty(group, temp_board):
                    for gx, gy in group:
                        temp_board[gy, gx] = self.EMPTY

        #after captures, check if our own group has liberties
        group = self.get_group(x, y, temp_board)
        return not self.has_liberty(group, temp_board)


    def count_territory(self, color):
        territory = 0
        for x in range(self.size):
            for y in range(self.size):
                if self.board[y, x] != self.EMPTY:
                    continue
                surrounded = True
                for dx, dy in self.directions:
                    nx, ny = x + dx, y + dy
                    if self.is_on_board(nx, ny):
                        if self.board[ny, nx] != color and self.board[ny, nx] != self.EMPTY:
                            surrounded = False
                            break
                if surrounded:
                    territory += 1
        return territory

    #japanese style scoring because why not
    def score(self, color):
        territory = self.count_territory(color)
        return territory + self.captures[color] + (self.komi if color == self.WHITE else 0)

    def get_active_player(self):
        return self.current_player


    def __str__(self):
        board_str = ""
        for y in range(self.size):
            row = ""
            for x in range(self.size):
                stone = self.board[y, x]
                if stone == self.BLACK:
                    row += "X "
                elif stone == self.WHITE:
                    row += "O "
                else:
                    row += ". "
            board_str += f"{y+1:2d} {row}\n"
        cols = " ".join([chr(ord("A") + i) for i in range(self.size)])
        board_str += "   " + cols + "\n"
        return board_str