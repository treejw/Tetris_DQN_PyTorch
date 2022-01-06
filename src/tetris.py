import numpy as np
from PIL import Image
import cv2
from matplotlib import style
import torch
import random
from datetime import datetime

style.use('ggplot')


class Tetromino:
    def __init__(self):
        self.pieces = [1, 2, 3, 4, 5, 6, 7]

        colors = [         # RGB
            [0, 0, 0],     # background
            [240, 55, 55],
            [110, 200, 55],
            [250, 180, 55],
            [250, 170, 170],
            [170, 170, 250],
            [50, 80, 250],
            [50, 200, 250],
        ]
        pieces_detail = [
            [],

            [[1, 1],
             [1, 1]],

            [[0, 2, 0],
             [2, 2, 2]],

            [[0, 3, 3],
             [3, 3, 0]],

            [[4, 4, 0],
             [0, 4, 4]],

            [[5, 5, 5, 5]],

            [[0, 0, 6],
             [6, 6, 6]],

            [[7, 0, 0],
             [7, 7, 7]]
        ]
        self.colors = [np.array(c[::-1], dtype=np.uint8) for c in colors]
        self.pieces_detail = [np.array(p, dtype=np.uint8) for p in pieces_detail]

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        return self.pieces_detail[idx]

    def random_choice(self):
        piece_id = random.choice(self.pieces)
        return piece_id, self.pieces_detail[piece_id]


class Tetris:
    tetromino = Tetromino()

    def __init__(self, height=20, width=10, block_size=20, out_images=None):
        self.height, self.width, self.block_size = height, width, block_size
        self.board_size = (height, width)
        # info_board
        h, w, c = (self.block_size*5), (self.width * self.block_size), 3
        bg_color = (255, 255, 255)
        self.text_color = (0, 0, 0)
        self.info_board = np.ones((h, w, c), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
        self.out_images = out_images
        self.reset()

    def reset(self):
        self.board = np.zeros(self.board_size, dtype=np.uint8)
        self.score = 0
        self.num_pieces = 0
        self.cleared_lines = 0
        self.piece_id, self.piece = self.tetromino.random_choice()
        piece_h, piece_w = self.piece.shape
        self.current_pos = {
            'x': self.width//2 - piece_w//2,
            'y': 0
        }
        self.gameover = False
        return self.get_board_state(self.board)

    def rotate(self, piece):
        return np.rot90(piece)

    def get_board_state(self, board):
        state = (board != 0).astype(int)
        return torch.FloatTensor(state)

    def get_next_states(self):
        states = {}
        num_rotations = 0
        if self.piece_id == 1:
            num_rotations = 1
        elif self.piece_id in [3, 4, 5]:
            num_rotations = 2
        elif self.piece_id in [2, 6, 7]:
            num_rotations = 4

        curr_piece = self.piece.copy()
        for i in range(num_rotations):
            valid_xs = self.width - curr_piece.shape[1]
            for x in range(valid_xs+1):
                piece = curr_piece.copy()
                pos = {'x': x, 'y': 0}
                while not self.check_collision(piece, pos):
                    pos['y'] += 1
                _, piece = self.truncate(piece, pos)
                board = self.store(piece, pos)
                states[(x, i)] = self.get_board_state(board)
            curr_piece = self.rotate(curr_piece)
        return states

    def get_current_board_state(self):
        board = self.board.copy()
        h, w = self.piece.shape
        board[self.current_pos['y']:self.current_pos['y']+h, self.current_pos['x']:self.current_pos['x']+w] += self.piece
        return board

    def new_piece(self):
        self.piece_id, self.piece = self.tetromino.random_choice()
        piece_h, piece_w = self.piece.shape
        self.current_pos = {
            'x': self.width//2 - piece_w//2,
            'y': 0
        }
        if self.check_collision(self.piece, self.current_pos):
            self.gameover = True

    def check_collision(self, piece, pos):
        future_y = pos['y'] + 1
        h, w = piece.shape
        board_status = (self.board[future_y:future_y+h, pos['x']:pos['x']+w] != 0).astype(int)
        if board_status.shape != piece.shape:
            return True
        overlap = (board_status*2) - np.where(piece>1, 1, piece) == 1
        if np.sum(overlap) > 0 or np.sum(np.array(range(h))+future_y > self.height-1) > 0:
            return True
        return False

    def truncate(self, piece, pos):
        def get_last_collision_row(h, board_status, piece):
            overlap = (board_status * 2) - np.where(piece > 1, 1, piece) == 1
            tmp = np.where(np.sum(overlap, axis=-1) >= 1, 1, np.sum(overlap, axis=-1))
            last_collision_row = h - (np.argmax(tmp[::-1]) + 1) if np.sum(tmp) != 0 else -1
            return last_collision_row

        gameover = False
        h, w = piece.shape
        board_status = (self.board[pos['y']:pos['y']+h, pos['x']:pos['x']+w] != 0).astype(int)
        last_collision_row = get_last_collision_row(h, board_status, piece)

        if pos['y'] - (h - last_collision_row) < 0 and last_collision_row > -1:
            while last_collision_row >= 0 and piece.shape[0] > 1:
                gameover = True
                piece = piece[1:, :]
                board_status = board_status[:piece.shape[0], :]
                last_collision_row = get_last_collision_row(piece.shape[0], board_status, piece)
        return gameover, piece

    def store(self, piece, pos):
        board = self.board.copy()
        h, w = piece.shape
        board_status = (board[pos['y']:pos['y']+h, pos['x']:pos['x']+w] != 0).astype(int)
        overlap = (board_status*2) - np.where(piece>1, 1, piece) == 1
        if np.sum(overlap) == 0:
            board[pos['y']:pos['y']+h, pos['x']:pos['x']+w] += piece
        return board

    def check_cleared_rows(self, board):
        to_delete = np.where(np.sum((board != 0).astype(int), axis=-1) == self.width)[0]
        if len(to_delete) > 0:
            board = self.remove_row(board, to_delete)
        return len(to_delete), board

    def remove_row(self, board, indices):
        for i in indices:
            board = np.concatenate((np.zeros((1, self.width), dtype=np.uint8), board[:i], board[i+1:]))
        return board

    def step(self, action, render=True, save_frame=None):
        x, num_rotations = action
        self.current_pos = {"x": x, "y": 0}
        for _ in range(num_rotations):
            self.piece = self.rotate(self.piece)

        while not self.check_collision(self.piece, self.current_pos):
            self.current_pos["y"] += 1
            if render:
                self.render(save_frame)

        overflow, piece = self.truncate(self.piece, self.current_pos)
        self.board = self.store(piece, self.current_pos)
        if overflow:
            self.gameover = True
            if render:
                self.render(save_frame, done=True)

        lines_cleared, self.board = self.check_cleared_rows(self.board)
        score = 1 + (lines_cleared ** 2) * self.width
        self.score += score
        self.num_pieces += 1
        self.cleared_lines += lines_cleared
        if not self.gameover:
            self.new_piece()
        if self.gameover:
            self.score -= 2

        return score, self.gameover

    def render(self, save_frame=None, done=False):
        if not self.gameover:
            img = np.expand_dims(self.get_current_board_state(), axis=-1)
            for piece_id in self.tetromino.pieces:
                img = np.where(img == piece_id, self.tetromino.colors[piece_id], img)
        else:
            img = np.expand_dims(self.board, axis=-1)
            for piece_id in self.tetromino.pieces:
                img = np.where(img == piece_id, self.tetromino.colors[piece_id], img)

        img = img[..., ::-1]
        img = Image.fromarray(img, "RGB")

        img = img.resize((self.width * self.block_size, self.height * self.block_size), 0)
        img = np.array(img)
        img[[i * self.block_size for i in range(self.height)], :, :] = 0
        img[:, [i * self.block_size for i in range(self.width)], :] = 0

        img = np.concatenate((self.info_board, img), axis=0)

        cv2.putText(img, "Score:", (self.block_size, self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2, color=self.text_color)
        cv2.putText(img, str(self.score),
                    (7 * self.block_size, self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2, color=self.text_color)

        cv2.putText(img, "N Pieces:", (self.block_size, 2 * self.block_size+int(self.block_size / 2)),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2, color=self.text_color)
        cv2.putText(img, str(self.num_pieces),
                    (7 * self.block_size, 2 * self.block_size + int(self.block_size / 2)),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2, color=self.text_color)

        cv2.putText(img, "Lines:", (self.block_size, 4 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2, color=self.text_color)
        cv2.putText(img, str(self.cleared_lines),
                    (7 * self.block_size, 4 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2, color=self.text_color)

        if save_frame:
            if done:
                img = np.array(Image.fromarray(img, "RGB").convert('L'))
            cv2.imwrite(f"{self.out_images}/{datetime.now().strftime('%H_%M_%S_%f')}.jpg", img)

        cv2.imshow("DQN Tetris", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    height, width, block_size = 18, 10, 20
    fps = 300

    env = Tetris(height, width, block_size)

    while True:
        action = [random.choice(range(width-4)), 0]
        _, done = env.step(action, render=True, save_frame=True)

        if done:
            # out.release()
            break

