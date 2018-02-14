import numpy as np

PIECE_SIZE = 16

def generate(board):
    result = np.zeros((3, 128, 128))
    result[1] = 1

    for y in range(len(board)):
        for x in range(len(board[y])):
            if board[y, x] > 0:
                draw_piece(result, y + 0.5, x + 0.5, (board[y, x] - 1,) * 3)
    return result

def draw_piece(board, y, x, color):
    yx = np.array(np.meshgrid(np.arange(board.shape[1]), np.arange(board.shape[2]))).transpose((1, 2, 0))\
            - [y * PIECE_SIZE, x * PIECE_SIZE]
    for c in range(len(color)):
        board[c, (yx ** 2).sum(axis=2) < (PIECE_SIZE / 2) ** 2] = color[c]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.array([
        [0, 2, 2, 2, 2, 0, 1, 0],
        [0, 0, 2, 2, 2, 1, 0, 1],
        [0, 1, 2, 2, 2, 1, 0, 1],
        ])

    b = generate(x)
    plt.imshow(b.transpose(1, 2, 0))
    plt.show()