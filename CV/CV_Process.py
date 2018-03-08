import os
import sys
import argparse

from ast import literal_eval

def try_play(board, y, x, col):
    board = [line[:] for line in board]

    if board[y][x] != 0:
        return "already something there"

    board[y][x] = col
    has_flipped = False
    for dx, dy in [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]:
        x_, y_ = x, y
        place = False

        while True:
            x_ += dx
            y_ += dy
            if y_ < 0 or y_ >= len(board) or x_ < 0 or x_ >= len(board[y_]):
                break
            if board[y_][x_] == col:
                place = True
                break
            if board[y_][x_] == 0:
                break

        if place:
            has_flipped = True
            x_, y_ = x, y
            while True:
                x_ += dx
                y_ += dy
                if y_ < 0 or y_ >= len(board) or x_ < 0 or x_ >= len(board[y_]):
                    break
                if board[y_][x_] == 0:
                    board[y_][x_] = col
                else:
                    break

    if not has_flipped:
        return "No flipped pieces"

    return board

parser = argparse.ArgumentParser(description="Take the result of CV_Gui and convert into training data")

parser.add_argument(
        "--input", "-i",
        type=str, required=True,
        help="The path to read the input from. Corresponds to --output-moves in CV_Gui")

parser.add_argument(
        "--output", "-o",
        type=str, default="--",
        help="Where to put the result (not required, default is STDOUT which also can be specified using `--`)")

args = parser.parse_args()

with open(args.input, "r") as input_file:
    content = input_file.read()
    try:
        boards = literal_eval(content)
    except Exception as e:
        print("Malformed data in {}:".format(args.input))
        print("    {}".format(e))

    if set(boards.keys()) != set(range(max(boards) + 1)):
        print("Missing: {}".format(set(range(max(boards))) - set(boards.keys())))


moves = []
for i in range(len(boards) - 1):
    curr, nxt = boards[i], boards[i + 1]

    pos = None
    for y in range(len(curr)):
        for x in range(len(curr[y])):
            if curr[y][x] == 0 and nxt[y][x] != 0:
                if pos == None:
                    pos = (len(curr[y]) - x - 1, y, nxt[y][x] - 1)
                else:
                    print("Invalid at {}-{}: too many new pieces ({} and {})"
                        .format(
                        i,
                        i + 1,
                        pos,
                        (len(curr[y]) - x - 1, y, nxt[y][x] - 1))
                        )
                    exit()
    if pos == None:
        print("Invalid at {}: no new piece".format(i + 1))
        exit()
    else:
        # for y in range(len(curr)):
        #     print(curr[y])

        # print(curr, pos[1], 7 - pos[0], pos[2] + 1)
        placed = try_play(curr, pos[1], 7 - pos[0], pos[2] + 1)
        if type(placed) == str:
            print("Invalid move at {}-{}: {}".format(i, i + 1, placed))
            exit()
        else:
            if placed == nxt:
                wrongs = []
                for y in range(len(placed)):
                    for x in range(len(placed[y])):
                        if placed[y][x] != nxt[y][x]:
                            wrongs.append((y, x))
                print("Invalid move at {}-{}: Not correct pieces at {}".format(", ".join(map(str, wrongs))))
                exit()
        moves.append(pos)

if args.output == "--":
    output = sys.stdout
else:
    output = open(args.output, "w")

for (x, y, col) in moves:
    output.write("{},{},{}\n".format(x, y, col))

sys.stderr.write("Finished with no errors\n".format(args.output))
