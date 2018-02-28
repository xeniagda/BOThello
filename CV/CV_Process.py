import os
import argparse

from ast import literal_eval

parser = argparse.ArgumentParser(description="Take the result of CV_Gui and convert into training data")

parser.add_argument(
        "--input", "-i",
        type=str, required=True,
        help="The path to read the input from. Corresponds to --output-moves in CV_Gui")

parser.add_argument(
        "--output", "-o",
        type=str, required=True,
        help="Where to put the result")

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
                    print("Invalid at {}: too many new pieces".format(i + 1))
                    exit()
    if pos == None:
        print("Invalid at {}: no new piece".format(i + 1))

    moves.append(pos)

with open(args.output, "w") as output_file:
    for (x, y, col) in moves:
        output_file.write("{},{},{}\n".format(x, y, col))

print("Done")
