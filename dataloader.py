import os
import numpy as np
import PIL.Image


def parse_move(move_):
    (i, move) = move_
    x = int(move[0])
    y = int(move[2])
    return [x, y, (i + 1) % 2]

def load_images_from(dataset_name):
    moves = open("dataset/" + dataset_name + "/input").readlines()
    moves = list(map(parse_move, enumerate(moves)))
    print(moves)

    imgs = list(filter(lambda x: x.endswith(".jpg"), os.listdir("dataset/" + dataset_name)))

    res = [() for i in range(len(imgs))]

    for img in imgs:
        i = int(img[3:img.index(".")])
        content = PIL.Image.open("dataset/" + dataset_name + "/" + img)
        data = (np.array(content.getdata(), dtype="float32") / 256).reshape(128,128,3)[:,:,0:3].transpose((2, 0, 1))
        res[i] = (data, moves[i - 1])

    res_ = []
    for i in range(1, len(res)):
        res_.append((res[i - 1][0], *res[i]))

    return res_


