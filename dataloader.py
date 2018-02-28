import os
import numpy as np
import PIL.Image

DATASET_PATH = "dataset"

def parse_move(move_):
    (i, move) = move_

    x = int(move.split(",")[0])
    y = int(move.split(",")[1])
    col = int(move.split(",")[2])
    return [x, y, col]

def load_images_from(dataset_name):
    moves = open(os.path.join(DATASET_PATH, dataset_name, "input")).readlines()
    moves = list(map(parse_move, enumerate(moves)))

    imgs = list(filter(lambda x: x.endswith(".jpg"), os.listdir(os.path.join(DATASET_PATH, dataset_name))))

    res = [() for i in range(len(imgs))]

    for img in imgs:
        i = int(img[3:img.index(".")])
        content = PIL.Image.open(os.path.join(DATASET_PATH, dataset_name, img))
        data = (np.array(content.getdata(), dtype="float32") / 256).reshape(128,128,3)[:,:,0:3].transpose((2, 0, 1))
        res[i] = (os.path.join(DATASET_PATH, dataset_name, img), data, moves[i - 1])

    res_ = []
    for i in range(1, len(res)):
        res_.append((res[i][0], res[i - 1][1], *res[i][1:]))

    return res_

def load_all_images():
    data = []
    for data_dir in os.listdir(DATASET_PATH):
        if os.path.isdir(os.path.join(DATASET_PATH, data_dir)):
            print("Loading", data_dir)
            data.extend(load_images_from(data_dir))
    return data

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import random

    data = load_all_images()
    random.shuffle(data)
    for (path, before, after, move) in data[:10]:
        print("path:", path)
        print(move)
        plt.subplot(211); plt.imshow(before.transpose((1, 2, 0)))
        plt.subplot(212); plt.imshow(after.transpose((1, 2, 0)))
        plt.show()
