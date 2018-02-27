import os
import random
import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np
import PIL
import dataloader

PATH = "network.tar"
BATCH_SIZE = 8


"""
    X: -1 x 3 x 128 x 128
    Conv(50, 5) + Maxpool(4)
    -1 x 50 x 31 x 31
    Conv(4) + Maxpool(3)
    -1 x 4 x 9 x 9
    Reshape(324)
    -1 x 324

    Connected(512)
    -1 x 512

    Append(Y: -1 x 65) [0-64: where, 65: whom (0=black, 1=white)]
    -1 x 577
    Connected(256)
    -1 x 256

    Reshape(4, 8, 8)
    -1 x 4 x 8 x 8
    Conv(50) + Upscale(4)
    -1 x 50 x 32 x 32
    Conv(3) + Upscale(4)
    -1 x 3 x 128 x 128
"""

class BOThello(nn.Module):
    def __init__(self):
        super(BOThello, self).__init__()

        self.conv_in_1 = nn.Conv2d(3, 50, 5)
        self.conv_in_2 = nn.Conv2d(50, 4, 5)

        self.conn_1 = nn.Linear(324, 512)
        self.conn_2 = nn.Linear(577, 256)

        self.conv_out_1 = nn.Conv2d(4, 50, 6)
        self.conv_out_2 = nn.Conv2d(50, 3, 8)

    # x = image [-1 x 3 x 128 x 128], y = player action [-1 x 65]
    def forward(self, x, y):
        x = self.conv_in_1(x)
        x = F.relu(F.max_pool2d(x, (4, 4)))
        x = self.conv_in_2(x)
        x = F.relu(F.max_pool2d(x, (3, 3)))

        x = x.view(-1, 324)
        x = self.conn_1(x)
        x = F.relu(x)

        x = torch.cat((x, y), dim=1)

        x = self.conn_2(x)
        x = F.relu(x)
        x = x.view(-1, 4, 8, 8)

        x = F.upsample(x, scale_factor=4)
        x = self.conv_out_1(x)
        x = F.upsample(x, scale_factor=5)
        x = self.conv_out_2(x)

        # x = F.sigmoid(x)

        return x

bot = BOThello()

def into_traindata(data):
    xs = np.array([]) # Image inputs
    ys = np.array([]) # Move inputs
    zs = np.array([]) # Wanted outputs

    for (path, before, after, move) in data:
        if len(xs) == 0:
            xs = np.array([before])
        else:
            xs = np.append([before], xs, axis=0)

        y = np.array(np.zeros((65,)), dtype="float32")
        y[move[0] * 8 + move[1]] = 1
        y[-1] = move[2]
        if len(ys) == 0:
            ys = np.array([y])
        else:
            ys = np.append([y], ys, axis=0)

        if len(zs) == 0:
            zs = np.array([after])
        else:
            zs = np.append([after], zs, axis=0)

    return (xs, ys, zs)



#plt.imshow(img1.transpose((1, 2, 0)))



losses = []
crit = nn.MSELoss()
optimizer = opt.SGD(bot.parameters(), lr=0.03, momentum=0.5)
epoch = 0

if os.path.isfile(PATH):
    data = input("Load data? [Y/n] ")
    if len(data) == 0 or data[0].lower() != "n":
        state = torch.load(PATH)
        bot.load_state_dict(state["state"])
        optimizer.load_state_dict(state["opt"])
        losses = state.get("losses", [])
        epoch = state.get("epoch", 0)


if __name__ == "__main__":
    data = dataloader.load_all_images()
    random.shuffle(data)

    while True:
        print(" === Starting epoch", epoch, "===")

        epoch_losses = []
        for j in range(0, len(data), BATCH_SIZE):
            batch_end = min(j + BATCH_SIZE, len(data))

            (train_xs, train_ys, train_zs) = into_traindata(data[j:batch_end])
            image = Variable(torch.from_numpy(train_xs))
            action = Variable(torch.from_numpy(train_ys))
            wanted = Variable(torch.from_numpy(train_zs))

            print("Batch {:>2} - {:<2} / {:>2}".format(j, batch_end, len(data)))

            res = bot(image, action)
            print("    Generated")

            optimizer.zero_grad()

            loss = crit(res, wanted)
            loss.backward()
            print("    Loss = {:.4f}".format(loss.data[0]))
            epoch_losses.append(loss.data[0])
            optimizer.step()


            print("    Done")

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print("Average loss: {:.4f}".format(epoch))
        losses.append(epoch_loss)

        epoch += 1

        print("Saving")

        state = {
                "state": bot.state_dict(),
                "opt": optimizer.state_dict(),
                "losses": losses,
                "epoch": epoch
                }
        torch.save(state, PATH)

