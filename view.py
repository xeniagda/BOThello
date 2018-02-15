from main import *
import matplotlib.pyplot as plt
import numpy as np
import random


# plt.ion()
#fig.axis([0, 3000, 0, 0.3])
# fig1 = plt.figure(1)
# ax = fig1.add_subplot(111)
# progress, = ax.plot([], [], 'b-')

# plt.ylim(0, 0.05)
# plt.xlim(0, 1000)


# progress.set_ydata(losses)
# progress.set_xdata([x * 50 for x in range(len(losses))])
# fig1.canvas.draw()

data = dataloader.load_images_from("game0")
(train_xs, train_ys, train_zs) = into_traindata(data)

image = Variable(torch.from_numpy(train_xs))
action = Variable(torch.from_numpy(train_ys))

print("Generating")
res = np.array(bot(image, action).data)


plt.plot(losses)
plt.show()

for i in range(10):
    i = random.randrange(0, len(res))

    inp = train_xs[i]
    out = res[i]

    plt.subplot(121)
    plt.imshow(np.clip(inp, 0, 1).transpose((1, 2, 0)))
    plt.subplot(122)
    plt.imshow(np.clip(out, 0, 1).transpose((1, 2, 0)))
    plt.show()


# while True:
#     plt.pause(0.5)

