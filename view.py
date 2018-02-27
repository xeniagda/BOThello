from main import *
import matplotlib.pyplot as plt
import numpy as np
import random

data = dataloader.load_all_images()
random.shuffle(data)
(train_xs, train_ys, train_zs) = into_traindata(data[:10])

image = Variable(torch.from_numpy(train_xs))
action = Variable(torch.from_numpy(train_ys))

print("Generating")
res = np.array(bot(image, action).data)


plt.plot(losses)
plt.show()

for i in range(10):
    inp = train_xs[i]
    out = res[i]

    plt.subplot(121)
    plt.imshow(np.clip(inp, 0, 1).transpose((1, 2, 0)))
    plt.subplot(122)
    plt.imshow(np.clip(out, 0, 1).transpose((1, 2, 0)))
    plt.show()

