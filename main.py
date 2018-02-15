import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np
import matplotlib.pyplot as plt
import PIL

img1 = PIL.Image.open("Othello.png")
img1 = (np.array(img1.getdata(), dtype="float32") / 256).reshape(128,128,4)[:,:,0:3].transpose((2, 0, 1))
img2 = PIL.Image.open("Othello_3_2.png")
img2 = (np.array(img2.getdata(), dtype="float32") / 256).reshape(128,128,4)[:,:,0:3].transpose((2, 0, 1))

#plt.subplot(211)
#plt.imshow(img1.transpose((1, 2, 0)))
#plt.subplot(212)
#plt.imshow(img2.transpose((1, 2, 0)))
#plt.show()

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
	
	# x = image, y = player action
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
		
		return x
	
bot = BOThello()


image = Variable(torch.from_numpy(img1.reshape(1, *img1.shape)))
player_action = torch.zeros(1, 65)
player_action[0, 3 * 8 + 2] = 1
res = bot(image, Variable(player_action))


#plt.imshow(np.array(img1).transpose((1, 2, 0)))
#plt.show()
#plt.imshow(np.array(res.data)[0].transpose((1, 2, 0)))
#plt.show()

plt.ion();
#fig.axis([0, 3000, 0, 0.3])
fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
progress, = ax.plot([], [], 'b-') 
losses = []

plt.ylim(0, 0.05)
plt.xlim(0, 1000)

#plt.imshow(img1.transpose((1, 2, 0)))

crit = nn.MSELoss()
optimizer = opt.SGD(bot.parameters(), lr=0.03, momentum=0.5)

for i in range(1000): # Train the network on img1->img2
	for j in range(50):
		image = Variable(torch.from_numpy(img1.reshape(1, *img1.shape)))
		wanted = Variable(torch.from_numpy(img2.reshape(1, *img2.shape)))
		
		player_action = torch.zeros(1, 65)
		player_action[0, 3 * 8 + 2] = 1
		res = bot(image, Variable(player_action))
		
		optimizer.zero_grad()
		
		loss = crit(res, wanted)
		loss.backward()
		optimizer.step()	
	print(loss.data[0])
	losses.append(loss.data[0])
	progress.set_ydata(losses)
	progress.set_xdata([x*50 for x in range(len(losses))])
	fig1.canvas.draw()

	resImg = np.clip(np.array(res.data)[0], 0, 1)
	
	print(resImg.shape)
	plt.figure(2)
	plt.imshow(resImg.transpose((1, 2, 0)))
	plt.show()
	plt.pause(0.2)


print("Done!")

plt.imshow(np.array(img1).transpose((1, 2, 0)))
plt.show()
plt.imshow(np.array(res.data)[0].transpose((1, 2, 0)))
plt.show()


while True:
	plt.pause(0.5)

