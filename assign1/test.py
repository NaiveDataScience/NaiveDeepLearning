import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt

class DoubleReluNet(nn.Module):

    def __init__(self):
        super(DoubleReluNet, self).__init__()
        
        self.fc1 = nn.Linear(1, 100)
        # self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

def get_batch(f, batch_size=10):
	inp = np.random.randn(batch_size, 1).astype(np.float64) * 10
	inp.sort(axis=0)
	expect = np.vectorize(f)(inp)
	print(inp)
	print(expect)
	return torch.from_numpy(inp), torch.from_numpy(expect)

def fit(func, Net, inp, expect, train_config, save_name=''):
	net = Net().double()
	optimizer = optim.Adam(net.parameters(), 
					lr = train_config["learning_rate"])
	# in your training loop:
	for i in range(train_config["epoch_number"]):
		optimizer.zero_grad()   # zero the gradient buffers
		output = net(inp)

		loss = nn.MSELoss()(output, expect)
		loss.backward()
		optimizer.step()    # Does the update
		if i % 10000 == 0:
			print (loss)

	if save_name:
		torch.save(net, save_name)
	return net

def get_square_data():
	square = np.arange(-20, 21, 5).reshape(9, 1).astype(np.float64)
	test = torch.from_numpy(square)
	return test


def test(net, test_data, f, plt_config):
	numpy_y = net(test_data).detach().numpy().T[0]
	numpy_x = test_data.detach().numpy().T[0]

	plt.figure(figsize=(12, 8))
	plt.plot(numpy_x, numpy_y, color='red', label='fitting', lw='2')
	plt.plot(numpy_x, list(map(f, numpy_x)), color='blue', label='expecting', lw='2')

	plt.legend()
	plt.title(plt_config["pic_title"])
	plt.savefig(plt_config["pic_name"])


def square(x):
	return x * x

def sino(x):
	return math.sin(x)

if __name__ == '__main__':
	inp, expect = get_batch(square)
	net = fit(square, DoubleReluNet, inp, expect, {
		"learning_rate": 0.01,
		"epoch_number": 100000 
		}, save_name="square.pkl")

	test(net, inp, square, {
		"pic_name" : "y = x ^ 2: Train",
		"pic_title": "y = x ^ 2: Train"
		})

	test_data = get_square_data()
	net = torch.load("square.pkl")
	test(net, test_data, square, {
		"pic_name" : "y = x ^ 2: Test",
		"pic_title": "y = x ^ 2: Test"
		})
	# fit(sino, net)
	



