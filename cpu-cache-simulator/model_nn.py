import numpy as np
import torch
import torch.optim
import torch.nn as nn
import sys
import os 
from sklearn.utils import shuffle
from sklearn import preprocessing
from scipy import sparse


def parse_data(file):
	f = open(file)
	onehot_encoder = preprocessing.OneHotEncoder()
	int_encoder = preprocessing.LabelEncoder()
	data = []
	for line in f.readlines():
		data.append(line.split())

	data = np.array(data)
	addr_int = int_encoder.fit_transform(data[:,0].ravel())
	addr_oneHot = onehot_encoder.fit_transform(addr_int.reshape(-1,1)).toarray()

	feat = []
	for i in range(data.shape[0]):
		cur = []
		cur.append(float(data[i,1]))
		cur.append(int(data[i,2]))
		cur.append(int(data[i,3]))
		cur.append(int(data[i,4]))
		feat.append(cur)

	feat = np.array(feat)
	return np.concatenate((addr_oneHot, feat), axis=1)


def main(file):
	data = parse_data(file)

	m, n = data.shape

	# dimensions of the nn
	n_in, n_h1, n_h2, n_out = n - 1, 100, 100, 1
	# condition of converge
	thres = 0.001
	model = nn.Sequential(nn.Linear(n_in, n_h1),
		nn.Tanh(),
		nn.Linear(n_h1, n_h2),
		nn.ReLU(),
		nn.Linear(n_h2, n_out),
		nn.Sigmoid()
		)

	criterion = torch.nn.BCELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

	prev_loss = 0

	# train the model on training data

	data = parse_data(file)
	# print(data.shape)
	shuffle(data)
	# business_id, useful, review_count, bag_of_words
	label = torch.from_numpy(data[:,n-1].reshape(-1,1)).float()
	features = torch.from_numpy(data[:, 0:n-1]).float()
	print(label.shape)
	print(features.shape)
	for epoch in range(100):
		y_pred = model(features)

		loss = criterion(y_pred, label)
		if(epoch % 100 == 0):
			print('epoch: ', epoch,' loss: ', loss.item())
			# converges, break
		
		if abs(prev_loss - loss.item()) <= thres:
			break
		prev_loss = loss.item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	pred = model(features).detach().numpy().reshape(-1,1)
	reuse = []
	for i in pred:
		if i < 0.5:
			reuse.append(0)
		else:
			reuse.append(1)

	print(sum(reuse == data[:,n-1]))


if __name__ == "__main__":
	main(sys.argv[1])

