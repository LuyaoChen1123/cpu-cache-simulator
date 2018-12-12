import numpy as np
import torch
import torch.optim
import torch.nn as nn
import sys
import os 
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy import sparse


def parse_data(file):
	f = open(file)
	onehot_encoder = preprocessing.OneHotEncoder()
	int_encoder = preprocessing.LabelEncoder()
	data = []
	for line in f.readlines():
		data.append(line.split())

	data = np.array(data)
	addr = data[:, 0]
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
	return np.concatenate((addr_oneHot, feat), axis=1), addr

def train_model(data):
	m, n = data.shape
	# address, miss rate, past reuse count, past reuse distance, reused(label)
	label = torch.from_numpy(data[:,n-1].reshape(-1,1)).float()
	features = torch.from_numpy(data[:, 0:n-1]).float()
	print(label.shape)
	print(features.shape)

	# dimensions of the nn
	n_in, n_h1, n_h2, n_out = n - 1, 300, 300, 1
	# condition of converge
	thres = 0.0005
	model = nn.Sequential(nn.Linear(n_in, n_h1),
		nn.Tanh(),
		nn.Linear(n_h1, n_h2),
		nn.ReLU(),
		nn.Linear(n_h2, n_out),
		nn.Sigmoid()
		)

	# BCE is cross-entropy loss
	criterion = torch.nn.BCELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

	prev_loss = 0

	# train the model on training data
	for epoch in range(1000):
		y_pred = model(features)

		loss = criterion(y_pred, label)
		if(epoch % 100 == 0):
			print('epoch: ', epoch,' loss: ', loss.item())
			# converges, break
			# if abs(prev_loss - loss.item()) <= thres:
			# 	break
			prev_loss = loss.item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	return model, features, label


def main(file):
	data, addr = parse_data(file)
	train, test = train_test_split(data, test_size=0.7, random_state=42)
	# data = np.concatenate((train, test), axis=0)
	m, n = data.shape
	print(data.shape)
	print(train.shape)
	print(test.shape)

	model, feat_train, label_train = train_model(train)
	feat_test = torch.from_numpy(test[:, 0:n-1]).float()

	# pred_train = model(feat_train).detach().numpy().ravel()
	# pred_test = model(feat_test).detach().numpy().reshape(-1,1)

	pred = model(torch.from_numpy(data[:, 0:n-1]).float()).detach().numpy().reshape(-1,1)
	
	reuse = []
	result = []
	for i in range(len(pred)):
		if pred[i] < 0.5:
			reuse.append(0)
		else:
			reuse.append(1)
			result.append('< ' + addr[i])

	with open("pred_nn.txt", "w") as f:
		for line in result:
			f.write(line + "\n")

	# print(reuse)
	print('The ratio of correctly predicted reuse: ' + str(sum(reuse == data[:,n-1]) / m))
	# print('Ratio on train: ' + str(sum(pred_train == train[:,n-1]) / train.shape[0]))
	# print('Ratio on test: ' + str(sum(pred_test == test[:,n-1]) / test.shape[0]))




if __name__ == "__main__":
	main(sys.argv[1])

