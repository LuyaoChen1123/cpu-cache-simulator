import numpy as np

import sys
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

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

def main(file):
	filename = file.split(".")[0]
	data, addr = parse_data(file)
	validation_spilit = 0.7
	nb_validation_samples = int(validation_spilit*data.shape[0])
	train = data[:-nb_validation_samples]
	test = data[-nb_validation_samples:]
	train_addr = addr[:-nb_validation_samples]
	# train_addr = train_addr.reshape((train_addr.shape[0],1))
	test_addr = addr[-nb_validation_samples:]

	# train, test = train_test_split(data, test_size=0.7)
	print(train.shape)
	print(train_addr.shape)
	# train_matrix = np.concatenate((train_addr, train), axis=1)
	np.save(filename+"_train_addr.npy", train_addr)
	np.save(filename+"_test_addr.npy", test_addr)
	np.save(filename+"_train.npy", train)
	np.save(filename+"_test.npy", test)

if __name__ == "__main__":
	main(sys.argv[1])
