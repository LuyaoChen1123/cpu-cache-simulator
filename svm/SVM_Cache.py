from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing
import numpy as np
from sklearn import svm, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

def tuning(X_Train, trainLabel):
    para = []
    for i in range(1, 20):
    # for i in range(140, 180):
        para.append(i / 100)

    maxScore = 0
    for c in para:
        clf = svm.LinearSVC(C=c, dual=False)
        clf.fit(X_Train, trainLabel)
        score = cross_val_score(clf, X_Train, trainLabel, cv=5).mean()
        print(c, score)
        if (score > maxScore):
            maxScore = score
            maxC = c

    print("Max accuracy score is", maxScore)
    print("c with max score is", maxC)

    return maxC


def main():
	addresses = []
	label = []
	features = []
	with open("res.txt", "r") as file:
		for line in file:
			feat = []
			line = line.rstrip().split(' ')
			addresses.append(line[0])
			label.append(int(line[-1]))
			feat.append(float(line[1]))
			feat.append(int(line[2]))
			feat.append(float(line[3]))
			features.append(feat)

	lenc = preprocessing.LabelEncoder()
	labeledAddr = lenc.fit_transform(addresses)
	labeledAddr = labeledAddr.reshape(len(labeledAddr), 1)
	hotEnc = preprocessing.OneHotEncoder()
	encodedAddr = hotEnc.fit_transform(labeledAddr).toarray()

	features = np.concatenate((encodedAddr, features), axis=1)

	maxC = tuning(features, label)

	print("Using c =", maxC)
	clf = svm.LinearSVC(C=maxC, dual=False)

	clf.fit(features, label)
	predict = clf.predict(features)
	print(accuracy_score(predict, label))


if __name__ == '__main__':
    main()