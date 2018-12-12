from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing
import numpy as np
import sys
from sklearn import svm, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def tuning(X_Train, trainLabel):
    para = []
    for i in range(1, 100):
    # for i in range(140, 180):
        para.append(i / 100)

    maxScore = 0
    for c in para:
        svmClassifier = svm.LinearSVC(C=c, dual=False)
        svmClassifier.fit(X_Train, trainLabel)
        score = cross_val_score(svmClassifier, X_Train, trainLabel, cv=5).mean()
        # print(c, score)
        if (score > maxScore):
            maxScore = score
            maxC = c

    print("Max accuracy score is", maxScore)
    print("c with max score is", maxC)

    return maxC


def main(filename):
	addresses = []
	label = []
	features = []
	with open(filename, "r") as file:
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
	X_train, X_test, y_train, y_test = train_test_split(features, label, 
		test_size=0.7)

	maxC = tuning(X_train, y_train)
	print("Using c =", maxC)
	svmClassifier = svm.LinearSVC(C=maxC, dual=False)
	svmClassifier.fit(X_train, y_train)

	randomForestClassifier = RandomForestClassifier(n_estimators=50)
	randomForestClassifier.fit(X_train, y_train)
	
	predict = svmClassifier.predict(X_test)
	# print(predict.shape)
	print("Accuracy on test set for SVM", accuracy_score(predict, y_test))

	predict = svmClassifier.predict(features)
	print("Accuracy on entire dataset for SVM", accuracy_score(predict, label))
	with open("pred_svm.txt", "w") as f:
		for i in range(len(predict)):
			if predict[i] == 1:
				f.write("< " + addresses[i] + "\n")

	# predict = randomForestClassifier.predict(X_test)
	# # print(predict.shape)
	# print("Accuracy on test set for RandomForest", accuracy_score(predict, y_test))

	predict = randomForestClassifier.predict(features)
	print("Accuracy on entire dataset for RandomForest", accuracy_score(predict, label))
	with open("pred_rf.txt", "w") as f:
		for i in range(len(predict)):
			if predict[i] == 1:
				f.write("< " + addresses[i] + "\n")

	# for i in range(len(predict)):
	# 	if predict[i] != label[i]:
	# 		print(addresses[i], label[i], predict[i])


if __name__ == '__main__':
    main(sys.argv[1])
