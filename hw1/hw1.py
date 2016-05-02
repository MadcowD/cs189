#This file uses Anaconda (scikit-learn plus scipy)
import scipy.io
from sklearn import svm
from sklearn.preprocessing import normalize
import numpy as np


#Printing courtesy of scikit-learn website
from skprint import *
from sklearn.metrics import confusion_matrix


def experiment12(a, ttrain_data, ttrain_labe, tvalid_data, tvalid_labe):
	digit_classifier = svm.SVC(C=2.8)
	digit_classifier.fit(ttrain_data[0:a],ttrain_labe[0:a])
	print("(%s, %s)" % (a, digit_classifier.score(tvalid_data,tvalid_labe)))
	pred = digit_classifier.predict(tvalid_data)
	print(confusion_matrix(pred,tvalid_labe))

def kfold_cv(k, X, Y, classifier):
	shuffle = np.random.permutation(np.arange(X.shape[0]))
	X, Y = X[shuffle], Y[shuffle]
	Xvalid = list([list(X[w:w+int(len(X)/k)]) for w in range(0,len(X), int(len(X)/k))])
	Yvalid = list([list(Y[w:w+int(len(Y)/k)]) for w in range(0,len(Y), int(len(Y)/k))])
	Xtrain = [list(X[0:w*int(len(X)/k)]) + list(X[(w+1)*int(len(X)/k):]) for w in range(k)]
	Ytrain = [list(Y[0:w*int(len(Y)/k)]) + list(Y[(w+1)*int(len(Y)/k):]) for w in range(k)]


	net_score = 0

	for Xtpart, Ytpart, Xvpart, Yvpart  in zip(Xtrain, Ytrain, Xvalid, Yvalid):
		classifier.fit(Xtpart, Ytpart)
		net_score += classifier.score(Xvpart, Yvpart)

	return net_score/k


trainset = scipy.io.loadmat("./data/digit-dataset/train.mat")
testset = scipy.io.loadmat("./data/digit-dataset/test.mat")

unpd =  trainset['train_images']
unpl =  trainset['train_labels']
tdata = np.array([np.array(unpd[:,:,i].flatten()) for i in range(len(unpd[0,0]))])
tdivide = 1/255.0*2
tlabel = unpl.ravel()
#NORMALIZE AND CENTER AVERAGE INTENSITY AROUND 0 :)
tdata = tdata *tdivide
tdata = tdata - 1

shuffle = np.random.permutation(np.arange(tdata.shape[0]))
tdata, tlabel = tdata[shuffle], tlabel[shuffle]


tvalid_data=tdata[1:10000]
tvalid_labe=tlabel[1:10000]
ttrain_labe=tlabel[10000:]
ttrain_data=tdata[10000:]

#Use the the one versus one decision function
# fromc (Knerr et al., 1990)
#
#print("Experiment 12")
#experiment12(100, ttrain_data,ttrain_labe, tvalid_data, tvalid_labe)
#experiment12(200, ttrain_data,ttrain_labe, tvalid_data, tvalid_labe)
#experiment12(500, ttrain_data,ttrain_labe, tvalid_data, tvalid_labe)
#experiment12(1000, ttrain_data,ttrain_labe, tvalid_data, tvalid_labe)
#experiment12(2000, ttrain_data,ttrain_labe, tvalid_data, tvalid_labe)
#experiment12(5000, ttrain_data,ttrain_labe, tvalid_data, tvalid_labe)
#experiment12(10000, ttrain_data,ttrain_labe, tvalid_data, tvalid_labe)

print("Experiment 3")

left=1.0
right=3.0
best=-1.0

numdata=10000

digit_classifier = svm.SVC(C=left)
leval = (kfold_cv(10, tdata[0:numdata], tlabel[0:numdata], digit_classifier))
digit_classifier = svm.SVC(C=right)
reval = (kfold_cv(10, tdata[0:numdata], tlabel[0:numdata], digit_classifier))
for i in range(4):
	digit_classifier = svm.SVC(C=(left+right)/2.0)
	meval = (kfold_cv(10, tdata[0:numdata], tlabel[0:numdata], digit_classifier))
	if leval < reval:
		left = (left+right)/2.0
	else:
		right = (left+right)/2.0

print((left+right)/2.0)

