import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from IPython import display
import random 
import matplotlib

# LOAD TRAINING DATA
train_mat = scipy.io.loadmat('data/mnist_data/images.mat')
train_images_raw = train_mat['images'].T
train_images_corrected = []
for i in range(len(train_images_raw)):
    train_images_corrected.append(train_images_raw[i].T.ravel())
train_images_corrected = np.array(train_images_corrected)

#Now let's try kmeans clustering
k = 10
mu = [np.random.rand(28*28) for i in range(k)]
k_classes = [[] for i in range(k)]

# 1 reclassify
# 2 reset mean.

#A single iteration of reclassification.

changed = True
truth = [True for i in range(k)]
while changed:


	for image in train_images_corrected:
		index, closest, dif  = 0, mu[0], 100000000000
		for i, mu_i in enumerate(mu):
			dist = np.linalg.norm(mu_i - image)
			if dist < dif:
				index,closest,dif = i, mu_i, dist

		k_classes[index].append(image)


	for i in range(len(mu)):
		temp = np.mean(k_classes[i], axis=0)
		truth[i] = temp != mu[i]
		mu[i] = temp

	for val in truth:
		if val is True:
			changed = True
			break
		else:
			changed = False


