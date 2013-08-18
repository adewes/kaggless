# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pylab as pl
import pandas as pd
import numpy as np

from sklearn import datasets, svm, metrics

# <codecell>

print "Loading training data..."
digits_train = np.loadtxt(open("train.csv","rb"),delimiter=",",skiprows=1)

targets_train = np.zeros(digits_train.shape[0],dtype = np.int64)

targets_train[:] = digits_train[:,0]
features_train= digits_train[:,1:]

print "Loading test data..."
features_test = np.loadtxt(open("test.csv","rb"),delimiter=",",skiprows=1)

# <codecell>

digits = datasets.load_digits()
n_samples = len(digits.images)
features_train = digits.images.reshape((n_samples, -1))
targets_train = digits.target

# <codecell>

from scipy.ndimage.interpolation import zoom


def zoom_features(features,dim):
    zoomed_features = np.zeros((features.shape[0],dim*dim))
    n = int(floor(sqrt(len(features[0]))))
    for i in range(0,len(features)):
        feature_image = features[i].reshape((n,n))
        result = zoom(feature_image,dim/float(n))
        zoomed_features[i] = result.reshape(dim*dim)
    return zoomed_features

zoomed_features_train = features_train
zoomed_features_test = features_test
#zoomed_features_train = zoom_features(features_train,27)
#zoomed_features_test = zoom_features(features_test,27)

# <codecell>

plt.figure(figsize = (12,6))
for i in range(0,32):
    image = zoomed_features_train[i]
    dim = floor(sqrt(image.shape[0]))
    image = image.reshape((dim,dim))
    label = targets_train[i]
    pl.subplot(4, 8, i + 1)
    pl.axis('off')
    pl.winter()
    pl.imshow(image,interpolation='nearest')
    pl.title('Training: %i' % label)

# <codecell>

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(features_train)

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.000001)

n_training = 1000

# We learn the digits on the first half of the digits
classifier.fit(zoomed_features_train[:n_training], targets_train[:n_training])

# <codecell>

targets_train_validation = classifier.predict(zoomed_features_train[n_training:n_training+20])
print targets_train_validation,"\n",targets_train[n_training:n_training+20]

# <codecell>

# Now predict the value of the digit on the second half:
targets_train_validation = classifier.predict(zoomed_features_train[n_training:])

print "Confusion matrix:\n%s" % metrics.confusion_matrix(targets_train[n_training:], targets_train_validation)

print "\n\nMisclassifications: %g %%" % (sum(targets_train_validation != targets_train[n_training:])/float(len(targets_train_validation))*100)

# <codecell>

targets_test = targets_train_validation

# <codecell>

targets_test = classifier.predict(zoomed_features_test)

# <codecell>

plt.figure(figsize = (12,6))
for i in range(0,32):
    image = features_test[i]
    dim = floor(sqrt(image.shape[0]))
    image = image.reshape((dim,dim))
    label = targets_test[i]
    pl.subplot(4, 8, i + 1)
    pl.axis('off')
    pl.winter()
    pl.imshow(image,interpolation='nearest')
    pl.title('Training: %i' % label)

# <codecell>

with open("result.csv","wb") as f:
    for target in targets_test:
        f.write("%d\n" % target)

