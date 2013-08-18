# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pylab as pl
import pandas as pd
import numpy as np

from sklearn import datasets, svm, metrics
from sklearn.ensemble import RandomForestClassifier

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer,SigmoidLayer

# <codecell>

print "Loading training data..."
digits_train = np.loadtxt(open("train.csv","rb"),delimiter=",",skiprows=1)

targets_train = np.zeros(digits_train.shape[0],dtype = np.int64)

targets_train[:] = digits_train[:,0]
features_train= digits_train[:,1:]

print "Loading test data..."
features_test = np.loadtxt(open("test.csv","rb"),delimiter=",",skiprows=1)

# <codecell>

#Alternatively, use the digits dataset from scikit-learn
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
        zoomed_features[i] = result.reshape(dim*dim)/256.0
    return zoomed_features

def enrich_features(features):
    features = features.resize(features.shape[0]+1)
    features[-1] = sum(features[:2]/float(len(features)))

zoomed_features_train = zoom_features(features_train,16)
zoomed_features_test = zoom_features(features_test,16)

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

alldata = ClassificationDataSet(zoomed_features_train.shape[1], 1, nb_classes = 10)
n_samples = int(len(zoomed_features_train))
for i in xrange(n_samples):
    alldata.addSample(zoomed_features_train[i], [targets_train[i]])

# <codecell>

tstdata, trndata = alldata.splitWithProportion( 0.2 )

# <codecell>

trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

# <codecell>

print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
#print trndata['input'][0], trndata['target'][0], trndata['class'][0]

# <codecell>

fnn = buildNetwork( trndata.indim, trndata.indim*2, trndata.outdim, outclass=SoftmaxLayer ,hiddenclass = SigmoidLayer)

# <codecell>

trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, lrdecay = 1.0, verbose=True, weightdecay=0.001)

# <codecell>

for i in range(0,3):
    trainer.trainEpochs(1)

# <codecell>

trainer.testOnClassData()[:20],list(targets_train[:20])

# <codecell>

trnresult = percentError( trainer.testOnClassData(),trndata['class'] )
tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )
print "epoch: %4d" % trainer.totalepochs, \
       "  train error: %5.2f%%" % trnresult, \
       "  test error: %5.2f%%" % tstresult

# <codecell>

# Now predict the value of the digit on the second half:
targets_train_validation = classifier.predict(zoomed_features_train[n_training:])

print "Confusion matrix:\n%s" % metrics.confusion_matrix(targets_train[n_training:], targets_train_validation)

print "\n\nMisclassifications: %g %%" % (sum(targets_train_validation != targets_train[n_training:])/float(len(targets_train_validation))*100)

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

