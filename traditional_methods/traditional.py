
# coding: utf-8

# In[1]:

import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsOneClassifier
import time


# In[2]:

train_data=np.fromfile("./mnist/mnist_train/mnist_train_data",dtype=np.uint8)
train_label=np.fromfile("./mnist/mnist_train/mnist_train_label",dtype=np.uint8)
test_data=np.fromfile("./mnist/mnist_test/mnist_test_data",dtype=np.uint8)
test_label=np.fromfile("./mnist/mnist_test/mnist_test_label",dtype=np.uint8)


# In[3]:

fig_w=45
train_data=train_data.reshape(-1,fig_w*fig_w)
test_data=test_data.reshape(-1,fig_w*fig_w)
print train_data.shape
print test_data.shape


# In[4]:

train_data_norm=train_data/255.
test_data_norm=test_data/255.


# In[6]:

pca=PCA(n_components=120)
train_data_reduce=pca.fit_transform(train_data_norm)
test_data_reduce=pca.transform(test_data_norm)


# In[9]:

accuList=[]
runtimeList=[]
for kernel in ['linear','poly','rbf','sigmoid']:
    print kernel,
    tic=time.time()
    clf=OneVsOneClassifier(SVC(kernel=kernel),n_jobs=20)
    clf.fit(train_data_reduce,train_label)
    toc=time.time()
    print 'train time:',toc-tic,
    runtimeList.append(toc-tic)
    tic=time.time()
    test_pred=clf.predict(test_data_reduce)
    toc=time.time()
    print 'test time:',toc-tic,
    runtimeList[-1]+=toc-tic
    accu=(test_pred==test_label).sum()/float(len(test_label))
    print accu
    accuList.append(accu)


# In[19]:

train_data_half_aug=np.fromfile("./mnist/mnist_train/mnist_train_cc1.0_data",dtype=np.uint8)
test_data_half_aug=np.fromfile("./mnist/mnist_test/mnist_test_cc1.0_data",dtype=np.uint8)
train_data_aug=np.fromfile("./mnist/mnist_train/mnist_train_cc1.0_crop45_data",dtype=np.uint8)
test_data_aug=np.fromfile("./mnist/mnist_test/mnist_test_cc1.0_crop45_data",dtype=np.uint8)


# In[20]:

fig_w=45
train_data_half_aug=train_data_half_aug.reshape(-1,fig_w*fig_w)
test_data_half_aug=test_data_half_aug.reshape(-1,fig_w*fig_w)
train_data_aug=train_data_aug.reshape(-1,fig_w*fig_w)
test_data_aug=test_data_aug.reshape(-1,fig_w*fig_w)


# In[21]:

train_data_half_aug=train_data_half_aug/255.
test_data_half_aug=test_data_half_aug/255.
train_data_aug=train_data_aug/255.
test_data_aug=test_data_aug/255.


# In[22]:

pca=PCA(n_components=120)
train_data_reduce=pca.fit_transform(train_data_half_aug)
test_data_reduce=pca.transform(test_data_half_aug)


# In[23]:

tic=time.time()
clf=OneVsOneClassifier(SVC(kernel='rbf'),n_jobs=20)
clf.fit(train_data_reduce,train_label)
toc=time.time()
print 'train time:',toc-tic,
tic=time.time()
test_pred=clf.predict(test_data_reduce)
toc=time.time()
print 'test time:',toc-tic,
accu=(test_pred==test_label).sum()/float(len(test_label))
print accu


# In[13]:

pca=PCA(n_components=120)
train_data_reduce=pca.fit_transform(train_data_aug)
test_data_reduce=pca.transform(test_data_aug)


# In[15]:

tic=time.time()
clf=OneVsOneClassifier(SVC(kernel='rbf'),n_jobs=20)
clf.fit(train_data_reduce,train_label)
toc=time.time()
print 'train time:',toc-tic,
tic=time.time()
test_pred=clf.predict(test_data_reduce)
toc=time.time()
print 'test time:',toc-tic,
accu=(test_pred==test_label).sum()/float(len(test_label))
print accu

