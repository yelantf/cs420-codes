
# coding: utf-8

# In[13]:

import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsOneClassifier
import time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
get_ipython().magic(u'matplotlib inline')


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


# In[5]:

train_data_norm=train_data/255.
test_data_norm=test_data/255.


# In[4]:

for i in xrange(20,401,20):
    pca=PCA(n_components=i)
    train_data_reduce=pca.fit_transform(train_data)
    test_data_reduce=pca.transform(test_data)
    print i,
    tic=time.time()
    clf=OneVsOneClassifier(LinearSVC(),n_jobs=10)
    clf.fit(train_data_reduce,train_label)
    toc=time.time()
    print 'train time:',toc-tic,
    tic=time.time()
    test_pred=clf.predict(test_data_reduce)
    toc=time.time()
    print 'test time:',toc-tic,
    print (test_pred==test_label).sum()/float(len(test_label))


# In[7]:

accuList=[]
runtimeList=[]
for i in xrange(5,201,5):
    pca=PCA(n_components=i)
    train_data_reduce=pca.fit_transform(train_data_norm)
    test_data_reduce=pca.transform(test_data_norm)
    print i,
    tic=time.time()
    clf=OneVsOneClassifier(LinearSVC(),n_jobs=10)
    clf.fit(train_data_reduce,train_label)
    toc=time.time()
    print 'train time:',toc-tic,
    runtimeList.append(toc-tic)
    tic=time.time()
    test_pred=clf.predict(test_data_reduce)
    toc=time.time()
    print 'test time:',toc-tic,
    accu=(test_pred==test_label).sum()/float(len(test_label))
    accuList.append(accu)
    runtimeList[-1]+=toc-tic
    print accu


# In[8]:

for i in xrange(205,251,5):
    pca=PCA(n_components=i)
    train_data_reduce=pca.fit_transform(train_data_norm)
    test_data_reduce=pca.transform(test_data_norm)
    print i,
    tic=time.time()
    clf=OneVsOneClassifier(LinearSVC(),n_jobs=10)
    clf.fit(train_data_reduce,train_label)
    toc=time.time()
    print 'train time:',toc-tic,
    runtimeList.append(toc-tic)
    tic=time.time()
    test_pred=clf.predict(test_data_reduce)
    toc=time.time()
    print 'test time:',toc-tic,
    accu=(test_pred==test_label).sum()/float(len(test_label))
    accuList.append(accu)
    runtimeList[-1]+=toc-tic
    print accu


# In[14]:

plt.figure()
plt.plot(np.arange(5,251,5),accuList)
plt.xlabel('dimension')
plt.ylabel('accuracy')
plt.savefig('pca-accu.pdf')


# In[16]:

plt.figure()
plt.plot(np.arange(5,251,5),runtimeList,color='coral')
plt.xlabel('dimension')
plt.ylabel('runtime')
plt.savefig('pca-runtime.pdf')


# In[ ]:



