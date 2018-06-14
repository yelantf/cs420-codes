
# coding: utf-8

# In[34]:

import numpy as np
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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


# In[63]:

train_data_half_aug=np.fromfile("./mnist/mnist_train/mnist_train_cc1.0_data",dtype=np.uint8)
test_data_half_aug=np.fromfile("./mnist/mnist_test/mnist_test_cc1.0_data",dtype=np.uint8)
train_data_aug=np.fromfile("./mnist/mnist_train/mnist_train_cc1.0_crop45_data",dtype=np.uint8)
test_data_aug=np.fromfile("./mnist/mnist_test/mnist_test_cc1.0_crop45_data",dtype=np.uint8)


# In[64]:

fig_w=45
train_data=train_data.reshape(-1,fig_w*fig_w)
test_data=test_data.reshape(-1,fig_w*fig_w)
train_data_half_aug=train_data_half_aug.reshape(-1,fig_w*fig_w)
test_data_half_aug=test_data_half_aug.reshape(-1,fig_w*fig_w)
train_data_aug=train_data_aug.reshape(-1,fig_w*fig_w)
test_data_aug=test_data_aug.reshape(-1,fig_w*fig_w)


# In[65]:

train_data_half_aug=train_data_half_aug/255.
test_data_half_aug=test_data_half_aug/255.
train_data_norm=train_data/255.
test_data_norm=test_data/255.
train_data_aug=train_data_aug/255.
test_data_aug=test_data_aug/255.


# In[39]:

checknum=50
alphaList=[]
accuList=[]
runtimeList=[]
for i in xrange(checknum+1):
    alpha=i/float(checknum)
    alphaList.append(alpha)
    print 'alpha:{}'.format(alpha),
    tic=time.time()
    bayes=MultinomialNB(alpha=alpha)
    bayes.fit(train_data_norm,train_label)
    test_pred=bayes.predict(test_data_norm)
    toc=time.time()
    print 'runtime',toc-tic,
    runtimeList.append(toc-tic)
    accu=(test_pred==test_label).sum()/float(len(test_label))
    accuList.append(accu)
    print accu


# In[66]:

tic=time.time()
bayes=MultinomialNB()
bayes.fit(train_data_half_aug,train_label)
test_pred=bayes.predict(test_data_half_aug)
toc=time.time()
print 'runtime',toc-tic,
accu=(test_pred==test_label).sum()/float(len(test_label))
print accu


# In[58]:

tic=time.time()
bayes=MultinomialNB()
bayes.fit(train_data_aug,train_label)
test_pred=bayes.predict(test_data_aug)
toc=time.time()
print 'runtime',toc-tic,
accu=(test_pred==test_label).sum()/float(len(test_label))
print accu


# In[7]:

knn=KNeighborsClassifier(n_neighbors=10,n_jobs=10)
knn.fit(train_data_norm,train_label)
test_pred=knn.predict(test_data_norm)
print (test_pred==test_label).sum()/float(len(test_label))


# In[11]:

tic=time.time()
knn=KNeighborsClassifier(n_neighbors=10,n_jobs=10)
knn.fit(train_data_norm,train_label)
test_pred=knn.predict(test_data_norm)
print (test_pred==test_label).sum()/float(len(test_label))
toc=time.time()
print toc-tic


# In[26]:

pca=PCA(n_components=50)
train_data_reduce=pca.fit_transform(train_data_norm)
test_data_reduce=pca.transform(test_data_norm)


# In[29]:

accuList=[]
runtimeList=[]
for i in xrange(1,51):
    tic=time.time()
    knn=KNeighborsClassifier(n_neighbors=i,n_jobs=10)
    knn.fit(train_data_reduce,train_label)
    test_pred=knn.predict(test_data_reduce)
    toc=time.time()
    print 'k={}'.format(i),
    print 'runtime',toc-tic,
    runtimeList.append(toc-tic)
    accu=(test_pred==test_label).sum()/float(len(test_label))
    accuList.append(accu)
    print accu


# In[32]:

plt.figure()
plt.plot(np.arange(1,51),accuList)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.savefig('k-accu.pdf')


# In[67]:

pca=PCA(n_components=50)
train_data_reduce=pca.fit_transform(train_data_half_aug)
test_data_reduce=pca.transform(test_data_half_aug)


# In[68]:

tic=time.time()
knn=KNeighborsClassifier(n_neighbors=5,n_jobs=10)
knn.fit(train_data_reduce,train_label)
test_pred=knn.predict(test_data_reduce)
toc=time.time()
print 'runtime',toc-tic,
accu=(test_pred==test_label).sum()/float(len(test_label))
print accu


# In[59]:

pca=PCA(n_components=50)
train_data_reduce=pca.fit_transform(train_data_aug)
test_data_reduce=pca.transform(test_data_aug)


# In[60]:

tic=time.time()
knn=KNeighborsClassifier(n_neighbors=5,n_jobs=10)
knn.fit(train_data_reduce,train_label)
test_pred=knn.predict(test_data_reduce)
toc=time.time()
print 'runtime',toc-tic,
accu=(test_pred==test_label).sum()/float(len(test_label))
print accu


# In[33]:

tic=time.time()
treeclf=DecisionTreeClassifier()
treeclf.fit(train_data_norm,train_label)
test_pred=treeclf.predict(test_data_norm)
print (test_pred==test_label).sum()/float(len(test_label))
toc=time.time()
print toc-tic


# In[69]:

tic=time.time()
treeclf=DecisionTreeClassifier()
treeclf.fit(train_data_half_aug,train_label)
test_pred=treeclf.predict(test_data_half_aug)
print (test_pred==test_label).sum()/float(len(test_label))
toc=time.time()
print toc-tic


# In[61]:

tic=time.time()
treeclf=DecisionTreeClassifier()
treeclf.fit(train_data_aug,train_label)
test_pred=treeclf.predict(test_data_aug)
print (test_pred==test_label).sum()/float(len(test_label))
toc=time.time()
print toc-tic


# In[46]:

accuList=[]
runtimeList=[]
for n_model in xrange(5,251,5):
    print '{} trees'.format(n_model),
    tic=time.time()
    forestclf=RandomForestClassifier(n_estimators=n_model,n_jobs=10)
    forestclf.fit(train_data_norm,train_label)
    test_pred=forestclf.predict(test_data_norm)
    toc=time.time()
    print "runtime",toc-tic,
    runtimeList.append(toc-tic)
    accu=(test_pred==test_label).sum()/float(len(test_label))
    print accu
    accuList.append(accu)


# In[49]:

plt.figure()
plt.plot(np.arange(5,251,5),accuList)
plt.xlabel('the number of trees')
plt.ylabel('accuracy')
plt.savefig('forest-accu.pdf')


# In[53]:

plt.figure()
plt.plot(np.arange(5,251,5),runtimeList,color='coral')
plt.xlabel('the number of trees')
plt.ylabel('runtime')
plt.savefig('forest-runtime.pdf')


# In[70]:

tic=time.time()
forestclf=RandomForestClassifier(n_estimators=200,n_jobs=10)
forestclf.fit(train_data_half_aug,train_label)
test_pred=forestclf.predict(test_data_half_aug)
toc=time.time()
print "runtime",toc-tic,
accu=(test_pred==test_label).sum()/float(len(test_label))
print accu


# In[62]:

tic=time.time()
forestclf=RandomForestClassifier(n_estimators=200,n_jobs=10)
forestclf.fit(train_data_aug,train_label)
test_pred=forestclf.predict(test_data_aug)
toc=time.time()
print "runtime",toc-tic,
accu=(test_pred==test_label).sum()/float(len(test_label))
print accu


# In[ ]:



