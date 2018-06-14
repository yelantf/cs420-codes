# coding: utf-8

import cv2
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

# keep the largest connected componets and remove others.
def largest_cc(datafile,ratio=1.0):

    # read data
    data=np.fromfile(datafile,dtype=np.uint8)
    fig_w=45
    data=data.reshape(-1,fig_w,fig_w)
    data_num=data.shape[0]

    # convert to binary image
    data255=data.copy()
    data255[data>0]=255

    # process each image
    for ind in xrange(data_num):
        # compute all connected components.
        _,markers=cv2.connectedComponents(data255[ind])
        components=[markers==markid for markid in np.unique(markers)[1:]]

        # calculate the area of each cc and keep the largest one.
        comparea=np.sum(components,axis=(1,2))
        maxarea=comparea.max()
        for i in xrange(len(components)):
            if comparea[i]<maxarea*ratio:
                data[ind][components[i]]=0
    
    # save data
    strlst=datafile.split('_')
    strlst.insert(-1,"cc{}".format(ratio))
    newdatafile='_'.join(strlst)
    data.ravel().tofile(newdatafile)
    return newdatafile

def crop_center(datafile,newfig_w=45):

    # read data
    data=np.fromfile(datafile,dtype=np.uint8)
    fig_w=45
    data=data.reshape(-1,fig_w,fig_w)
    data_num=data.shape[0]

    # generate empty images
    newdata=np.zeros([data_num,newfig_w,newfig_w],dtype=np.uint8)
    
    # crop and pad for each image
    for ind in xrange(data_num):

        # find the bounding box of largest cc.
        imgdata=data[ind]
        M,N=np.where(imgdata>0)
        top,bot=M.min(),M.max()
        left,right=N.min(),N.max()

        # crop the largest cc out.
        newimg=imgdata[top:bot+1,left:right+1]
        height=bot-top+1
        width=right-left+1

        if height>newfig_w:
            # if larger than new width, crop the center part
            height_s=(height-newfig_w)/2
            newimg=newimg[height_s:height_s+newfig_w,:]
        else:
            # ifl less than the new width, pad to the new width
            pad_pre=(newfig_w-height)/2
            pad_post=newfig_w-height-pad_pre
            padtup=((pad_pre,pad_post),(0,0))
            newimg=np.pad(newimg,padtup,'constant')
        
        if width>newfig_w:
            width_s=(width-newfig_w)/2
            newimg=newimg[:,width_s:width_s+newfig_w]
        else:
            pad_pre=(newfig_w-width)/2
            pad_post=newfig_w-width-pad_pre
            padtup=((0,0),(pad_pre,pad_post))
            newimg=np.pad(newimg,padtup,'constant')
        newdata[ind]=newimg
    strlst=datafile.split('_')
    strlst.insert(-1,"crop{}".format(newfig_w))
    newdatafile='_'.join(strlst)
    newdata.ravel().tofile(newdatafile)
    return newdatafile

def location_data(datafile):
    data=np.fromfile(datafile,dtype=np.uint8)
    fig_w=45
    data=data.reshape(-1,fig_w,fig_w)
    data_num=data.shape[0]
    newdata=np.zeros([data_num,4],dtype=np.float32)
    for ind in xrange(data_num):
        imgdata=data[ind]
        M,N=np.where(imgdata>0)
        top,bot=M.min(),M.max()
        left,right=N.min(),N.max()
        newdata[ind][0]=(left+right)/2.0
        newdata[ind][1]=(top+bot)/2.0
        newdata[ind][2]=right-left+1
        newdata[ind][3]=bot-top+1
    strlst=datafile.split('_')
    strlst[-1]="loc"
    newdatafile='_'.join(strlst)
    newdata.ravel().tofile(newdatafile)
    return newdatafile

traindata="mnist/mnist_train/mnist_train_data"
newtrain=largest_cc(traindata)
crop_center(newtrain)

testdata="mnist/mnist_test/mnist_test_data"
newtest=largest_cc(testdata)
crop_center(newtest)

location_data(newtrain)

location_data(newtest)



