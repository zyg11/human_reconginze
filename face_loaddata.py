import os
from PIL import Image
import numpy as np

#读取文件夹下的400张图片，图片为灰度图，所以为1通道，
#如果是将彩色图作为输入,则将1替换为3，
# 并且data[i,:,:,:] = arr改为data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
def load_data():
    # num=4200
    data=np.empty((400,1,57,47),dtype="float32")
    label=np.empty((400,),dtype='uint8')
    imgs=os.listdir("E:/keras_data/face/train")
    num=len(imgs)
    for i in range(num):
        img=Image.open("E:/keras_data/face/train/"+imgs[i])
        arr=np.asarray(img,dtype="float32")
        data[i,:,:,:]=arr
        label[i]=int(imgs[i].split('.')[0])#这里有问题
    data = data.reshape(42000, 28, 28, 1)
    return data,label