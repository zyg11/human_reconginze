from keras.models import load_model
from PIL import Image
import numpy as np
from keras.utils import np_utils

img_rows,img_cols=57 , 47
nb_classes=40
"""
加载图像数据的函数,dataset_path即图像olivettifaces的路径
加载olivettifaces后，划分为train_data,valid_data,test_data三个数据集
函数返回train_data,valid_data,test_data以及对应的label
图片大小是1190*942，一共有20*20张人脸，故每张人脸大小是（1190/20）*（942/20）即57*47=2679
"""
def load_data(dataset_path):
    img=Image.open(dataset_path)
    img_ndarray=np.array(img,dtype='float64')/256
    faces=np.empty((400,2679))
    for row in range(20):#行
        for column in range(20):#列
            faces[row*20+column]=np.ndarray.flatten(img_ndarray[row*57:(row+1)*57,column*47:(column+1)*47])
    label=np.empty(400)#标记
    for i in range(400):
        label[i*10:i*10+10]=i
    label=label.astype(np.int)
    #分成训练集、验证集、测试集，大小如下
    train_data=np.empty((320,2679))
    train_lable=np.empty((320))
    valid_data=np.empty((40,2679))
    valid_label=np.empty((40))
    test_data=np.empty((40,2679))
    test_label=np.empty((40))

    for i in range(40):#64×64
        train_data[i*8:i*8+8]=faces[i*10:i*10+8]# 320/40=8 前8个
        train_lable[i*8:i*8+8]=label[i*10:i*10+8]# 320/40=8 前8个
        valid_data[i]=faces[i*10+8]# 40/40=1 第9个
        valid_label[i]=label[i*10+8]
        test_data[i]=faces[i*10+9] #40/40=1 第9个
        test_label[i]=label[i*10+9]
    rval=[(train_data,train_lable),(valid_data,valid_label),(test_data,test_label)]
    return rval

(x_train,y_train),(x_val,y_val),(x_test,y_test)=load_data('E:keras_data/face/olivettifaces.gif')
x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
x_val=x_val.reshape(x_val.shape[0],img_rows,img_cols,1)
x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)#报错，此处应该是小括号不是中括号
y_test = np_utils.to_categorical(y_test, nb_classes)

model=load_model('E:/keras_data/face/olivettifaces_model.h5')
score=model.evaluate(x_test,y_test,verbose=1)
print('Test score:',score[0])
print('Test accuracy:',score[1])
