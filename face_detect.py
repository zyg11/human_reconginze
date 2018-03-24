#olivettifaces人脸数据库，LnNet5的CNN模型
import os
import sys
import time
import numpy as np
# np.random.seed(1337)  # for reproducibility
from PIL import Image
import matplotlib.pyplot as plt

from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.models import load_model
from keras.utils import np_utils
from keras.optimizers import SGD
nb_classes = 40
nb_epoch = 50
batch_size = 16
img_rows,img_cols=57,47
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

def Net_model():
    model=Sequential()
    model.add(Conv2D(32,(3,3),input_shape=(img_rows,img_cols,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.5))最好再全连接层Droupt

    model.add(Flatten())
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes,activation='softmax'))

    sgd=SGD(lr=0.006,momentum=0.9,decay=1e-6,nesterov=True)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        metrics=['accuracy']
    )
    return model

def train_model(model,x_train,y_train,x_val,y_val):
    history_fit=model.fit(
        x_train,y_train,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        validation_data=(x_val,y_val)
    )
    model.save('E:/keras_data/face/olivettifaces_model.h5')
    return history_fit

def test_model(x,y):#是否合理？
    model=load_model('E:/keras_data/face/olivettifaces_model.h5')
    score=model.evaluate(x,y)
    print('Test score:',score[0])
    print('Test accuracy:',score[1])
    return score
#画图函数
def plot_training(history):
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs=range(len(acc))
    plt.plot(epochs,acc,'b')
    plt.plot(epochs,val_acc,'r')
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(epochs,loss,'b')
    plt.plot(epochs,val_loss,'r')
    plt.title('Training and validation loss')
    plt.show()

if __name__ =='__main__':
    (x_train,y_train),(x_val,y_val),(x_test,y_test)=load_data('E:/keras_data/face/olivettifaces.gif')
    x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
    x_val=x_val.reshape(x_val.shape[0],img_rows,img_cols,1)
    x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)#报错，此处应该是小括号不是中括号
    print('x_train_shape',x_train.shape)
    print(x_train.shape[0],'train_samples')
    print(x_val.shape[0],'val_samples')
    print(x_test.shape[0],'test_samples')
    # convert class vectors to binary class matrices
    y_train=np_utils.to_categorical(y_train,nb_classes)
    y_val=np_utils.to_categorical(y_val,nb_classes)
    y_test=np_utils.to_categorical(y_test,nb_classes)

    model=Net_model()
    history_fit=train_model(model,x_train,y_train,x_val,y_val)
    #画图
    plot_training(history_fit)
    score = test_model(x_test, y_test)
    # classes = model.predict_classes(x_test, verbose=1)
    # test_accuracy = np.mean(np.equal(y_test, classes))
    # print("accuarcy:", test_accuracy)
