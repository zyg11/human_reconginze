#进行预测
#定义层
import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
#指定图片尺寸
target_size=(47,57)#fixed size for faces

#预测函数
#输入:model,图片,目标尺寸
#输出：预测，predict
def predict(model,img,target_size):
    """Run model prediction on image
    Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
    :param model:
    :param img:
    :param target_size:
    :return:list of predicted labels and their probabilities
    """
    if img.size!=target_size:
        img=img.resize(target_size)
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    # x=preprocess_input(x)#此处需仔细检查
    preds=model.predict(x)
    return preds[0]
#画图函数
# 预测之后画图，这里默认是猫狗，也可以改label

def plot_preds(image,preds):
    """isplays image and the top-n predicted probabilities in a bar graph
    :param image:PIL image
    :param preds:list of predicted labels and their probabilities
    :param labels:
    :return:
    """
    # labels = ("bus", "dinasours", "elephants", "flowers", "horse")
    labels=("0","1","2","3","4","5","6","7","8","9",
            "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
            "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
            "30", "31", "32", "33", "34", "35", "36", "37", "38", "39")
    image=np.array(image)
    plt.imshow(image,cmap='gray')
    plt.axis('off')
    plt.figure()
    y_pos = np.arange(len(labels))
    error = np.random.rand(len(labels))
    plt.barh(y_pos, preds, align='center', alpha=0.5)  # 有问题
    plt.yticks(y_pos, labels)  #
    plt.xlabel('Probability')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()

#载入模型
# model=load_model('E:/keras_data/mnist/first_CNN_model.h5')
model=load_model('E:/keras_data/face/olivettifaces_model.h5')
#本地图片
img=Image.open('187.gif')
preds=predict(model, img, target_size)
print(preds)
plot_preds(img, preds)