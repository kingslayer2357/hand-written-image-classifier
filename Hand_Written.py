# -*- coding: utf-8 -*-
"""
Created on Fri May 29 23:06:17 2020

@author: kingslayer
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

convert("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
"mnist_train.csv", 60000)
convert("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte",
"mnist_test.csv", 10000)


training_set=pd.read_csv("mnist_train.csv")
test_set=pd.read_csv("mnist_test.csv")


training_set.rename(columns={"5":"labels"},inplace=True)
test_set.rename(columns={"7":"labels"},inplace=True)


train2=training_set.drop(columns=["labels"])
test2=test_set.drop(columns=["labels"])


train2=train2.as_matrix()
plt.imshow(train2[7,:].reshape(28,28))


test2=test2.as_matrix()
plt.imshow(test2[70,:].reshape(28,28))


X_train=train2.astype("uint8")
X_test=test2.astype("uint8")

y_train=training_set["labels"]
y_test=test_set["labels"]

plt.hist(y_train)
plt.hist(y_test)

#Feature Scaling
X_train=X_train/255
X_test=X_test/255


X_train= X_train.reshape((59999,28,28,1))
X_test=X_test.reshape((9999,28,28,1))




#Model
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout


cnn=Sequential()

cnn.add(Conv2D(64,(3,3),activation="relu",input_shape=(28,28,1)))
cnn.add(MaxPooling2D(2,2))
cnn.add(Dropout(0.3))

cnn.add(Conv2D(128,(3,3),activation="relu"))
cnn.add(MaxPooling2D(2,2))
cnn.add(Dropout(0.3))

cnn.add(Flatten())

cnn.add(Dense(units=512,activation="relu"))
cnn.add(Dense(units=512,activation="relu"))
cnn.add(Dropout(0.3))
cnn.add(Dense(units=512,activation="relu"))
cnn.add(Dense(units=10,activation="softmax"))

cnn.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

cnn.fit(X_train,y_train,batch_size=100,epochs=3,validation_split=0.2)

y_pred=cnn.predict_classes(X_test)

#CM
from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred)
report=classification_report(y_test,y_pred)
print(report)

