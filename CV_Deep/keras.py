import numpy as np
import tensorflow
import tensorflow.python.keras
import tensorflow.keras

from tensorflow.keras.models import Sequential 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional import MaxPooling2D
from tensorflow.python.keras.layers.convolutional import ZeroPadding2D
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt




WIDTH = 80
HEIGHT = 60
LR = 1e-3      # Learning Rate
EPOCHS = 8     
 
# 모델의 이름을 아래와 같이 Learning_Rate, 이름, Epochs 정보를 추가해서 알아보기 쉽게 한다
MODEL_NAME = 'pygta5-car-{}-{}-{}-epoch'.format(LR, 'alexnetv2', EPOCHS)
 
 

 
# 학습데이터를 불러온 다음
train_data = np.load('training_data_v2.npy', allow_pickle=True)

# 원하는 크기로 Test, Train 데이터를 나누고
train = train_data[:-100]
test = train_data[-100:]

# 데이터(영상)와 정답(키보드데이터)를 분리한다
X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = [i[1] for i in train]
 
test_X = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_Y = [i[1] for i in test]

'''
val_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.2, height_shift_range = 0.2,
zoom_range=0.2, horizontal_flip= True, vertical_flip=True)

val_generator = val_datagen.flow_from_directory(
'C:\\Projects\\keras_talk\\test_set',
target_size=(224, 224),
batch_size=128,
class_mode='binary')
'''
#알렉스넷 모델 생성
model = Sequential()

#Alexnet - 계층 1 : 11x11 필터를 96개를 사용, strides = 4, 활화화함수 = relu,
# 입력 데이터 크기 224x224 , 3x3 크기의 풀리계층 사용

model.add(Conv2D(96, (11,11), strides=4, input_shape=(80,60,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3),strides=1))
model.add(BatchNormalization())

#Alexnet - 계층 2 : 5X5 필터를 256개 사용 , strides = 1, 활화화함수 = relu, 3x3 크기의 풀리계층 사용
model.add(ZeroPadding2D(2))
model.add(Conv2D(256,(5,5), strides=1, activation='relu'))

model.add(MaxPooling2D(pool_size=(3,3),strides=1))
model.add(BatchNormalization())

#Alexnet - 계층 3 : 3x3 필터를 384개 사용, strides =1 , 활성화함수 = relu
model.add(ZeroPadding2D(1))
model.add(Conv2D(384,(3,3), strides=1, activation='relu'))


#Alexnet - 계층 4 : 3x3 필터를 384개 사용, strides =1 , 활성화함수 = relu
model.add(ZeroPadding2D(1))
model.add(Conv2D(384,(3,3), strides=1, activation='relu'))


#Alexnet - 계층 5 : 3x3 필터를 256개 사용, strides =1 , 활성화함수 = relu, 3x3 크기의 풀리계층 사용
model.add(ZeroPadding2D(1))
model.add(Conv2D(256,(3,3), strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=2))

#계산을 위해서 1차원 배열로 전환
model.add(Flatten())

#Alexnet - 계층 6 : 4096개의 출력뉴런, 활성화함수 = relu
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

#Alexnet - 계층 7 : 4096게의 출력뉴런, 활성화함수 = relu
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

#Alexnet - 계층 8 : 1개의 출력뉴런, 활성화함수 = sigmoid
model.add(Dense(1, activation='sigmoid'))

#학습과정 설정 - 손실함수는 크로스엔트로피, 가중치 검색은 아담
sgd = SGD(lr=LR,decay=5e-4, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer='sgd',metrics=['accuracy'])

model.summary()


#Alexnet - 학습하기
hist = model.fit(X, Y, epochs=EPOCHS,batch_size=len(X))
print(len(X))
#모델 저장하기
model.save('Alexnet2.h5')
