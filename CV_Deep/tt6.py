from __future__ import division, print_function, absolute_import
import numpy as np
#import tflearn
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import cv2

def alexnet(width, height, lr):
    network = input_data(shape=[None, width, height, 1], name='input')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 3, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=lr, name='targets')
 
    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')
 
    return model


 
# 기본 설정 
WIDTH = 80
HEIGHT = 60
LR = 1e-3      # Learning Rate
EPOCHS = 8     
 
# 모델의 이름을 아래와 같이 Learning_Rate, 이름, Epochs 정보를 추가해서 알아보기 쉽게 한다
MODEL_NAME = 'pygta5-car-{}-{}-{}-epoch.model'.format(LR, 'alexnetv2', EPOCHS)
 
 
# alexnet 객체를 생성한다
model = alexnet(WIDTH, HEIGHT, LR)
 
# 학습데이터를 불러온 다음
train_data = np.load('training_data_v2.npy',allow_pickle=True)

# 원하는 크기로 Test, Train 데이터를 나누고
train = train_data[:-100]
test = train_data[-100:]

# 데이터(영상)와 정답(키보드데이터)를 분리한다
X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = [i[1] for i in train]
 
test_X = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_Y = [i[1] for i in test]
 
 
# 학습을 시작한다
model.fit({'input':X}, {'targets':Y}, n_epoch=EPOCHS, 
        validation_set=({'input':test_X}, {'targets':test_Y}), 
        snapshot_step=100, show_metric=True, run_id=MODEL_NAME)
 
 
# 파이썬스크립트가 있는 경로에 log 폴더를 만들고 
# 학습을 하면서 새로운 cmd창에 아래 명령어를 치면 실시간으로 loss, accuracy 그래프를 확인할 수 있다
# tensorboard --logdir=foo:E:/gitrepo/lockdpwn/python_archive/pygta5/log

cap = cv2.VideoCapture(0)
while cap.isOpened():
    sucess, test = cap.read()
    
    if sucess:
        cv2.imshow('Camera Capture', test)
        screen = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (80,60))
        test_data = np.array([i for i in screen]).reshape(1, 80, 60,1)
        print(model.predict(test_data))
        key = cv2.waitKey(1)
        if(key == 27):
            break
            
cap.release()
cv2.destroyAllWindows() 

# 학습이 끝나면 모델을 저장한다
model.save(MODEL_NAME)