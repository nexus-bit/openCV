import numpy as np
import cv2
import time
import os

# 어떤 키(전진,좌회전,우회전)이 눌렸는지를 체크하고 행렬형태로 반환하는 함수
def keys_to_output(keys):
    # [A,W,D]
    output = [0,0,0]
    if(keys == 97):
        output[0] = 1
    elif (keys == 119):
        output[2] = 1
    else:
        output[1] = 1

    return output


file_name = 'training_data.npy'

# 이미 파일이 존재하면 기존데이터를 불러오고
if os.path.isfile(file_name):
    print('File exist, loading previous data!')
    training_data = list(np.load(file_name, allow_pickle=True))
else:
    print('File does not exist, starting fresh')
    training_data = []
 
 
 
def main():
    #last_time = time.time()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
    # 무한루프를 돌면서
        sucess, frame = cap.read()
        if sucess:   
            height, width = frame.shape[:2]
            pt3 = (int(width), int(height))
            pt4 = (0, 0)
            cv2.rectangle(frame, pt4, pt3, (0, 255, 0))
            
            screen = frame
            screen = cv2.resize(screen, (80,60))
            cv2.imshow('Camera Capture', screen)
            # 추출한 영상을 흑백으로 한 다음 80x60 사이즈로 축소시킨다
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            
    
            # 조작하는 사람이 어떤 키를 누르는지 지켜보다가
            keys = cv2.waitKey(1)
            # 특정키를 누르면 [0,1,0] 같은 행렬로 반환한다
            if(keys == 27):
                break
            output = keys_to_output(keys)
            # 그리고 학습데이터에 저장한다
            training_data.append([screen, output])
    
        
            # While 루프의 속도를 체크하는 코드
            #print('Frame took {} seconds'.format(time.time()-last_time))
            #last_time = time.time()
    
    
            # 500번의 루프마다 file.npy에 저장한다
            if len(training_data) % 500 == 0:
                print(len(training_data))
                np.save(file_name, training_data)
        

    cap.release()
    cv2.destroyAllWindows() 
main()
