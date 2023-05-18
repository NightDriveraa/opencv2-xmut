import cv2
import os
import numpy as np
from util import *

face_cascade = cv2.CascadeClassifier('resource/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 从摄像头获取人脸数据
def save_face_data(name):
    cap = cv2.VideoCapture(0) # 打开摄像头

    count = 0 # 用于统计采样次数

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 转为灰度图

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 检测灰度图中的人脸

        for (x, y, w, h) in faces:
            # 在摄像头画面中用矩形框标记人脸
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 从灰度图中提取人脸并保存到数据集文件夹
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            cv2.imwrite("dataset/" + name + '_' + str(count) + ".jpg", roi_gray)

            count += 1

        cv2.imshow('frame', frame)

        # 按下q或保存30张人脸后退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count >= 30:
            break

    # 释放所有窗口
    cap.release()
    cv2.destroyAllWindows()


# 加载本地图片作为人脸数据
def save_face_data_2(image_path, name):

    # 检测数据集文件夹是否存在，不在则创建
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    # 读取图片并转化为灰度图
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测人脸位置
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    count = 0

    # 提取人脸灰度图并进行存储
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        cv2.imwrite("dataset/" + name + '_' + str(count) + ".jpg", roi_gray)

        count += 1

    print(str(count) + " face(s) saved")


# 训练人脸识别器
def train():
    dataset_path = 'dataset'

    faces = []
    labels = []

    # 遍历数据集文件夹
    for dirName, subdirList, fileList in os.walk(dataset_path):
        for fileName in fileList:
            if fileName.endswith(".jpg"):
                path = os.path.join(dirName, fileName)

                # 读取数据集标签(即输入的Name)
                label = int(fileName.split('_')[0])

                # 图片读取/灰度转换
                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # 添加人脸数据列表与标签列表
                faces.append(gray)
                labels.append(label)

    recognizer.train(faces, np.array(labels))
    recognizer.save('recognizer/trainingData.yml')

# 加载训练好的识别器模型
def recognize():
    recognizer.read('recognizer/trainingData.yml')

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # 检测摄像头画面中的人脸
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # 使用矩形框标记摄像头中的人脸
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # 使用识别器模型对人脸灰度图进行识别，返回可能对应的标签与置信度
            label, confidence = recognizer.predict(roi_gray)

            # 输出标签与置信度
            print("Label: " + str(label))
            print("Confidence: " + str(confidence))

            # 置信度低于80才判定为准入
            if confidence < 80:
                cv2.putText(frame, "Access Granted", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Access Denied", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('frame', frame)

        # 按q可退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放所有窗口
    cap.release()
    cv2.destroyAllWindows()

while True:
    print("1. Save face data")
    print("2. Train recognizer")
    print("3. Recognize face")
    print("q. Quit")

    choice = input("Enter choice: ")

    if choice == '1':
        # 输入本地图片路径用于训练
        # image_path = input("Enter image path: ")
        # name = input("Enter name: ")
        # save_face_data_2(image_path, name)

        # 摄像头捕获图片用于训练
        name = input("Enter name: ")
        save_face_data(name)

    elif choice == '2':
        train()
    elif choice == '3':
        recognize()
    elif choice == 'q':
        break
    else:
        print("Invalid choice")