import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import ImageTk, Image
import os
import numpy as np
from util.dbutil import *

face_cascade = cv2.CascadeClassifier('resource/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


class SaveFaceDataWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.title("Save Face Data")
        self.geometry("400x200")
        self.create_widgets()

    def create_widgets(self):
        self.label1 = tk.Label(self, text="姓名:")
        self.label1.pack(side="top")
        self.entry1 = tk.Entry(self)
        self.entry1.pack(side="top")


        # self.button1 = tk.Button(self, text="Select image", command=self.select_image)
        # self.button1.pack(side="top")

        self.button2 = tk.Button(self, text="Save", command=self.save_face_data)
        self.button2.pack(side="top")

    # def select_image(self):
    #     self.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
    #                                                filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    #     self.label2 = tk.Label(self, text="Selected image: {}".format(self.filename))
    #     self.label2.pack(side="top")

    def save_face_data(self):
        # 获取输入的姓名
        user_name = self.entry1.get()
        # 插入输入库，并获取自增后的主键
        user = User(0,user_name,1,None)
        user_id = User.insertUser(user)

        cap = cv2.VideoCapture(0)  # 打开摄像头

        count = 0  # 用于统计采样次数

        while True:
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 检测灰度图中的人脸

            for (x, y, w, h) in faces:
                # 在摄像头画面中用矩形框标记人脸
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 从灰度图中提取人脸并保存到数据集文件夹
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                cv2.imwrite("dataset/" + str(user_id) + '_' + str(count) + ".jpg", roi_gray)

                count += 1

            cv2.imshow('frame', frame)

            # 按下q或保存30张人脸后退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if count >= 30:
                break

        # 保存用户彩照
        cv2.imwrite("UserData/image/" + str(user_id) + ".jpg", roi_color)
        user = User(user_id,user_name,1,"dataset/" + str(user_id) + ".jpg")
        User.updateUser(user)

        # 释放所有窗口
        cap.release()
        cv2.destroyAllWindows()

        self.label3 = tk.Label(self, text="Face data saved successfully.")
        self.label3.pack(side="top")

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.label1 = tk.Label(self, text="1. Save face data")
        self.label1.pack(side="top")
        self.button1 = tk.Button(self, text="Open image", command=self.open_image)
        self.button1.pack(side="top")
        self.button2 = tk.Button(self, text="Save face data", command=self.open_save_face_data_window)
        self.button2.pack(side="top")

        self.label2 = tk.Label(self, text="2. Train recognizer")
        self.label2.pack(side="top")
        self.button3 = tk.Button(self, text="Train", command=self.train)
        self.button3.pack(side="top")

        self.label3 = tk.Label(self, text="3. Recognize face")
        self.label3.pack(side="top")
        self.button4 = tk.Button(self, text="Start recognition", command=self.recognize)
        self.button4.pack(side="top")

        self.quit = tk.Button(self, text="Quit", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    # 打开本地图片
    def open_image(self):
        self.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                                   filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.img = Image.open(self.filename)
        self.img = self.img.resize((350, 350), Image.ANTIALIAS)
        self.imgtk = ImageTk.PhotoImage(self.img)
        self.panel = tk.Label(self, image=self.imgtk)
        self.panel.image = self.imgtk
        self.panel.pack(side="bottom")

    # 从本地图片中提取人脸并保存到文件夹中
    def save_face_data(self):
        name = input("Enter name: ")
        cap = cv2.VideoCapture(0)  # 打开摄像头

        count = 0  # 用于统计采样次数

        while True:
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图

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
        print("Face data saved successfully.")
        # else:
        #     print("Please open an image first.")

    # 训练人脸识别器
    def train(self):
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
        print("Training completed.")

    # 开始人脸识别
    def recognize(self):
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

    def open_save_face_data_window(self):
        SaveFaceDataWindow(self)

root = tk.Tk()
root.geometry("800x600+{}+{}".format(int(root.winfo_screenwidth()/2-400), int(root.winfo_screenheight()/2-300)))
app = Application(master=root)
app.mainloop()