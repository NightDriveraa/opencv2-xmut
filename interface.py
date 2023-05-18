import tkinter as tk
from tkinter import filedialog, ttk
import tkinter.messagebox as messagebox
import cv2
from PIL import ImageTk, Image, ImageFont, ImageDraw
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
        user = User(None,user_name,1,None)
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

class ModifyUser(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.title("modify_user_info")
        self.geometry("400x300")
        self.create_widgets()

    def create_widgets(self):
        # 在窗口中添加控件
        self.user_id_combobox = ttk.Combobox(self, values=self.get_user_id_list())
        self.user_id_combobox.pack(side="top", padx=10, pady=10)
        self.user_id_combobox.bind("<<ComboboxSelected>>", self.show_user_info)

        self.name_label = tk.Label(self, text="姓名: ")
        self.name_label.pack(side="top", padx=10, pady=5)
        self.new_name_entry = tk.Entry(self)
        self.new_name_entry.pack(side="top", padx=10, pady=5)

        self.allow_label = tk.Label(self, text="是否放行: ")
        self.allow_label.pack(side="top", padx=10, pady=5)
        self.new_allow_entry = tk.Entry(self)
        self.new_allow_entry.pack(side="top", padx=10, pady=5)

        # 保存修改后的信息
        save_button = tk.Button(self, text="Save", command=self.save_info)
        save_button.pack(side="top", padx=10, pady=10)

    def get_user_id_list(self):
        users_id = []
        users = User.getAllUser()
        for user in users:
            users_id.append(user.user_id)
        return users_id

    def show_user_info(self, event):
        # 获取用户选择的 ID
        selected_user_id = self.user_id_combobox.get()

        # 从数据库中读取用户信息
        user_info = self.read_user_info(selected_user_id)

        # 显示用户信息
        self.name_label.config(text="姓名: " + user_info["name"])
        self.new_name_entry.delete(0, tk.END)
        self.new_name_entry.insert(0, user_info["name"])

        self.allow_label.config(text="是否放行: " + user_info["allowed"])
        self.new_allow_entry.delete(0, tk.END)
        self.new_allow_entry.insert(0, user_info["allowed"])

    def read_user_info(self, user_id):
        user_info = User.getUser(user_id)
        if user_info.allowed:
            allow = '是'
        else:
            allow = '否'
        return {"user_id": user_info.user_id, "name": user_info.user_name, "allowed": allow, "image": user_info.image}

    def save_info(self):
        # 获取用户信息
        user_id = self.user_id_combobox.get()
        new_name = self.new_name_entry.get()
        new_allow = self.new_allow_entry.get()
        if new_allow == '是':
            new_allow = 1
        else:
            new_allow = 0

        # 保存修改后的信息
        user = User.getUser(user_id)
        user.user_name = new_name
        user.allowed = new_allow
        User.updateUser(user)
        messagebox.showinfo(title="修改用户信息", message="用户信息已成功更新！")


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.label1 = tk.Label(self, text="1. Save face data",font=("simsun", 18),height=2)
        self.label1.pack(side="top")
        self.button1 = tk.Button(self, text="Open image", command=self.open_image,font=("simsun", 18),height=1)
        self.button1.pack(side="top")
        self.button2 = tk.Button(self, text="Save face data", command=self.open_save_face_data_window,font=("simsun", 18),height=1)
        self.button2.pack(side="top")

        self.label2 = tk.Label(self, text="2. Train recognizer",font=("simsun", 18),height=2)
        self.label2.pack(side="top")
        self.button3 = tk.Button(self, text="Train", command=self.train,font=("simsun", 18),height=1)
        self.button3.pack(side="top")

        self.label3 = tk.Label(self, text="3. Recognize face",font=("simsun", 18),height=2)
        self.label3.pack(side="top")
        self.button4 = tk.Button(self, text="开始门禁识别", command=self.recognize,font=("simsun", 18),height=1)
        self.button4.pack(side="top")

        self.label4 = tk.Label(self, text="置信度阈值:",font=("simsun", 18),height=2)
        self.label4.pack(side="top")
        self.entry1 = tk.Entry(self)
        self.entry1.insert(0, str(70))
        self.entry1.pack(side="top")

        self.button5 = tk.Button(self, text="管理用户", command=self.open_modify_user_window,
                                 font=("simsun", 18), height=1)
        self.button5.pack(side="top")

        self.quit = tk.Button(self, text="Quit", fg="red",
                              command=self.master.destroy,font=("simsun", 18),height=1)
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
        messagebox.showinfo(title="Training complete", message="Training has been completed successfully!")

        print("Training completed.")

    # 开始人脸识别
    def recognize(self):
        recognizer.read('recognizer/trainingData.yml')
        # 打开摄像头
        cap = cv2.VideoCapture(0)

        confidence_threshold = int(self.entry1.get())

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
                # print("Label: " + str(label))
                # print("Confidence: " + str(confidence))

                font_path = "simsun.ttc"
                font = ImageFont.truetype(font_path, 30)

                # 置信度低于70才判定为核准成功
                if confidence < confidence_threshold :
                    # 根据返回的标签获取用户信息
                    print(label)
                    user_info = User.getUser(label)
                    user_name = user_info.user_name


                    # 将图像转换为 PIL.Image 对象
                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # 在 PIL.Image 对象上绘制中文
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((x, y - 40), user_name, font=font, fill=(0, 255, 0))

                    if user_info.allowed:
                        draw.text((x + 60, y + 210), '通行', font=font, fill=(0, 255, 0))
                    else:
                        draw.text((x + 60, y + 210), '禁入', font=font, fill=(255, 0, 0))

                    # 将 PIL.Image 对象转换为图像
                    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                    # cv2.putText(frame, user_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                else:
                    # 将图像转换为 PIL.Image 对象
                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # 在 PIL.Image 对象上绘制中文
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((x, y - 40), '陌生人', font=font, fill=(255, 0, 0))
                    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


            cv2.imshow('frame', frame)

            # 按q可退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                # 释放所有窗口
        cap.release()
        cv2.destroyAllWindows()

    def open_save_face_data_window(self):
        SaveFaceDataWindow(self)

    def open_modify_user_window(self):
        ModifyUser(self)

root = tk.Tk()
root.geometry("800x600+{}+{}".format(int(root.winfo_screenwidth()/2-400), int(root.winfo_screenheight()/2-300)))
app = Application(master=root)
app.mainloop()