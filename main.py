import os
# os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np
import cv2
from tkinter import *
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
from tkinter import filedialog
import pickle
import math
import dlib
from cv2 import WINDOW_NORMAL

face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'model.h5')

detector = dlib.get_frontal_face_detector() #Face detector
model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
emotions = ["HAPPY", "CONTEMPT", "ANGER", "DISGUST", "FEAR", "SADNESS", "SURPRISE", "NEUTRAL"]

root = Tk()  # create root window
root.title("Basic GUI Layout")  # title of the GUI window
root.geometry("610x500+300+50")
root.resizable(width=True, height=True)
# root.maxsize(1000, 900)  # specify the max size the window can expand to
root.config(bg="skyblue")  # specify background color

# Create left and right frames
left_frame = Frame(root, width=350, height=350, bg='grey')
left_frame.grid(row=0, column=0, padx=10, pady=5)
right_frame = Frame(root, width=260, height=350, bg='grey')
right_frame.grid(row=0, column=1, padx=10, pady=5)
img_f = Frame(right_frame, width=260, height=260)
img_f.grid(row=0, column=1, padx=5, pady=5)
tool_f = Frame(right_frame, width=180, height=185)
tool_f.grid(row=1, column=1, padx=5, pady=5)
tool_bar = Frame(left_frame, width=180, height=185)
tool_bar.grid(row=2, column=0, padx=5, pady=5)
open_i = Frame(left_frame, width=250, height=250)
open_i.grid(row=1, column=0, padx=5, pady=5)
n = tk.StringVar()
monthchoosen = ttk.Combobox(left_frame, width=43, textvariable=n)
monthchoosen['values'] = ('Nhận Dạng Biểu Cảm Bằng CNN',
                          'Nhận Dạng Biểu Cảm Bằng SVM',
                          'So Sánh Nhận Dạng Biểu Cảm Bằng CNN & KNN'
                          )

monthchoosen.grid(row=0, column=0, padx=5, pady=5)
monthchoosen.current(0)


def get_landmarks_with_point(image, frame):
    detections = detector(image, 1)
    #For all detected face instances individually
    for k,d in enumerate(detections):
        #get facial landmarks with prediction model
        shape = model(image, d)
        xpoint = []
        ypoint = []
        for i in range(17, 68):
            # if (i == 27) | (i == 30):
            #     cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), thickness=2)
            xpoint.append(float(shape.part(i).x))
            ypoint.append(float(shape.part(i).y))

        #center points of both axis
        xcenter = np.mean(xpoint)
        ycenter = np.mean(ypoint)
        #Calculate distance between particular points and center point
        xdistcent = [(x-xcenter) for x in xpoint]
        ydistcent = [(y-ycenter) for y in ypoint]

        #prevent divided by 0 value
        if xpoint[11] == xpoint[14]:
            angle_nose = 0
        else:
            #point 14 is the tip of the nose, point 11 is the top of the nose brigde
            angle_nose = int(math.atan((ypoint[11]-ypoint[14])/(xpoint[11]-xpoint[14]))*180/math.pi)

        #Get offset by finding how the nose brigde should be rotated to become perpendicular to the horizontal plane
        if angle_nose < 0:
            angle_nose += 90
        else:
            angle_nose -= 90

        landmarks = []
        for cx,cy,x,y in zip(xdistcent, ydistcent, xpoint, ypoint):
            #Add the coordinates relative to the centre of gravity
            landmarks.append(cx)
            landmarks.append(cy)

            #Get the euclidean distance between each point and the centre point (the vector length)
            meanar = np.asarray((ycenter,xcenter))
            centpar = np.asarray((y,x))
            dist = np.linalg.norm(centpar-meanar)

            #Get the angle the vector describes relative to the image, corrected for the offset that the nosebrigde has when the face is not perfectly horizontal
            if x == xcenter:
                angle_relative = 0
            else:
                angle_relative = (math.atan(float(y-ycenter)/(x-xcenter))*180/math.pi) - angle_nose
                #print(anglerelative)
            landmarks.append(dist)
            landmarks.append(angle_relative)

    if len(detections) < 1:
        #If no face is detected set the data to value "error" to catch detection errors
        landmarks = "error"
    return landmarks

def show_webcam_and_run(model, emotions, window_size=None, window_name='webcam', update_time=10):
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    #Set up some required objects
    vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if vc.isOpened():
        ret, frame = vc.read()
    else:
        print("webcam not found")
        return

    while ret:
        training_data = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)

        #Get Point and Landmarks
        landmarks_vectorised = get_landmarks_with_point(clahe_image, frame)
        #print(landmarks_vectorised)
        if landmarks_vectorised == "error":
            pass
        else:
            #Predict emotion
            training_data.append(landmarks_vectorised)
            npar_pd = np.array(training_data)
            """prediction_emo_set = model.predict_proba(npar_pd)
            if cv2.__version__ != '3.1.0':
                prediction_emo_set = prediction_emo_set[0]
            print(zip(model.classes_, prediction_emo_set))"""
            prediction_emo_set = model.predict_proba(npar_pd)
            prediction_emo = model.predict(npar_pd)
            if cv2.__version__ != '3.1.0':
                prediction_emo = prediction_emo[0]
            label_position = (50, 50)
            cv2.putText(frame, emotions[prediction_emo]+":"+format(max(prediction_emo_set[0])*100, '.2f')+"%", label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(window_name, frame)  #Display the frame
        ret, frame = vc.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):   #Exit program when user press 'q'
            break
def select_video():
    if monthchoosen.get() == "Nhận Dạng Biểu Cảm Bằng CNN":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            _, frame = cap.read()
            labels = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    prediction = classifier.predict(roi)[0]
                    print(prediction)
                    label = emotion_labels[prediction.argmax()]
                    label_position = (x, y)
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Emotion Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    elif monthchoosen.get() == "Nhận Dạng Biểu Cảm Bằng SVM":
        pkl_file = open('models\model1.pkl', 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()
        window_name = 'WEBCAM (press q to exit)'
        show_webcam_and_run(data, emotions, window_size=(800, 600), window_name=window_name, update_time=8)
def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename
def open_img():
    if monthchoosen.get() == "Nhận Dạng Biểu Cảm Bằng CNN":
        x = openfn()
        f = open(x, 'r')
        Label(tool_bar, text=os.path.basename(f.name)).grid(row=1, column=1, padx=5, pady=5)
        image = Image.open(x)
        Label(tool_bar, text=image.size).grid(row=2, column=1, padx=5, pady=5)
        image = image.resize((250, 250), Image.ANTIALIAS)
        Label(tool_bar, text=image.size).grid(row=3, column=1, padx=5, pady=5)
        image = ImageTk.PhotoImage(image)
        tyhoang = Label(open_i, image=image)
        tyhoang.grid(row=0, column=0, padx=5, pady=5)
        tyhoang.image = image
        # tyhoang.pack()

        # Display image in right_frame
        imgs = cv2.imread(x)

        gray = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(imgs, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]

        # convert back to Image object
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(imgs)

        image = im.resize((250, 250), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)
        tyhoang1 = Label(img_f, image=image)
        tyhoang1.grid(row=0, column=0, padx=5, pady=5)
        tyhoang1.image = image
        # tyhoang1.pack()

        Label(tool_f, text="Angry:").grid(row=1, column=0, padx=5, pady=5, ipadx=10)
        Label(tool_f, text="Disgust:").grid(row=2, column=0, padx=5, pady=5)
        Label(tool_f, text="Fear:").grid(row=3, column=0, padx=5, pady=5)
        Label(tool_f, text="Happy:").grid(row=4, column=0, padx=5, pady=5)
        Label(tool_f, text="Neutral:").grid(row=5, column=0, padx=5, pady=5)
        Label(tool_f, text="Sad:").grid(row=6, column=0, padx=5, pady=5)
        Label(tool_f, text="Surprise:").grid(row=7, column=0, padx=5, pady=5)

        Label(tool_f, text=format(prediction[0]*100, '.2f')+"%").grid(row=1, column=1, padx=5, pady=5, ipadx=10)
        Label(tool_f, text=format(prediction[1]*100, '.2f')+"%").grid(row=2, column=1, padx=5, pady=5, ipadx=10)
        Label(tool_f, text=format(prediction[2]*100, '.2f')+"%").grid(row=3, column=1, padx=5, pady=5, ipadx=10)
        Label(tool_f, text=format(prediction[3]*100, '.2f')+"%").grid(row=4, column=1, padx=5, pady=5, ipadx=10)
        Label(tool_f, text=format(prediction[4]*100, '.2f')+"%").grid(row=5, column=1, padx=5, pady=5, ipadx=10)
        Label(tool_f, text=format(prediction[5]*100, '.2f')+"%").grid(row=6, column=1, padx=5, pady=5, ipadx=10)
        Label(tool_f, text=format(prediction[6]*100, '.2f')+"%").grid(row=7, column=1, padx=5, pady=5, ipadx=10)
    elif monthchoosen.get() == "Nhận Dạng Biểu Cảm Bằng SVM":
        pkl_file = open('models\model1.pkl', 'rb')
        model = pickle.load(pkl_file)
        pkl_file.close()
        x = openfn()
        f = open(x, 'r')
        Label(tool_bar, text=os.path.basename(f.name)).grid(row=1, column=1, padx=5, pady=5)
        image = Image.open(x)
        Label(tool_bar, text=image.size).grid(row=2, column=1, padx=5, pady=5)
        image = image.resize((250, 250), Image.ANTIALIAS)
        Label(tool_bar, text=image.size).grid(row=3, column=1, padx=5, pady=5)
        image = ImageTk.PhotoImage(image)
        tyhoang = Label(open_i, image=image)
        tyhoang.grid(row=0, column=0, padx=5, pady=5)
        tyhoang.image = image


        training_data = []
        image = cv2.imread(x)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(gray)

        # Get Point and Landmarks
        landmarks_vectorised = get_landmarks_with_point(clahe_image, image)
        # print(landmarks_vectorised)
        if landmarks_vectorised == "error":
            pass
        else:
            # Predict emotion
            training_data.append(landmarks_vectorised)
            npar_pd = np.array(training_data)
            # prediction_emo_set = model.predict(npar_pd)
            # if cv2.__version__ != '3.1.0':
            #     prediction_emo_set = prediction_emo_set[0]
            # print(zip(model.classes_, prediction_emo_set))
            prediction_emo = model.predict(npar_pd)
            if cv2.__version__ != '3.1.0':
                prediction_emo = prediction_emo[0]
            print(emotions[prediction_emo])

        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



# # Create tool bar frame

# Example labels that serve as placeholders for other widgets
Button(tool_bar, text='open Image', command=open_img).grid(row=0, column=0, padx=5, pady=3,
                                                  ipadx=10)  # ipadx is padding inside the Label widget
Button(tool_bar, text='open Video', command=select_video).grid(row=0, column=1, padx=5, pady=3, ipadx=10)
Label(tool_bar, text="name image:").grid(row=1, column=0, padx=5, pady=5)
Label(tool_bar, text="Size").grid(row=2, column=0, padx=5, pady=5)
Label(tool_bar, text="Resize").grid(row=3, column=0, padx=5, pady=5)
root.mainloop()