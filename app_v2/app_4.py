import streamlit as st
from streamlit_webrtc import webrtc_streamer
import mediapipe as mp
import time
import logging
import queue
import cv2
import av
import pickle
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

logger = logging.getLogger(__name__)
logging.basicConfig(filename='myapp.log', filemode='w', level=logging.INFO, datefmt='%H:%M:%S')
labels_dict = {0: 'Switch ON', 1: 'Switch OFF', 2: 'Adjust',
               3: 'MicroWave', 4: 'Light', 5: 'TV', 6: 'AirCondition',
               7: 'TeaPot', 8: 'Curtain', 9: 'Music'}

st.title("Hello world")
logger.info("START")

mpHands = mp.solutions.hands            # подключаем раздел распознавания рук
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

#result_queue: "queue.Queue[List[Detection]]" = queue.Queue()
points = [0 for i in range(42)]
f = [0 for i in range(5)]
switch_on = None
start_session = 0
flag = None
message = ''


def diff(point1, point2):
    return abs(point1 - point2)


def msg_recognition(msg, timeout=7):
    global flag, start_session, message
    global switch_on

    if flag:
        if msg == "#" or (time.time() - start_session) > timeout:
            flag = False
            message = ''
            switch_on = None
        else:
            logger.info(f"AAAAAAAAAA {f}")
            if get_act() != None:
                if not f[1] and not f[2] and not f[3] and f[4] and diff(points[5][0], points[4][0]) > 15:
                    message_operation("Teapot")
                elif diff(points[12][0], points[16][0]) > 20 and f[1] and f[2] and f[3] and f[4] and (diff(points[8][0],points[12][0]) < 4) and (diff(points[16][0], points[20][0]) < 4):
                    message_operation("Music")
                elif diff(points[4][0], points[5][0]) > 30 and f[1] and f[2] and f[3] and f[4]:
                    message_operation("Curtain")
                elif f[2] and f[3] and diff(points[12][0], points[16][0])<4\
                        and diff(points[8][0], points[12][0]) > 10\
                        and diff(points[16][0], points[20][0]) > 10:
                    message_operation("Microwave ovn")
                logger.info(f"BBBBBBBBBB {f}")
            else:
                pass

    else:
        if msg == "$":
            flag = True
            start_session = time.time()


def get_act():
    global switch_on
    
    if switch_on == None:
        if f[0] and not f[1] and not f[2] and not f[3] and not f[4] and points[4][1] < points[6][1]:
            switch_on = True
        elif f[0] and not f[1] and not f[2] and not f[3] and not f[4] and points[4][1] > points[6][1]:
            switch_on = False
    return switch_on 


def message_operation(info):
    global switch_on
    global message
    if switch_on:
        logger.info(f"Act ON Info: {info}")
        message = f"Switch ON {info}"
    else:
        logger.info(f"Act OFF Info: {info}")
        message = f"Switch OFF {info}"
    #switch_on = None


def video_frame_callback(frame):
    global flag
    global message
    
    img = frame.to_ndarray(format="bgr24")
    height, width, color = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    data_aux = []
    x_ = []
    y_ = []
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            for id, point in enumerate(handLms.landmark):
                width, height = int(point.x), int(point.y)
                points[2*id] = width
                points[2*id+1] = height

        prediction = model.predict([np.asarray(points)])
        logger.info(f"AcAAAAAAaA: {prediction}")
        predicted_character = labels_dict[int(prediction[0])]

    #if flag:
        cv2.putText(img, predicted_character, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 0, 0), 1)

    #result_queue.put(img)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="sample",
                video_frame_callback=video_frame_callback,
                media_stream_constraints={"video": True, "audio": False})

