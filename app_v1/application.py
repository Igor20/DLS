import streamlit as st
from streamlit_webrtc import webrtc_streamer
import mediapipe as mp
import time
import logging
import cv2
import av

logger = logging.getLogger(__name__)
logging.basicConfig(filename='app.log', filemode='w', level=logging.INFO)

st.title("MVP SmartHome")
logger.info("START")

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

points = [(0, 0) for i in range(21)]
f = [0 for i in range(5)]
switch_on = None
flag = None
message = ''
start_session = 0
prev_msg = ['', 0]


def diff(point1, point2):
    return abs(point1 - point2)


def msg_recognition(msg, timeout=7):
    global flag, start_session, message
    global switch_on, prev_msg

    if flag:
        if msg == "#" or (time.time() - start_session) > timeout:
            prev_msg = ['', 0]
            flag = False
            message = ''
            switch_on = None
        else:
            if get_act() != None:
                if f[1] and f[2] and f[3] and f[4]:   #All fingers
                    if diff(points[12][0], points[16][0]) > 30:
                        message_operation("Music")
                    else:
                        message_operation("Curtain")
                elif 1 in f:                #Part fingers
                    if f[4] and diff(points[5][0], points[4][0]) > 20:
                        message_operation("Teapot")
                    elif not f[2] and f[1] and diff(points[10][0], points[4][0]) > 45:
                        message_operation("Light")
                    elif f[1] and f[2] and diff(points[8][0], points[12][0]) > 15:
                        message_operation("TV")
                else:                       #Close fingers
                    if diff(points[4][0], points[20][0]) < 7 and points[3][1] > points[11][1]:
                        message_operation("MicroWave")
                    elif diff(points[4][1], points[6][1]) < 7:
                        message_operation("AirCondition")
    else:
        if msg == "$":
            flag = True
            switch_on = None
            start_session = time.time()


def get_act():
    global switch_on
    
    if switch_on == None:
        if f[0] and not f[1] and not f[2] and not f[3] and not f[4] and diff(points[8][0], points[2][0]) > 50:
            switch_on = False
        elif f[0] and not f[1] and not f[2] and not f[3] and not f[4]:
            switch_on = True
    return switch_on 


def message_operation(info):
    global switch_on
    global message
    valid_message(info)

    if switch_on:
        logger.info(f"Act ON Info: {info}")
        if prev_msg[1] > 6:
            message = f"Switch ON {info}"
    else:
        logger.info(f"Act OFF Info: {info}")
        if prev_msg[1] > 6:
            message = f"Switch OFF {info}"


def valid_message(info):
    global prev_msg

    if prev_msg[0] != info:
        prev_msg[0] = info
        prev_msg[1] = 0
    else:
        prev_msg[1] += 1


def video_frame_callback(frame):
    global flag
    global message
    
    img = frame.to_ndarray(format="bgr24")
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            for id, point in enumerate(handLms.landmark):
                width, height, color = img.shape
                width, height = int(point.x * height), int(point.y * width)
                points[id] = (width, height)

        ref_position = diff(points[0][1], points[5][1]) * 1.5
        f[0] = 1 if diff(points[4][1], points[17][1]) > ref_position else 0
        f[1] = 1 if diff(points[0][1], points[8][1]) > ref_position else 0
        f[2] = 1 if diff(points[0][1], points[12][1]) > ref_position else 0
        f[3] = 1 if diff(points[0][1], points[16][1]) > ref_position else 0
        f[4] = 1 if diff(points[0][1], points[20][1]) > ref_position else 0

        logger.info(f)
        if not f[0] and f[1] and not f[2] and not f[3] and not f[4]:
            logger.info("Start recv process")
            msg_recognition('$')
        elif not f[0] and f[1] and not f[2] and not f[3] and f[4]:
            logger.info("Stop recv process")
            msg_recognition('#')
        msg_recognition(f)

    if flag:
        cv2.putText(img, message, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 0, 0), 1)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="sample",
                video_frame_callback=video_frame_callback,
                media_stream_constraints={"video": True, "audio": False})

