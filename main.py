import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def getDistance(coordinates):
    distance = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[1]))
    return distance

def setVolume(distance):
    volMin, volMax = volume.GetVolumeRange()[:2]
    vol = np.interp(distance, [15, 220], [volMin, volMax])
    volume.SetMasterVolumeLevel(vol, None)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mpHands = mp.solutions.hands
indexes = [4, 8]

with mpHands.Hands(
    model_complexity = 0,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as hands:
    while cap.isOpened():
        succes, image = cap.read()

        if not succes:
            print('Ignoring empty camera frame.')
            continue #If it's a video use break

        height, width, _ = image.shape
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        coordinates = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for index in indexes:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates.append([x, y])

                    cv2.circle(image, (x, y), 2, (0, 255, 255), 2)
                    cv2.circle(image, (x, y), 1, (128, 0, 250), 2)

            distance = getDistance(coordinates)
            setVolume(distance)
                
        cv2.imshow('Volume Controller', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()