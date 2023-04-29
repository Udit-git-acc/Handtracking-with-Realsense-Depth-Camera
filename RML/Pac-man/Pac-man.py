import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from pynput.keyboard import Key, Controller

keyboard = Controller()
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
model = load_model('C:\d_drive\ITR Project\RML\Pac-man\mp_hand_gesture')
f = open('C:\d_drive\ITR Project\RML\Pac-man\gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    x, y, c = frame.shape
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    className = ''
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            className = classNames[classID]
            
            # key mapping
            if className=='thumbs up':
                keyboard.press(Key.up)
                keyboard.release(Key.up)
            elif className=='okay':
                keyboard.press(Key.left)
                keyboard.release(Key.left)
            elif className=='thumbs down':
                keyboard.press(Key.down)
                keyboard.release(Key.down)
            elif className=='peace':
                keyboard.press(Key.right)
                keyboard.release(Key.right)

    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow("Output", frame) 
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()