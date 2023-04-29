import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import pyrealsense2 as rs

bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]


blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

kernel = np.ones((5,5),np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

paintWindow = np.zeros((720,1280,3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), (255,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), (0,255,0), 2)
paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), (0,0,255), 2)
paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), (0,255,255), 2)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)


mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.6)
mpDraw = mp.solutions.drawing_utils

p=rs.pipeline()
p.start()
# cap = cv2.VideoCapture(0)
ret = True
while ret:
    # ret, frame = cap.read()
    frames=p.wait_for_frames()
    bgr_frame, depth_frame = frames.get_color_frame(),frames.get_depth_frame()
    
    frame = np.asanyarray(bgr_frame.get_data())
    depth_frame_np = np.asanyarray(depth_frame.get_data())
    x, y, c = frame.shape
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    framergb = cv2.rectangle(framergb, (40,1), (140,65), (0,0,0), 2)
    framergb = cv2.rectangle(framergb, (160,1), (255,65), (255,0,0), 2)
    framerbg = cv2.rectangle(framergb, (275,1), (370,65), (0,255,0), 2)
    framerbg = cv2.rectangle(framergb, (390,1), (485,65), (0,0,255), 2)
    framerbg = cv2.rectangle(framergb, (505,1), (600,65), (0,255,255), 2)
    cv2.putText(framerbg, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(framerbg, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(framerbg, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(framerbg, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(framerbg, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 1280)
                lmy = int(lm.y * 720)

                landmarks.append([lmx, lmy])


            mpDraw.draw_landmarks(framergb, handslms, mpHands.HAND_CONNECTIONS)
        fore_finger = (landmarks[8][0],landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0],landmarks[4][1])
        cv2.circle(frame, center, 3, (0,255,0),-1)
        print(center[1]-thumb[1])
        if (thumb[1]-center[1]<30):
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

        elif center[1] <= 65:
            if 40 <= center[0] <= 140: # Clear Button
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paintWindow[67:,:,:] = 255
            elif 160 <= center[0] <= 255:
                    colorIndex = 0 # Blue
            elif 275 <= center[0] <= 370:
                    colorIndex = 1 # Green
            elif 390 <= center[0] <= 485:
                    colorIndex = 2 # Red
            elif 505 <= center[0] <= 600:
                    colorIndex = 3 # Yellow
        else :
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(center)

    else:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1

    points = [bpoints, gpoints, rpoints, ypoints]
    # for j in range(len(points[0])):
    #         for k in range(1, len(points[0][j])):
    #             if points[0][j][k - 1] is None or points[0][j][k] is None:
    #                 continue
    #             cv2.line(paintWindow, points[0][j][k - 1], points[0][j][k], colors[0], 2)
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cy = j
                cx = i
                depth_mm = depth_frame_np[cy, cx]
                max_depth = np.max(depth_frame_np)
                # min_depth = np.min(depth_frame_np)
                print ("Depth is ", depth_mm)
                print ("Max is ", max_depth)
                # depth_mm = int((depth_mm/max_depth)*100)
                # print ("Depth_mm", depth_mm)
                # if depth_mm == 0:
                #     depth_mm = 1
                # else:
                #     continue
                if depth_mm<3000:
                    cv2.line(framergb, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                # elif 2000<= depth_mm < 4000:
                #     cv2.line(framergb, points[i][j][k - 1], points[i][j][k], colors[i], 4)
                #     cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 4)
                elif 3000 <= depth_mm < 7000:
                    cv2.line(framergb, points[i][j][k - 1], points[i][j][k], colors[i], 6)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 6)
                # elif 8000 <= depth_mm < 10000:
                #     cv2.line(framergb, points[i][j][k - 1], points[i][j][k], colors[i], 8)
                #     cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 8)
                else:
                    cv2.line(framergb, points[i][j][k - 1], points[i][j][k], colors[i], 10)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 10)
    cv2.imshow("Output", framergb) 
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break

# cap.release()
cv2.destroyAllWindows()