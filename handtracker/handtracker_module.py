import time
import cv2
import mediapipe as mp
class handDetector():
    "initializing the variables"
    def __init__(self,mode = False, maxHands = 2, model_complexity = 1,detectionConfidence = 0.5,trackconfidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionconfidence = detectionConfidence
        self.trackconfidence = trackconfidence
        self.model_complexity = model_complexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.model_complexity,self.detectionconfidence,self.trackconfidence)
        self.mpDraw = mp.solutions.drawing_utils
    """just drawing the hands which are being detected"""
    def find_hands(self,img,draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)
        return img
    """will return the positions of landmarks
    if no id_no is given it will return a list of list of all the positions
    otherwise if id_no is given 0-20 then only that id will be returned"""
    def find_position(self,img,handno = 0,id_no = -1,draw = False):
        pos_list = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[handno]
            for id, lm in enumerate(my_hand.landmark):
                if (id_no == -1):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pos_list.append([id,cx,cy])
                    if draw:
                        cv2.circle(img, (cx, cy),
                                   10, (0, 0, 0), cv2.FILLED)
                else:
                    if(id==id_no):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        pos_list.append([cx,cy])
                        if draw:
                            cv2.circle(img, (cx,cy),
                                       10, (0,0,0), cv2.FILLED)
        return pos_list


def main():
    prevtime = 0
    curtime = 0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        list_of_positions = detector.find_position(img,0)
        if(len(list_of_positions)!=0):
            print(list_of_positions)
        curtime = time.time()
        fps = 1 / (curtime - prevtime)
        prevtime = curtime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (255, 255, 0), 3)
        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()