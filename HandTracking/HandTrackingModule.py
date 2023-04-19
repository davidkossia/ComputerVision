import cv2
import mediapipe as mp
import time

class handDetector:

    THUMB_LANDMARKS = [1, 2, 3, 4]
    INDEX_LANDMARKS = [5, 6, 7, 8]
    MIDDLE_LANDMARKS = [9, 10, 11, 12]
    RING_LANDMARKS = [13, 14, 15, 16]
    PINKY_LANDMARKS = [17, 18, 19, 20]

    # static image mode used for still images or series of images with different hands
    def __init__(self, static_image_mode=False, no_hands=2, complexity=1, min_detect_con=0.5, min_track_con=0.5):
        self.mode = static_image_mode
        self.num_hands = no_hands
        self.complexity = complexity
        self.min_detect_con = min_detect_con
        self.min_track_con = min_track_con

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.num_hands, self.complexity,
                                        self.min_detect_con, self.min_track_con)
        self.mpDraw = mp.solutions.drawing_utils

    # detect the hands in the image
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                                self.mpHands.HAND_CONNECTIONS)
        return img

    # get list of landmarks for a single hand
    def findPosition(self, img, hand_no=0, draw=True):

        lmList =[]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw and id == 0:
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        return lmList

    def theBird(self, img, draw=True):
        pass
# runs when used as script
# current function: display live camera feed with hand tracing on and
# and print THUMB_TIP (4) landmark (x,y) coordinates to terminal
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        detector.findHands(img)
        lmList = detector.findPosition(img)

        # track specific landark
        if len(lmList)  != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1/(pTime-cTime)
        pTIme = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.namedWindow('mandem learnin innit', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('mandem learnin innit', cv2.flip(img, 1))
        cv2.resizeWindow('mandem learnin innit', 800, 450)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()