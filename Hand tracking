@@ -0,0 +1,86 @@
import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize the MediaPipe Hands module
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # Convert image to RGB format (required by MediaPipe)
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # Process hand detection on the image
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Draw landmarks and connections on the image
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                # Calculate the pixel coordinates of the landmark
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if draw:
                    # Draw circles at landmark positions
                    cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

        return lmList

def main():
    pTime = 0  # previous time
    cTime = 0  # current time
    cap = cv.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        # Flip the image horizontally for a more intuitive user experience
        img = cv.flip(img, 1)
        # Perform hand detection on the image
        img = detector.findHands(img)
        # Find the positions of landmarks on the detected hand
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            # Print the position of a specific landmark (e.g., the tip of the index finger)
            print(lmList[4])

        # Calculate and display frames per second (FPS) on the image
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Display the image with landmarks and FPS information
        cv.imshow("Image", img)

        # Exit the loop if the 'q' key is pressed
        if cv.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    main()
 112 changes: 112 additions & 0 deletions112  
Hand_Tracking/VolumeHandControl.py
@@ -0,0 +1,112 @@
import cv2 as cv
import time
import numpy as np
import HandTrackingModule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam = 640
hCam = 480

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# for fps calculation
pTime = 0  # previous time
cTime = 0  # current time

detector = htm.handDetector(detectionCon=0.7)
#dtectionCon = 0.7 means that the hand has to be 70% visible to be detected

# Github pycaw

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
#volume.SetMasterVolumeLevel(0, None)
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    img = cv.flip(img, 1) # to avoid mirror effect, we flip the image
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:
       # print(lmlist[4]) #print the position of the tip of the index finger
        x1, y1 = lmlist[4][1], lmlist[4][2] #x1 and y1 are the coordinates of the tip of the index finger
        x2, y2 = lmlist[8][1], lmlist[8][2] #x2 and y2 are the coordinates of the tip of the middle finger
        cx, cy = (x1+x2)//2, (y1+y2)//2 #cx and cy are the coordinates of the center of the line between the tip of the index finger and the tip of the middle finger

        cv.circle(img, (x1,y1), 15, (255, 0, 255), cv.FILLED) #draw a circle on the tip of the index finger  
        cv.circle(img, (x2,y2), 15, (255, 0, 255), cv.FILLED) #draw a circle on the tip of the middle finger
        cv.line(img, (x1,y1), (x2,y2), (255, 0, 255), 3) #draw a line between the tip of the index finger and the tip of the middle finger
        cv.circle(img, (cx,cy), 15, (255, 0, 255), cv.FILLED) #draw a circle on the center of the line between the tip of the index finger and the tip of the middle finger
        length = math.hypot(x2-x1, y2-y1) #calculate the length of the line between the tip of the index finger and the tip of the middle finger
       # print(length)

        # Hand range 50 - 300
        # Volume range -65 - 0
        vol = np.interp(length, [50,200], [minVol, maxVol]) #interpolate the length of the line between the tip of the index finger and the tip of the middle finger to the volume range
        volBar = np.interp(length, [50,250], [400, 120]) #interpolate the length of the line between the tip of the index finger and the tip of the middle finger to the volume bar range
        volPer = np.interp(length, [50,200], [0, 100]) #interpolate the length of the line between the tip of the index finger and the tip of the middle finger to the volume percentage range

        print(f'length: {int(length)}, vol: {vol}')
        volume.SetMasterVolumeLevel(vol, None)

        if length<50:
            cv.circle(img, (cx,cy), 15, (0, 255, 0), cv.FILLED)
            cv.putText(img,f' Muted', (10, 130), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        if length > 50 and length < 200:
           cv.putText(img,f' {int(volPer)}', (20, 130), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3) #print the volume percentage on the screen

        #adding a volume bar
        cv.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)

        # to make the volume bar move smoothly, we use cv.FILLED and to keep the volume bar in the range of 50 to 250, we use if statement
        if length < 200:
         cv.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv.FILLED)
        else:
          cv.rectangle(img, (50, 150), (85, 400), (255, 0, 0), cv.FILLED)
          cv.putText(img,f' Max Volume', (10, 130), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)



    #fps calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img,f'FPS: {int(fps)}', (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)



    cv.imshow("Image", img)
    if cv.waitKey(1) == ord('q'):
        break


















 86 changes: 86 additions & 0 deletions86  
Main(Final)/HandTrackingModule.py
@@ -0,0 +1,86 @@
import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize the MediaPipe Hands module
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # Convert image to RGB format (required by MediaPipe)
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # Process hand detection on the image
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Draw landmarks and connections on the image
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                # Calculate the pixel coordinates of the landmark
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if draw:
                    # Draw circles at landmark positions
                    cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

        return lmList

def main():
    pTime = 0  # previous time
    cTime = 0  # current time
    cap = cv.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        # Flip the image horizontally for a more intuitive user experience
        img = cv.flip(img, 1)
        # Perform hand detection on the image
        img = detector.findHands(img)
        # Find the positions of landmarks on the detected hand
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            # Print the position of a specific landmark (e.g., the tip of the index finger)
            print(lmList[4])

        # Calculate and display frames per second (FPS) on the image
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Display the image with landmarks and FPS information
        cv.imshow("Image", img)

        # Exit the loop if the 'q' key is pressed
        if cv.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    main()
 166 changes: 166 additions & 0 deletions166  
Main(Final)/ObjectWithGesture(Main).py
@@ -0,0 +1,166 @@
# Main File for Object Detection and Hand Gesture Volume Control
import cv2 as cv
import time
import numpy as np
import HandTrackingModule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import cvlib as cvl
from cvlib.object_detection import draw_bbox
from gtts import gTTS
import pygame


wCam = 640
hCam = 480

cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# for fps calculation
pTime = 0  # previous time
cTime = 0  # current time
detected_objects = set()  # To store the names of detected objects, in object detection

detector = htm.handDetector(detectionCon=0.7) #dtectionCon = 0.7 means that the hand has to be 70% visible to be detected

# Github  repository: " pychaw " used for controling the volume of the device
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()   # Not required for this project
#volume.GetMasterVolumeLevel()  # Not required for this project
volRange = volume.GetVolumeRange()
#volume.SetMasterVolumeLevel(0, None)  # Not required for this project

minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while True:
    # Code starts for volume control-------------------------------------------------------------------------------------------------------------
    success, img = cap.read()
    img = cv.flip(img, 1) # to avoid mirror effect, we flip the image
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:
       # print(lmlist[4]) #print the position of the tip of the index finger
        x1, y1 = lmlist[4][1], lmlist[4][2] #x1 and y1 are the coordinates of the tip of the index finger
        x2, y2 = lmlist[8][1], lmlist[8][2] #x2 and y2 are the coordinates of the tip of the middle finger
        cx, cy = (x1+x2)//2, (y1+y2)//2 #cx and cy are the coordinates of the center of the line between the tip of the index finger and the tip of the middle finger

        cv.circle(img, (x1,y1), 15, (255, 0, 255), cv.FILLED) #draw a circle on the tip of the index finger  
        cv.circle(img, (x2,y2), 15, (255, 0, 255), cv.FILLED) #draw a circle on the tip of the middle finger
        cv.line(img, (x1,y1), (x2,y2), (255, 0, 255), 3) #draw a line between the tip of the index finger and the tip of the middle finger
        cv.circle(img, (cx,cy), 15, (255, 0, 255), cv.FILLED) #draw a circle on the center of the line between the tip of the index finger and the tip of the middle finger
        length = math.hypot(x2-x1, y2-y1) #calculate the length of the line between the tip of the index finger and the tip of the middle finger
       # print(length)

        # Hand range 50 - 300
        # Volume range -65 - 0
        vol = np.interp(length, [50,200], [minVol, maxVol]) #interpolate the length of the line between the tip of the index finger and the tip of the middle finger to the volume range
        volBar = np.interp(length, [50,250], [400, 120]) #interpolate the length of the line between the tip of the index finger and the tip of the middle finger to the volume bar range
        volPer = np.interp(length, [50,200], [0, 100]) #interpolate the length of the line between the tip of the index finger and the tip of the middle finger to the volume percentage range

        print(f'length: {int(length)}, volume: {int(volPer)}')
        volume.SetMasterVolumeLevel(vol, None)

        if length<50:
            cv.circle(img, (cx,cy), 15, (0, 255, 0), cv.FILLED)
            cv.putText(img,f' Muted', (10, 130), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        if length > 50 and length < 200:
           cv.putText(img,f' {int(volPer)}', (20, 130), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3) #print the volume percentage on the screen

        #adding a volume bar
        cv.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)

        # to make the volume bar move smoothly, we use cv.FILLED and to keep the volume bar in the range of 50 to 250, we use if statement
        if length < 200:
         cv.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv.FILLED)
        else:
          cv.rectangle(img, (50, 150), (85, 400), (255, 0, 0), cv.FILLED)
          cv.putText(img,f' Max Volume', (10, 130), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)



    #fps calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img,f'FPS: {int(fps)}', (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)



# code starts for object detection-------------------------------------------------------------------------------------------------------------

    #ret, frame = cap.read()
    bbox, label, conf = cvl.detect_common_objects(img)
    output_image = draw_bbox(img, bbox, label, conf)

    cv.imshow("Object and Hand Gesture  Voulme Control", output_image)  # This line is added to show the object detection and hand gesture volume control in one window

    new_objects = set(label) - detected_objects  # Calculate new detected objects
    for obj_label in new_objects:
        print("New object detected:", obj_label)
        obj_label = "New object detected: "+obj_label
        language = "en"
        output = gTTS(text=obj_label, lang=language, slow=False)
        output.save("./sounds/output.mp3")

        pygame.init()

        # Load the MP3 file
        mp3_file = "./sounds/output.mp3"
        pygame.mixer.music.load(mp3_file)

       # Play the MP3
        pygame.mixer.music.play()

        # Allow time for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.delay(100)

        # Clean up resources
        pygame.quit()



    detected_objects.update(new_objects)  # Update the set of detected objects










   # cv.imshow("Object and Hand Gesture  Voulme Control", output_image)
    if cv.waitKey(1) == ord('q'):
        break



cap.release()
cv.destroyAllWindows()















 Binary file addedBIN +2.15 KB 
Main(Final)/__pycache__/HandTrackingModule.cpython-39.pyc
Binary file not shown.
 77 changes: 77 additions & 0 deletions77  
Object_Detection/Object_detection.py
@@ -0,0 +1,77 @@
import os
import cv2 as cv
import cvlib as cvl
from cvlib.object_detection import draw_bbox
from gtts import gTTS
import pygame
import time

# Initialize previous time (pTime) and current time (cTime) for FPS calculation
pTime = 0
cTime = 0

# Open the default camera (camera index 0)
video = cv.VideoCapture(0)

# Set to store the names of detected objects
detected_objects = set()

# Infinite loop for real-time object detection
while True:
    # Read a frame from the video feed
    ret, frame = video.read()
    frame = cv.flip(frame, 1)

    # Detect common objects in the frame using cvlib
    bbox, label, conf = cvl.detect_common_objects(frame)

    # Draw bounding boxes and labels on the frame
    output_image = draw_bbox(frame, bbox, label, conf)

    # Calculate FPS (Frames Per Second)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display the FPS on the frame
    cv.putText(frame, f'FPS: {int(fps)}', (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Show the frame with object detection results
    cv.imshow("Real-time object detection", output_image)

    # Calculate new detected objects
    new_objects = set(label) - detected_objects

    # Process new detected objects
    for obj_label in new_objects:
        print("New object detected:", obj_label)

        # Convert object label to speech and save as an MP3 file
        obj_label = "New object detected: " + obj_label
        language = "en"
        output = gTTS(text=obj_label, lang=language, slow=False)
        output.save("./sounds/output.mp3")

        # Initialize and play the MP3 audio using Pygame
        pygame.init()
        mp3_file = "./sounds/output.mp3"
        pygame.mixer.music.load(mp3_file)
        pygame.mixer.music.play()

        # Allow time for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.delay(100)

        # Clean up resources
        pygame.quit()

    # Update the set of detected objects
    detected_objects.update(new_objects)

    # Exit the loop if 'q' key is pressed
    if cv.waitKey(1) == ord('q'):
        break

# Release the video feed and close all windows
video.release()
cv.destroyAllWindows()
